
#include "mkl_lapacke.h"

#include <ginkgo/ginkgo.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "netcdf.h"
#include <cuda.h>


using precision = double;
using real_precision = gko::remove_complex<precision>;
using vec = gko::matrix::Dense<precision>;
using real_vec = gko::matrix::Dense<real_precision>;
using mtx = gko::matrix::Csr<real_precision>;
using VecPtr = std::shared_ptr<vec>;

/* -----------------------------------------------------------------------------
 * NetCDF error reporting.
 *
 * Kept from the original (which used this same handle_error() for its
 * NetCDF state-file reads). It only prints; it does not exit, exactly as
 * before.
 * ---------------------------------------------------------------------------*/
static void
handle_error (int status)
{
  if (status != NC_NOERR)
    printf ("%s\n", nc_strerror (status));
}

/* -----------------------------------------------------------------------------
 * Lanczos-based spectral bounds via small tridiagonal diagonalization.
 * Given a Hermitian A, produce upper and lower bounds on lambda_min and
 * lambda_max. 
 * ---------------------------------------------------------------------------*/
struct LanczosSpecBounds
{
  double mu_min = 0.0, mu_max = 0.0, beta_k1 = 0.0;
  double lam_min_lower = 0.0, lam_min_upper = 0.0;
  double lam_max_lower = 0.0, lam_max_upper = 0.0;
};

static LanczosSpecBounds
lanczos_bounds (std::shared_ptr<gko::matrix::Csr<real_precision> > A, int k_max, double tol, std::shared_ptr<gko::Executor> exec, std::shared_ptr<gko::Executor> host_exec)
{
  // get size
  const auto n = A->get_size ()[0];
  LanczosSpecBounds out;

  // --- v0: random, normalized (host -> device) ---
  auto v = vec::create (exec, gko::dim<2>{ n, 1 });
  auto vm = vec::create (exec, gko::dim<2>{ n, 1 });
  auto t = vec::create (exec, gko::dim<2>{ n, 1 });
  auto w = vec::create (exec, gko::dim<2>{ n, 1 });

  // initial random vector and normalize
  auto v_h = vec::create (host_exec, gko::dim<2>{ n, 1 });
  {
    // tjc This is not a random vector since the same seed is being used every time
    std::mt19937 gen (42);
    std::normal_distribution<double> N01 (0.0, 1.0);
    for (gko::size_type i = 0; i < n; ++i)
      v_h->at (i, 0) = static_cast<precision> (N01 (gen)); // real or complex
    auto nh = real_vec::create (host_exec, gko::dim<2>{ 1, 1 });
    v_h->compute_norm2 (nh);
    const double inv = nh->at (0, 0) > 0.0 ? 1.0 / nh->at (0, 0) : 1.0;
    v_h->scale (gko::initialize<vec> ({ static_cast<precision> (inv) }, host_exec));
  }

  // initial
  v->copy_from (v_h.get ());
  vm->fill (static_cast<precision> (0.0));

  // store diagonals and off diagonals, for tridiagonal
  std::vector<double> alphas;
  alphas.reserve (k_max);
  std::vector<double> betas;
  betas.reserve (k_max - 1);

  // scalars
  auto dot_c = vec::create (exec, gko::dim<2>{ 1, 1 });
  auto rnorm = real_vec::create (exec, gko::dim<2>{ 1, 1 });

  double beta_i = 0.0;
  int k_eff = 0;

  // for number of k's
  for (int i = 0; i < k_max; ++i)
    {
      A->apply (v, t); // Av

      v->compute_conj_dot (t, dot_c); // v*Av Rayleigh q on current vector
      const double alpha_i = std::real (gko::clone (host_exec, dot_c)->at (0, 0));
      // store diagonal
      alphas.push_back (alpha_i);

      w->copy_from (t.get ()); // Av

      // w = Av-alphav-betav-1
      if (i > 0)
        w->add_scaled (gko::initialize<vec> ({ static_cast<precision> (-beta_i) }, exec), vm.get ());
      w->add_scaled (gko::initialize<vec> ({ static_cast<precision> (-alpha_i) }, exec), v.get ());

      // offdiagonal coefficient
      w->compute_norm2 (rnorm);
      const double beta_ip1 = gko::clone (host_exec, rnorm)->at (0, 0);

      // safety if krylov grows
      if (beta_ip1 < tol)
        {
          out.beta_k1 = 0.0;
          k_eff = i + 1;
          break;
        }
      // stores Bi+1
      if (i < k_max - 1)
        betas.push_back (beta_ip1);
      out.beta_k1 = beta_ip1;
      // normalize and iterate
      vm->copy_from (v.get ());
      v->copy_from (w.get ());
      const double inv = beta_ip1 > 0.0 ? 1.0 / beta_ip1 : 1.0;
      v->scale (gko::initialize<vec> ({ static_cast<precision> (inv) }, exec));
      // end of step
      beta_i = beta_ip1;
      k_eff = i + 1;
    }


  // gather diagonals and off diagonals
  const int m = k_eff;
  std::vector<double> d (alphas.begin (), alphas.begin () + m);
  std::vector<double> e;
  if (m >= 2)
    e.assign (betas.begin (), betas.begin () + (m - 1));

  // Diagonalize T matrix size K x K
  std::vector<double> Z (m * m, 0.0);
  int info = LAPACKE_dstev (LAPACK_COL_MAJOR, 'V', m, d.data (), (m >= 2 ? e.data () : nullptr),
                            Z.data (), m);
  if (info != 0)
    std::cerr << "[Lanczos] dstev failed, info=" << info << "\n";

  // store extremal Ritz
  out.mu_min = d.front ();
  out.mu_max = d.back ();
  out.lam_max_lower = out.mu_max;
  out.lam_min_upper = out.mu_min;

  // calculating magnitude of ritz vector
  auto last_row_mag = [&] (int j) { return std::abs (Z[(m - 1) + j * m]); };
  // find eigenvector
  int j_max = 0;
  for (int j = 1; j < m; ++j)
    if (d[j] > d[j_max])
      j_max = j;

  // safeguards
  double max_all = 0.0;
  for (int j = 0; j < m; ++j)
    max_all = std::max (max_all, last_row_mag (j));
  // this creates a conservative guess for max
  const double bnd2_up = out.mu_max + last_row_mag (j_max) * out.beta_k1;
  const double bnd7_up = out.mu_max + max_all * out.beta_k1;
  // keep m 3
  double bnd4_up = bnd7_up;
  if (m >= 3)
    {
      double max_last3 = 0.0;
      for (int j = m - 3; j < m; ++j)
        max_last3 = std::max (max_last3, last_row_mag (j));
      bnd4_up = out.mu_max + max_last3 * out.beta_k1;
    }
  // safeguard for tiny scales
  const double scale_k = last_row_mag (j_max) * out.beta_k1;
  const double alg1_up = (scale_k < 1e-12 && m >= 3) ? bnd4_up : 0.5 * (bnd2_up + bnd7_up);

  // safeguard
  double temple_up = std::numeric_limits<double>::infinity ();
  double temple_lo = -std::numeric_limits<double>::infinity ();

  // ensure nonzero gap between ritz values
  if (m >= 2)
    {
      const double gap_up = d[m - 1] - d[m - 2];
      const double gap_lo = d[1] - d[0];
      if (gap_up > 0.0)
        {
          const double r_top = scale_k;
          temple_up = d[m - 1] + (r_top * r_top) / gap_up;
        }
      if (gap_lo > 0.0)
        {
          const double r_bot = out.beta_k1 * last_row_mag (0);
          temple_lo = d[0] - (r_bot * r_bot) / gap_lo;
        }
    }

  // picking upper and lower
  out.lam_max_upper = std::min (alg1_up, temple_up);
  out.lam_min_lower = std::max (out.mu_min - max_all * out.beta_k1, temple_lo);

  // if things failed
  if (info != 0)
    {
      out.lam_max_upper = out.mu_max + out.beta_k1;
      out.lam_min_lower = out.mu_min - out.beta_k1;
    }
  return out;
}

/* -----------------------------------------------------------------------------
 * print_mem: reports free/used/total CUDA device memory for the given GPU.
 * ---------------------------------------------------------------------------*/
static void
print_mem (const std::string &label, int gpuID)
{
  size_t free, total;
  CUdevice dev;
  cuDeviceGet (&dev, gpuID);
  cuMemGetInfo (&free, &total);
  free /= (1024 * 1024 * 1024);
  total /= (1024 * 1024 * 1024);
  std::cout << "[MEM] " << label
            << ": total=" << total << " GB"
            << "  free=" << free << " GB"
            << "  used=" << (total - free) << " GB\n";
}

/* -----------------------------------------------------------------------------
 * Step 2: Chebyshev moments of the rescaled matrix.
 *
 * mu_n = <r0 | T_n(A~) | r0>, averaged over num_random_vecs random start
 * vectors, via the three-term recurrence r_{n+1} = 2*A~*r_n - r_{n-1}
 * (eqs. 30-32, Rev. Mod. Phys. 78, 275 (2006)).
 * ---------------------------------------------------------------------------*/
static std::vector<double>
compute_chebyshev_moments (std::shared_ptr<gko::LinOp> scaled_matrix, gko::size_type N, int num_moments, int num_random_vecs, std::shared_ptr<gko::Executor> exec, std::shared_ptr<gko::Executor> this_exec)
{
  std::vector<double> moments (num_moments, 0.0);

  for (int r = 0; r < num_random_vecs; ++r)
    {
      std::normal_distribution<> dist (0.0, 1.0);
      // tjc not random
      std::mt19937 gen (42);
      // initializes random vector r_0

      auto r0_h = vec::create (this_exec, gko::dim<2>{ N, 1 });
      for (gko::size_type i = 0; i < N; ++i)
        {
          r0_h->at (i, 0) = dist (gen);
        }

      // normalize on host
      auto nh = real_vec::create (this_exec, gko::dim<2>{ 1, 1 });
      r0_h->compute_norm2 (nh);
      auto inv_norm_scalar = precision{ 1.0 / nh->at (0, 0) };
      r0_h->scale (gko::initialize<vec> ({ inv_norm_scalar }, this_exec));

      // send to device
      auto r0 = clone (exec, r0_h);

      // equations 30-32 https://doi.org/10.1103/RevModPhys.78.275, a simplification of 29
      // initialize r1 = A r_0, r_prev = r_0, r_curr = r_1
      auto r_prev = r0->clone ();
      auto r_curr = vec::create (exec, gko::dim<2>{ N, 1 });
      scaled_matrix->apply (r_prev, r_curr);

      auto mu0 = vec::create (exec, gko::dim<2>{ 1, 1 });
      //<r_0|r_0>
      r0->compute_dot (r0, mu0);
      moments[0] += gko::clone (this_exec, mu0)->at (0, 0);
      //<r_0|Ar_0>
      auto mu1 = vec::create (exec, gko::dim<2>{ 1, 1 });
      r0->compute_dot (r_curr, mu1);
      moments[1] += gko::clone (this_exec, mu1)->at (0, 0);

      // recurrence via formula r_n+1 = 2*A*r_n - r_n-1
      for (int n = 2; n < num_moments; ++n)
        {
          auto r_next = vec::create (exec, gko::dim<2>{ N, 1 });
          scaled_matrix->apply (r_curr, r_next);
          r_next->scale (gko::initialize<vec> ({ 2.0 }, exec));
          r_next->add_scaled (gko::initialize<vec> ({ -1.0 }, exec), r_prev);

          auto mu_n = vec::create (exec, gko::dim<2>{ 1, 1 });
          r0->compute_dot (r_next, mu_n);
          moments[n] += gko::clone (this_exec, mu_n)->at (0, 0);

          // next iteration
          r_prev = std::move (r_curr);
          r_curr = std::move (r_next);
        }
    }

  // normalize moments
  for (auto &mu : moments)
    {
      mu /= static_cast<double> (num_random_vecs);
    }
  return moments;
}

/* -----------------------------------------------------------------------------
 * Step 4: Jackson kernel -- damps the Gibbs oscillations of the truncated
 * Chebyshev series (eq. 71, Rev. Mod. Phys. 78, 275 (2006)).
 * ---------------------------------------------------------------------------*/
static std::vector<double>
apply_jackson_kernel (const std::vector<double> &moments)
{
  const int num_moments = static_cast<int> (moments.size ());
  std::vector<double> filtered_moments (num_moments);
  const double cot_pi_over_Np1 = 1.0 / std::tan (M_PI / (num_moments + 1));
  for (int n = 0; n < num_moments; ++n)
    {
      double theta = M_PI * n / (num_moments + 1);
      double g_n = ((num_moments - n + 1) * std::cos (theta) + std::sin (theta) * cot_pi_over_Np1) / (num_moments + 1);
      filtered_moments[n] = g_n * moments[n];
    }
  return filtered_moments;
}

/* -----------------------------------------------------------------------------
 * Step 5: spectral density reconstruction (eq. 18, Rev. Mod. Phys. 78, 275
 * (2006), kernel already applied to mu_n), rescaled back to the physical
 * energy axis E = a*x + b, then normalized so the grid values sum to 1
 *
 * energies/rho are output parameters, filled in place.
 * ---------------------------------------------------------------------------*/
static void
reconstruct_dos (const std::vector<double> &filtered_moments, int num_points, double a, double b, std::vector<double> &energies, std::vector<double> &rho)
{
  const int num_moments = static_cast<int> (filtered_moments.size ());
  energies.assign (num_points, 0.0);
  std::vector<double> rhos (num_points);

  for (int i = 0; i < num_points; ++i)
    {
      double E_scaled = -1.0 + 2.0 * i / static_cast<double> (num_points - 1);
      // reconstruct spectral density
      double rho_val = filtered_moments[0];
      for (int n = 1; n < num_moments; ++n)
        {
          rho_val += 2.0 * filtered_moments[n] * std::cos (n * std::acos (E_scaled));
        }
      rho_val /= (M_PI * std::sqrt (1.0 - E_scaled * E_scaled));
      // rescaling to the original scale
      double E_actual = a * E_scaled + b;

      energies[i] = E_actual;
      rhos[i] = rho_val;
    }

  std::vector<double> rhoE (num_points);
  for (int i = 0; i < num_points; ++i)
    {
      rhoE[i] = std::isfinite (rhos[i]) ? (rhos[i] / a) : 0.0; // converts back to actual energies from [-1,1]
    }

  rho.assign (num_points, 0.0);
  double sum_rho = 0.0;
  for (double v : rhoE)
    if (std::isfinite (v))
      sum_rho += v;

  double sum_p = 0.0;
  for (int i = 0; i < num_points; ++i)
    {
      double p_i = (sum_rho > 0.0 && std::isfinite (rhoE[i])) ? (rhoE[i] / sum_rho) : 0.0;
      rho[i] = p_i;
      sum_p += p_i;
    }
  std::cout << "Sum of all p_i values: " << sum_p << std::endl;
}

/* -----------------------------------------------------------------------------
 * Trapezoid-rule integral of the (piecewise-linear) DOS over [lo, hi].
 * ---------------------------------------------------------------------------*/
static double
integrate_rho (const std::vector<double> &E_scaled, const std::vector<double> &rho, double lo, double hi)
{
  double I = 0.0;
  for (size_t i = 0; i + 1 < E_scaled.size (); ++i)
    {
      double x1 = E_scaled[i], x2 = E_scaled[i + 1];
      if (x2 <= lo || x1 >= hi)
        continue;
      double xa = std::max (lo, x1), xb = std::min (hi, x2);
      if (xb <= xa)
        continue;
      double r1 = rho[i], r2 = rho[i + 1];
      // trapezoid
      double rxa = r1 + (r2 - r1) * (xa - x1) / (x2 - x1);
      double rxb = r1 + (r2 - r1) * (xb - x1) / (x2 - x1);
      I += 0.5 * (rxa + rxb) * (xb - xa);
    }
  return I;
}

/* -----------------------------------------------------------------------------
 * Expected number of eigenvalues in [center - half_width, center + half_width].
 * ---------------------------------------------------------------------------*/
static double
expected_eigs_window (const std::vector<double> &E_scaled, const std::vector<double> &rho, double center, double half_width, double full_lo, double full_hi, double I_full, double N_total)
{
  double a = std::max (full_lo, center - half_width);
  double b = std::min (full_hi, center + half_width);
  double I_T = integrate_rho (E_scaled, rho, a, b);
  return N_total * (I_T / I_full);
}

/* -----------------------------------------------------------------------------
 * Bisection on the half-width so the window centered at `center` is
 * expected to contain `target_count` eigenvalues.
 * ---------------------------------------------------------------------------*/
static double
bisect_half_width (const std::vector<double> &E_scaled, const std::vector<double> &rho, double center, double target_count, double full_lo, double full_hi, double I_full, double N_total, int num_iters)
{
  double lo = 0.0;
  double hi = 0.5 * (full_hi - full_lo);
  for (int it = 0; it < num_iters; ++it)
    {
      double mid = 0.5 * (lo + hi);
      double n_mid = expected_eigs_window (E_scaled, rho, center, mid, full_lo, full_hi, I_full, N_total);
      if (n_mid > target_count)
        hi = mid; // window includes too many eigenvalues -> shrink
      else
        lo = mid; // window too small -> grow
    }
  return 0.5 * (lo + hi);
}

/* -----------------------------------------------------------------------------
 * clamp11: clamp x to [-1, 1]. 
 * ---------------------------------------------------------------------------*/
static double
clamp11 (double x)
{
  return std::max (-1.0, std::min (1.0, x));
}

/* -----------------------------------------------------------------------------
 * Step 6a: Chebyshev window-filter coefficients w_n = g_n * c_win_n, where
 * c_win_n are the Chebyshev coefficients of the ideal window indicator on
 * [lower_use, upper_use] and g_n is the Lanczos sigma kernel (mu = 2).
 * ---------------------------------------------------------------------------*/
static std::vector<double>
build_window_coefficients (double lower_use, double upper_use, double alpha, double beta, int NP)
{
  std::vector<double> g (NP + 1, 1.0);
  for (int n = 1; n <= NP; ++n)
    {
      double x = M_PI * n / (NP + 1);
      double s = std::sin (x) / x; // sinc
      g[n] = s * s;                // mu = 2
    }

  double u = clamp11 (alpha * lower_use + beta);
  double v = clamp11 (alpha * upper_use + beta);
  if (u > v)
    std::swap (u, v);

  std::vector<double> c_win (NP + 1, 0.0);
  auto th_u = std::acos (u);
  auto th_v = std::acos (v);

  c_win[0] = (th_u - th_v) / M_PI;
  for (int ncoef = 1; ncoef <= NP; ++ncoef)
    {
      c_win[ncoef] = (2.0 / (M_PI * ncoef)) * (std::sin (ncoef * th_v) - std::sin (ncoef * th_u));
    }

  std::vector<double> w (NP + 1, 0.0);
  for (int ncoef = 0; ncoef <= NP; ++ncoef)
    w[ncoef] = g[ncoef] * c_win[ncoef];
  return w;
}

/* -----------------------------------------------------------------------------
 * print_norm: prints ||v|| for a device vector. Only used from the DEBUG==1
 * blocks inside chebyshev_filter_pass, exactly as in the original.
 * ---------------------------------------------------------------------------*/
static void
print_norm (const std::string &label, VecPtr v, std::shared_ptr<gko::Executor> exec, std::shared_ptr<gko::Executor> this_exec)
{
  auto nrm = real_vec::create (exec, gko::dim<2>{ 1, 1 });
  v->compute_norm2 (nrm);
  double val = gko::clone (this_exec, nrm)->at (0, 0);
  std::cout << label << " ||v|| = " << std::setprecision (12) << val << "\n";
}

/* -----------------------------------------------------------------------------
 * Step 6b: Chebyshev filter pass.
 *
 * For every search vector x in X, accumulate y = sum_j w_j T_j(A~) x via the
 * three-term recurrence T_j = 2 A~ T_{j-1} - T_{j-2}, normalize y, and write
 * it back over X[k]. 
 * ---------------------------------------------------------------------------*/
static void
chebyshev_filter_pass (std::vector<VecPtr> &X, std::shared_ptr<gko::LinOp> scaled_matrix, const std::vector<double> &w, gko::size_type N, std::shared_ptr<gko::Executor> exec, std::shared_ptr<gko::Executor> this_exec)
{
  const int NP = static_cast<int> (w.size ()) - 1;
  const int NSloc = (int) X.size ();
  const gko::size_type nloc = N;

  // move w to the GPU (dimension NP + 1)
  auto wV_local = vec::create (this_exec, gko::dim<2>{ static_cast<gko::size_type> (NP + 1), 1 });
  for (int i = 0; i <= NP; i++)
    wV_local->at (i, 0) = w[i];
  auto wV = gko::share (clone (exec, wV_local));

  // reusable for (w0, w1, w2, w[j])
  auto two = gko::share (gko::initialize<vec> ({ precision{ 2.0 } }, exec));
  auto minus_one = gko::share (gko::initialize<vec> ({ precision{ -1.0 } }, exec));
  auto coeff = gko::share (gko::initialize<vec> ({ precision{ 0.0 } }, exec));

  bool DEBUG_FILTER = true;

  auto t1 = gko::share (vec::create (exec, gko::dim<2>{ nloc, 1 }));
  auto t2 = gko::share (vec::create (exec, gko::dim<2>{ nloc, 1 }));
  auto y = gko::share (vec::create (exec, gko::dim<2>{ nloc, 1 }));

  for (int k = 0; k < NSloc; ++k)
    {
      if (k == 0 || k == NSloc - 1)
        {
          std::cout << "----------------------" << std::endl;
        }
      if (k % 50 == 0)
        {
          std::cout << " we are on vector " << k << "/" << NSloc << std::endl;
        }

      // t1 = T1(A~) X[k] = A~ X[k]
      scaled_matrix->apply (X[k], t1);
      // t2 = T2(A~) X[k] = 2 A~ t1 - X[k]
      scaled_matrix->apply (t1, t2);
      t2->scale (two);
      t2->add_scaled (minus_one, X[k]);

      // y = w0*T0 + w1*T1 + w2*T2
      gko::span rowSpan (0, 1);
      gko::span colSpan (0, 1);
      auto w0_h = wV->create_submatrix (rowSpan, colSpan);
      y->copy_from (X[k]);
      y->scale (w0_h);

      gko::span rowSpan1 (1, 2);
      auto w1_h = wV->create_submatrix (rowSpan1, colSpan);
      y->add_scaled (w1_h, t1);

      gko::span rowSpan2 (2, 3);
      auto w2_h = wV->create_submatrix (rowSpan2, colSpan);
      y->add_scaled (w2_h, t2);

      // Recurrence: T_j from j=3..NP; accumulate y += w[j]*T_j
      auto tjm2 = t1; // T_{j-2}
      auto tjm1 = t2; // T_{j-1}
      auto tj = gko::share (vec::create (exec, gko::dim<2>{ nloc, 1 }));

      for (int j = 3; j <= NP; ++j)
        {
          // tj = 2 A~ tjm1 - tjm2
          scaled_matrix->apply (tjm1, tj);
          tj->scale (two);
          tj->add_scaled (minus_one, tjm2);

          gko::span rowSpanj (j, j + 1);
          auto wj_h = wV->create_submatrix (rowSpanj, colSpan);
          y->add_scaled (wj_h, tj);

          // shift window: T_{j-2} <- T_{j-1}, T_{j-1} <- T_j
          tjm2.swap (tjm1);
          tjm1.swap (tj);


        }

      // Normalize y on the device
      auto y_norm = real_vec::create (exec, gko::dim<2>{ 1, 1 });
      y->compute_norm2 (y_norm);
      double yn = gko::clone (this_exec, y_norm)->at (0, 0);

      auto inv_h = gko::initialize<vec> ({ precision{ (yn > 0.0) ? 1.0 / yn : 1.0 } }, this_exec);
      coeff->copy_from (inv_h.get ());
      y->scale (coeff.get ());

      // Write back: X[k] = y
      X[k] = gko::share (clone (exec, y));
    }
}

/* -----------------------------------------------------------------------------
 * Step 7: SVQB block orthonormalization.
 *   (1) S = W^T W
 *   (2) D = diag(S)^{1/2},  D^{-1} = diag(S)^{-1/2}
 *   (3) S = D^{-1} S D^{-1}
 *   (4) S = Q /\ Q^T                     (eigendecomposition on host, LAPACK)
 *   (5) T = D^{-1} Q /\^{-1/2}           (scale by inverse sqrt of eigenvalues)
 *   (6) Y = W T                         (columns of Y are orthonormal)
 * ---------------------------------------------------------------------------*/
static int
svqb (std::vector<VecPtr> &X, gko::size_type N, std::shared_ptr<gko::Executor> exec, std::shared_ptr<gko::Executor> this_exec, int gpuID)
{
  const int nrows = static_cast<int> (N);
  const int r_blk = static_cast<int> (X.size ());
  print_mem ("start of svqb ", gpuID);
  auto W_real = real_vec::create (this_exec, gko::dim<2>{ nrows, r_blk });

  // Fill W from device vectors X[k]
  for (int k = 0; k < r_blk; ++k)
    {
      auto xh = gko::clone (this_exec, X[k]); // device -> host
      for (int i = 0; i < nrows; ++i)
        {
          W_real->at (i, k) = static_cast<double> (std::real (xh->at (i, 0)));
        }
    }

  auto Wt = real_vec::create (this_exec, gko::dim<2>{ r_blk, nrows }); // W^T
  W_real->conj_transpose (Wt);

  auto S_real = real_vec::create (this_exec, gko::dim<2>{ r_blk, r_blk }); // (1) S = W^T W
  Wt->apply (W_real, S_real);

  // (2) D^{-1} from diagonal of S
  std::vector<double> Dinv (r_blk, 1.0);
  for (int i = 0; i < r_blk; ++i)
    {
      double sii = std::max (1e-300, static_cast<double> (S_real->at (i, i)));
      Dinv[i] = 1.0 / std::sqrt (sii);
    }

  // (3) S = D^{-1} S D^{-1}  (host)
  std::vector<double> Shat (r_blk * r_blk);
  for (int j = 0; j < r_blk; ++j)
    for (int i = 0; i < r_blk; ++i)
      Shat[i + j * r_blk] = Dinv[i] * static_cast<double> (S_real->at (i, j)) * Dinv[j];

  std::vector<double> evals (r_blk);
  int info = LAPACKE_dsyevd (LAPACK_COL_MAJOR, 'V', 'U',
                             r_blk, Shat.data (), r_blk, evals.data ()); // (4) S = Q /\ Q^T

  if (info != 0)
    {
      std::cerr << "[SVQB] dsyevd failed, info=" << info << "\n";
      return 0;
    }

  // tolerance so it's not malformed at all
  double lam_max = 0.0;
  for (int j = 0; j < r_blk; ++j)
    lam_max = std::max (lam_max, std::abs (evals[j]));
  double tol = lam_max * 1e-18;

  std::vector<int> keep;
  keep.reserve (r_blk);

  for (int j = 0; j < r_blk; ++j)
    {
      double lj = evals[j]; // raw eigenvalue from dsyevd
      double abs_lj = std::abs (lj);

      // symmetric test: drop only if |lambda| is truly tiny
      if (abs_lj > tol)
        {
          double scl = 1.0 / std::sqrt (abs_lj); // scale by |lambda|^{-1/2}
          for (int i = 0; i < r_blk; ++i)
            Shat[i + j * r_blk] *= scl; // Q lambda^{-1/2}
          keep.push_back (j);
        }
    }

  int r = static_cast<int> (keep.size ());
  if (r == 0)
    return 0;

  auto T_real = real_vec::create (this_exec, gko::dim<2>{ r_blk, r }); // T = D^{-1} Q .\^{-1/2}
  for (int jj = 0; jj < r; ++jj)
    {
      int j = keep[jj];
      for (int i = 0; i < r_blk; ++i)
        T_real->at (i, jj) = Dinv[i] * Shat[i + j * r_blk];
    }

  auto Y_real = real_vec::create (this_exec, gko::dim<2>{ nrows, r }); // (6) Y = W T
  W_real->apply (T_real, Y_real);

  // Clone Y columns back to device and replace X
  std::vector<VecPtr> X_new (r);
  for (int k = 0; k < r; ++k)
    {
      auto col_h = vec::create (this_exec, gko::dim<2>{ nrows, 1 });
      for (int i = 0; i < nrows; ++i)
        col_h->at (i, 0) = real_precision{ static_cast<double> (Y_real->at (i, k)) };
      X_new[k] = gko::share (gko::clone (exec, col_h)); // host -> device
    }
  X = std::move (X_new);
  print_mem ("after svqb", gpuID);
  return r;
}

/* -----------------------------------------------------------------------------
 * Step 8: Rayleigh-Ritz extraction and residuals.
 *   Y = [X], AY = A*Y, H = sym(Y^T A Y), eig(H) = U theta U^T.
 *   Ritz vectors v_j = Y u_j; residuals r_j = ||A v_j - theta_j v_j||_2.
 * ---------------------------------------------------------------------------*/
struct RitzResult
{
  std::vector<VecPtr> v;   // Ritz vectors
  std::vector<double> lam; // Ritz values theta_j (host)
  std::vector<double> rn;  // residual norms ||A v_j - theta_j v_j||_2
};

static RitzResult
rayleigh_ritz (const std::vector<VecPtr> &X, std::shared_ptr<mtx> A, gko::size_type N, std::shared_ptr<gko::Executor> exec, std::shared_ptr<gko::Executor> this_exec, int gpuID)
{
  RitzResult R;
  const int r = (int) X.size ();
  const gko::size_type nloc = N;
  print_mem ("begin RR", gpuID);

  auto Y = gko::share (vec::create (exec, gko::dim<2>{ nloc, r })); // pack X columns into dense matrix Y
  for (int j = 0; j < r; ++j)
    {
      auto Yj = Y->create_submatrix (gko::span{ 0, nloc }, gko::span{ static_cast<gko::size_type> (j), static_cast<gko::size_type> (j + 1) });
      Yj->copy_from (X[j].get ());
    }
  auto AY = gko::share (vec::create (exec, gko::dim<2>{ nloc, r })); // AY = A*Y
  A->apply (Y, AY);

  auto YH = gko::share (vec::create (exec, gko::dim<2>{ r, nloc })); // Y^H
  Y->conj_transpose (YH);

  auto Hc = gko::share (vec::create (exec, gko::dim<2>{ r, r })); // Hc = Y^H * A * Y
  YH->apply (AY, Hc);

  auto Hc_h = gko::clone (this_exec, Hc); // host copy
  std::vector<double> H (r * r);

  for (int j = 0; j < r; ++j)
    {
      for (int i = 0; i < r; ++i)
        {
          const auto hij = Hc_h->at (i, j);
          const auto hji = Hc_h->at (j, i);
          H[i + j * r] = 0.5 * (std::real (hij) + std::real (hji));
        }
    }

  std::vector<double> theta (r, 0.0); // eigenvalues (Ritz values)
  int info_H = LAPACKE_dsyev (LAPACK_COL_MAJOR, 'V', 'U', r, H.data (), r, theta.data ());

  if (info_H != 0)
    {
      std::cerr << "[RR] dsyev failed, info=" << info_H << "\n";
      return R;
    }

  // Build Ritz vectors V = Y * U (device), where columns of U are eigenvectors from dsyev
  auto U_h = gko::matrix::Dense<precision>::create (this_exec, gko::dim<2>{ r, r });
  for (int j = 0; j < r; ++j)
    for (int i = 0; i < r; ++i)
      U_h->at (i, j) = real_precision{ H[i + j * r] }; // copy U (real) into complex container

  auto U = gko::clone (exec, U_h); // device r x r
  auto V = gko::matrix::Dense<precision>::create (exec, gko::dim<2>{ nloc, r });
  Y->apply (U, V); // V = Y * U  (device)

  auto Av = gko::matrix::Dense<precision>::create (exec, gko::dim<2>{ nloc, r });
  AY->apply (U, Av);

  // residuals: ||Av(:,j) - theta[j] * V(:,j)||_2
  R.v.resize (r);
  R.rn.resize (r);
  for (int j = 0; j < r; ++j)
    {
      auto Vj = V->create_submatrix (gko::span{ 0, nloc }, gko::span{ static_cast<gko::size_type> (j), static_cast<gko::size_type> (j + 1) });
      auto Avj = Av->create_submatrix (gko::span{ 0, nloc }, gko::span{ static_cast<gko::size_type> (j), static_cast<gko::size_type> (j + 1) });
      auto tmp = gko::matrix::Dense<precision>::create (exec, gko::dim<2>{ nloc, 1 });
      tmp->copy_from (Avj.get ()); // tmp = Av(:,j)
      auto neglam = gko::initialize<gko::matrix::Dense<precision> > ({ real_precision{ -theta[j] } }, exec);
      tmp->add_scaled (neglam.get (), Vj.get ()); // tmp = Av(:,j) - theta_j V(:,j)

      auto rn = gko::matrix::Dense<real_precision>::create (exec, gko::dim<2>{ 1, 1 });
      tmp->compute_norm2 (rn);
      R.rn[j] = gko::clone (this_exec, rn)->at (0, 0); // scalar back to host

      auto vj = gko::matrix::Dense<precision>::create (exec, gko::dim<2>{ nloc, 1 });
      vj->copy_from (Vj.get ()); // save Ritz vector
      R.v[j] = gko::share (std::move (vj));
    }

  R.lam.assign (theta.begin (), theta.end ());
  print_mem ("end of RR", gpuID);
  return R;
}

/* -----------------------------------------------------------------------------
 * Step 9: an accepted, converged, in-window Ritz pair.
 * ---------------------------------------------------------------------------*/
struct AcceptedPair
{
  double lam;
  double rn;
  std::vector<double> vec;
};

/* -----------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------*/
int
main (int argc, char *argv[])
{
  using namespace std::chrono;

  if (argc < 9)
    {
      std::cerr << "Usage: " << argv[0]
                << " <matrix.mtx> <gpu_id> <NP> <NT> <max_outer> <target_frac> <mc_target> <ns_max>\n";
      return EXIT_FAILURE;
    }

  int c = 1;
  char *matFile = argv[c++];
  int gpuID = atoi (argv[c++]);
  int NP = atoi (argv[c++]);
  int NT = atoi (argv[c++]);
  int max_outer = atoi (argv[c++]);
  double target_frac = atof (argv[c++]);
  int mc_target = atoi (argv[c++]);
  int ns_max = atoi (argv[c++]);

  std::cout << gko::version_info::get () << std::endl;
  std::cout << std::scientific << std::setprecision (8) << std::showpos;

  /************** Create the CUDA executor ************************/
  int device_id = gpuID;
  auto exec = gko::CudaExecutor::create (device_id, gko::ReferenceExecutor::create ());
  auto this_exec = exec->get_master ();

  /********** Load the matrix from a MatrixMarket / plain-text file *********/
  std::ifstream mat_stream (matFile);
  if (!mat_stream)
    {
      std::cerr << "Cannot open matrix file: " << matFile << "\n";
      return EXIT_FAILURE;
    }
  auto A = share (gko::read<mtx> (mat_stream, exec));
  const gko::int64 N = A->get_size ()[0];
  std::cout << "Loaded " << matFile << ": N = " << N
            << ", nnz = " << A->get_num_stored_elements () << "\n";

  double min_lower_use = std::numeric_limits<double>::infinity ();
  double max_upper_use = -std::numeric_limits<double>::infinity ();

  std::string exp_hitsFile = "ritz_hits_gpu" + std::to_string (gpuID) + ".nc";

  int retval, nc_hits_id, hit_dimid, hits_varid;
  retval = nc_create (exp_hitsFile.c_str (), NC_CLOBBER, &nc_hits_id);
  handle_error (retval);
  retval = nc_def_dim (nc_hits_id, "n", NC_UNLIMITED, &hit_dimid);
  handle_error (retval);
  retval = nc_def_var (nc_hits_id, "DATA", NC_DOUBLE, 1, &hit_dimid, &hits_varid);
  handle_error (retval);
  retval = nc_enddef (nc_hits_id);
  handle_error (retval);
  size_t hit_index = 0;

  auto start = high_resolution_clock::now ();

  // --- Spectral bounds via short Lanczos + LAPACKE on T_k ---
  const int k_lanczos = 30; // 8-16 is usually plenty
  const double tol_lanc = 1e-20;

  auto L = lanczos_bounds (A, k_lanczos, tol_lanc, exec, this_exec);

  const real_precision min_eig = static_cast<real_precision> (L.lam_min_lower);
  const real_precision max_eig = static_cast<real_precision> (L.lam_max_upper);

  std::cout << "Lanczos bounds (k=" << k_lanczos << "):\n"
            << "  mu_min=" << L.mu_min << "  mu_max=" << L.mu_max
            << "  beta_{k+1}=" << L.beta_k1 << "\n"
            << "  Lower bound for lambda_min(A): " << L.lam_min_lower << "\n"
            << "  Upper bound for lambda_max(A): " << L.lam_max_upper << "\n";

  // input min and max, epsilon defines the shift to get the scale to be -1,1
  const real_precision epsilon = 1e-2;
  const real_precision E_min = min_eig;
  const real_precision E_max = max_eig;

  // equations 26 and 27 https://doi.org/10.1103/RevModPhys.78.275
  // transforming the scale of Emin and Emax to (-1,1)
  auto a = (E_max - E_min) / (2.0 - epsilon);
  auto b = (E_max + E_min) / 2.0;

  // New alpha and beta for: (A - bI) / a
  auto alpha = real_precision{ 1.0 } / a;
  auto beta = -b / a;

  // Construct rescaled matrix: alpha * A + beta * I
  auto scaled_matrix = share (gko::Combination<real_precision>::create (gko::initialize<vec> ({ alpha }, exec), A, gko::initialize<vec> ({ beta }, exec), gko::matrix::Identity<real_precision>::create (exec, N)));

  // === Step 2: Chebyshev Moments ===
  const int num_moments = 100;
  std::cout << "num_moments = " << num_moments << std::endl;
  const int num_random_vecs = 1;

  std::vector<double> moments = compute_chebyshev_moments (scaled_matrix, N, num_moments, num_random_vecs, exec, this_exec);

  // === Step 4: Jackson Kernel ===
  std::vector<double> filtered_moments = apply_jackson_kernel (moments);

  // === Step 5: Spectral Density Reconstruction ===
  const int num_points = 0.3 * N;
  std::cout << "number of points " << num_points << std::endl;

  std::vector<double> energies, rho;
  reconstruct_dos (filtered_moments, num_points, a, b, energies, rho);

  std::ofstream rho_out ("spectrum.txt");
  rho_out << "# E_scaled rho(E_scaled)\n";
  for (int i = 0; i < num_points; ++i)
    rho_out << energies[i] << " " << rho[i] << "\n";
  rho_out.close ();

  std::cout << "----------------------------------" << std::endl;
  std::cout << "Wrote spectral density to spectrum.txt\n";

  /*
     Chebyshev Filter Diagonalization starts here
  */

  std::cout << "----------------------------------" << std::endl;
  std::cout << "Starting Adaptive Window" << std::endl;
  std::cout << "----------------------------------" << std::endl;

  // Energy grid from KPM, rho already defined
  std::vector<double> E_scaled = energies;

  std::random_device rd;
  std::mt19937 gen (rd ());

  const double full_lo = E_min;
  const double full_hi = E_max;
  const double I_full = integrate_rho (E_scaled, rho, E_min, E_max);

  // -------------------------------
  // Build initial global window
  // -------------------------------
  double E_center = 0;
  const double target_N = target_frac * static_cast<double> (N);

  double half_width = bisect_half_width (E_scaled, rho, E_center, target_N, full_lo, full_hi, I_full, N, 60);

  // final symmetric window around E_center
  double big_lower = std::max (full_lo, E_center - half_width);
  double big_upper = std::min (full_hi, E_center + half_width);

  double I_big = integrate_rho (E_scaled, rho, big_lower, big_upper);
  double N_big = N * (I_big / I_full);


  std::uniform_real_distribution<double> dist3 (big_lower, big_upper);
  std::uniform_real_distribution<double> uni01 (0.0, 1);

  // --- Monte Carlo controls ---
  int MC_TARGET = mc_target; // Set how many eigenvectors to sample

  int mc_count = 0;
  double sum_rel_res = 0.0;
double best_rel_res = std::numeric_limits<double>::infinity ();
double worst_rel_res = 0.0;
long long total_duplicates_found = 0;
long long total_dedup_candidates = 0;

  // Counts consecutive rejected rejection-sampling draws. Declared outside
  // the while loop so it actually accumulates across
  // draws instead of resetting to 0 every pass. Capped at max_outer, so the
  // same knob that limits filter/SVQB/RR retries also limits how long we'll
  // keep rejection-sampling before warning and giving the counter a rest.

  int mc_draw_attempts = 0; 
  

  while (mc_count < MC_TARGET)
    {
      // One MC draw -> bracket & acceptance test
      //  - Locate [E_i,E_{i+1}] such that E_i <= z_new <= E_{i+1}.
      //  - Interpolate rho(z_new) linearly and accept if u < rho(z_new).
      double z_new = dist3 (gen);      // randomly chosen energy
      double random_num = uni01 (gen); // random number

      auto it = std::upper_bound (E_scaled.begin (), E_scaled.end (), z_new);
      int lower_idx = std::max (0, (int) (it - E_scaled.begin ()) - 1);

      double E1 = E_scaled[lower_idx];
      double E2 = E_scaled[lower_idx + 1];

      double rho1 = rho[lower_idx];
      double rho2 = rho[lower_idx + 1];

      double interpolated_rho = rho1 + (rho2 - rho1) * ((z_new - E1) / (E2 - E1));

      

      // monte carlo check
if (random_num >= interpolated_rho)
  {
    mc_draw_attempts++;
    if (mc_draw_attempts >= 1000)
      {
        std::cout << "Failed the MC draw after 1000 attempts\n";
        mc_draw_attempts = 0;
      }
    continue;
  }

      mc_draw_attempts = 0;   // reset the moment a draw is accepted



      const int target_eigs = NT;

      // Center the window on z_new
      E_center = z_new;

      // Bisection to find half-width giving target_eigs eigenvalues
      double win_half_width = bisect_half_width (E_scaled, rho, E_center, target_eigs, full_lo, full_hi, I_full, N, 50);

      // Final window bounds
      double lower_use = std::max (big_lower, E_center - win_half_width);
      double upper_use = std::min (big_upper, E_center + win_half_width);

      // NOTE: rho is invariant across the whole run; recomputing max_rho on
      // every Monte-Carlo draw is wasteful but kept exactly as original.
      double max_rho = *std::max_element (rho.begin (), rho.end ());

      // Final values
      const int NS = std::min (2 * NT, ns_max);

      // mean rho over [lower_use, upper_use] -> how many pairs to keep from this window
      double sum_win = 0.0;
      int count_win = 0;
      for (int i = 0; i < num_points; ++i)
        {
          if (energies[i] >= lower_use && energies[i] <= upper_use)
            {
              sum_win += rho[i];
              count_win++;
            }
        }
      double mean_rho_window = (count_win > 0) ? sum_win / count_win : 0.0;
      double p = mean_rho_window / max_rho;
      int n_accept = std::max ((int) (p * NT), 20);

      std::cout << "Adapted Window: [" << lower_use << ", " << upper_use << "]\n"
                << "NT=" << NT << "  NS=" << NS << "  NP=" << NP << std::endl;

      min_lower_use = std::min (min_lower_use, lower_use);
      max_upper_use = std::max (max_upper_use, upper_use);

      // Random search block (NS vectors) & normalization
      std::vector<VecPtr> search_vectors;
      search_vectors.reserve (NS);
      std::normal_distribution<double> normal (0.0, 1.0);

      print_mem ("before making search vectors", gpuID);

      for (int k = 0; k < NS; ++k)
        {
          auto work = vec::create (this_exec, gko::dim<2>{ N, 1 });
          for (gko::size_type i = 0; i < N; ++i)
            {
              double re = normal (gen);
              work->get_values ()[i] = real_precision{ re };
            }
          long double s = 0.0L;
          for (gko::size_type i = 0; i < N; ++i)
            s += std::norm (work->get_values ()[i]);
          double inv_norm = (s > 0.0L) ? 1.0 / std::sqrt ((double) s) : 1.0;
          for (gko::size_type i = 0; i < N; ++i)
            work->get_values ()[i] *= inv_norm;
          search_vectors.push_back (gko::share (clone (exec, work)));
        }

      print_mem ("after making search vectors", gpuID);

      // Window coefficients (Lanczos kernel * step function on [lower_use, upper_use])
      std::vector<double> w = build_window_coefficients (lower_use, upper_use, alpha, beta, NP);

      // Restart loop & acceptance test:
      //  Iterate filter -> SVQB -> RR; accept in-window Ritz pairs whose
      //  residual <= tau_keep.
      const double tau_keep = 1e-3;

      std::vector<AcceptedPair> all_accepted;

      int outer = 1;
      while ((int) all_accepted.size () < n_accept && outer <= max_outer)
        {
          all_accepted.clear ();

          print_mem ("before CFD", gpuID);
          chebyshev_filter_pass (search_vectors, scaled_matrix, w, N, exec, this_exec); // Step 6
          print_mem ("after CFD", gpuID);

          int r_now = svqb (search_vectors, N, exec, this_exec, gpuID); // Step 7
          if (r_now == 0)
            {
              // refill with fresh randoms if completely dropped
              search_vectors.clear ();
              std::mt19937 rng2 (1234 + outer);
              std::normal_distribution<double> N01 (0.0, 1.0);
              while ((int) search_vectors.size () < NS)
                {
                  auto col_h = vec::create (this_exec, gko::dim<2>{ N, 1 });
                  long double ss = 0.0L;
                  for (gko::size_type i = 0; i < N; ++i)
                    {
                      double re = N01 (rng2);
                      col_h->at (i, 0) = real_precision{ re };
                      ss += re * re;
                    }
                  double invn = (ss > 0.0L) ? 1.0 / std::sqrt ((double) ss) : 1.0;
                  for (gko::size_type i = 0; i < N; ++i)
                    col_h->at (i, 0) *= invn;
                  auto col_d = gko::share (clone (exec, col_h));
                  search_vectors.push_back (col_d);
                }
              continue; // try again
            }

          RitzResult R = rayleigh_ritz (search_vectors, A, N, exec, this_exec, gpuID); // Step 8

          // Print the middle 20 Ritz pairs
          std::cout << "\n[Ritz pairs @ outer " << outer << "]" << std::endl;
          std::cout << " index        lambda_tilde              ||r||_2          rel_res" << std::endl;
          int total = (int) R.lam.size ();
          int mid = total / 2;
          int print_start = std::max (0, mid - 10);
          int print_end = std::min (total, mid + 10);
          for (int j = print_start; j < print_end; ++j)
            {
              double rel_res = (std::abs (R.lam[j]) > 0) ? R.rn[j] / std::abs (R.lam[j]) : R.rn[j];
              std::cout << "  " << std::setw (4) << j
                        << "  " << std::setprecision (12) << std::setw (20) << R.lam[j]
                        << "  " << std::setprecision (6) << std::setw (12) << R.rn[j]
                        << "  " << std::setprecision (6) << std::setw (12) << rel_res
                        << std::endl;
            }

          // Collect in-window Ritz pairs passing the residual test, deduplicated
          std::vector<int> accepted_indices;
          int duplicates_removed = 0;

          for (int j = 0; j < (int) R.lam.size (); ++j)
            {
              if (R.lam[j] >= lower_use && R.lam[j] <= upper_use)
                {
                  double rel_res = R.rn[j] / std::abs (R.lam[j]);
                  if (rel_res <= tau_keep)
                    {
                      bool duplicate = false;
                      for (int i = 0; i < (int) accepted_indices.size (); ++i)
                        {
                          int k = accepted_indices[i];
                          double lam_diff = std::abs (R.lam[j] - R.lam[k]) / std::abs (R.lam[k]);
                          if (lam_diff < 1e-5)
                            {
                              if (R.rn[j] < R.rn[k])
                                {
                                  accepted_indices.erase (accepted_indices.begin () + i);
                                  accepted_indices.push_back (j);
                                }
                              duplicate = true;
                              duplicates_removed++;
                              break;
                            }
                        }
                      if (!duplicate)
                        accepted_indices.push_back (j);
                    }
                }
            }

          std::cout << "[RR] Accepted " << accepted_indices.size ()
                    << " in-window Ritz pairs with residual <= " << tau_keep << std::endl;

                total_duplicates_found += duplicates_removed;
                total_dedup_candidates += (int) accepted_indices.size () + duplicates_removed;

          if (!accepted_indices.empty ())
            {
              for (int jj : accepted_indices)
                {
                  auto v_h = gko::clone (this_exec, R.v[jj]);
                  const auto nloc = v_h->get_size ()[0];

                  AcceptedPair ap;
                  ap.lam = R.lam[jj];
                  ap.rn = R.rn[jj];
                  const auto *vals = v_h->get_const_values ();
                  ap.vec.assign (vals, vals + nloc);
                  all_accepted.push_back (ap);
                }
              if (outer == max_outer)
                {
                  std::cout << "[RR] Reached max_outer=" << max_outer << " with " << mc_count << " total accepted pairs.\n";
                  break;
                }
            }
          outer++;
        }

      if ((int) all_accepted.size () > n_accept)
        {
          std::shuffle (all_accepted.begin (), all_accepted.end (), gen);
          all_accepted.resize (n_accept);
        }

      std::cout << "[SUBSAMPLE] Keeping " << all_accepted.size ()
                << " of converged pairs (n_accept=" << n_accept << ")\n";

      // Write accepted pairs to the NetCDF output.
      // Each accepted pair contributes 3 doubles to the "DATA" variable:
      // lambda, residual norm, and the first component of the Ritz vector
      
      // The full Ritz vector is available in ap.vec if you want to compute things with it
      // e.g. loop over ap.vec and nc_put_var1_double each entry.
      for (int k = 0; k < (int) all_accepted.size (); ++k)
        {
          const auto &ap = all_accepted[k];

          nc_put_var1_double (nc_hits_id, hits_varid, &hit_index, &ap.lam);
          hit_index++;
          nc_put_var1_double (nc_hits_id, hits_varid, &hit_index, &ap.rn);
          hit_index++;
          nc_put_var1_double (nc_hits_id, hits_varid, &hit_index, &ap.vec[0]);
          hit_index++;

          mc_count++;
          const double rel_res = std::abs (ap.lam) > 0.0 ? ap.rn / std::abs (ap.lam) : ap.rn;
        sum_rel_res += rel_res;
        best_rel_res = std::min (best_rel_res, rel_res);
        worst_rel_res = std::max (worst_rel_res, rel_res);
        }
      nc_sync (nc_hits_id);
      std::cout << "[Progress] Total eigenvectors collected: " << mc_count << " / " << MC_TARGET << "\n";
    }

  std::cout << "----------------------------------" << std::endl;
  std::cout << "Adaptive window summary:\n"
            << "  min_lower   = " << std::setprecision (17) << min_lower_use << "\n"
            << "  max_upper   = " << std::setprecision (17) << max_upper_use << "\n";

            std::cout << "----------------------------------" << std::endl;
  std::cout << "Eigenvector summary:\n"
            << "  eigenvectors found = " << mc_count << "\n";
  if (mc_count > 0)
   {
      const double mean_rel_res = sum_rel_res / mc_count;
      std::cout << "  mean accuracy = " << std::setprecision (6) << (1.0 - mean_rel_res) * 100.0 << "%"
                << "  (mean relative residual = " << mean_rel_res << ")\n"
                << "  best  accuracy = " << (1.0 - best_rel_res) * 100.0 << "%"
                << "  (best  relative residual = " << best_rel_res << ")\n"
                << "  worst accuracy = " << (1.0 - worst_rel_res) * 100.0 << "%"
                << "  (worst relative residual = " << worst_rel_res << ")\n";
    }
    std::cout << "Duplicate summary:\n"
          << "  duplicates found = " << total_duplicates_found
          << " out of " << total_dedup_candidates << " converged candidates";
    if (total_dedup_candidates > 0)
        std::cout << "  (" << std::setprecision (4)
            << (100.0 * total_duplicates_found / total_dedup_candidates) << "%)";
        std::cout << "\n";  

  auto stop = high_resolution_clock::now ();
  auto duration = duration_cast<microseconds> (stop - start);

  std::cout << "----------------------------------" << std::endl;
  std::cout << "Time to run code: "
            << duration.count () / 1000000 << " seconds" << std::endl;

  nc_close (nc_hits_id);
  return EXIT_SUCCESS;
}
