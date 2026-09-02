# A Method for Finding Eigenstates Near an Energy of Interest For Large Sparse Matrices: An Example

This repository is intended to be supplementary material for paper : (paper goes here). If you find this prior to paper submission, please email me at nchlandaATbrynmawr.edu

This program is meant to give researchers a easy method for finding eigenstates near an energy of interest for large *sparse* hamiltonians, ones too large for exact diagonalization, using Ginkgo (https://github.com/ginkgo-project/ginkgo). Ginkgo is an high-performance numerical linear algebra library with a focus on sparse linear systems. This code also contains The Kernel Polynomial Method (KPM, RevModPhys.78.275) and Chebyshev Filter Diagonalization (CFD, j.jcp.2016.08.027) if you would just like to use our implementation of Ginkgo for those methods specifically. 

Furthermore, if you are interested in Ginkgo, this is a great way to see how to implement various things.

 # What problem this solves


  If you desire information about specific eigenstates of your system, but your dimension size is much greater than what is possible via exact diagonalizaiton, this method gives you a way around that.


  # Example: Testing the weak Eigenstate Thermalization Hypothesis (wETH)

  We apply this method to testing the wETH, more information of our technique can be found here in this paper (cite). However, briefly, we estimate the density of eigenvalues across the whole spectrum (KPM), repeatedly pick a random energy somewhere in a desired spectrum (exp. middle 1/3), and create a small window surrounding that energy, then use a polynomial filter + a small dense eigensolver to pull out the eigenpairs that live there (CFD). It keeps doing this until it has collected as many eigenpairs as you asked for (mc_target), each passing a residual check.

 
 # The 10-step pipeline

   For a visual representation, see flowchartvertical via this repository.
   
 
   1. Load a sparse matrix A from a MatrixMarket (.mtx / plain text) file.
   
   2. Spectral bounds: a short Lanczos run + a small tridiagonal eigenproblem
      (LAPACK dstev) gives bounds on lambda_min and lambda_max
      of A.
      [lanczos_bounds()]
   
   3. Rescale A to A~ = alphaA + betaI so its whole spectrum maps inside
      [-1, 1] -- Chebyshev polynomials are well-behaved on that domain.
      [main(), rescaling block, right after step 2]
  
   4. KPM: estimate the density of states (DOS) of A~ by computing Chebyshev
      moments of a random vector, damping them with a Jackson kernel (to
      suppress Gibbs oscillations), and reconstructing a DOS curve on the original energy bounds then
      Written to spectrum.txt.
      [compute_chebyshev_moments(), apply_jackson_kernel(), reconstruct_dos()]
  
   5. Monte-Carlo targeting: repeatedly draw a candidate energy from the
      reconstructed DOS by rejection sampling, so energies in denser parts
      of the spectrum get proposed more often. 
      [main(), Monte-Carlo while loop]
  
   6. Adaptive window: for each accepted draw, bisect on the integrated DOS
      to find a half-width so the window is expected to contain a set number of
      eigenvalues (NT); NS = min(2NT, ns_max) random search vectors are sized
      to that window.
      [bisect_half_width(), expected_eigs_window(), integrate_rho()]
  
   7. Chebyshev window filter: a degree-NP polynomial
      is applied to all NS search vectors, suppressing everything outside
      the window and amplifying what's inside.
      [build_window_coefficients(), chebyshev_filter_pass()]
 
  8. SVQB: orthonormalizes the filtered block.
      [svqb()]
 
  9. Rayleigh-Ritz + accept: project A onto the orthonormal block, get
      Ritz pairs + residuals
      [rayleigh_ritz()]
  
  10. Repeat: steps 7-10 (up to a set max_outer times) until
      enough in-window pairs converge below tau_keep. Deduplicate near-
      identical eigenvalues, and append the results to the NetCDF output file.
      [main(), outer refinement loop]
 
  # Input

  A real-symmetric sparse matrix in MatrixMarket coordinate text format
  (see the "matrix.mtx" argument below).
 
  # Outputs

  **spectrum.txt**: Two columns: energy, normalized DOS value. This is the KPM density-of-states estimate       from step 4, written once near the start of the run.

                        
  **ritz_hits_gpu<ID>.nc**:  One "DATA" variable holding, per accepted pair: lambda (the eigenvalue), the residual norm ||Av - lambdav||, and the first component of the Ritz vector v. The full Ritz vector is available in memory (AcceptedPair::vec) if you want to save it too -- see the comment at the NetCDF write loop.
                          
  **(console)**: Progress prints throughout the run (spectral bounds, per-window status, Ritz-pair tables, acceptance/duplicate/subsample counts), plus a final summary block (window coverage, eigenvector count + accuracy, duplicate rate, and timer).

# Command Line

  **Compile**: "make" in terminal which uses Makefile
 
  **Run**: ./ matrixname.mtx <gpu_id> <NP> <NT> <max_outer> <target_frac> <mc_target> <ns_max>
  
  **Example**: ./wETH matrix_out.txt 0 100 200 5 0.5 1000 600
 
  **matrix.mtx**: real-symmetric sparse matrix, MatrixMarket text format.
  
 **gpu_id**: which CUDA device to use (as reported by nvidia-smi).
  
  **NP**: Chebyshev filter polynomial order (step 7). Higher NP = sharper window edges, but more matrix-vector products per search vector per outer iteration -- scale this to how finely you need to resolve a window relative to the full spectral width, not to N.
  
  **NT**: target number of eigenvalues expected per search window (step 6); also used directly as the residual-table/console label and in the NS_MAX/NS sizing below.
  
  **max_outer**: max number of filter/SVQB/Rayleigh-Ritz refinement cycles to try per window (step 9) before giving up on that window and moving to the next Monte-Carlo draw.
  
  **target_frac**: fraction (0,1] of the whole spectrum covered by the single global sampling window that Monte-Carlo draws are pulled from (computed once, before the per-window search begins). exp: 0.5 will contain half of the spectrum centered on zero
  
  **mc_target**: total number of Ritz pairs to collect before the program stops (the loop keeps drawing/searching windows until this many pairs have been written to the NetCDF file).
  
  **ns_max**: hard cap on the number of Chebyshev-filtered search vectors per window; NS = min(2NT, ns_max). Pick this to fit your GPU's memory (each search vector is N doubles).

  # Libraries Required

  To run this program you will need *Ginkgo* and *Lapack*. Ginkgo is an high-performance numerical linear algebra library with a focus on sparse linear systems, we use this to handle all matrix vector multiplication, scaling and overall handeling of vectors and matricies. We use Lapack to solve small eigenproblems.


