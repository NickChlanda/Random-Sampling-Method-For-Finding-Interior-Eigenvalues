#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# define the C compiler to use
CC = mpic++

# define any compile-time flags
CFLAGS = -O3 -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++17 -w

# define any directories containing header files other than /usr/include
#
INCLUDES = -I/usr/local/cuda/include -I/usr/local/include/gsl  -I/usr/local/boost_1_77_0 -I/usr/include -I/usr/include/openmpi-x86_64/

# define linking flags
LFLAGS = -fPIC -fopenmp # -Wl

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = -L/usr/lib64 -L/usr/local/lib -L/usr/local/cuda/lib64 -L/opt/intel/oneapi/mkl/latest/lib/intel64 -L/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin -lnetcdf -lmkl_intel_thread -lginkgo -lginkgo_omp -lginkgo_cuda -lginkgo_reference -lginkgo_dpcpp -lginkgo_device -lcuda -liomp5 -lginkgo_hip -lmkl_core -lmkl_gf_ilp64


#removed
# -lblas -llapack -lcudadevrt -lcublas -lcudart -lcusparse -lcublas -lstdc++ -lm -lpthread -lgfortran -lgsl 


# define the C source files
SRCS = wETH-stat-GitHub.cpp

# define headers
HEAD = 

# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.cpp=.o)

# define the executable file
MAIN = wETH

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: depend clean

all:    $(MAIN) $(HEAD)


$(MAIN): $(OBJS) $(HEAD)
		$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.cpp.o: $(HEAD)
		$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
		$(RM) *.o *~ $(MAIN)

depend: $(SRCS) $(HEAD)
		makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
