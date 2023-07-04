##########################################
# CODE DIRECTORIES AND FILES
##########################################
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
BIN_DIR := ./bin
SRC_DIR := ./src
LIB_DIR := ./lib
BUILD_DIR = ./obj
LIBS := mod_constants.f90 \
	mod_misc_ml.f90 \
	mod_types.f90 \
	mod_activation_gaussian.f90 \
	mod_activation_linear.f90 \
	mod_activation_piecewise.f90 \
	mod_activation_relu.f90 \
	mod_activation_leaky_relu.f90 \
	mod_activation_sigmoid.f90 \
	mod_activation_tanh.f90 \
	mod_misc.f90 \
	mod_tools_infile.f90 \
	mod_normalisation.f90 \
	mod_loss_categorical.f90
OBJS := $(addprefix $(LIB_DIR)/,$(LIBS))
#$(info VAR is $(OBJS))
SRCS := inputs.f90 \
	convolution.f90 \
	pooling.f90 \
	fullyconnected.f90 \
	softmax.f90
MAIN := main.f90
SRCS := $(OBJS) $(SRCS) $(MAIN)
OBJS := $(addprefix $(SRC_DIR)/,$(SRCS))


##########################################
# COMPILER CHOICE SECTION
##########################################
FFLAGS = -O2
#PPFLAGS = -cpp
FC=gfortran
ifeq ($(FC),ifort)
	PPFLAG = -cpp
	MPFLAG = -qopenmp
	MODULEFLAG = -module
	DEVFLAGS = -check all -warn #all
	DEBUGFLAGS = -check all -fpe0 -warn -tracekback -debug extended # -check bounds
	OPTIMFLAG = -O3
else
	PPFLAG = -cpp
	MPFLAG = -fopenmp
	MODULEFLAG = -J
	DEVFLAGS = -g -fbacktrace -fcheck=all -fbounds-check #-g -static -ffpe-trap=invalid
	DEBUGFLAGS = -fbounds-check
	MEMFLAG = -mcmodel=large
	OPTIMFLAG = -O3 -march=native
endif


##########################################
# LAPACK SECTION
##########################################
MKLROOT?="/usr/local/intel/parallel_studio_xe_2017/compilers_and_libraries_2017/linux/mkl/lib/intel64_lin"
LLAPACK = $(MKLROOT)/libmkl_lapack95_lp64.a \
	-Wl,--start-group \
	$(MKLROOT)/libmkl_intel_lp64.a \
	$(MKLROOT)/libmkl_sequential.a \
	$(MKLROOT)/libmkl_core.a \
	-Wl,--end-group \
	-lpthread

#$(MKLROOT)/libmkl_scalapack_lp64.a \
#$(MKLROOT)/libmkl_solver_lp64_sequential.a \


##########################################
# COMPILATION SECTION
##########################################
INSTALL_DIR?=$(HOME)/bin
NAME = cnn_dev
programs = $(BIN_DIR)/$(NAME)
programs_mp = $(BIN_DIR)/$(NAME)_mp

.PHONY: all debug install uninstall dev optim mp mp_optim mp_dev clean

all: $(programs)

$(BIN_DIR):
	mkdir -p $@

$(BUILD_DIR):
	mkdir -p $@

$(BIN_DIR)/$(NAME): $(OBJS) | $(BIN_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAG) $(MEMFLAG) $(MODULEFLAG) $(BUILD_DIR) $(OBJS) -o $@

install: $(OBJS) | $(INSTALL_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAG) $(MEMFLAG) $(MODULEFLAG) $(BUILD_DIR) $(OBJS) -o $(INSTALL_DIR)/$(NAME)

debug: $(OBJS) | $(BIN_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAG) $(MEMFLAG) $(DEBUGFLAGS) $(MODULEFLAG) $(BUILD_DIR) $(OBJS) -o $(programs)

dev: $(OBJS) | $(BIN_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAG) $(MEMFLAG) $(DEVFLAGS) $(MODULEFLAG) $(BUILD_DIR) $(OBJS) -o $(programs)

optim: $(OBJS) | $(BIN_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAG) $(OPTIMFLAG) $(DEVFLAGS) $(MODULEFLAG) $(BUILD_DIR) $(OBJS) -o $(programs)

mp_dev: $(OBJS) | $(BIN_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAG) $(MEMFLAG) $(DEVFLAGS) $(MPFLAG) $(MODULEFLAG) $(BUILD_DIR) $(OBJS) -o $(programs_mp)

mp: $(OBJS) | $(BIN_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAG) $(MEMFLAG) $(MPFLAG) $(MODULEFLAG) $(BUILD_DIR) $(OBJS) -o $(programs_mp)

mp_optim: $(OBJS) | $(BIN_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAG) $(OPTIMFLAG) $(MPFLAG) $(MODULEFLAG) $(BUILD_DIR) $(OBJS) -o $(programs_mp)

clean: $(BUILD_DIR) $(BIN_DIR)
	rm -r $(BUILD_DIR)/ $(BIN_DIR)/

uninstall: $(INSTALL_DIR)/$(NAME)
	rm $(INSTALL_DIR)/$(NAME)
