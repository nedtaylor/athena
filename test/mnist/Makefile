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
	mod_misc.f90
OBJS := $(addprefix $(LIB_DIR)/,$(LIBS))
SRCS := inputs.f90
MAIN := main.f90
SRCS := $(OBJS) $(SRCS) $(MAIN)
OBJS := $(addprefix $(SRC_DIR)/,$(SRCS))


##########################################
# COMPILER CHOICE SECTION
##########################################
FC=gfortran
ifeq ($(FC), $(filter $(FC), ifort ifx))
	PPFLAGS = -stand f18 -cpp
	MPFLAGS = -qopenmp
	MODULEFLAGS = -module
	DEVFLAGS = -check all -warn #all
	DEBUGFLAGS = -check all -fpe0 -warn -tracekback -debug extended # -check bounds
	OPTIMFLAGS = -O3
else ifeq ($(FC), $(filter $(FC), gfortran gcc)) 
	PPFLAGS = -cpp
	MPFLAGS = -fopenmp
	MODULEFLAGS = -J
	WARNFLAGS = -Wall
	DEVFLAGS = -g -fbacktrace -fcheck=all -fbounds-check -Og #-g -static -ffpe-trap=invalid
	DEBUGFLAGS = -fbounds-check
	MEMFLAGS = -mcmodel=large
	OPTIMFLAGS = -O3 -march=native
	FASTFLAGS = -Ofast -march=native
else ifeq ($(FC), $(filter $(FC), nag nagfor nagfmcheck)) 
	PPFLAGS = -f2018 -fpp
	MPFLAGS = -openmp
	MODULEFLAGS = -mdir ${BUILD_DIR} -I
	WARNFLAGS = -Wall
	DEVFLAGS = -g -mtrace -C=all -colour -O0 #-g -static -ffpe-trap=invalid
	DEBUGFLAGS = -C=array
	MEMFLAGS = -mcmodel=large
	OPTIMFLAGS = -O3
	FASTFLAGS = -Ofast
else
#	echo "Compiler" ${FC} "not known"
#	exit
endif


##########################################
# LAPACK SECTION
##########################################
MKLROOT?="/usr/local/intel/parallel_studio_xe_2017/compilers_and_libraries_2017/linux/mkl/lib/intel64"
LLAPACK = $(MKLROOT)/libmkl_lapack95_lp64.a \
        -Wl,--start-group \
        $(MKLROOT)/libmkl_intel_lp64.a \
        $(MKLROOT)/libmkl_sequential.a \
        $(MKLROOT)/libmkl_core.a \
        -Wl,--end-group \
        -lpthread

#$(MKLROOT)/libmkl_scalapack_lp64.a \
#$(MKLROOT)/libmkl_solver_lp64_sequential.a
LBLAS = /home/links/hm556/.local/lapack/libblas.a #$(MKLROOT)/libmkl_blas95_lp64.a
LLAPACK = /home/links/hm556/.local/lapack/liblapack.a
LIBRARY = -fexternal-blas $(LBLAS)

ATHENAROOT = "${HOME}/.local/athena"
LATHENA = -I$(ATHENAROOT)/include -L$(ATHENAROOT)/lib -gdwarf-2 -lathena


##########################################
# COMPILATION SECTION
##########################################
INSTALL_DIR?=$(HOME)/bin
NAME = athena_test

CFLAGS =


ifeq ($(findstring bigmem,$(MAKECMDGOALS)),bigmem)
	CFLAGS+=$(MEMFLAGS)
endif
ifeq ($(findstring debug,$(MAKECMDGOALS)),debug)
	CFLAGS+=$(DEBUGFLAGS)
endif
ifeq ($(findstring dev,$(MAKECMDGOALS)),dev)
	CFLAGS+=$(DEVFLAGS)
endif
ifeq ($(findstring mp,$(MAKECMDGOALS)),mp)
	CFLAGS+=$(MPFLAGS)
	NAME:=$(NAME)_mp
	LATHENA:=$(LATHENA)_omp
endif
ifeq ($(findstring address,$(MAKECMDGOALS)),address)
	CFLAGS:=$(filter-out -fsanitize=leak, $(CFLAGS))
	CFLAGS+=-fsanitize=address
endif
ifeq ($(findstring memcheck,$(MAKECMDGOALS)),memcheck)
	CFLAGS:=$(filter-out -fsanitize=address, $(CFLAGS))
	CFLAGS:=$(filter-out -Og, $(CFLAGS))
	CFLAGS+=-fsanitize=leak
endif
ifeq ($(findstring optim,$(MAKECMDGOALS)),optim)
	CFLAGS+=$(OPTIMFLAGS)
endif
ifeq ($(findstring fast,$(MAKECMDGOALS)),fast)
	CFLAGS+=$(FASTFLAGS)
#	LIBFILES:=$(addprefix $(SRC_DIR)/$(LIB_DIR)/,$(LIBS))
#	OBJS:=$(filter-out $(LIBFILES), $(OBJS))
#	$(FC) -O5 -march=native -ffree-form -c $(LIBFILES) $(MODULEFLAG) $(BUILD_DIR)
endif


.PHONY: all install build uninstall clean #mp debug dev optim memcheck bigmem

programs = $(BIN_DIR)/$(NAME)
all: $(programs)


build: all
	@:

%:
	@:
#	$(FC) $(PPFLAGS) $(CFLAGS) $(MODULEFLAGS) $(BUILD_DIR) $(OBJS) -o $(programs)

$(BIN_DIR):
	mkdir -p $@

$(BUILD_DIR):
	mkdir -p $@

$(programs): $(OBJS) | $(BIN_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAGS) $(CFLAGS) $(MODULEFLAGS) $(BUILD_DIR) $(OBJS) $(LATHENA) -o $@

install: $(OBJS) | $(INSTALL_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAGS) $(CFLAGS) $(MODULEFLAGS) $(BUILD_DIR) $(OBJS) $(LATHENA) -o $(programs)

clean: 
	rm -rf $(BUILD_DIR)/ $(BIN_DIR)/

uninstall: $(INSTALL_DIR)/$(NAME)
	rm $(INSTALL_DIR)/$(NAME)
