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
	mod_misc.f90 \
	mod_random.f90 \
	mod_optimiser.f90 \
	mod_types.f90 \
	mod_metrics.f90 \
	mod_misc_ml.f90 \
	mod_activation_gaussian.f90 \
	mod_activation_linear.f90 \
	mod_activation_piecewise.f90 \
	mod_activation_relu.f90 \
	mod_activation_leaky_relu.f90 \
	mod_activation_sigmoid.f90 \
	mod_activation_tanh.f90 \
	mod_activation_none.f90 \
	mod_activation_softmax.f90 \
	mod_initialiser_glorot.f90 \
	mod_initialiser_he.f90 \
	mod_initialiser_lecun.f90 \
	mod_initialiser_zeros.f90 \
	mod_activation.f90 \
	mod_initialiser.f90 \
	mod_tools_infile.f90 \
	mod_normalisation.f90 \
	mod_batch_norm.f90 \
	mod_loss_categorical.f90 \
	mod_base_layer.f90 \
	mod_conv2d_layer.f90 \
	mod_conv3d_layer.f90 \
	mod_maxpool2d_layer.f90 \
	mod_maxpool3d_layer.f90 \
	mod_full_layer.f90 \
	mod_flatten2d_layer.f90 \
	mod_flatten3d_layer.f90 \
	mod_input1d_layer.f90 \
	mod_input3d_layer.f90 \
	mod_input4d_layer.f90 \
	mod_container_layer.f90 \
	mod_container_layer_sub.f90 \
	mod_network.f90
SRCS := atelos.f90
OBJS := $(addprefix $(LIB_DIR)/,$(LIBS))
OBJS := $(OBJS) $(SRCS)
OBJS := $(addprefix $(SRC_DIR)/,$(OBJS))

LIBS_o := $(addprefix $(BUILD_DIR)/,$(LIBS))
LIBS_o := $(LIBS_o:.f90=.o)
SRCS_o := $(addprefix $(BUILD_DIR)/,$(SRCS))
SRCS_o := $(SRCS_o:.f90=.o)

OBJS_mod := $(LIBS_o:.o=.mod) $(SRCS_o:.o=.mod)


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
LIBRARY_NAME = libatelos.a

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


.PHONY: all install build uninstall clean modlist #mp debug dev optim memcheck bigmem

libraries = $(BIN_DIR)/$(LIBRARY_NAME)
all: $(libraries)


build: all
	@:

%:
	@:
#	$(FC) $(PPFLAGS) $(CFLAGS) $(MODULEFLAGS) $(BUILD_DIR) $(OBJS) -o $(libraries)

$(BIN_DIR):
	mkdir -p $@

$(BUILD_DIR):
	mkdir -p $@

$(libraries): $(LIBS_o) $(SRCS_o) modlist | $(BIN_DIR)
	@echo "Creating library $@"
	ar rcs $@ $(LIBS_o) $(SRCS_o) $(shell cat $(BUILD_DIR)/mod_files.txt)
	cp $(BUILD_DIR)/atelos.mod $(BIN_DIR)/.


# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.f90 | $(BUILD_DIR)
	$(FC) $(PPFLAGS) $(CFLAGS) $(MODULEFLAGS) $(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/$(LIB_DIR)/%.f90 | $(BUILD_DIR)
	$(FC) $(PPFLAGS) $(CFLAGS) $(MODULEFLAGS) $(BUILD_DIR) -c $< -o $@


# Rule to generate the list of .mod files
modlist: | $(BUILD_DIR)
	@find $(BUILD_DIR) -type f -name "*.mod" > $(BUILD_DIR)/mod_files.txt


install: $(OBJS) | $(INSTALL_DIR) $(BUILD_DIR)
	$(FC) $(PPFLAGS) $(CFLAGS) $(MODULEFLAGS) $(BUILD_DIR) $(OBJS) -o $(libraries)

clean: 
	rm -rf $(BUILD_DIR)/ $(BIN_DIR)/

uninstall: $(INSTALL_DIR)/$(NAME)
	rm $(INSTALL_DIR)/$(NAME)
