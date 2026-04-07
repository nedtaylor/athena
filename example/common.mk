##########################################
# CODE DIRECTORIES AND FILES
##########################################
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
ROOT_DIR ?= $(abspath $(mkfile_dir)/../..)
BIN_DIR ?= ./bin
SRC_DIR ?= ./src
LIB_DIR ?= ../example_library/src
BUILD_DIR ?= ./obj
NAME ?= $(notdir $(patsubst %/,%,$(mkfile_dir)))
MAIN ?= main.f90
LIBS ?=
SRCS ?=

LIB_FILES := $(addprefix $(LIB_DIR)/,$(LIBS))
LOCAL_FILES := $(addprefix $(SRC_DIR)/,$(SRCS))
MAIN_FILE := $(SRC_DIR)/$(MAIN)
SOURCES := $(LIB_FILES) $(LOCAL_FILES) $(MAIN_FILE)


##########################################
# BUILD CACHE DISCOVERY
##########################################
REQUIRED_MODS ?=
PREFERRED_MOD_FILE := $(if $(filter athena_wandb,$(REQUIRED_MODS)),$(shell ls -td $(ROOT_DIR)/build/gfortran_*/athena_wandb.mod 2>/dev/null | head -n 1))
PRIMARY_BUILD_REL := $(shell grep -m1 -o 'build/gfortran_[A-F0-9]*' $(ROOT_DIR)/build/compile_commands.json 2>/dev/null)
PRIMARY_BUILD_DIR := $(if $(PREFERRED_MOD_FILE),$(patsubst %/,%,$(dir $(PREFERRED_MOD_FILE))),$(if $(PRIMARY_BUILD_REL),$(ROOT_DIR)/$(PRIMARY_BUILD_REL)))
DIFFSTRUC_BUILD_REL := $(shell grep -o 'build/gfortran_[A-F0-9]*' $(ROOT_DIR)/build/compile_commands.json 2>/dev/null | sed -n '2p')
DIFFSTRUC_BUILD_DIR := $(if $(DIFFSTRUC_BUILD_REL),$(ROOT_DIR)/$(DIFFSTRUC_BUILD_REL))

ATHENA_MOD_FILE := $(or $(wildcard $(PRIMARY_BUILD_DIR)/athena.mod),$(shell ls -td $(ROOT_DIR)/build/gfortran_*/athena.mod 2>/dev/null | head -n 1))
ATHENA_MOD_DIR := $(patsubst %/,%,$(dir $(ATHENA_MOD_FILE)))
COREUTILS_MOD_FILE := $(shell ls -td $(ROOT_DIR)/build/gfortran_*/coreutils.mod 2>/dev/null | head -n 1)
COREUTILS_MOD_DIR := $(patsubst %/,%,$(dir $(COREUTILS_MOD_FILE)))
GRAPHSTRUC_MOD_FILE := $(shell ls -td $(ROOT_DIR)/build/gfortran_*/graphstruc.mod 2>/dev/null | head -n 1)
GRAPHSTRUC_MOD_DIR := $(patsubst %/,%,$(dir $(GRAPHSTRUC_MOD_FILE)))
ATOMSTRUC_MOD_FILE := $(shell ls -td $(ROOT_DIR)/build/gfortran_*/atomstruc.mod 2>/dev/null | head -n 1)
ATOMSTRUC_MOD_DIR := $(patsubst %/,%,$(dir $(ATOMSTRUC_MOD_FILE)))
DIFFSTRUC_MOD_FILE := $(or $(wildcard $(DIFFSTRUC_BUILD_DIR)/diffstruc.mod),$(shell ls -td $(ROOT_DIR)/build/gfortran_*/diffstruc.mod 2>/dev/null | head -n 1))
DIFFSTRUC_MOD_DIR := $(patsubst %/,%,$(dir $(DIFFSTRUC_MOD_FILE)))
ATHENA_WANDB_MOD_FILE := $(or $(wildcard $(ATHENA_MOD_DIR)/athena_wandb.mod),$(shell ls -td $(ROOT_DIR)/build/gfortran_*/athena_wandb.mod 2>/dev/null | head -n 1))
ATHENA_WANDB_MOD_DIR := $(patsubst %/,%,$(dir $(ATHENA_WANDB_MOD_FILE)))
WF_MOD_FILE := $(shell ls -td $(ROOT_DIR)/build/gfortran_*/wf.mod 2>/dev/null | head -n 1)
WF_MOD_DIR := $(patsubst %/,%,$(dir $(WF_MOD_FILE)))

ATHENA_OBJ_DIR := $(ATHENA_MOD_DIR)/athena
COREUTILS_OBJ_DIR := $(patsubst %/,%,$(dir $(shell ls -td $(ROOT_DIR)/build/gfortran_*/athena/build_dependencies_coreutils_src_coreutils.f90.o 2>/dev/null | head -n 1)))
GRAPHSTRUC_OBJ_DIR := $(patsubst %/,%,$(dir $(shell ls -td $(ROOT_DIR)/build/gfortran_*/athena/build_dependencies_graphstruc_src_graphstruc.f90.o 2>/dev/null | head -n 1)))
ATOMSTRUC_OBJ_DIR := $(patsubst %/,%,$(dir $(shell ls -td $(ROOT_DIR)/build/gfortran_*/athena/build_dependencies_atomstruc_src_atomstruc.f90.o 2>/dev/null | head -n 1)))
DIFFSTRUC_OBJ_DIR := $(patsubst %/,%,$(dir $(shell ls -td $(ROOT_DIR)/build/gfortran_*/athena/build_dependencies_diffstruc_src_diffstruc.f90.o 2>/dev/null | head -n 1)))
WANDB_OBJ_DIR := $(patsubst %/,%,$(dir $(shell ls -td $(ROOT_DIR)/build/gfortran_*/athena/build_dependencies_wandb-fortran_src_wf.f90.o 2>/dev/null | head -n 1)))

ATHENA_OBJS := $(wildcard $(ATHENA_OBJ_DIR)/src_athena*.o)
COREUTILS_OBJS := $(wildcard $(COREUTILS_OBJ_DIR)/build_dependencies_coreutils_src_*.o)
GRAPHSTRUC_OBJS := $(wildcard $(GRAPHSTRUC_OBJ_DIR)/build_dependencies_graphstruc_src_*.o)
ATOMSTRUC_OBJS := $(wildcard $(ATOMSTRUC_OBJ_DIR)/build_dependencies_atomstruc_src_*.o)
DIFFSTRUC_OBJS := $(wildcard $(DIFFSTRUC_OBJ_DIR)/build_dependencies_diffstruc_src_*.o)
WANDB_OBJS := $(wildcard $(WANDB_OBJ_DIR)/build_dependencies_wandb-fortran_*.o)

EXTRA_MOD_DIRS ?=
EXTRA_DEP_OBJS ?=
EXTRA_LIBS ?=
RUN_ENV ?=
RUN_DIR ?= $(ROOT_DIR)
RUN_ARGS ?=
SETUP_ENV_SCRIPT ?=
SETUP_ENV_ABS := $(if $(strip $(SETUP_ENV_SCRIPT)),$(abspath $(SETUP_ENV_SCRIPT)))
CPP_MACROS ?=
WANDD_C_SRC := $(ROOT_DIR)/build/dependencies/wandb-fortran/src/wf_c.c
WANDD_C_OBJ := $(if $(filter athena_wandb,$(REQUIRED_MODS)),$(BUILD_DIR)/wf_c.o)

ALL_MOD_DIRS := $(strip $(ATHENA_MOD_DIR) $(COREUTILS_MOD_DIR) \
	$(GRAPHSTRUC_MOD_DIR) $(ATOMSTRUC_MOD_DIR) $(DIFFSTRUC_MOD_DIR) \
	$(WF_MOD_DIR) \
	$(ATHENA_WANDB_MOD_DIR) $(EXTRA_MOD_DIRS))
INCLUDE_FLAGS := $(addprefix -I,$(sort $(ALL_MOD_DIRS)))
CPP_FLAGS := $(addprefix -D,$(CPP_MACROS))
DEPENDENCY_OBJS := $(sort $(ATHENA_OBJS) $(COREUTILS_OBJS) $(GRAPHSTRUC_OBJS) \
	$(ATOMSTRUC_OBJS) $(DIFFSTRUC_OBJS) $(WANDB_OBJS) $(WANDD_C_OBJ) $(EXTRA_DEP_OBJS))

SETUP_ENV_CMD := $(if $(strip $(SETUP_ENV_ABS)),source $(SETUP_ENV_ABS) >/dev/null && unset DYLD_LIBRARY_PATH && ,)


##########################################
# COMPILER CHOICE SECTION
##########################################
UNAME_S := $(shell uname -s)
NTHREADS=4
FC=gfortran
CC ?= cc
ifneq (,$(filter ifort% ifx%, $(FC)))
	PPFLAGS = -stand f18 -cpp
	MPFLAGS = -qopenmp -ftree-parallelize-loops=$(NTHREADS)
	MODULEFLAGS = -module
	DEVFLAGS = -check all -warn
	DEBUGFLAGS = -check all -fpe0 -warn -traceback -debug extended
	OPTIMFLAGS = -O3
else ifneq (,$(filter gfortran% gcc%, $(FC)))
	PPFLAGS = -cpp -gdwarf-2
	MPFLAGS = -fopenmp -ftree-parallelize-loops=$(NTHREADS)
	MODULEFLAGS = -J
	WARNFLAGS = -Wall
	DEVFLAGS = -g -fbacktrace -fcheck=all -fbounds-check -Og
	DEBUGFLAGS = -fbounds-check
	MEMFLAGS = -mcmodel=large
	OPTIMFLAGS = -O3 -march=native
	FASTFLAGS = -Ofast -march=native
else ifneq (,$(filter nag% nagfor% nagfmcheck%, $(FC)))
	PPFLAGS = -f2018 -fpp
	MPFLAGS = -openmp
	MODULEFLAGS = -mdir ${BUILD_DIR} -I
	WARNFLAGS = -Wall
	DEVFLAGS = -g -mtrace -C=all -colour -O0
	DEBUGFLAGS = -C=array
	MEMFLAGS = -mcmodel=large
	OPTIMFLAGS = -O3
	FASTFLAGS = -Ofast
endif


##########################################
# EXTERNAL LIBRARIES
##########################################
ifeq ($(UNAME_S),Darwin)
	EXT_LIBS ?= -framework Accelerate
else
	EXT_LIBS ?=
endif


##########################################
# COMPILATION SECTION
##########################################
INSTALL_DIR ?= $(HOME)/bin
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
$(info NTHREADS = $(NTHREADS))
	CFLAGS+=$(MPFLAGS)
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
endif


.PHONY: all install build uninstall clean cleanall run check-build-config

programs = $(BIN_DIR)/$(NAME)
all: $(programs)

build: all
	@:

run: $(programs)
	cd $(RUN_DIR) && $(SETUP_ENV_CMD)$(RUN_ENV) $(abspath $(programs)) $(RUN_ARGS)

%:
	@:

check-build-config:
	@test -n "$(ATHENA_MOD_DIR)" || (echo "Missing athena.mod in build cache" && exit 1)
	@test -n "$(ATHENA_OBJS)" || (echo "Missing athena object files in build cache" && exit 1)

$(BIN_DIR):
	mkdir -p $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/wf_c.o: $(WANDD_C_SRC) | $(BUILD_DIR)
	$(SETUP_ENV_CMD)$(CC) $$FPM_CFLAGS -c $< -o $@

$(programs): $(SOURCES) $(WANDD_C_OBJ) | $(BIN_DIR) $(BUILD_DIR) check-build-config
	$(SETUP_ENV_CMD)$(FC) $(PPFLAGS) $(CFLAGS) $(MODULEFLAGS) $(BUILD_DIR) \
		$(CPP_FLAGS) $(INCLUDE_FLAGS) $$FPM_CFLAGS $(SOURCES) \
		$(DEPENDENCY_OBJS) $(EXT_LIBS) $(EXTRA_LIBS) $$FPM_LDFLAGS -o $@

install: $(programs) | $(INSTALL_DIR)
	cp $(programs) $(INSTALL_DIR)/$(NAME)

$(INSTALL_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)/ $(BIN_DIR)/

cleanall: clean

uninstall:
	rm -f $(INSTALL_DIR)/$(NAME)
