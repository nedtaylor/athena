##########################################
# CODE DIRECTORIES AND FILES
##########################################
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
BIN_DIR := ./bin
SRC_DIR := ./src
LIB_DIR := ../../example_library/src
BUILD_DIR = ./obj
LIBS := mod_constants.f90
OBJS := $(addprefix $(LIB_DIR)/,$(LIBS))
MAIN := main.f90
SRCS := $(OBJS) $(MAIN)
OBJS := $(addprefix $(SRC_DIR)/,$(SRCS))


##########################################
# COMPILER CHOICE SECTION
##########################################
NTHREADS=4
FC=gfortran
ifneq (,$(filter ifort% ifx%, $(FC)))
	PPFLAGS = -stand f18 -cpp
	MPFLAGS = -qopenmp -ftree-parallelize-loops=$(NTHREADS)
	MODULEFLAGS = -module
	DEVFLAGS = -check all -warn #all
	DEBUGFLAGS = -check all -fpe0 -warn -tracekback -debug extended # -check bounds
	OPTIMFLAGS = -O3
else ifneq (,$(filter gfortran% gcc%, $(FC)))
	PPFLAGS = -cpp -gdwarf-2
	MPFLAGS = -fopenmp -ftree-parallelize-loops=$(NTHREADS)
	MODULEFLAGS = -J
	WARNFLAGS = -Wall
	DEVFLAGS = -g -fbacktrace -fcheck=all -fbounds-check -Og #-g -static -ffpe-trap=invalid
	DEBUGFLAGS = -fbounds-check
	MEMFLAGS = -mcmodel=large
	OPTIMFLAGS = -O3 -march=native
	FASTFLAGS = -Ofast -march=native
else ifneq (,$(filter nag% nagfor% nagfmcheck%, $(FC)))
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
endif


##########################################
# EXTERNAL DEPENDENCIES
##########################################
ATHENA_INSTALL_PATH=../../bin


##########################################
# LIBATHENA
##########################################
LIBATHENA = $(ATHENA_INSTALL_PATH)/lib/libathena.a
ATHENA_INCLUDES = $(ATHENA_INSTALL_PATH)/include


##########################################
# EXTERNAL LIBRARIES
##########################################
LIBS_PATH_COMPILER =
LIBS_PATH_LINKER =
EXT_LIBS =


##########################################
# MAIN TARGET
##########################################
TARGET := rnn_timeseries


##########################################
# BUILD TARGET
##########################################
all: libathena $(TARGET)

$(TARGET): $(BUILD_DIR) $(BIN_DIR) $(MAIN) $(LIBS)
	$(FC) $(MPFLAGS) $(DEVFLAGS) $(MEMFLAGS) -o $(BIN_DIR)/$(TARGET) \
	$(MODULEFLAGS) $(BUILD_DIR) \
	-I$(ATHENA_INCLUDES) \
	$(OBJS) \
	$(LIBATHENA) \
	$(LIBS_PATH_COMPILER) \
	$(LIBS_PATH_LINKER) \
	$(EXT_LIBS)

libathena:
	cd ../.. && fpm install --prefix $(mkfile_dir)/$(ATHENA_INSTALL_PATH)


##########################################
# DIRECTORY CREATION
##########################################
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)


##########################################
# CLEANING
##########################################
clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod

cleanall:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.mod $(BIN_DIR)/$(TARGET)
	rm -f rnn_timeseries_model.txt


##########################################
# RUN TARGET
##########################################
run: all
	./$(BIN_DIR)/$(TARGET)


.PHONY: all clean cleanall run libathena
