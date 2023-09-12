cmake_minimum_required(VERSION 3.17.5)

set( CMAKE_INSTALL_PREFIX "."
  CACHE STRING "Select where to install the library." )

project(athena NONE)



set( LIB_NAME ${PROJECT_NAME} )
set( PROJECT_DESCRIPTION
  "Fortran neural network" )
set( PROJECT_URL "https://https://git.exeter.ac.uk/hepplestone/convolutional_neural_network" )
set( CMAKE_CONFIGURATION_TYPES "Release" "Dev" )
set( CMAKE_BUILD_TYPE "Release"
  CACHE STRING "Select which configuration to build." )

set(CMAKE_Fortran_COMPILER gfortran)  # Change this to your desired compiler
set(CMAKE_Fortran_STANDARD 2018)

enable_language(Fortran)

# Define the sources
set(SRC_DIR "src")
set(LIB_DIR "${SRC_DIR}/lib")

set(LIBS
    mod_constants.f90
    mod_misc.f90
    mod_random.f90
    mod_optimiser.f90
    mod_types.f90
    mod_metrics.f90
    mod_misc_ml.f90
    mod_activation_gaussian.f90
    mod_activation_linear.f90
    mod_activation_piecewise.f90
    mod_activation_relu.f90
    mod_activation_leaky_relu.f90
    mod_activation_sigmoid.f90
    mod_activation_tanh.f90
    mod_activation_none.f90
    mod_activation_softmax.f90
    mod_initialiser_glorot.f90
    mod_initialiser_he.f90
    mod_initialiser_lecun.f90
    mod_initialiser_zeros.f90
    mod_activation.f90
    mod_initialiser.f90
    mod_tools_infile.f90
    mod_normalisation.f90
    mod_batch_norm.f90
    mod_loss_categorical.f90
    mod_base_layer.f90
    mod_conv2d_layer.f90
    mod_conv3d_layer.f90
    mod_maxpool2d_layer.f90
    mod_maxpool3d_layer.f90
    mod_full_layer.f90
    mod_flatten2d_layer.f90
    mod_flatten3d_layer.f90
    mod_input1d_layer.f90
    mod_input3d_layer.f90
    mod_input4d_layer.f90
    mod_container_layer.f90
    mod_container_layer_sub.f90
    mod_network.f90
)
foreach(lib ${LIBS})
    list(APPEND PREPENDED_LIBS "${LIB_DIR}/${lib}")
endforeach()
message(STATUS "Modified LIBS: ${PREPENDED_LIBS}")

set(SRCS athena.f90)
foreach(src ${SRCS})
    list(APPEND PREPENDED_SRCS "${SRC_DIR}/${src}")
endforeach()
message(STATUS "Modified SRCS: ${PREPENDED_SRCS}")


set(CPPFLAGS "")
set(CFLAGS "")
set(MODULEFLAGS "")
set(MPFLAGS "")
set(WARNFLAGS "")
set(DEVFLAGS "")
set(DEBUGFLAGS "")
set(MEMFLAGS "")
set(OPTIMFLAGS "")
set(FASTFLAGS "")

if (CMAKE_Fortran_COMPILER MATCHES ".*gfortran.*" OR CMAKE_Fortran_COMPILER MATCHES ".*gcc.*")
    message(STATUS "Using gfortran compiler")
    set(PPFLAGS -cpp)
    set(MPFLAGS -fopenmp)
    set(WARNFLAGS -Wall)
    set(DEVFLAGS -fbacktrace -fcheck=all -fbounds-check -Og)
    set(DEBUGFLAGS -fbounds-check)
    set(MEMFLAGS -mcmodel=large)
    set(OPTIMFLAGS -O3 -march=native)
    set(FASTFLAGS -Ofast -march=native)
elseif (CMAKE_Fortran_COMPILER MATCHES ".*nag.*")
    message(STATUS "Using nag compiler")
    set(PPFLAGS -f2018 -fpp)
    set(MPFLAGS -openmp)
    set(WARNFLAGS -Wall)
    set(DEVFLAGS -g -mtrace -C=all -colour -O0)
    set(DEBUGFLAGS -C=array)
    set(MEMFLAGS -mcmodel=large)
    set(OPTIMFLAGS -O3)
    set(FASTFLAGS -Ofast)
elseif (CMAKE_Fortran_COMPILER MATCHES ".*ifort.*" OR CMAKE_Fortran_COMPILER MATCHES ".*ifx.*")
    message(STATUS "Using intel compiler")
    set(PPFLAGS -stand f18 -cpp)
    set(MPFLAGS -qopenmp)
    set(DEVFLAGS -check all -warn)
    set(DEBUGFLAGS -check all -fpe0 -warn -tracekback -debug extended)
    set(OPTIMFLAGS -O3)
else()
    # Code for other Fortran compilers
    message(STATUS "Using a different Fortran compiler")
endif()

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${PPFLAGS}")

# Add custom build types for different configurations
add_compile_options("$<$<CONFIG:dev>:${DEVFLAGS}>")
add_compile_options("$<$<CONFIG:debug>:${DEBUGFLAGS}>")
add_compile_options("$<$<CONFIG:warn>:${WARNFLAGS}>")
add_compile_options("$<$<CONFIG:optim>:${OPTIMFLAGS}>")
add_compile_options("$<$<CONFIG:fast>:${FASTFLAGS}>")
add_compile_options("$<$<CONFIG:mp>:${MPFLAGS}>")


# Create the library
add_library(athena STATIC ${PREPENDED_LIBS} ${PREPENDED_SRCS})
set_target_properties(athena PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

# # Link libraries and compile options
# target_link_libraries(athena PRIVATE ${CFLAGS})

# Installation
set(INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/bin)  # Change this to your desired installation path
install(TARGETS athena DESTINATION ${INSTALL_DIR})
