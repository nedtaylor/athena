cmake_minimum_required(VERSION 3.17.5)

# define build environments
set( CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local/athena"
  CACHE STRING "Select where to install the library." )
execute_process(COMMAND pwd OUTPUT_VARIABLE CURRENT_WORKING_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
message("Current working directory: ${CURRENT_WORKING_DIR}")
set( CMAKE_BUILD_PREFIX ${CURRENT_WORKING_DIR}
  CACHE STRING "Select where to build the library." )

# set the project name
project(athena NONE)

# set the library name
set( LIB_NAME ${PROJECT_NAME} )
set( PROJECT_DESCRIPTION
  "Fortran neural network" )
set( PROJECT_URL "https://github.com/nedtaylor/athena" )
set( CMAKE_CONFIGURATION_TYPES "Release" "Parallel" "Serial" "Dev" "Debug" "Parallel_Dev"
  CACHE STRING "List of configurations types." )
set( CMAKE_BUILD_TYPE "Release"
  CACHE STRING "Select which configuration to build." )

# change name based on parallel
if (CMAKE_BUILD_TYPE MATCHES "Parallel*")
   project(athena_omp NONE) # change project name to parallel
   message(FATAL_ERROR "Configuration stopped because Parallel is not yet set up")
endif()

# set compiler
set(CMAKE_Fortran_COMPILER gfortran
  CACHE STRING "Select fortran compiler." )  # Change this to your desired compiler
set(CMAKE_Fortran_STANDARD 2018)

# set language
enable_language(Fortran)

# set coverage compiler flags
if (CMAKE_BUILD_TYPE MATCHES "Debug*" OR CMAKE_BUILD_TYPE MATCHES "Dev*")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
      include(CodeCoverage)
      setup_target_for_coverage_gcovr_html(
        NAME coverage
        EXECUTABLE ctest
        EXCLUDE "${PROJECT_SOURCE_DIR}/test/*")
  endif()
endif()

# enable testing
enable_testing()

# set options for building tests and examples
option(BUILD_TESTS "Build the unit tests" ON)
option(BUILD_EXAMPLES "Build the examples" ON)

# Define the sources
set(SRC_DIR src)
set(LIB_DIR ${SRC_DIR}/lib)

set(LIB_FILES
    mod_constants.f90
    mod_misc.f90
    mod_random.f90
    mod_clipper.f90
    mod_regulariser.f90
    mod_lr_decay.f90
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
    mod_initialiser_ones.f90
    mod_initialiser_zeros.f90
    mod_initialiser_ident.f90
    mod_initialiser_gaussian.f90
    mod_activation.f90
    mod_initialiser.f90
    mod_tools_infile.f90
    mod_normalisation.f90
    mod_loss.f90
    mod_accuracy.f90
    mod_base_layer.f90
    mod_base_layer_sub.f90
    mod_batchnorm1d_layer.f90
    mod_batchnorm2d_layer.f90
    mod_batchnorm3d_layer.f90
    mod_conv1d_layer.f90
    mod_conv2d_layer.f90
    mod_conv3d_layer.f90
    mod_dropout_layer.f90
    mod_dropblock2d_layer.f90
    mod_dropblock3d_layer.f90
    mod_avgpool1d_layer.f90
    mod_avgpool2d_layer.f90
    mod_avgpool3d_layer.f90
    mod_maxpool1d_layer.f90
    mod_maxpool2d_layer.f90
    mod_maxpool3d_layer.f90
    mod_full_layer.f90
    mod_flatten1d_layer.f90
    mod_flatten2d_layer.f90
    mod_flatten3d_layer.f90
    mod_flatten4d_layer.f90
    mod_input1d_layer.f90
    mod_input2d_layer.f90
    mod_input3d_layer.f90
    mod_input4d_layer.f90
    mod_container_layer.f90
    mod_container_layer_sub.f90
    mod_network.f90
    mod_network_sub.f90
)
foreach(lib ${LIB_FILES})
    list(APPEND PREPENDED_LIB_FILES ${LIB_DIR}/${lib})
endforeach()
message(STATUS "Modified LIB_FILES: ${PREPENDED_LIB_FILES}")

if (CMAKE_BUILD_TYPE MATCHES "Parallel*")
    set(SRC_FILES athena_omp.f90)
else()
    set(SRC_FILES athena.f90)
endif()
foreach(src ${SRC_FILES})
    list(APPEND PREPENDED_SRC_FILES ${SRC_DIR}/${src})
endforeach()
message(STATUS "Modified SRC_FILES: ${PREPENDED_SRC_FILES}")

# initialise flags
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

# set flags based on compiler
if (CMAKE_Fortran_COMPILER MATCHES ".*gfortran.*" OR CMAKE_Fortran_COMPILER MATCHES ".*gcc.*")
    message(STATUS "Using gfortran compiler")
    set(PPFLAGS -cpp)
    set(MPFLAGS -fopenmp)
    set(WARNFLAGS -Wall)
    set(DEVFLAGS -g -fbacktrace -fcheck=all -fbounds-check -Og)
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
    set(PPFLAGS -fpp)
    set(MPFLAGS -qopenmp)
    set(WARNFLAGS -warn all)
    set(DEVFLAGS -check all -warn)
    set(DEBUGFLAGS -check all -fpe0 -warn -tracekback -debug extended)
    set(MEMFLAGS -mcmodel=large)
    set(OPTIMFLAGS -O3)
    set(FASTFLAGS -Ofast)
else()
    # Code for other Fortran compilers
    message(STATUS "Using a different Fortran compiler")
endif()

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${PPFLAGS}")

# Add custom build types for different configurations
# add_compile_options("$<$<CONFIG:dev>:${DEVFLAGS}>")
# add_compile_options("$<$<CONFIG:debug>:${DEBUGFLAGS}>")
# add_compile_options("$<$<CONFIG:warn>:${WARNFLAGS}>")
# add_compile_options("$<$<CONFIG:optim>:${OPTIMFLAGS}>")
# add_compile_options("$<$<CONFIG:fast>:${FASTFLAGS}>")
# add_compile_options("$<$<CONFIG:mp>:${MPFLAGS}>")


# create the library
add_library(${PROJECT_NAME} STATIC ${PREPENDED_LIB_FILES} ${PREPENDED_SRC_FILES})
set(MODULE_DIR ${CMAKE_BUILD_PREFIX}/modules)
set_target_properties(${PROJECT_NAME} PROPERTIES Fortran_MODULE_DIRECTORY ${MODULE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC)

# replace ".f90" with ".mod"
string(REGEX REPLACE "\\.[^.]*$" ".mod" MODULE_FILES "${SRC_FILES}")

# installation
install(FILES ${MODULE_DIR}/${MODULE_FILES} DESTINATION include)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

# set compile options based on different build configurations
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Release>:${OPTIMFLAGS}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Parallel>:${OPTIMFLAGS}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Parallel>:${MPFLAGS}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Dev>:${DEVFLAGS}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:${DEBUGFLAGS}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Parallel_Dev>:${MPFLAGS}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Parallel_Dev>:${DEVFLAGS}>")

# include the build test directory
if(BUILD_TESTS)
  add_subdirectory(test)
endif()

# add coverage compiler flags
if ( ( CMAKE_Fortran_COMPILER MATCHES ".*gfortran.*" OR CMAKE_Fortran_COMPILER MATCHES ".*gcc.*" ) AND 
     ( CMAKE_BUILD_TYPE MATCHES "Debug*" OR CMAKE_BUILD_TYPE MATCHES "Dev*" ) )
  append_coverage_compiler_flags()
endif()