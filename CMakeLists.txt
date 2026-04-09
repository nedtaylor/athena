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

# extract version from fpm.toml
file(READ "${CMAKE_SOURCE_DIR}/fpm.toml" FPM_TOML)
string(REGEX MATCH
    "version[ \t]*=[ \t]*\"([^\"]+)\""
    _match
    "${FPM_TOML}"
)
if(NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "Could not extract version from fpm.toml")
endif()
set(ATHENA_VERSION "${CMAKE_MATCH_1}")
message(STATUS "athena version: ${ATHENA_VERSION}")


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
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
      include(CodeCoverage)
      setup_target_for_coverage_gcovr_html(
        NAME coverage
        EXECUTABLE ctest
        EXCLUDE "${PROJECT_SOURCE_DIR}/test/*")
  endif()
endif()


# Disable testing for dependencies - both standard variables
set(BUILD_TESTING OFF)
set(BUILD_TESTS OFF)

# include CPM module
find_package(Git REQUIRED)
set(GIT_EXECUTABLE "${GIT_EXECUTABLE}" CACHE FILEPATH "Path to git executable" FORCE)
include(cmake/CPM.cmake)

# add dependencies
CPMAddPackage(
  NAME coreutils
  VERSION 0.1.0
  GITHUB_REPOSITORY nedtaylor/coreutils
)
CPMAddPackage(
  NAME diffstruc
  VERSION 1.2.0
  GITHUB_REPOSITORY nedtaylor/diffstruc
)
CPMAddPackage(
  NAME graphstruc
  VERSION 0.2.1
  GITHUB_REPOSITORY nedtaylor/graphstruc
)

# Set up inter-dependency module paths after targets are created
# The .mod files are generated in the build directory, not in a module directory
set(COREUTILS_MOD_DIR "${CMAKE_BINARY_DIR}/_deps/coreutils-build")
set(DIFFSTRUC_MOD_DIR "${CMAKE_BINARY_DIR}/_deps/diffstruc-build")
set(GRAPHSTRUC_MOD_DIR "${CMAKE_BINARY_DIR}/_deps/graphstruc-build")

if(TARGET diffstruc AND TARGET coreutils)
  # Make diffstruc wait for coreutils to be built first
  add_dependencies(diffstruc coreutils)
  # Add coreutils build directory where .mod files are actually generated
  target_include_directories(diffstruc BEFORE PUBLIC ${COREUTILS_MOD_DIR})
endif()

if(TARGET graphstruc AND TARGET coreutils)
  # Make graphstruc wait for coreutils to be built first
  add_dependencies(graphstruc coreutils)
  # Add coreutils build directory where .mod files are actually generated
  target_include_directories(graphstruc BEFORE PUBLIC ${COREUTILS_MOD_DIR})
endif()

# Re-enable BUILD_TESTS for our project
set(BUILD_TESTS ON)

# Only enable testing and options when this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  enable_testing()

  # set options for building tests and examples
  option(BUILD_TESTS "Build the unit tests" ON)
  option(BUILD_EXAMPLES "Build the examples" ON)
  option(BUILD_WANDB_EXAMPLES "Build W&B examples (requires ATHENA_ENABLE_WANDB=ON)" OFF)
  option(ATHENA_ENABLE_WANDB "Enable Weights & Biases logging support via wandb_network_type (_WANDB macro)" OFF)
else()
  # When used as a dependency, disable tests by default
  set(BUILD_TESTS OFF)
  set(BUILD_EXAMPLES OFF)
  set(BUILD_WANDB_EXAMPLES OFF)
endif()

# Backward/forward-compatible alias for W&B support
if(BUILD_WANDB_EXAMPLES AND NOT BUILD_EXAMPLES)
  message(FATAL_ERROR "BUILD_WANDB_EXAMPLES requires BUILD_EXAMPLES=ON")
endif()


# Define the sources
set(SRC_DIR src)
set(LIB_DIR ${SRC_DIR}/athena)
# Automatically find all Fortran files in athena/
file(GLOB LIB_FILES "${LIB_DIR}/*.f90" "${LIB_DIR}/*.F90")

# Collect the main source file
file(GLOB SRC_FILES "${SRC_DIR}/*.f90")

# Combine all source files
set(ALL_SOURCES ${SRC_FILES} ${LIB_FILES})

# Optional: print for debugging
message(STATUS "LIB_FILES: ${LIB_FILES}")
message(STATUS "SRC_FILES: ${SRC_FILES}")
message(STATUS "ALL_SOURCES: ${ALL_SOURCES}")

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
add_library(${PROJECT_NAME} STATIC ${ALL_SOURCES})
set(MODULE_DIR ${CMAKE_BUILD_PREFIX}/mod)
set_target_properties(${PROJECT_NAME} PROPERTIES Fortran_MODULE_DIRECTORY ${MODULE_DIR})

# Add dependency module directories
target_include_directories(${PROJECT_NAME} PUBLIC
  ${MODULE_DIR}
  ${COREUTILS_MOD_DIR}
  ${DIFFSTRUC_MOD_DIR}
  ${GRAPHSTRUC_MOD_DIR}
)
target_link_libraries(${PROJECT_NAME} PUBLIC)



# replace ".f90" with ".mod"
string(REGEX REPLACE "\\.[^.]*$" ".mod" MODULE_FILES "${SRC_FILES}")
# remove directory path from module files
foreach(i ${MODULE_FILES})
    get_filename_component(fname ${i} NAME)
    list(APPEND MODULE_FILES_CLEAN ${fname})
endforeach()
set(MODULE_FILES ${MODULE_FILES_CLEAN})

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

# add library dependencies
add_dependencies(${PROJECT_NAME} coreutils diffstruc graphstruc)
target_link_libraries(${PROJECT_NAME} coreutils diffstruc graphstruc)

# --------------------------------------------------------------------------- #
#  Optional: Weights & Biases support  (-DATHENA_ENABLE_WANDB=ON)             #
# --------------------------------------------------------------------------- #
if(ATHENA_ENABLE_WANDB)
  message(STATUS "W&B support enabled (-D_WANDB will be set)")

  CPMAddPackage(
    NAME wandb_fortran_dep
    GITHUB_REPOSITORY nedtaylor/wandb-fortran
    GIT_TAG development
    OPTIONS
      "BUILD_TESTING OFF"
  )

  if(NOT TARGET wandb_fortran)
    message(FATAL_ERROR "CPM failed to provide target 'wandb_fortran' from nedtaylor/wandb-fortran@development")
  endif()

  # Activate the _WANDB code-paths in athena
  target_compile_definitions(${PROJECT_NAME} PRIVATE _WANDB)

  # The athena library now needs the wf module
  add_dependencies(${PROJECT_NAME} wandb_fortran)
  target_link_libraries(${PROJECT_NAME} wandb_fortran)

  get_target_property(WF_MOD_DIR wandb_fortran Fortran_MODULE_DIRECTORY)
  if(WF_MOD_DIR)
    target_include_directories(${PROJECT_NAME} PUBLIC ${WF_MOD_DIR})
  endif()

  if(BUILD_EXAMPLES AND BUILD_WANDB_EXAMPLES)
    file(GLOB EXAMPLE_LIBRARY_SOURCES "${CMAKE_SOURCE_DIR}/example/example_library/src/*.f90")

    add_executable(wandb_sine "${CMAKE_SOURCE_DIR}/example/wandb_sine/src/main.f90")
    target_include_directories(wandb_sine PRIVATE ${MODULE_DIR} ${WF_MOD_DIR})
    target_link_libraries(wandb_sine PRIVATE ${PROJECT_NAME} wandb_fortran)

    add_executable(wandb_sweep "${CMAKE_SOURCE_DIR}/example/wandb_sweep/src/main.f90")
    target_include_directories(wandb_sweep PRIVATE ${MODULE_DIR} ${WF_MOD_DIR})
    target_link_libraries(wandb_sweep PRIVATE ${PROJECT_NAME} wandb_fortran)

    add_executable(wandb_network_sine "${CMAKE_SOURCE_DIR}/example/wandb_network_sine/src/main.F90")
    target_compile_definitions(wandb_network_sine PRIVATE _WANDB)
    target_include_directories(wandb_network_sine PRIVATE ${MODULE_DIR} ${WF_MOD_DIR})
    target_link_libraries(wandb_network_sine PRIVATE ${PROJECT_NAME} wandb_fortran)

    add_executable(wandb_pinn_burgers
      "${CMAKE_SOURCE_DIR}/example/wandb_pinn_burgers/src/main.f90"
      ${EXAMPLE_LIBRARY_SOURCES}
    )
    target_include_directories(wandb_pinn_burgers PRIVATE ${MODULE_DIR} ${WF_MOD_DIR})
    target_link_libraries(wandb_pinn_burgers PRIVATE ${PROJECT_NAME} wandb_fortran)
  endif()
elseif(BUILD_WANDB_EXAMPLES)
  message(FATAL_ERROR "BUILD_WANDB_EXAMPLES requires ATHENA_ENABLE_WANDB=ON")
endif()

# include the build test directory
if(BUILD_TESTS)
  add_subdirectory(test)
endif()

# add coverage compiler flags
if ( ( CMAKE_Fortran_COMPILER MATCHES ".*gfortran.*" OR CMAKE_Fortran_COMPILER MATCHES ".*gcc.*" ) AND
     ( CMAKE_BUILD_TYPE MATCHES "Debug*" OR CMAKE_BUILD_TYPE MATCHES "Dev*" ) )
  append_coverage_compiler_flags()
endif()
