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

# include FetchContent module
include(FetchContent)

# Disable testing for dependencies - both standard variables
set(BUILD_TESTING OFF)
set(BUILD_TESTS OFF)

# set up the coreutils dependency
FetchContent_Declare(
    coreutils
    GIT_REPOSITORY https://github.com/nedtaylor/coreutils.git
    GIT_TAG        main
    FIND_PACKAGE_ARGS NAMES coreutils
)
FetchContent_MakeAvailable(coreutils)
find_package(coreutils REQUIRED)

message(STATUS "coreutils home directory: ${coreutils_DIR}")
message(STATUS "coreutils include directory: ${coreutils_INCLUDE_DIRS}")
message(STATUS "coreutils source directory: ${coreutils_BUILD_DIRS}")

# set up the diffstruc dependency
FetchContent_Declare(
    diffstruc
    GIT_REPOSITORY https://github.com/nedtaylor/diffstruc.git
    GIT_TAG        v1.1.0
    FIND_PACKAGE_ARGS NAMES diffstruc
)
FetchContent_MakeAvailable(diffstruc)
find_package(diffstruc REQUIRED)

message(STATUS "diffstruc home directory: ${diffstruc_DIR}")
message(STATUS "diffstruc include directory: ${diffstruc_INCLUDE_DIRS}")
message(STATUS "diffstruc source directory: ${diffstruc_BUILD_DIRS}")

# set up the graphstruc dependency
FetchContent_Declare(
    graphstruc
    GIT_REPOSITORY https://github.com/nedtaylor/graphstruc.git
    GIT_TAG        v0.2.0
    FIND_PACKAGE_ARGS NAMES graphstruc
)
FetchContent_MakeAvailable(graphstruc)
# find_library(ATHENA_LIBRARY NAMES athena)
find_package(graphstruc REQUIRED)
# include_directories(${GRAPHSTRUC_INCLUDE_DIR})

message(STATUS "graphstruc home directory: ${graphstruc_DIR}")
message(STATUS "graphstruc include directory: ${graphstruc_INCLUDE_DIRS}")
message(STATUS "graphstruc source directory: ${graphstruc_BUILD_DIRS}")

# Re-enable BUILD_TESTS for our project
set(BUILD_TESTS ON)

# Only enable testing and options when this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  enable_testing()

  # set options for building tests and examples
  option(BUILD_TESTS "Build the unit tests" ON)
  option(BUILD_EXAMPLES "Build the examples" ON)
else()
  # When used as a dependency, disable tests by default
  set(BUILD_TESTS OFF)
  set(BUILD_EXAMPLES OFF)
endif()


# Define the sources
set(SRC_DIR src)
set(LIB_DIR ${SRC_DIR}/athena)
# Automatically find all Fortran files in athena/
file(GLOB LIB_FILES "${LIB_DIR}/*.f90")

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

# add library dependencies
add_dependencies(${PROJECT_NAME} coreutils)
target_link_libraries(${PROJECT_NAME} coreutils)
add_dependencies(${PROJECT_NAME} diffstruc)
target_link_libraries(${PROJECT_NAME} diffstruc)
add_dependencies(${PROJECT_NAME} graphstruc)
target_link_libraries(${PROJECT_NAME} graphstruc)

# include the build test directory
if(BUILD_TESTS)
  add_subdirectory(test)
endif()

# add coverage compiler flags
if ( ( CMAKE_Fortran_COMPILER MATCHES ".*gfortran.*" OR CMAKE_Fortran_COMPILER MATCHES ".*gcc.*" ) AND
     ( CMAKE_BUILD_TYPE MATCHES "Debug*" OR CMAKE_BUILD_TYPE MATCHES "Dev*" ) )
  append_coverage_compiler_flags()
endif()
