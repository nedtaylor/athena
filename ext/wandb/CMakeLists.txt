cmake_minimum_required(VERSION 3.17.5)
project(athena_wandb C Fortran)

# --------------------------------------------------------------------------- #
#  Find Python (interpreter + development libs)                                #
# --------------------------------------------------------------------------- #
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 lib:     ${Python3_LIBRARIES}")


# --------------------------------------------------------------------------- #
#  Build the wandb bridge library (C + Fortran)                                #
# --------------------------------------------------------------------------- #
add_library(athena_wandb STATIC
  wandb_c.c
  athena_wandb.f90
)

# C source needs the Python headers
target_include_directories(athena_wandb PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(athena_wandb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Fortran .mod output directory
set(WANDB_MOD_DIR ${CMAKE_CURRENT_BINARY_DIR}/mod)
set_target_properties(athena_wandb PROPERTIES
  Fortran_MODULE_DIRECTORY ${WANDB_MOD_DIR}
)
target_include_directories(athena_wandb PUBLIC ${WANDB_MOD_DIR})

# Link Python
target_link_libraries(athena_wandb PUBLIC ${Python3_LIBRARIES})


# --------------------------------------------------------------------------- #
#  Install                                                                     #
# --------------------------------------------------------------------------- #
install(TARGETS athena_wandb DESTINATION lib)
install(DIRECTORY ${WANDB_MOD_DIR}/ DESTINATION include
  FILES_MATCHING PATTERN "*.mod")
