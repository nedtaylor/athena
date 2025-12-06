message(STATUS "Building tests for ${PROJECT_NAME}")

# Only find test files in THIS project's test directory, not dependencies
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.f90")

# Filter to ensure we only get files from this specific directory
list(FILTER TEST_SOURCES INCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/test_.*\\.f90$")


message(STATUS "Test sources found: ${TEST_SOURCES}")

# Build each test
foreach(test_file ${TEST_SOURCES})
  # Extract the filename without extension
  get_filename_component(test_name ${test_file} NAME_WE)

  message(STATUS "Found test: ${test_name}")

  add_executable(${test_name} ${test_file})

  # Specify the include directories
  target_include_directories(${test_name} PRIVATE "${CMAKE_BUILD_PREFIX}" "${CMAKE_BUILD_PREFIX}/mod")

  target_link_libraries(${test_name} PRIVATE athena)

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
