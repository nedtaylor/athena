message(STATUS "Building tests")
foreach(execid
  conv2d_layer
  conv2d_network
  full_layer
  full_network
  )
  add_executable(test_${execid} test_${execid}.f90)
  # # Specify the include directories
  target_include_directories(test_${execid} PRIVATE "${CMAKE_BUILD_PREFIX}" "${CMAKE_BUILD_PREFIX}/modules")

  target_link_libraries(test_${execid} PRIVATE athena)

  add_test(NAME test_${execid} COMMAND test_${execid})
endforeach()
