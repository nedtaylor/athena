message(STATUS "Building tests")
foreach(execid
  input_layer
  conv2d_layer
  conv2d_network
  conv3d_layer
  conv3d_network
  full_layer
  full_network
  maxpool2d_layer
  maxpool3d_layer
  avgpool2d_layer
  avgpool3d_layer
  batchnorm2d_layer
  batchnorm3d_layer
  dropout_layer
  dropblock2d_layer
  dropblock3d_layer
  flatten1d_layer
  flatten2d_layer
  flatten3d_layer
  flatten4d_layer
  activations
  initialisers
  padding
  shuffle
  split
  lr_decay
  regulariser
  misc
  normalisation
  metrics
  accuracy
  network
  random
  optimiser
  clipper
  loss
  )
  add_executable(test_${execid} test_${execid}.f90)
  # # Specify the include directories
  target_include_directories(test_${execid} PRIVATE "${CMAKE_BUILD_PREFIX}" "${CMAKE_BUILD_PREFIX}/modules")

  target_link_libraries(test_${execid} PRIVATE athena)

  add_test(NAME test_${execid} COMMAND test_${execid})
endforeach()
