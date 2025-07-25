program test_onnx
  !! Test program for ONNX export and import functionality
  use athena__constants, only: real32
  use athena

  implicit none

  type(network_type) :: network, network_imported

  ! Create a simple network
  call network%add(input_layer_type(input_shape=[21,21,3]))
  call network%add(conv2d_layer_type( &
       kernel_size=[3,3], &
       num_filters=1 &
  ))
  call network%add(maxpool2d_layer_type( &
       pool_size=[2,2], &
       stride=[2,2] &
  ))
  call network%add(full_layer_type(num_outputs=64))
  call network%add(actv_layer_type('relu'))
  call network%add(full_layer_type(num_outputs=10))

  call network%compile( &
       optimiser = sgd_optimiser_type( &
            learning_rate = 1.E-2_real32 &
       ), & ! Compile the network with the Adam optimiser
       loss_method = "mse", &
       accuracy_method = "mse", &
       verbose = 1 &
  )
  ! write(*,*) network%get_params()

  ! Export to ONNX
  call write_onnx('test_model.onnx', network)
  print *, 'Network exported to test_model.onnx'

  write(*,*) 'ONNX test completed successfully!'

end program test_onnx
