program onnx_example
  use athena
  implicit none

  type(network_type) :: network_write, network_read
  character(256) :: onnx_file

  ! Create a simple network
  write(*,*) "Creating a simple test network..."

  call network_write%add(conv2d_layer_type( &
       input_shape=[28,28,1], &
       num_filters=6, &
       kernel_size=3, &
       activation="relu"))

  call network_write%add(maxpool2d_layer_type( &
       pool_size=2, &
       stride=2))

  call network_write%add(full_layer_type( &
       num_outputs=12, &
       activation="relu"))

  call network_write%add(full_layer_type( &
       num_outputs=10, &
       activation="softmax"))

  call network_write%compile( &
       optimiser=base_optimiser_type(learning_rate=0.01), &
       loss_method="categorical_crossentropy", &
       batch_size=1)

  ! Write network to ONNX
  onnx_file = "test_network.onnx"
  write(*,*) "Writing network to ONNX file: ", trim(onnx_file)
  call write_onnx(onnx_file, network_write)
  write(*,*) "ONNX file written successfully"

  ! Read network from ONNX
  write(*,*) ""
  write(*,*) "Reading network from ONNX file..."
  network_read = read_onnx(onnx_file)


  call network_read%compile( &
       optimiser=base_optimiser_type(learning_rate=0.01), &
       loss_method="categorical_crossentropy", &
       batch_size=1)
  write(*,*) size(network_read%model)
  call network_read%print_summary()
  call write_onnx("test_network_second_pass.onnx", network_read)

  write(*,*) ""
  write(*,*) "Network read completed"
  write(*,*) "Number of layers in original network: ", network_write%num_layers
  write(*,*) "Number of layers in read network: ", network_read%num_layers

end program onnx_example
