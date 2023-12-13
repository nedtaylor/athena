program test_conv2d_network
  use athena, only: &
       network_type, &
       conv2d_layer_type, &
       base_optimiser_type
  implicit none

  type(network_type) :: network

  real, allocatable, dimension(:,:,:,:) :: input_data, output
  logical :: success = .true.

  write(*,*) "test_conv2d_network"
  !! create network
  call network%add(conv2d_layer_type( &
       input_shape=[32, 32, 3], &
       num_filters = 16, &
       kernel_size = 3 &
       ))
  call network%add(conv2d_layer_type( &
       num_filters = 32, &
       kernel_size = 3 &
       ))
    
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=1.0), &
       loss_method="mse", metrics=["loss"], verbose=1, &
       batch_size=1)

  if(network%num_layers.ne.3)then
    success = .false.
    write(*,*) "conv2d network should have 3 layers"
  end if

  call network%set_batch_size(1)
  allocate(input_data(32, 32, 3, 1))
  input_data = 0.0

  call network%forward(input_data)
  call network%model(3)%layer%get_output(output)

  if(all(shape(output).ne.[28,28,32,1]))then
     success = .false.
     write(*,*) "conv2d network output shape should be [28,28,32]"
  end if

  !! check for any fails
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_conv2d_network passed all tests'
  else
     write(*,*) 'test_conv2d_network failed one or more tests'
     stop 1
  end if

end program test_conv2d_network