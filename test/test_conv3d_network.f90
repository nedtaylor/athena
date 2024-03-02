program test_conv3d_network
  use athena, only: &
       network_type, &
       conv3d_layer_type, &
       base_optimiser_type
  implicit none

  type(network_type) :: network

  real, allocatable, dimension(:,:,:,:,:) :: input_data, output
  logical :: success = .true.

  write(*,*) "test_conv3d_network"
  !! create network
  call network%add(conv3d_layer_type( &
       input_shape=[12, 12, 12, 3], &
       num_filters = 16, &
       kernel_size = 3 &
       ))
  call network%add(conv3d_layer_type( &
       num_filters = 32, &
       kernel_size = 3 &
       ))
    
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=1.0), &
       loss_method="mse", metrics=["loss"], verbose=1, &
       batch_size=1)

  if(network%num_layers.ne.3)then
    success = .false.
    write(*,*) "conv3d network should have 3 layers"
  end if

  call network%set_batch_size(1)
  allocate(input_data(12, 12, 12, 3, 1))
  input_data = 0.0

  call network%forward(input_data)
  call network%model(3)%layer%get_output(output)

  if(any(shape(output).ne.[8,8,8,32,1]))then
     success = .false.
     write(*,*) "conv3d network output shape should be [8,8,8,32]"
  end if

  !! check for any fails
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_conv3d_network passed all tests'
  else
     write(0,*) 'test_conv3d_network failed one or more tests'
     stop 1
  end if

end program test_conv3d_network