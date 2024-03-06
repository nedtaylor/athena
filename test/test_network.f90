program test_network
  use athena, only: &
       full_layer_type, &
       input1d_layer_type, &
       network_type, &
       base_optimiser_type, &
       maxpool2d_layer_type, &
       conv2d_layer_type, &
       batchnorm2d_layer_type, &
       dropblock2d_layer_type, &
       flatten2d_layer_type, &
       conv3d_layer_type
   use loss, only: &
       compute_loss_bce, &
       compute_loss_cce, &
       compute_loss_mae, &
       compute_loss_mse, &
       compute_loss_nll, &
       compute_loss_function
   implicit none

  type(network_type) :: network, network3
  type(network_type), allocatable :: network2
  real, allocatable, dimension(:) :: gradients
  real, allocatable, dimension(:,:) :: x, y
  
  real, parameter :: learning_rate = 0.1

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  integer :: i, n
  real :: rtmp1
  logical :: success = .true.
  procedure(compute_loss_function), pointer :: get_loss => null()


  !! set random seed
  seed_size = 8
  call random_seed(size=seed_size)
  seed = [1,1,1,1,1,1,1,1]
  call random_seed(put=seed)

  !! create network
  ! call network%add(input1d_layer_type(input_shape=[1]))
  call network%add(full_layer_type( &
       num_inputs=3, num_outputs=5, activation_function="tanh"))
  call network%add(full_layer_type( &
       num_outputs=2, activation_function="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", metrics=["loss"], verbose=1)
  call network%set_batch_size(1)

  !! create train data
  x = reshape([0.2, 0.4, 0.6], [3,1])
  y = reshape([0.123456, 0.246802], [2,1])

  !! train network
  write(*,*) "Training network"
  call network%train(x, y, num_epochs=600, batch_size=1, verbose=0)
  write(*,*) "Network trained"

  if(network%metrics(1)%val.gt.1.E-3) then
     write(*,*) "Training loss higher than expected"
     success = .false.
  end if
  if(network%metrics(2)%val.lt.0.95) then
     write(*,*) "Training accuracy higher than expected"
     success = .false.
  end if


  !! create test data
  x = reshape([0.4, 0.6, 0.8], [3,1])
  y = reshape([0.370368, 0.493824], [2,1])
  call network%test(x, y)
  if(network%loss.gt.1.E-1) then
     write(*,*) "Test loss higher than expected"
     success = .false.
  end if
  if(network%accuracy.lt.0.7) then
     write(*,*) "Test accuracy higher than expected"
     success = .false.
  end if

  !! check network allocation
  allocate(network2, source=network_type(layers=network%model, batch_size=4))
  if(network2%batch_size.ne.4) then
     write(*,*) "Batch size not set correctly"
     success = .false.
  end if

  !! check gradients
  call network2%set_gradients(0.1)
  if(any(abs(network2%get_gradients()-0.1).gt.1.E-6)) then
     write(*,*) "Gradients not set correctly"
     success = .false.
  end if
  allocate(gradients(network%get_num_params()))
  gradients = 0.2
  call network%set_gradients(gradients)
  if(any(abs(network%get_gradients()-0.2).gt.1.E-6)) then
     write(*,*) "Gradients not set correctly"
     success = .false.
  end if
  
  !! check network copy
  call network3%copy(network)
  if(abs(network3%metrics(1)%val-network%metrics(1)%val).gt.1.E-6) then
     write(*,*) "Network copy failed"
     success = .false.
  end if
  if(size(network3%model).ne.size(network%model)) then
     write(*,*) "Network copy failed"
     success = .false.
  end if
  rtmp1 = network3%metrics(1)%val
  
  !! check network reduce
  call network3%reduce(network)
  if(abs(network3%metrics(1)%val-(network%metrics(1)%val+rtmp1)).gt.1.E-6) then
     write(*,*) "Network reduction failed"
     success = .false.
  end if

  !! check adding all layer types
  call network3%reset()
  call network3%add(maxpool2d_layer_type())
  call network3%add(conv2d_layer_type())
  call network3%add(batchnorm2d_layer_type())
  call network3%add(dropblock2d_layer_type(block_size=3, rate=0.1))
  call network3%add(flatten2d_layer_type())

  !! check automatic flatten layer adding
  call network3%reset()
  call network3%add(conv2d_layer_type(input_shape=[3,3,3]))
  call network3%add(full_layer_type(num_outputs=5))
  call network3%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", metrics=["loss"], verbose=1)

  !! check automatic flatten layer adding
  call network3%reset()
  call network3%add(conv3d_layer_type(input_shape=[3,3,3,3]))
  call network3%add(full_layer_type(num_outputs=5))
  call network3%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", metrics=["loss"], verbose=1)

  !! check loss procedure setting
  call network3%set_loss("binary_crossentropy")
  get_loss => compute_loss_bce
  if (.not.associated(network3%get_loss, get_loss)) then
     write(*,*) "BCE loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("categorical_crossentropy")
  get_loss => compute_loss_cce
  if (.not.associated(network3%get_loss, get_loss)) then
     write(*,*) "CCE loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("mean_absolute_error")
  get_loss => compute_loss_mae
  if (.not.associated(network3%get_loss, get_loss)) then
     write(*,*) "MAE loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("mean_squared_error")
  get_loss => compute_loss_mse
  if (.not.associated(network3%get_loss, get_loss)) then
     write(*,*) "MSE loss method not set correctly"
     success = .false.
  end if
  call network3%set_loss("negative_loss_likelihood")
  get_loss => compute_loss_nll
  if (.not.associated(network3%get_loss, get_loss)) then
     write(*,*) "NLL loss method not set correctly"
     success = .false.
  end if

!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_network passed all tests'
  else
     write(0,*) 'test_network failed one or more tests'
     stop 1
  end if


end program test_network
