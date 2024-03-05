program test_network
  use athena, only: &
       full_layer_type, &
       input1d_layer_type, &
       network_type, &
       base_optimiser_type
  implicit none

  type(network_type) :: network
  real, allocatable, dimension(:,:) :: x, y
  
  real, parameter :: learning_rate = 0.1

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  integer :: i, n
  logical :: success = .true.


  !! set random seed
  seed_size = 8
  call random_seed(size=seed_size)
  seed = [1,1,1,1,1,1,1,1]
  call random_seed(put=seed)

  write(*,*) "Simple function approximation using a fully-connected neural network"
  write(*,*) "--------------------------------------------------"
  write(*,*) "Based on example provided in the neural-fortran code"

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
