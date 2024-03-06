program test_full_network
  use athena, only: &
       network_type, &
       full_layer_type, &
       sgd_optimiser_type
  implicit none

  type(network_type) :: network

  real, allocatable, dimension(:) :: output_1d
  real, allocatable, dimension(:,:) :: output_2d

  logical :: success = .true.

  call network%add(full_layer_type( &
       num_inputs=1, &
       num_outputs=1, &
       kernel_initialiser='ones' &
       ))

  call network%compile( &
       optimiser=sgd_optimiser_type(learning_rate=1.0), &
       loss_method='mse', &
       metrics=['loss'], &
       batch_size = 1, &
       verbose=1 &
       )

  !! check network has correct number of layers
  if(network%num_layers .ne. 2)then
     success = .false.
     write(0,*) 'network has wrong number of layers'
  end if

  !! train network
  training: block
     real, dimension(1,1) :: x, y
     real :: tol = 1.E-3
     integer :: n
     integer, parameter :: num_iterations = 1000

     x(1,1) = 0.124
     y(1,1) = 0.765

     train_loop: do n=1, num_iterations
        call network%forward(x)
        call network%backward(y)
        call network%update()
        if(all(abs(network%predict(x)-y) .lt. tol)) exit train_loop
      end do train_loop

      if(n.gt.num_iterations)then
         success = .false.
         write(0,*) 'network failed to converge'
      end if

  end block training

  call network%model(1)%layer%get_output(output_1d)
  call network%model(1)%layer%get_output(output_2d)
  if(any(abs(output_1d - reshape(output_2d, [size(output_2d)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if

  !! reset network
  call network%reset()
  call network%add(full_layer_type(num_inputs=760, num_outputs=30))
  call network%add(full_layer_type(20))
  call network%add(full_layer_type(10))

  call network%compile( &
       optimiser=sgd_optimiser_type(learning_rate=1.0), &
       loss_method='mse', &
       metrics=['loss'], &
       batch_size = 1, &
       verbose=1 &
       )
  
  !! check network has correct number of layers
  if(network%num_layers .ne. 4)then
     success = .false.
     write(0,*) 'network has wrong number of layers'
  end if


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_full_network passed all tests'
  else
     write(0,*) 'test_full_network failed one or more tests'
     stop 1
  end if

end program test_full_network