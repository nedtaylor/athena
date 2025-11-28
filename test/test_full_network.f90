program test_full_network
  use coreutils, only: real32
  use athena, only: &
       network_type, &
       full_layer_type, &
       sgd_optimiser_type
  use diffstruc, only: array_type
  implicit none

  type(network_type) :: network

  real, allocatable, dimension(:) :: output_1d
  real, allocatable, dimension(:,:) :: output_2d
  type(array_type), pointer :: loss

  logical :: success = .true.


!-------------------------------------------------------------------------------
! set up network
!-------------------------------------------------------------------------------
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


!-------------------------------------------------------------------------------
! manually train network
!-------------------------------------------------------------------------------
  training: block
    real, dimension(1,1) :: x, y
    real :: tol = 1.E-3
    integer :: n
    integer, parameter :: num_iterations = 1000

    x(1,1) = 0.124
    y(1,1) = 0.765

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(source=y)
    train_loop: do n = 1, num_iterations
       call network%forward(x)
       loss => network%loss_backward(1, 1)
       call loss%grad_reverse()
       call network%update()
       if(all(abs(network%predict(x)-y) .lt. tol)) exit train_loop
       call loss%nullify_graph()
       loss => null()
    end do train_loop

    if(n.gt.num_iterations)then
       success = .false.
       write(0,*) 'network failed to converge'
    end if

  end block training


!-------------------------------------------------------------------------------
! check output request using rank 1 and rank 2 arrays is consistent
!-------------------------------------------------------------------------------
  call network%extract_output(output_1d)
  call network%extract_output(output_2d)
  if(any(abs(output_1d - reshape(output_2d, [size(output_2d)])) .gt. 1.E-6))then
     success = .false.
     write(0,*) 'output_1d and output_2d are not consistent'
  end if


!!!-----------------------------------------------------------------------------
!!! check adding layers works as expected
!!! check compile adds input_layer at the start
!!!-----------------------------------------------------------------------------
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


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_full_network passed all tests'
  else
     write(0,*) 'test_full_network failed one or more tests'
     stop 1
  end if

end program test_full_network
