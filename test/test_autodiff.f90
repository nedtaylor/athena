program test_autodiff
  !! Test program for autodifferentiation functionality
  use athena__constants, only: real32
  use athena__misc_types, only: array_type, operator(+), &
       operator(*), sin, cos, exp
  implicit none

  type(array_type) :: x, y, z, result
  logical :: success = .true.

  write(*,*) 'Testing autodifferentiation functionality...'

  ! Initialize arrays
  write(*,*) "test0"
  call x%allocate([3, 1])
  write(*,*) "test1"
  call y%allocate([3, 1])
  call z%allocate([3, 1])

  ! Set values
  x%val(:,1) = [1.0_real32, 2.0_real32, 3.0_real32]
  y%val(:,1) = [0.5_real32, 1.5_real32, 2.5_real32]
  z%val(:,1) = [2.0_real32, 3.0_real32, 4.0_real32]

  ! Enable gradient computation
  call x%set_requires_grad(.true.)
  call y%set_requires_grad(.true.)
  call z%set_requires_grad(.true.)

  ! Perform some operations: result = sin(x * y) + exp(z)
  result = sin( x * y ) + exp(z) + 3._real32

  write(*,*) 'Forward pass completed.'
  write(*,*) 'Result values:', result%val(:,1)

  ! Perform backward pass
  call result%grad_reverse()

  write(*,*) 'Backward pass completed.'

  ! Check if gradients were computed
  if(associated(x%grad)) then
     write(*,*) 'SUCCESS: Gradients computed for x'
     write(*,*) 'x gradient:', x%grad%val(:,1)
  else
     write(*,*) 'ERROR: No gradients computed for x'
     success = .false.
  end if

  ! Check if gradients were computed
  if(associated(y%grad)) then
     write(*,*) 'SUCCESS: Gradients computed for y'
     write(*,*) 'y gradient:', y%grad%val(:,1)
  else
     write(*,*) 'ERROR: No gradients computed for y'
     success = .false.
  end if

  ! Check if gradients were computed
  if(associated(z%grad)) then
     write(*,*) 'SUCCESS: Gradients computed for z'
     write(*,*) 'z gradient:', z%grad%val(:,1)
  else
     write(*,*) 'ERROR: No gradients computed for z'
     success = .false.
  end if

  ! Test basic arithmetic operations
  call test_basic_operations()


  call test_network()

  if(success) then
     write(*,*) 'All autodiff tests passed!'
  else
     write(*,*) 'Some autodiff tests failed!'
     stop 1
  end if

contains

  subroutine test_basic_operations()
    !! Test basic arithmetic operations
    type(array_type) :: a, b, c

    write(*,*) 'Testing basic operations...'

    ! Initialize
    call a%allocate([2, 1])
    call b%allocate([2, 1])

    a%val(:,1) = [2.0_real32, 3.0_real32]
    b%val(:,1) = [4.0_real32, 5.0_real32]

    call a%set_requires_grad(.true.)
    call b%set_requires_grad(.true.)

    ! Test addition
    c = a + b
    write(*,*) 'Addition result:', c%val(:,1)

    ! Test multiplication
    c = a * b
    write(*,*) 'Multiplication result:', c%val(:,1)

    write(*,*) 'Basic operations test completed.'

  end subroutine test_basic_operations


  subroutine test_network()
    use athena, only: &
         network_type, &
         full_layer_type, &
         sgd_optimiser_type
    implicit none

    type(network_type) :: network


    write(*,*) "tar0"
    call network%add(full_layer_type( &
         num_inputs=2, &
         num_outputs=1, &
         kernel_initialiser='ones', &
         activation_function='sigmoid' &
    ))
    write(*,*) "tar1"
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=1.0), &
         loss_method='mse', &
         accuracy_method='mse', &
         metrics=['loss'], &
         batch_size = 1, &
         verbose=1 &
    )
    training: block
      type(array_type), dimension(1,1) :: x, y
      real :: tol = 1.E-3
      integer :: n
      integer, parameter :: num_iterations = 100

      write(*,*) 'Testing basic operations...'

      ! Initialize
      call x(1,1)%allocate([2, 2])
      call y(1,1)%allocate([1, 2])

      x(1,1)%val(:,1) = [0.4,0.124]
      y(1,1)%val(:,1) = [0.765]
      x(1,1)%val(:,2) = [0.5,0.125]
      y(1,1)%val(:,2) = [0.8]

      call network%train(x, y, num_iterations)
      write(*,*) shape(network%model(1)%layer%output(1,1)%val)
      write(*,*) network%model(1)%layer%output(1,1)%val(:,1)
      write(*,*) "associated:", associated(network%model(2)%layer%output(1,1)%grad)

      if(network%epoch.ge.num_iterations)then
         success = .false.
         write(0,*) 'network failed to converge'
      end if

      if(associated(network%model(2)%layer%output(1,1)%grad)) then
         write(*,*) 'SUCCESS: Gradients computed for x'
         write(*,*) 'x gradient:', network%model(2)%layer%output(1,1)%grad%val(:,1)
      else
         write(*,*) 'ERROR: No gradients computed for x'
         success = .false.
      end if

    end block training
  end subroutine test_network


end program test_autodiff
