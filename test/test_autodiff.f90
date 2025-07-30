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
  write(*,*) "test2"
  x%val(:,1) = [1.0_real32, 2.0_real32, 3.0_real32]
  y%val(:,1) = [0.5_real32, 1.5_real32, 2.5_real32]
  z%val(:,1) = [2.0_real32, 3.0_real32, 4.0_real32]

  ! Enable gradient computation
  write(*,*) "test3"
  call x%set_requires_grad(.true.)
  call y%set_requires_grad(.true.)
  call z%set_requires_grad(.true.)

  ! Perform some operations: result = sin(x * y) + exp(z)
  write(*,*) "test4"
  result = sin( x * y ) + exp(z)

  write(*,*) 'Forward pass completed.'
  write(*,*) 'Result values:', result%val(:,1)

  ! Perform backward pass
  write(*,*) "test5"
  call result%backward()

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

end program test_autodiff
