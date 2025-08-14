program test_autodiff_simple
  !! Simple test to verify autodiff memory management
  use athena__misc_types!, only: array_type
  use athena__constants, only: real32
  implicit none

  type(array_type) :: a, b
  type(array_type) :: c, d
  integer :: i
  logical :: success = .true.

  write(*,*) "Testing basic autodiff memory management..."

  ! Test 1: Basic allocation and operations
  call a%allocate([2, 2])
  call b%allocate([2, 2])

  a%val = reshape([1.0_real32, 2.0_real32, 3.0_real32, 4.0_real32], [2, 2])
  b%val = reshape([0.5_real32, 1.0_real32, 1.5_real32, 2.0_real32], [2, 2])

  call a%set_requires_grad(.true.)
  call b%set_requires_grad(.true.)

  ! Test simple operations
  c = a + b
!   if(.not. associated(c)) then
!      write(*,*) "ERROR: Addition failed"
!      success = .false.
!   else
!      write(*,*) "Addition successful, result shape:", size(c%val, 1), size(c%val, 2)
!   end if
  write(*,*) associated(c%left_operand)

  d = c * 2.0_real32
!   if(.not. associated(d)) then
!      write(*,*) "ERROR: Scalar multiplication failed"
!      success = .false.
!   else
!      write(*,*) "Scalar multiplication successful"
!   end if

  ! Test gradient computation
  write(*,*) "Testing gradient computation..."
  call d%backward()

  if(.not. associated(a%grad)) then
     write(*,*) "ERROR: Gradient not computed for a"
     success = .false.
  else
     write(*,*) "Gradient computed for a, max value:", maxval(abs(a%grad%val))
  end if

  if(.not. associated(b%grad)) then
     write(*,*) "ERROR: Gradient not computed for b"
     success = .false.
  else
     write(*,*) "Gradient computed for b, max value:", maxval(abs(b%grad%val))
  end if

  ! Test memory cleanup
  write(*,*) "Testing memory cleanup..."
!   deallocate(c, d)
  call a%deallocate()
  call b%deallocate()

  if(success) then
     write(*,*) "All basic tests PASSED"
  else
     write(*,*) "Some tests FAILED"
  end if

  write(*,*) "Test completed."

end program test_autodiff_simple
