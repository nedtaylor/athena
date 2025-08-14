program test_autodiff_memory
  !! Test program to verify memory management fixes in autodiff
  use athena__misc_types
  use athena__constants, only: real32
  implicit none

  type(array_type) :: a, b
  type(array_type) :: d, e
  type(array_type), pointer :: c
  real(real32), dimension(3,2) :: test_data_a, test_data_b
  integer :: i

  write(*,*) "Testing autodiff memory management fixes..."

  ! Initialize test data
  test_data_a = reshape([1.0_real32, 2.0_real32, 3.0_real32, &
                        4.0_real32, 5.0_real32, 6.0_real32], [3, 2])
  test_data_b = reshape([0.5_real32, 1.5_real32, 2.5_real32, &
                        3.5_real32, 4.5_real32, 5.5_real32], [3, 2])

  ! Test 1: Basic array creation and operations
  write(*,*) "Test 1: Basic operations"
  call a%allocate([3, 2])
  call b%allocate([3, 2])
  a%val = test_data_a
  b%val = test_data_b
  a%requires_grad = .true.
  b%requires_grad = .true.

  ! Test addition
  allocate(c)
  c = a + b
  c => c * 2.0_real32
  write(*,*) "Addition result shape:", c%allocated, size(c%val)

  ! Test multiple operations
  d = c * 2.0_real32
  e = d - a

  write(*,*) "Multiple operations successful"

  ! Test 2: Gradient computation
  write(*,*) "Test 2: Gradient computation"
  call e%backward()

  if(associated(a%grad)) then
     write(*,*) "Gradient computed for a:", maxval(abs(a%grad%val))
  else
     write(*,*) "ERROR: Gradient not computed for a"
  end if

  if(associated(b%grad)) then
     write(*,*) "Gradient computed for b:", maxval(abs(b%grad%val))
  else
     write(*,*) "ERROR: Gradient not computed for b"
  end if

  ! Test 3: Memory cleanup test
  write(*,*) "Test 3: Memory cleanup"
!   deallocate(c, d, e)
  call a%deallocate()
  call b%deallocate()

  ! Test 4: Repeated operations to check for memory leaks
  write(*,*) "Test 4: Repeated operations"
  do i = 1, 100
     call a%allocate([2, 2])
     call b%allocate([2, 2])
     a%val = 1.0_real32
     b%val = 2.0_real32
     a%requires_grad = .true.

     c = a + b
     d = c * c
     call d%backward()

   !   deallocate(c, d)
     call a%deallocate()
     call b%deallocate()
     call c%deallocate()
     call d%deallocate()
  end do

  write(*,*) "All tests completed successfully!"

end program test_autodiff_memory
