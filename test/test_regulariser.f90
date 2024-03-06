program test_mod_regulariser
  use regulariser
  implicit none

  type(l1_regulariser_type) :: l1_regulariser
  type(l2_regulariser_type) :: l2_regulariser
  type(l1l2_regulariser_type) :: l1l2_regulariser

  real :: learning_rate = 0.1E0
  logical :: success = .true.

  real, dimension(1) :: params, gradient, &
       expected_gradient_l1, expected_gradient_l2, expected_gradient_l1l2

  ! Initialize params and gradient here...
  gradient = 1.0E0
  expected_gradient_l1   = gradient + 1.E-3
  expected_gradient_l2   = gradient + 1.E-3
  expected_gradient_l1l2 = gradient + 2.E-3

  call l1_regulariser%regularise(params, gradient, learning_rate)
  ! Check the result
  call check(gradient, expected_gradient_l1, success)

  call l2_regulariser%regularise(params, gradient, learning_rate)
  ! Check the result
  call check(gradient, expected_gradient_l2, success)

  call l1l2_regulariser%regularise(params, gradient, learning_rate)
  ! Check the result
  call check(gradient, expected_gradient_l1l2, success)


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_shuffle passed all tests'
  else
     write(0,*) 'test_shuffle failed one or more tests'
     stop 1
  end if


contains

subroutine check(actual, expected, success)
  implicit none
  real, dimension(:), intent(in) :: actual
  real, dimension(:), intent(in) :: expected
  real :: diff
  integer :: i
  logical, intent(inout) :: success

  if (size(actual) .ne. size(expected)) then
    print *, "Error: Arrays are not the same size."
    stop 1
  end if

  do i = 1, size(actual)
    diff = abs(actual(i) - expected(i))
    if (diff .gt. 1.E-6) then
      write(*,*) "Error: Arrays are not approximately equal."
      write(*,*) "Index: ", i
      write(*,*) "Actual: ", actual(i)
      write(*,*) "Expected: ", expected(i)
      success = .false.
    end if
  end do
end subroutine check

end program test_mod_regulariser