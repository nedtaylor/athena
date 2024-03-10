program test_mod_regulariser
  use regulariser
  implicit none

  type(l1_regulariser_type) :: l1_regulariser
  type(l2_regulariser_type) :: l2_regulariser
  type(l1l2_regulariser_type) :: l1l2_regulariser

  real :: learning_rate = 0.1E0
  logical :: success = .true.

  real, dimension(1) :: params, gradient, expected_gradient

  !! initialize parameters
  params = 1.E0

  !! test l1 regulariser
  gradient = 1.E0
  expected_gradient = gradient + 1.E-3
  write(*,*) "testing L1 regulariser"
  call l1_regulariser%regularise(params, gradient, learning_rate)
  call check(gradient, expected_gradient, success)

  !! test l2 regulariser
  gradient = 1.E0
  expected_gradient = gradient + 2.E-3
  write(*,*) "testing L2 regulariser"
  call l2_regulariser%regularise(params, gradient, learning_rate)
  call check(gradient, expected_gradient, success)

  !! test l1l2 regulariser
  gradient = 1.E0
  expected_gradient = gradient + 3.E-3
  write(*,*) "testing L1L2 regulariser"
  call l1l2_regulariser%regularise(params, gradient, learning_rate)
  call check(gradient, expected_gradient, success)


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_regulariser passed all tests'
  else
     write(0,*) 'test_regulariser failed one or more tests'
     stop 1
  end if


contains

!!!-----------------------------------------------------------------------------
!!! check gradients are as expected
!!!-----------------------------------------------------------------------------
subroutine check(actual, expected, success)
  implicit none
  real, dimension(:), intent(in) :: actual
  real, dimension(:), intent(in) :: expected
  real :: diff
  integer :: i
  logical, intent(inout) :: success

  if (size(actual) .ne. size(expected)) then
    write(0,*) "Gradient size not expected"
    success = .false.
  end if

  write(*,*) actual, expected
  do i = 1, size(actual)
    diff = abs(actual(i) - expected(i))
    if (diff .gt. 1.E-6) then
      write(0,*) "gradients not as expected"
      write(0,*) "Index: ", i
      write(0,*) "Actual: ", actual(i)
      write(0,*) "Expected: ", expected(i)
      success = .false.
    end if
  end do
end subroutine check

end program test_mod_regulariser
