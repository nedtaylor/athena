program test_metrics
  use metrics
  implicit none

  type(metric_dict_type) :: a, b, result, expected_result
  type(metric_dict_type) :: dict(2), source(2)

  integer :: i
  integer :: converged

  logical :: success = .true.

  ! Initialize a and b here...
  a%key = "loss"
  a%val = 10
  a%threshold = 5.0
  a%active = .true.
  a%history = [1, 2, 3, 4, 5]

  b%key = "loss"
  b%val = 20
  b%threshold = 10.0
  b%active = .false.
  b%history = [6, 7, 8, 9, 10]

  expected_result%key = "loss"
  expected_result%val = 30
  expected_result%threshold = 5.0
  expected_result%active = .true.
  expected_result%history = a%history

  result = a + b

  ! Check the result
  call check_metric_dict(result, expected_result)

  call metric_dict_alloc(dict, length=10)
  do i = 1, size(dict)
     if(size(dict(i)%history) .ne. 10) then
        write(*,*) "Error: metric_dict_alloc failed to allocate the correct size."
        success = .false.
     end if
  end do

  source(:) = a
  call metric_dict_alloc(dict, source = source)
  do i = 1, size(dict)
     if(size(dict(i)%history) .ne. size(source(i)%history)) then
        write(*,*) "Error: metric_dict_alloc failed to allocate the correct size."
        success = .false.
     end if
  end do

  a%history = [1, 1, 1, 1, 1]
  call a%check(plateau_threshold=10.E0, converged = converged)
  if (converged .ne. 1) then
     write(*,*) "Error: a%check failed to detect convergence."
     success = .false.
  end if

  a%history = [8, 8, 8, 8, 8]
  call a%check(plateau_threshold=10.E0, converged = converged)
  if (converged .ne. -1) then
     write(*,*) "Error: a%check failed to detect plateau."
     success = .false.
  end if

!!!-----------------------------------------------------------------------------
!!! Final printing array shuffle tests
!!!-----------------------------------------------------------------------------
  
  !! check for any fails
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_shuffle passed all tests'
  else
     write(0,*) 'test_shuffle failed one or more tests'
     stop 1
  end if

contains

  subroutine check_metric_dict(actual, expected)
    type(metric_dict_type), intent(in) :: actual
    type(metric_dict_type), intent(in) :: expected
  
    if (actual%key .ne. expected%key .or. &
         actual%val .ne. expected%val .or. &
         actual%threshold .ne. expected%threshold .or. &
         actual%active .neqv. expected%active .or. &
         any(actual%history .ne. expected%history)) then
       write(*,*) "Error: metric_dict_types are not equal."
       success = .false.
    end if
  end subroutine check_metric_dict
  
end program test_metrics
