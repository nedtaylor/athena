program test_shuffle
  use misc_ml, only: shuffle
  implicit none
  
  integer, parameter :: n = 10
  integer :: i
  integer, dimension(n) :: array, original_array
  logical :: success = .true.
  
  !! initialize the array
  array = (/ (i, i = 1, n) /)
  original_array = array ! Make a copy of the original array

  !! shuffle the array
  call shuffle(array)
  
  !! check if the array is shuffled
  if (all(array .eq. original_array)) then
     write(*,*) 'test_shuffle failed: array is not shuffled'
     success = .false.
  end if
  
  !! check if all original elements are still in the shuffled array
  do i = 1, n
     if (.not. any(array .eq. original_array(i))) then
        write(*,*) 'test_shuffle failed: original element', original_array(i), &
             'is missing in the shuffled array'
        success = .false.
     end if
  end do

  
  !! check for any fails
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_shuffle passed all tests'
  else
     write(*,*) 'test_shuffle failed one or more tests'
     stop 1
  end if


end program test_shuffle
