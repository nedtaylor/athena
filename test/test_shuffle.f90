program test_shuffle
  use misc_ml, only: shuffle
  implicit none

  integer :: i, j, seed_size
  integer, allocatable, dimension(:) :: iseed
  real :: r
  
  integer, parameter :: n = 3
  integer, parameter :: m = 5


  integer, dimension(n) :: array, original_array, array_tmp

  real, dimension(n, m) :: array_2d_original, array_2d_shuffled, &
       array_2d_shuffled_tmp
  logical :: success = .true.
  

!!!-----------------------------------------------------------------------------
!!! 1D array shuffle tests
!!!-----------------------------------------------------------------------------

  !! initialize the array
  array = (/ (i, i = 1, n) /)
  original_array = array ! Make a copy of the original array

  !! shuffle the array
  call shuffle(array)
  
  !! check if the array is shuffled
  if (all(array .eq. original_array)) then
     write(*,*) 'Array is not shuffled'
     success = .false.
  end if
  
  !! check if all original elements are still in the shuffled array
  do i = 1, n
     if (.not. any(array .eq. original_array(i))) then
        write(*,*) 'Original element', original_array(i), &
             'is missing in the shuffled array'
        success = .false.
     end if
  end do

  !! check that seed works
  array_tmp = original_array
  call shuffle(array_tmp, seed = 1)
  array = original_array
  call shuffle(array, seed = 1)

  !! check if array and array_tmp are the same
  if (all(array_tmp .ne. array)) then
     write(*,*) 'Shuffle seed does not work as intended'
     success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! 2D array shuffle tests
!!!-----------------------------------------------------------------------------

  !! test 2D array shuffling with dim
  do i = 1, n
     do j = 1, m
        array_2d_original(i, j) = i * m + j
     end do
  end do

  !! Shuffle the 2D array along the second dimension
  array_2d_shuffled = array_2d_original
  call shuffle(array_2d_shuffled, dim = 2, seed = 1)

  !! Check if the 2D array is shuffled along the second dimension
  if (all(array_2d_shuffled .eq. array_2d_original)) then
     write(*,*) '2D Array is not shuffled along the second dimension'
     success = .false.
  end if

  !! Check if all original elements are still in the shuffled 2D array
  do i = 1, n
    do j = 1, m
       if ( all(abs(array_2d_shuffled(i, :) - &
            array_2d_original(i, j)).gt.1.E-6) ) then
          write(*,*) 'Original element', array_2d_original(i, j), &
               'is missing in the shuffled 2D array'
          write(*,*) array_2d_shuffled(i,:)
          write(*,*) array_2d_original(i,j)
          success = .false.
       end if
    end do
  end do

  !! Check that seed works for 2D array shuffling
  array_2d_shuffled_tmp = array_2d_original
  call shuffle(array_2d_shuffled_tmp, dim = 2, seed = 4)
  array_2d_shuffled = array_2d_original
  call shuffle(array_2d_shuffled, dim = 2, seed = 4)

  !! Check if array_2d_shuffled and array_2d_shuffled_tmp are the same
  if (any(abs(array_2d_shuffled_tmp - array_2d_shuffled).gt.1.E-6)) then
     write(*,*) array_2d_shuffled_tmp
       write(*,*) array_2d_shuffled
     write(*,*) '2D Array shuffle seed does not work as intended'
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
     write(*,*) 'test_shuffle failed one or more tests'
     stop 1
  end if


end program test_shuffle
