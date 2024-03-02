program test_shuffle
  use misc_ml, only: shuffle
  implicit none

  integer :: i, j, k, l, seed_size
  integer, allocatable, dimension(:) :: iseed
  real :: r
  
  integer, parameter :: n = 3
  integer, parameter :: m = 5
  integer, parameter :: p = 5
  integer, parameter :: q = 5


  integer, dimension(n) :: array, original_array, array_tmp

  real, dimension(n, m) :: array_2d_original, array_2d_shuffled, &
       array_2d_shuffled_tmp

  real :: array_3d_original(n, m, p)
  real :: array_3d_shuffled(n, m, p)
  real :: array_3d_shuffled_tmp(n, m, p)

  real :: array_4d_original(n, m, p, q)
  real :: array_4d_shuffled(n, m, p, q)
  real :: array_4d_shuffled_tmp(n, m, p, q)

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
     write(*,*) '2D Array shuffle seed does not work as intended'
     success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! 3D array shuffle tests
!!!-----------------------------------------------------------------------------

  !! test 3D array shuffling
  do i = 1, n
     do j = 1, m
        do k = 1, p
           array_3d_original(i, j, k) = i * m * p + j * p + k
        end do
     end do
  end do

  !! Shuffle the 3D array along the third dimension
  array_3d_shuffled = array_3d_original
  call shuffle(array_3d_shuffled, dim = 3, seed = 1)

  !! Check if the 3D array is shuffled along the third dimension
  if (all(array_3d_shuffled .eq. array_3d_original)) then
     write(*,*) '3D Array is not shuffled along the third dimension'
     success = .false.
  end if

  !! Check if all original elements are still in the shuffled 3D array
  do i = 1, n
    do j = 1, m
       do k = 1, p
          if ( all(abs(array_3d_shuffled(i, j, :) - &
               array_3d_original(i, j, k)).gt.1.E-6) ) then
             write(*,*) 'Original element', array_3d_original(i, j, k), &
                  'is missing in the shuffled 3D array'
             write(*,*) array_3d_shuffled(i, j, :)
             write(*,*) array_3d_original(i, j, k)
             success = .false.
          end if
       end do
    end do
  end do

  !! Check that seed works for 3D array shuffling
  array_3d_shuffled_tmp = array_3d_original
  call shuffle(array_3d_shuffled_tmp, dim = 3, seed = 4)
  array_3d_shuffled = array_3d_original
  call shuffle(array_3d_shuffled, dim = 3, seed = 4)

  !! Check if array_3d_shuffled and array_3d_shuffled_tmp are the same
  if (any(abs(array_3d_shuffled_tmp - array_3d_shuffled).gt.1.E-6)) then
     write(*,*) '3D Array shuffle seed does not work as intended'
     success = .false.
  end if


  !!!-----------------------------------------------------------------------------
  !!! 4D array shuffle tests
  !!!-----------------------------------------------------------------------------
  
    !! test 3D array shuffling
    do i = 1, n
       do j = 1, m
          do k = 1, p
            do l = 1, q
               array_4d_original(i, j, k, l) = i * m * p * q + j * p * q + k * q + l
            end do
          end do
       end do
    end do
  
    !! Shuffle the 3D array along the third dimension
    array_4d_shuffled = array_4d_original
    call shuffle(array_4d_shuffled, dim = 4, seed = 1)
  
    !! Check if the 3D array is shuffled along the third dimension
    if (all(array_4d_shuffled .eq. array_4d_original)) then
       write(*,*) '4D Array is not shuffled along the third dimension'
       success = .false.
    end if
  
    !! Check if all original elements are still in the shuffled 4D array
    do i = 1, n
      do j = 1, m
         do k = 1, p
            do l = 1, q
               if ( all(abs(array_4d_shuffled(i, j, k, :) - &
                  array_4d_original(i, j, k, l)).gt.1.E-6) ) then
                  write(*,*) 'Original element', &
                       array_4d_original(i, j, k, l), &
                       'is missing in the shuffled 3D array'
                  write(*,*) array_4d_shuffled(i, j, k, :)
                  write(*,*) array_4d_original(i, j, k, l)
                  success = .false.
               end if
            end do
         end do
      end do
    end do
  
    !! Check that seed works for 4D array shuffling
    array_4d_shuffled_tmp = array_4d_original
    call shuffle(array_4d_shuffled_tmp, dim = 4, seed = 4)
    array_4d_shuffled = array_4d_original
    call shuffle(array_4d_shuffled, dim = 4, seed = 4)
  
    !! Check if array_4d_shuffled and array_4d_shuffled_tmp are the same
    if (any(abs(array_4d_shuffled_tmp - array_4d_shuffled).gt.1.E-6)) then
       write(*,*) '4D Array shuffle seed does not work as intended'
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
