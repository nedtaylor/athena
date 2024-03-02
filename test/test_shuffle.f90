program test_shuffle
  use misc_ml, only: shuffle
  implicit none

  integer :: i, j, k, l, o
  integer :: idim
  
  integer, parameter :: n = 10
  integer, parameter :: m = 3
  integer, parameter :: p = 3
  integer, parameter :: q = 3
  integer, parameter :: s = 3

  integer :: ilabel_original(n)
  integer :: ilabel_shuffled(n)
  real :: rlabel_original(n)
  real :: rlabel_shuffled(n)

  integer :: array_1d_original(n)
  integer :: array_1d_shuffled(n)
  integer :: array_1d_shuffled_tmp(n)

  real :: array_2d_original(n, m)
  real :: array_2d_shuffled(n, m)
  real :: array_2d_shuffled_tmp(n, m)

  integer :: iarray_3d_original(n, m, p)
  integer :: iarray_3d_shuffled(n, m, p)
  integer :: iarray_3d_shuffled_tmp(n, m, p)

  real :: rarray_3d_original(n, m, p)
  real :: rarray_3d_shuffled(n, m, p)
  real :: rarray_3d_shuffled_tmp(n, m, p)

  real :: array_4d_original(n, m, p, q)
  real :: array_4d_shuffled(n, m, p, q)
  real :: array_4d_shuffled_tmp(n, m, p, q)

  real :: array_5d_original(n, m, p, q, s)
  real :: array_5d_shuffled(n, m, p, q, s)
  real :: array_5d_shuffled_tmp(n, m, p, q, s)

  logical :: success = .true.
  

   ilabel_original = (/ (i, i = 1, n) /)
   rlabel_original = (/ (i, i = 1, n) /)

!!!-----------------------------------------------------------------------------
!!! 1D array shuffle tests
!!!-----------------------------------------------------------------------------

  !! initialize the array
  array_1d_original = (/ (i, i = 1, n) /)
  array_1d_shuffled = array_1d_original ! Make a copy of the original array

  !! shuffle the array
  call shuffle(array_1d_shuffled, seed = 1)
  
  !! check if the array is shuffled
  if (all(array_1d_shuffled .eq. array_1d_original)) then
     write(*,*) '1D Array is not shuffled'
     success = .false.
  end if
  
  !! check if all original elements are still in the shuffled array
  do i = 1, n
     if (all(array_1d_shuffled .ne. array_1d_original(i))) then
        write(*,*) 'Original element', array_1d_original(i), &
             'is missing in the shuffled 1D array'
        success = .false.
     end if
  end do

  !! check that seed works
  array_1d_shuffled_tmp = array_1d_original
  call shuffle(array_1d_shuffled_tmp, seed = 1)
  array_1d_shuffled = array_1d_original
  call shuffle(array_1d_shuffled, seed = 1)

  !! check if array and array_1d_shuffled_tmp are the same
  if (all(array_1d_shuffled_tmp .ne. array_1d_shuffled)) then
     write(*,*) 'Shuffle seed does not work as intended'
     success = .false.
  end if


!!!-----------------------------------------------------------------------------
!!! 2D array shuffle tests
!!!-----------------------------------------------------------------------------

  !! initialize the array
  do i = 1, n
     do j = 1, m
        array_2d_original(i, j) = i * m + j
     end do
  end do

  do idim = 1, 2
     !! Shuffle the 2D array along the second dimension
     array_2d_shuffled = array_2d_original
     call shuffle(array_2d_shuffled, dim = idim, seed = 1)

     !! Check if the 2D array is shuffled along the second dimension
     if (all(array_2d_shuffled .eq. array_2d_original)) then
        write(*,*) '2D Array is not shuffled along the second dimension'
        success = .false.
     end if

     !! Check that seed works for 2D array shuffling
     array_2d_shuffled_tmp = array_2d_original
     call shuffle(array_2d_shuffled_tmp, dim = idim, seed = 1)
     array_2d_shuffled = array_2d_original
     call shuffle(array_2d_shuffled, dim = idim, seed = 1)

     !! Check if array_2d_shuffled and array_2d_shuffled_tmp are the same
     if (any(abs(array_2d_shuffled_tmp - array_2d_shuffled).gt.1.E-6)) then
        write(*,*) '2D Array shuffle seed does not work as intended'
        success = .false.
     end if
  end do

  !! Check if all original elements are still in the shuffled 2D array
  do i = 1, n
     do j = 1, m
        if ( all(abs(array_2d_shuffled(i, :) - &
              array_2d_original(i, j)).gt.1.E-6) ) then
           write(*,*) 'Original element', array_2d_original(i, j), &
                 'is missing in the shuffled 2D array'
           success = .false.
        end if
     end do
  end do


  !!!-----------------------------------------------------------------------------
  !!! 3D integer array shuffle tests
  !!!-----------------------------------------------------------------------------
  
  !! initialize the array
    do i = 1, n
       do j = 1, m
          do k = 1, p
             iarray_3d_original(i, j, k) = i * m * p + j * p + k
          end do
       end do
    end do
  
    do idim = 1, 3
       !! Shuffle the 3D array along the third dimension
       iarray_3d_shuffled = iarray_3d_original
       call shuffle(iarray_3d_shuffled, dim = idim, seed = 1)

       !! Check if the 3D array is shuffled along the third dimension
       if (all(iarray_3d_shuffled .eq. iarray_3d_original)) then
          write(*,*) '3D Array is not shuffled along the third dimension'
          success = .false.
       end if

       !! Check that seed works for 3D array shuffling
       iarray_3d_shuffled_tmp = iarray_3d_original
       call shuffle(iarray_3d_shuffled_tmp, dim = idim, seed = 4)
       iarray_3d_shuffled = iarray_3d_original
       call shuffle(iarray_3d_shuffled, dim = idim, seed = 4)

       !! Check if iarray_3d_shuffled and iarray_3d_shuffled_tmp are the same
       if (any(iarray_3d_shuffled_tmp .ne. iarray_3d_shuffled)) then
          write(*,*) '3D Array shuffle seed does not work as intended'
          success = .false.
       end if
    end do

    iarray_3d_shuffled = iarray_3d_original
    ilabel_shuffled = ilabel_original
    call shuffle(iarray_3d_shuffled, ilabel_shuffled, dim = 1, seed = 1)

    !! Check if the label is shuffled
    if (all(ilabel_shuffled .eq. ilabel_original)) then
       write(*,*) '3D array label is not shuffled'
       success = .false.
    end if

    !! Check if all original elements are still in the shuffled 3D array
    do i = 1, n
       do j = 1, m
          do k = 1, p
             if ( all(iarray_3d_shuffled(:, j, k) .eq. &
                   iarray_3d_original(i, j, k)) ) then
                write(*,*) 'Original element', iarray_3d_original(i, j, k), &
                      'is missing in the shuffled 3D array'
                success = .false.
             end if
          end do
       end do
       if(any(iarray_3d_shuffled(i,:,:) .ne. &
            iarray_3d_original(ilabel_shuffled(i),:,:))) then
          write(*,*) '3D array and label shuffle inconsistency'
          success = .false.
       end if
    end do

    iarray_3d_shuffled = iarray_3d_original
    rlabel_shuffled = rlabel_original
    call shuffle(iarray_3d_shuffled, rlabel_shuffled, dim = 1, seed = 1)

    !! Check if the label is shuffled
    if (all(abs(rlabel_shuffled - rlabel_original).lt.1.E-6)) then
       write(*,*) '3D array label is not shuffled'
       success = .false.
    end if

    !! Check if all original elements are still in the shuffled 3D array
    do i = 1, n
       do j = 1, m
          do k = 1, p
             if ( all(iarray_3d_shuffled(:, j, k) .eq. &
                   iarray_3d_original(i, j, k)) ) then
                write(*,*) 'Original element', iarray_3d_original(i, j, k), &
                      'is missing in the shuffled 3D array'
                success = .false.
             end if
          end do
       end do
       if(any(iarray_3d_shuffled(i,:,:) .ne. &
            iarray_3d_original(nint(rlabel_shuffled(i)),:,:))) then
          write(*,*) '3D array and label shuffle inconsistency'
          success = .false.
       end if
    end do


!!!-----------------------------------------------------------------------------
!!! 3D real array shuffle tests
!!!-----------------------------------------------------------------------------

  !! initialize the array
  do i = 1, n
     do j = 1, m
        do k = 1, p
           rarray_3d_original(i, j, k) = i * m * p + j * p + k
        end do
     end do
  end do

  do idim = 1, 3
     !! Shuffle the 3D array along the third dimension
     rarray_3d_shuffled = rarray_3d_original
     call shuffle(rarray_3d_shuffled, dim = 3, seed = 1)

     !! Check if the 3D array is shuffled along the third dimension
     if (all(rarray_3d_shuffled .eq. rarray_3d_original)) then
        write(*,*) '3D Array is not shuffled along the third dimension'
        success = .false.
     end if

     !! Check that seed works for 3D array shuffling
     rarray_3d_shuffled_tmp = rarray_3d_original
     call shuffle(rarray_3d_shuffled_tmp, dim = idim, seed = 4)
     rarray_3d_shuffled = rarray_3d_original
     call shuffle(rarray_3d_shuffled, dim = idim, seed = 4)
   
     !! Check if rarray_3d_shuffled and rarray_3d_shuffled_tmp are the same
     if (any(abs(rarray_3d_shuffled_tmp - rarray_3d_shuffled).gt.1.E-6)) then
        write(*,*) '3D Array shuffle seed does not work as intended'
        success = .false.
     end if
  end do

  rarray_3d_shuffled = rarray_3d_original
  call shuffle(rarray_3d_shuffled, dim = 1, seed = 1)

  !! Check if all original elements are still in the shuffled 3D array
  do i = 1, n
    do j = 1, m
       do k = 1, p
          if ( all(abs(rarray_3d_shuffled(:, j, k) - &
               rarray_3d_original(i, j, k)).gt.1.E-6) ) then
             write(*,*) 'Original element', rarray_3d_original(i, j, k), &
                  'is missing in the shuffled 3D array'
             success = .false.
          end if
       end do
    end do
  end do


!!!-----------------------------------------------------------------------------
!!! 4D array shuffle tests
!!!-----------------------------------------------------------------------------
  
  !! initialize the array
    do i = 1, n
       do j = 1, m
          do k = 1, p
            do l = 1, q
               array_4d_original(i, j, k, l) = i * m * p * q + j * p * q + k * q + l
            end do
          end do
       end do
    end do
  
    do idim = 1, 4
       !! Shuffle the 4D array along the third dimension
       array_4d_shuffled = array_4d_original
       call shuffle(array_4d_shuffled, dim = idim, seed = 1)

       !! Check if the 4D array is shuffled along the third dimension
       if (all(array_4d_shuffled .eq. array_4d_original)) then
          write(*,*) '4D Array is not shuffled along the third dimension'
          success = .false.
       end if

       !! Check that seed works for 4D array shuffling
       array_4d_shuffled_tmp = array_4d_original
       call shuffle(array_4d_shuffled_tmp, dim = idim, seed = 4)
       array_4d_shuffled = array_4d_original
       call shuffle(array_4d_shuffled, dim = idim, seed = 4)

       !! Check if array_4d_shuffled and array_4d_shuffled_tmp are the same
       if (any(abs(array_4d_shuffled_tmp - array_4d_shuffled).gt.1.E-6)) then
          write(*,*) '4D Array shuffle seed does not work as intended'
          success = .false.
       end if
    end do

    array_4d_shuffled = array_4d_original
    ilabel_shuffled = ilabel_original
    call shuffle(array_4d_shuffled, ilabel_shuffled, dim = 1, seed = 1)

    !! Check if the label is shuffled
    if (all(ilabel_shuffled .eq. ilabel_original)) then
       write(*,*) '4D array label is not shuffled'
       success = .false.
    end if

    !! Check if all original elements are still in the shuffled 4D array
    do i = 1, n
       do j = 1, m
          do k = 1, p
             do l = 1, q
                if ( all(abs(array_4d_shuffled(:, j, k, l) - &
                   array_4d_original(i, j, k, l)).gt.1.E-6) ) then
                   write(*,*) 'Original element', &
                        array_4d_original(i, j, k, l), &
                        'is missing in the shuffled 4D array'
                   success = .false.
                end if
             end do
          end do
       end do
       if(any(abs(array_4d_shuffled(i,:,:,:) - &
            array_4d_original(ilabel_shuffled(i),:,:,:)) .gt. 1.E-6)) then
          write(*,*) '4D array and label shuffle inconsistency'
          success = .false.
       end if
    end do


!!!-----------------------------------------------------------------------------
!!! 5D array shuffle tests
!!!-----------------------------------------------------------------------------
  
   !! initialize the array
   do i = 1, n
      do j = 1, m
         do k = 1, p
           do l = 1, q
              do o = 1, s
                 array_5d_original(i, j, k, l, o) = &
                      i * m * p * q * s + &
                      j * p * q * s + &
                      k * q * s + &
                      l * s + &
                      o
              end do
           end do
         end do
      end do
   end do

   do idim = 1, 5
      !! Shuffle the 3D array along the third dimension
      array_5d_shuffled = array_5d_original
      call shuffle(array_5d_shuffled, dim = idim, seed = 1)

      !! Check if the 3D array is shuffled along the third dimension
      if (all(array_5d_shuffled .eq. array_5d_original)) then
         write(*,*) '5D Array is not shuffled along the third dimension'
         success = .false.
      end if

      !! Check that seed works for 5D array shuffling
      array_5d_shuffled_tmp = array_5d_original
      call shuffle(array_5d_shuffled_tmp, dim = idim, seed = 4)
      array_5d_shuffled = array_5d_original
      call shuffle(array_5d_shuffled, dim = idim, seed = 4)

      !! Check if array_5d_shuffled and array_5d_shuffled_tmp are the same
      if (any(abs(array_5d_shuffled_tmp - array_5d_shuffled).gt.1.E-6)) then
         write(*,*) '5D Array shuffle seed does not work as intended'
         success = .false.
      end if
   end do

   array_5d_shuffled = array_5d_original
   ilabel_shuffled = ilabel_original
   call shuffle(array_5d_shuffled, ilabel_shuffled, dim = 1, seed = 1)

   !! Check if the label is shuffled
   if (all(ilabel_shuffled .eq. ilabel_original)) then
      write(*,*) '5D array label is not shuffled'
      success = .false.
   end if

   !! Check if all original elements are still in the shuffled 5D array
   do i = 1, n
     do j = 1, m
        do k = 1, p
           do l = 1, q
              do o = 1, s
                 if ( all(abs(array_5d_shuffled(:, j, k, l, o) - &
                    array_5d_original(i, j, k, l, o)).gt.1.E-6) ) then
                    write(*,*) 'Original element', &
                         array_5d_original(i, j, k, l, o), &
                         'is missing in the shuffled 5D array'
                    success = .false.
                 end if
               end do
           end do
        end do
     end do
     if(any(abs(array_5d_shuffled(i,:,:,:,:) - &
          array_5d_original(ilabel_shuffled(i),:,:,:,:)) .gt. 1.E-6)) then
        write(*,*) '5D array and label shuffle inconsistency'
        success = .false.
     end if
   end do

   array_5d_shuffled = array_5d_original
   rlabel_shuffled = rlabel_original
   call shuffle(array_5d_shuffled, rlabel_shuffled, dim = 1, seed = 1)

   !! Check if the label is shuffled
   if (all(abs(rlabel_shuffled - rlabel_original).lt.1.E-6)) then
      write(*,*) '5D array label is not shuffled'
      success = .false.
   end if

   !! Check if all original elements are still in the shuffled 5D array
   do i = 1, n
     do j = 1, m
        do k = 1, p
           do l = 1, q
              do o = 1, s
                 if ( all(abs(array_5d_shuffled(:, j, k, l, o) - &
                    array_5d_original(i, j, k, l, o)).gt.1.E-6) ) then
                    write(*,*) 'Original element', &
                         array_5d_original(i, j, k, l, o), &
                         'is missing in the shuffled 5D array'
                    success = .false.
                 end if
               end do
           end do
        end do
     end do
     if(any(abs(array_5d_shuffled(i,:,:,:,:) - &
          array_5d_original(nint(rlabel_shuffled(i)),:,:,:,:)) .gt. 1.E-6)) then
        write(*,*) '5D array and label shuffle inconsistency'
        success = .false.
     end if
   end do


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

end program test_shuffle
