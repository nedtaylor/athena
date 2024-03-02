program test_split
  use misc_ml, only: split
  implicit none

  integer :: i, j, k, l, o
  integer :: itmp1
  
  real :: left_size = 0.25
  
  integer, parameter :: n = 100
  integer, parameter :: m = 3
  integer, parameter :: p = 3
  integer, parameter :: q = 3
  integer, parameter :: s = 3

  integer :: label(n)
  integer, allocatable :: label_left(:)
  integer, allocatable :: label_right(:)


  real :: rlabel(n)
  real, allocatable :: rlabel_left(:)
  real, allocatable :: rlabel_right(:)

  integer :: array_3d(n, m, p)
  integer, allocatable :: array_3d_left(:,:,:)
  integer, allocatable :: array_3d_right(:,:,:)

  real :: array_5d(n, m, p, q, s)
  real, allocatable :: array_5d_left(:,:,:,:,:)
  real, allocatable :: array_5d_right(:,:,:,:,:)

  logical :: success = .true.
  

!!!-----------------------------------------------------------------------------
!!! 3D array shuffle tests
!!!-----------------------------------------------------------------------------

  !! initialise array
  do i = 1, n
     label(i) = i
     rlabel(i) = i
     do j = 1, m
        do k = 1, p
           array_3d(i, j, k) = i * m * p + j * p + k
        end do
     end do
  end do

  !! split data
  call split(array_3d, label, array_3d_left, array_3d_right, &
       label_left, label_right, &
       dim = 1, left_size=left_size, seed = 1)

  itmp1 = nint(real(n) * 0.25)
  !! check if the left array is the correct size
  if (size(array_3d_left, 1) .ne. itmp1) then
     write(*,*) '3D array left size is not correct'
     success = .false.
  end if
  if (size(array_3d_left, 1) .ne. size(label_left)) then
     write(*,*) '3D array size and label size do not match'
     success = .false.
  end if
  if (size(array_3d_left,1) + size(array_3d_right,1) .ne. n .or. &
   size(label_left,1) + size(label_right,1) .ne. n) then
     write(*,*) '3D array label sizes do not add up to n'
     success = .false.
  end if

  do i = 1, size(array_3d_left, 1)
     if (any(array_3d_left(i,:,:) .ne. array_3d(label_left(i),:,:))) then
        write(*,*) '3D data split is not correct'
        success = .false.
     end if
  end do

  !! split data
  call split(array_3d, rlabel, array_3d_left, array_3d_right, &
       rlabel_left, rlabel_right, &
       dim = 1, left_size=left_size, seed = 1)

   itmp1 = nint(real(n) * 0.25)
   !! check if the left array is the correct size
   if (size(array_3d_left, 1) .ne. itmp1) then
      write(*,*) '3D array left size is not correct'
      success = .false.
   end if
   if (size(array_3d_left, 1) .ne. size(rlabel_left)) then
      write(*,*) '3D array size and label size do not match'
      success = .false.
   end if
   if (size(array_3d_left,1) + size(array_3d_right,1) .ne. n .or. &
   size(rlabel_left,1) + size(rlabel_right,1) .ne. n) then
      write(*,*) '3D array label sizes do not add up to n'
      success = .false.
   end if

   do i = 1, size(array_3d_left, 1)
      if (any(array_3d_left(i,:,:) .ne. &
           array_3d(nint(rlabel_left(i)),:,:))) then
         write(*,*) '3D data split is not correct'
         success = .false.
      end if
   end do

!!!-----------------------------------------------------------------------------
!!! 5D array shuffle tests
!!!-----------------------------------------------------------------------------

  !! initialise array
    do i = 1, n
      do j = 1, m
         do k = 1, p
           do l = 1, q
              do o = 1, s
                 array_5d(i, j, k, l, o) = &
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

  !! split data
  call split(array_5d, array_5d_left, array_5d_right, &
       dim = 1, left_size=left_size, seed = 1)

  itmp1 = nint(real(n) * 0.25)
  !! check if the left array is the correct size
  if (size(array_5d_left, 1) .ne. itmp1) then
     write(*,*) '5D array left size is not correct'
     success = .false.
  end if


  if (size(array_5d_left,1) + size(array_5d_right,1) .ne. n ) then !.or. &
      !  size(label_left,1) + size(label_right,1) .ne. n) then
     write(*,*) '5D array label sizes do not add up to n'
     success = .false.
  end if

  !! split data
  call split(array_5d, rlabel, array_5d_left, array_5d_right, &
       rlabel_left, rlabel_right, &
       dim = 1, left_size=left_size, seed = 1)


  if (size(array_5d_left, 1) .ne. size(rlabel_left)) then
     write(*,*) '5D array size and label size do not match'
     success = .false.
  end if
  do i = 1, size(array_5d_left, 1)
     if (any(abs(array_5d_left(i,:,:,:,:) - &
          array_5d(nint(rlabel_left(i)),:,:,:,:)) .gt. 1.E-6 ) ) then
        write(*,*) '5D data split is not correct'
        success = .false.
     end if
  end do


  call split(array_5d, array_5d_left, array_5d_right, &
       dim = 1, right_size=1.E0-left_size, seed = 1)
  if (size(array_5d_left, 1) .ne. itmp1) then
     write(*,*) '5D array left size is not correct'
     success = .false.
  end if
  call split(array_5d, rlabel, array_5d_left, array_5d_right, &
       rlabel_left, rlabel_right, &
       dim = 1, right_size=1.E0-left_size, seed = 1)
  if (size(array_5d_left, 1) .ne. itmp1) then
     write(*,*) '5D array left size is not correct'
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


end program test_split
