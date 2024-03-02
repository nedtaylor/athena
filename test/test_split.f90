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
     do j = 1, m
        do k = 1, p
           array_3d(i, j, k) = i * m * p + j * p + k
        end do
     end do
  end do

!   subroutine split_3Didata_1Dilist(data,label,left_data,right_data,&
!    left_label,right_label,dim,&
!    left_size,right_size,&
!    shuffle,seed,split_list)
! implicit none
! integer, dimension(:,:,:), intent(in) :: data
! integer, dimension(:), intent(in) :: label
! integer, allocatable, dimension(:,:,:), intent(out) :: left_data, right_data
! integer, allocatable, dimension(:), intent(out) :: left_label, right_label
! integer, intent(in) :: dim
! real(real12), optional, intent(in) :: left_size, right_size
! logical, optional, intent(in) :: shuffle
! integer, optional, intent(in) :: seed
! integer, optional, dimension(size(data,dim)), intent(out) :: split_list


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
  if (size(label_left,1) + size(label_right,1) .ne. n) then
     write(*,*) '3D array label sizes do not add up to n'
     success = .false.
  end if

  

!   !! Check if all original elements are still in the shuffled 3D array
!   do i = 1, n
!     do j = 1, m
!        do k = 1, p
!           if ( all(abs(array_3d_shuffled(i, j, :) - &
!                array_3d_original(i, j, k)).gt.1.E-6) ) then
!              write(*,*) 'Original element', array_3d_original(i, j, k), &
!                   'is missing in the shuffled 3D array'
!              success = .false.
!           end if
!        end do
!     end do
!   end do

!   !! Check that seed works for 3D array shuffling
!   array_3d_shuffled_tmp = array_3d_original
!   call shuffle(array_3d_shuffled_tmp, dim = 3, seed = 4)
!   array_3d_shuffled = array_3d_original
!   call shuffle(array_3d_shuffled, dim = 3, seed = 4)

!   !! Check if array_3d_shuffled and array_3d_shuffled_tmp are the same
!   if (any(abs(array_3d_shuffled_tmp - array_3d_shuffled).gt.1.E-6)) then
!      write(*,*) '3D Array shuffle seed does not work as intended'
!      success = .false.
!   end if


!!!-----------------------------------------------------------------------------
!!! 5D array shuffle tests
!!!-----------------------------------------------------------------------------
  
   !  !! test 5D array shuffling
   !  do i = 1, n
   !    do j = 1, m
   !       do k = 1, p
   !         do l = 1, q
   !            do o = 1, s
   !               array_5d_original(i, j, k, l, o) = &
   !                    i * m * p * q * s + &
   !                    j * p * q * s + &
   !                    k * q * s + &
   !                    l * s + &
   !                    o
   !            end do
   !         end do
   !       end do
   !    end do
   ! end do
 
   ! !! Shuffle the 3D array along the third dimension
   ! array_5d_shuffled = array_5d_original
   ! call shuffle(array_5d_shuffled, dim = 5, seed = 1)
 
   ! !! Check if the 3D array is shuffled along the third dimension
   ! if (all(array_5d_shuffled .eq. array_5d_original)) then
   !    write(*,*) '5D Array is not shuffled along the third dimension'
   !    success = .false.
   ! end if
 
   ! !! Check if all original elements are still in the shuffled 5D array
   ! do i = 1, n
   !   do j = 1, m
   !      do k = 1, p
   !         do l = 1, q
   !            do o = 1, s
   !               if ( all(abs(array_5d_shuffled(i, j, k, l, :) - &
   !                  array_5d_original(i, j, k, l, o)).gt.1.E-6) ) then
   !                  write(*,*) 'Original element', &
   !                       array_5d_original(i, j, k, l, o), &
   !                       'is missing in the shuffled 5D array'
   !                  success = .false.
   !               end if
   !             end do
   !         end do
   !      end do
   !   end do
   ! end do
 
   ! !! Check that seed works for 5D array shuffling
   ! array_5d_shuffled_tmp = array_5d_original
   ! call shuffle(array_5d_shuffled_tmp, dim = 5, seed = 4)
   ! array_5d_shuffled = array_5d_original
   ! call shuffle(array_5d_shuffled, dim = 5, seed = 4)
 
   ! !! Check if array_5d_shuffled and array_5d_shuffled_tmp are the same
   ! if (any(abs(array_5d_shuffled_tmp - array_5d_shuffled).gt.1.E-6)) then
   !    write(*,*) '5D Array shuffle seed does not work as intended'
   !    success = .false.
   ! end if


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


end program test_split
