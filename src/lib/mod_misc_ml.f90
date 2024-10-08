!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains various miscellaneous procedures relating to ...
!!! ... machine learning
!!! module includes the following procedures:
!!! - shuffle - randomly shuffle a 2D array along one dimension
!!! - split   - split data into train and test sets
!!!##################
!!! - set_padding - set padding any-rank 2D array
!!! - pad_data    - pad any-rank array
!!!#############################################################################
module misc_ml
  use constants, only: real12
  implicit none


!!!-----------------------------------------------------
!!! shuffle an array along one dimension
!!! optional index array is also shuffled
!!!-----------------------------------------------------
!!! data  = (I/R, io) input array to be shuffled
!!! dim   = (I, in) dimension along which to shuffle
!!! label = (I, io, opt) index array to be shuffled
!!! seed  = (I, in, opt) random seed
  interface shuffle
  procedure shuffle_1Dilist, &
       shuffle_2Drdata, shuffle_3Didata, shuffle_3Drdata, &
       shuffle_4Drdata, shuffle_5Drdata, &
       shuffle_3Didata_1Dilist, shuffle_3Didata_1Drlist, &
       shuffle_4Drdata_1Dilist, shuffle_5Drdata_1Dilist, shuffle_5Drdata_1Drlist
  end interface shuffle
!!!=====================================================


!!!-----------------------------------------------------
!!! split an array into train and test sets
!!! optional index array is also split
!!!-----------------------------------------------------
!!! data        = (I/R, in ) input array to be shuffled
!!! left_data   = (I/R, out) left split of data
!!! right_data  = (I/R, out) right split of data
!!! dim         = (I, in ) dimension along which to split
!!! label       = (I, in , opt) index array to be shuffled
!!! left_label  = (I, out, opt) left split of label
!!! right_label = (I, out, opt) right split of label
!!! left_size   = (I, in , opt) size of left split
!!! right_size  = (I, in , opt) size of right split
!!! shuffle     = (B, in , opt) shuffle data before splitting
!!! seed        = (I, in , opt) random seed
!!! split_list  = (I, out, opt) index array of split
  interface split
  procedure &
       split_3Didata_1Dilist, split_3Didata_1Drlist, &
       split_5Drdata, &
       split_5Drdata_1Drlist
  end interface split
!!!=====================================================


  private

  public :: shuffle, split
  public :: set_padding, pad_data


contains
!!!#####################################################
!!! shuffle an array along one dimension
!!!#####################################################
subroutine shuffle_1Dilist(data,seed)
implicit none
integer :: istart, num_data, seed_size
integer :: itmp1, i, j
real(real12) :: r
integer, allocatable, dimension(:) :: iseed

integer, optional, intent(in) :: seed
integer, dimension(:), intent(inout) :: data

!! set or get random seed
call random_seed(size=seed_size)
allocate(iseed(seed_size))
if(present(seed))then
   iseed = seed
   call random_seed(put=iseed)
else
   call random_seed(get=iseed)
end if

!! shuffle the data
num_data = size(data,dim=1)
istart=1
do i=1,num_data
  call random_number(r)
  j = istart + floor((num_data+1-istart)*r)
  if(i.eq.j) cycle
  itmp1   = data(j)
  data(j) = data(i)
  data(i) = itmp1
end do

end subroutine shuffle_1Dilist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_2Drdata(data,dim,seed)
implicit none
integer :: istart,seed_size
integer :: i,j,n_data,iother
integer :: i1s,i2s,i1e,i2e,j1s,j2s,j1e,j2e
real(real12) :: r
integer, allocatable, dimension(:) :: iseed
real(real12), allocatable, dimension(:,:) :: tlist

integer, intent(in) :: dim
real(real12), dimension(:,:), intent(inout) :: data

integer, optional, intent(in) :: seed

!! set or get random seed
call random_seed(size=seed_size)
allocate(iseed(seed_size))
if(present(seed))then
   iseed = seed
   call random_seed(put=iseed)
else
   call random_seed(get=iseed)
end if

!! shuffle the data
n_data = size(data,dim=dim)
if(dim.eq.1)then
  iother = 2
  i2s=1;i2e=size(data,dim=iother)
  j2s=1;j2e=size(data,dim=iother)
else
  iother = 1
  i1s=1;i1e=size(data,dim=iother)
  j1s=1;j1e=size(data,dim=iother)
end if
istart=1
allocate(tlist(1,size(data,dim=iother)))
do i=1,n_data
  call random_number(r)
  j = istart + floor((n_data+1-istart)*r)
  if(i.eq.j) cycle
  if(dim.eq.1)then
     i1s=i;i1e=i
     j1s=j;j1e=j
  else
     i2s=i;i2e=i
     j2s=j;j2e=j
  end if
  tlist(1:1,:) = reshape(data(i1s:i1e,i2s:i2e),shape=shape(tlist))
  data(i1s:i1e,i2s:i2e) = data(j1s:j1e,j2s:j2e)
  data(j1s:j1e,j2s:j2e) = reshape(tlist(1:1,:),&
       shape=shape(data(j1s:j1e,j2s:j2e)))
end do

end subroutine shuffle_2Drdata
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_3Drdata(data,dim,seed)
implicit none
integer :: istart,seed_size
integer :: i,j,n_data
real(real12) :: r
integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
integer, dimension(3,2) :: t_size
integer, allocatable, dimension(:) :: iseed
real(real12), allocatable, dimension(:,:,:) :: tlist

integer, intent(in) :: dim
real(real12), dimension(:,:,:), intent(inout) :: data

integer, optional, intent(in) :: seed

!! set or get random seed
call random_seed(size=seed_size)
allocate(iseed(seed_size))
if(present(seed))then
   iseed = seed
   call random_seed(put=iseed)
else
   call random_seed(get=iseed)
end if

n_data = size(data,dim=dim)
do i=1,3
  t_size(i,1) = 1
  jdx_s(i) = 1
  jdx_e(i) = size(data,dim=i)
  idx_s(i) = 1
  idx_e(i) = size(data,dim=i)
  if(i.eq.dim) then
     t_size(i,2) = 1
  else
     t_size(i,2) = size(data,dim=i)
  end if
end do

allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))

istart=1
do i=1,n_data
  call random_number(r)
  j = istart + floor((n_data+1-istart)*r)
  if(i.eq.j) cycle
  idx_s(dim) = i
  idx_e(dim) = i
  jdx_s(dim) = j
  jdx_e(dim) = j
  tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2)) = data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3))
  data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3)) = data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3))
  data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3)) = tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2))
end do

end subroutine shuffle_3Drdata
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_3Didata(data,dim,seed)
 implicit none
 integer :: istart,seed_size
 integer :: i,j,n_data
 real(real12) :: r
 integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
 integer, dimension(3,2) :: t_size
 integer, allocatable, dimension(:) :: iseed
 integer, allocatable, dimension(:,:,:) :: tlist

 integer, intent(in) :: dim
 integer, dimension(:,:,:), intent(inout) :: data

 integer, optional, intent(in) :: seed


 !! set or get random seed
 call random_seed(size=seed_size)
 allocate(iseed(seed_size))
 if(present(seed))then
    iseed = seed
    call random_seed(put=iseed)
 else
    call random_seed(get=iseed)
 end if

 !! get the size of the data
 n_data = size(data,dim=dim)
 do i=1,3
    t_size(i,1) = 1
    jdx_s(i) = 1
    jdx_e(i) = size(data,dim=i)
    idx_s(i) = 1
    idx_e(i) = size(data,dim=i)
    if(i.eq.dim) then
       t_size(i,2) = 1
    else
       t_size(i,2) = size(data,dim=i)
    end if
 end do
 allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))

 !! shuffle the data
 istart=1
 do i=1,n_data
    call random_number(r)
    j = istart + floor((n_data+1-istart)*r)
    if(i.eq.j) cycle
    idx_s(dim) = i
    idx_e(dim) = i
    jdx_s(dim) = j
    jdx_e(dim) = j
    tlist(&
         t_size(1,1):t_size(1,2),&
         t_size(2,1):t_size(2,2),&
         t_size(3,1):t_size(3,2)) = data(&
         idx_s(1):idx_e(1),&
         idx_s(2):idx_e(2),&
         idx_s(3):idx_e(3))
    data(&
         idx_s(1):idx_e(1),&
         idx_s(2):idx_e(2),&
         idx_s(3):idx_e(3)) = data(&
         jdx_s(1):jdx_e(1),&
         jdx_s(2):jdx_e(2),&
         jdx_s(3):jdx_e(3))
    data(&
         jdx_s(1):jdx_e(1),&
         jdx_s(2):jdx_e(2),&
         jdx_s(3):jdx_e(3)) = tlist(&
         t_size(1,1):t_size(1,2),&
         t_size(2,1):t_size(2,2),&
         t_size(3,1):t_size(3,2))
 end do
 
end subroutine shuffle_3Didata
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_4Drdata(data,dim,seed)
implicit none
integer :: istart,seed_size
integer :: i,j,n_data
real(real12) :: r
integer, dimension(4) :: idx_s,idx_e,jdx_s,jdx_e
integer, dimension(4,2) :: t_size
integer, allocatable, dimension(:) :: iseed
real(real12), allocatable, dimension(:,:,:,:) :: tlist

integer, intent(in) :: dim
real(real12), dimension(:,:,:,:), intent(inout) :: data

integer, optional, intent(in) :: seed


!! set or get random seed
call random_seed(size=seed_size)
allocate(iseed(seed_size))
if(present(seed))then
   iseed = seed
   call random_seed(put=iseed)
else
   call random_seed(get=iseed)
end if

!! get the size of the data
n_data = size(data,dim=dim)
do i=1,4
  t_size(i,1) = 1
  jdx_s(i) = 1
  jdx_e(i) = size(data,dim=i)
  idx_s(i) = 1
  idx_e(i) = size(data,dim=i)
  if(i.eq.dim) then
     t_size(i,2) = 1
  else
     t_size(i,2) = size(data,dim=i)      
  end if
end do
allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2),t_size(4,2)))

!! shuffle the data
istart=1
do i=1,n_data
  call random_number(r)
  j = istart + floor((n_data+1-istart)*r)
  idx_s(dim) = i
  idx_e(dim) = i
  jdx_s(dim) = j
  jdx_e(dim) = j
  tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2)) = data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4))
  data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4)) = data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4))
  data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4)) = tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2))
end do

end subroutine shuffle_4Drdata
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_5Drdata(data,dim,seed)
implicit none
integer :: istart,seed_size
integer :: i,j,n_data
real(real12) :: r
integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
integer, dimension(5,2) :: t_size
integer, allocatable, dimension(:) :: iseed
real(real12), allocatable, dimension(:,:,:,:,:) :: tlist

integer, intent(in) :: dim
real(real12), dimension(:,:,:,:,:), intent(inout) :: data

integer, optional, intent(in) :: seed


!! set or get random seed
call random_seed(size=seed_size)
allocate(iseed(seed_size))
if(present(seed))then
   iseed = seed
   call random_seed(put=iseed)
else
   call random_seed(get=iseed)
end if

!! get the size of the data
n_data = size(data,dim=dim)
do i=1,5
  t_size(i,1) = 1
  jdx_s(i) = 1
  jdx_e(i) = size(data,dim=i)
  idx_s(i) = 1
  idx_e(i) = size(data,dim=i)
  if(i.eq.dim) then
     t_size(i,2) = 1
  else
     t_size(i,2) = size(data,dim=i)
  end if
end do
allocate(tlist(&
    t_size(1,2),t_size(2,2),&
    t_size(3,2),t_size(4,2),&
    t_size(5,2)))

!! shuffle the data
istart=1
do i=1,n_data
  call random_number(r)
  j = istart + floor((n_data+1-istart)*r)
  idx_s(dim) = i
  idx_e(dim) = i
  jdx_s(dim) = j
  jdx_e(dim) = j
  tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2),&
       t_size(5,1):t_size(5,2)) = data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4),&
       idx_s(5):idx_e(5))
  data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4),&
       idx_s(5):idx_e(5)) = data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4),&
       jdx_s(5):jdx_e(5))
  data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4),&
       jdx_s(5):jdx_e(5)) = tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2),&
       t_size(5,1):t_size(5,2))
end do

end subroutine shuffle_5Drdata
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_3Didata_1Dilist(data,label,dim,seed)
 implicit none
 integer :: istart,seed_size
 integer :: i,j,n_data
 integer :: itmp1
 real(real12) :: r
 integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
 integer, dimension(3,2) :: t_size
 integer, allocatable, dimension(:) :: iseed
 integer, allocatable, dimension(:,:,:) :: tlist

 integer, intent(in) :: dim
 integer, dimension(:,:,:), intent(inout) :: data
 integer, dimension(:), intent(inout) :: label

 integer, optional, intent(in) :: seed


 !! set or get random seed
 call random_seed(size=seed_size)
 allocate(iseed(seed_size))
 if(present(seed))then
    iseed = seed
    call random_seed(put=iseed)
 else
    call random_seed(get=iseed)
 end if

 !! get the size of the data
 n_data = size(data,dim=dim)
 do i=1,3
    t_size(i,1) = 1
    jdx_s(i) = 1
    jdx_e(i) = size(data,dim=i)
    idx_s(i) = 1
    idx_e(i) = size(data,dim=i)
    if(i.eq.dim) then
       t_size(i,2) = 1
    else
       t_size(i,2) = size(data,dim=i)
    end if
 end do
 allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))

 !! shuffle the data
 istart=1
 do i=1,n_data
    call random_number(r)
    j = istart + floor((n_data+1-istart)*r)
    idx_s(dim) = i
    idx_e(dim) = i
    jdx_s(dim) = j
    jdx_e(dim) = j
    tlist(&
         t_size(1,1):t_size(1,2),&
         t_size(2,1):t_size(2,2),&
         t_size(3,1):t_size(3,2)) = data(&
         idx_s(1):idx_e(1),&
         idx_s(2):idx_e(2),&
         idx_s(3):idx_e(3))
    data(&
         idx_s(1):idx_e(1),&
         idx_s(2):idx_e(2),&
         idx_s(3):idx_e(3)) = data(&
         jdx_s(1):jdx_e(1),&
         jdx_s(2):jdx_e(2),&
         jdx_s(3):jdx_e(3))
    data(&
         jdx_s(1):jdx_e(1),&
         jdx_s(2):jdx_e(2),&
         jdx_s(3):jdx_e(3)) = tlist(&
         t_size(1,1):t_size(1,2),&
         t_size(2,1):t_size(2,2),&
         t_size(3,1):t_size(3,2))

    itmp1 = label(i)
    label(i) = label(j)
    label(j) = itmp1

 end do

end subroutine shuffle_3Didata_1Dilist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
 subroutine shuffle_3Didata_1Drlist(data,label,dim,seed)
   implicit none
   integer :: istart,seed_size
   integer :: i,j,n_data
   real(real12) :: rtmp1
   real(real12) :: r
   integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
   integer, dimension(3,2) :: t_size
   integer, allocatable, dimension(:) :: iseed
   integer, allocatable, dimension(:,:,:) :: tlist

   integer, intent(in) :: dim
   integer, dimension(:,:,:), intent(inout) :: data
   real(real12), dimension(:), intent(inout) :: label

   integer, optional, intent(in) :: seed


   !! set or get random seed
   call random_seed(size=seed_size)
   allocate(iseed(seed_size))
   if(present(seed))then
      iseed = seed
      call random_seed(put=iseed)
   else
      call random_seed(get=iseed)
   end if

   !! get the size of the data
   n_data = size(data,dim=dim)
   do i=1,3
      t_size(i,1) = 1
      jdx_s(i) = 1
      jdx_e(i) = size(data,dim=i)
      idx_s(i) = 1
      idx_e(i) = size(data,dim=i)
      if(i.eq.dim) then
         t_size(i,2) = 1
      else
         t_size(i,2) = size(data,dim=i)
      end if
   end do
   allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))

   !! shuffle the data
   istart=1
   do i=1,n_data
      call random_number(r)
      j = istart + floor((n_data+1-istart)*r)
      idx_s(dim) = i
      idx_e(dim) = i
      jdx_s(dim) = j
      jdx_e(dim) = j
      tlist(&
           t_size(1,1):t_size(1,2),&
           t_size(2,1):t_size(2,2),&
           t_size(3,1):t_size(3,2)) = data(&
           idx_s(1):idx_e(1),&
           idx_s(2):idx_e(2),&
           idx_s(3):idx_e(3))
      data(&
           idx_s(1):idx_e(1),&
           idx_s(2):idx_e(2),&
           idx_s(3):idx_e(3)) = data(&
           jdx_s(1):jdx_e(1),&
           jdx_s(2):jdx_e(2),&
           jdx_s(3):jdx_e(3))
      data(&
           jdx_s(1):jdx_e(1),&
           jdx_s(2):jdx_e(2),&
           jdx_s(3):jdx_e(3)) = tlist(&
           t_size(1,1):t_size(1,2),&
           t_size(2,1):t_size(2,2),&
           t_size(3,1):t_size(3,2))

      rtmp1    = label(i)
      label(i) = label(j)
      label(j) = rtmp1

   end do

 end subroutine shuffle_3Didata_1Drlist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_4Drdata_1Dilist(data,label,dim,seed)
implicit none
integer :: istart, seed_size
integer :: i,j,n_data
integer :: itmp1
real(real12) :: r
integer, dimension(4) :: idx_s,idx_e,jdx_s,jdx_e
integer, dimension(4,2) :: t_size
integer, allocatable, dimension(:) :: iseed
real(real12), allocatable, dimension(:,:,:,:) :: tlist

integer, intent(in) :: dim
real(real12), dimension(:,:,:,:), intent(inout) :: data
integer, dimension(:), intent(inout) :: label

integer, optional, intent(in) :: seed


!! set or get random seed
call random_seed(size=seed_size)
allocate(iseed(seed_size))
if(present(seed))then
   iseed = seed
   call random_seed(put=iseed)
else
   call random_seed(get=iseed)
end if

!! get the size of the data
n_data = size(data,dim=dim)
do i=1,4
  t_size(i,1) = 1
  jdx_s(i) = 1
  jdx_e(i) = size(data,dim=i)
  idx_s(i) = 1
  idx_e(i) = size(data,dim=i)
  if(i.eq.dim) then
     t_size(i,2) = 1
  else
     t_size(i,2) = size(data,dim=i)
  end if
end do
allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2),t_size(4,2)))

!! shuffle the data
istart=1
do i=1,n_data
  call random_number(r)
  j = istart + floor((n_data+1-istart)*r)
  idx_s(dim) = i
  idx_e(dim) = i
  jdx_s(dim) = j
  jdx_e(dim) = j
  tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2)) = data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4))
  data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4)) = data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4))
  data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4)) = tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2))

  itmp1 = label(i)
  label(i) = label(j)
  label(j) = itmp1

end do

end subroutine shuffle_4Drdata_1Dilist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_5Drdata_1Dilist(data,label,dim,seed)
implicit none
integer :: istart,seed_size
integer :: i,j,n_data
integer :: itmp1
real(real12) :: r
integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
integer, dimension(5,2) :: t_size
integer, allocatable, dimension(:) :: iseed
real(real12), allocatable, dimension(:,:,:,:,:) :: tlist

integer, intent(in) :: dim
real(real12), dimension(:,:,:,:,:), intent(inout) :: data
integer, dimension(:), intent(inout) :: label

integer, optional, intent(in) :: seed


!! set or get random seed
call random_seed(size=seed_size)
allocate(iseed(seed_size))
if(present(seed))then
   iseed = seed
   call random_seed(put=iseed)
else
   call random_seed(get=iseed)
end if

!! get the size of the data
n_data = size(data,dim=dim)
do i=1,5
  t_size(i,1) = 1
  jdx_s(i) = 1
  jdx_e(i) = size(data,dim=i)
  idx_s(i) = 1
  idx_e(i) = size(data,dim=i)
  if(i.eq.dim) then
     t_size(i,2) = 1
  else
     t_size(i,2) = size(data,dim=i)
  end if
end do
allocate(tlist(&
    t_size(1,2),t_size(2,2),&
    t_size(3,2),t_size(4,2),&
    t_size(5,2)))

!! shuffle the data
istart=1
do i=1,n_data
  call random_number(r)
  j = istart + floor((n_data+1-istart)*r)
  idx_s(dim) = i
  idx_e(dim) = i
  jdx_s(dim) = j
  jdx_e(dim) = j
  tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2),&
       t_size(5,1):t_size(5,2)) = data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4),&
       idx_s(5):idx_e(5))
  data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4),&
       idx_s(5):idx_e(5)) = data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4),&
       jdx_s(5):jdx_e(5))
  data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4),&
       jdx_s(5):jdx_e(5)) = tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2),&
       t_size(5,1):t_size(5,2))

  itmp1 = label(i)
  label(i) = label(j)
  label(j) = itmp1

end do

end subroutine shuffle_5Drdata_1Dilist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine shuffle_5Drdata_1Drlist(data,label,dim,seed,shuffle_list)
implicit none
integer :: istart,seed_size
integer :: i,j,n_data
real(real12) :: rtmp1
real(real12) :: r
integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
integer, dimension(5,2) :: t_size
integer, allocatable, dimension(:) :: iseed
real(real12), allocatable, dimension(:,:,:,:,:) :: tlist

integer, intent(in) :: dim
real(real12), dimension(:,:,:,:,:), intent(inout) :: data
real(real12), dimension(:), intent(inout) :: label

integer, optional, intent(in) :: seed
integer, optional, dimension(size(data,dim)), intent(out) :: shuffle_list


!! set or get random seed
call random_seed(size=seed_size)
allocate(iseed(seed_size))
if(present(seed))then
   iseed = seed
   call random_seed(put=iseed)
else
   call random_seed(get=iseed)
end if

!! get the size of the data
n_data = size(data,dim=dim)
do i=1,5
  t_size(i,1) = 1
  jdx_s(i) = 1
  jdx_e(i) = size(data,dim=i)
  idx_s(i) = 1
  idx_e(i) = size(data,dim=i)
  if(i.eq.dim) then
     t_size(i,2) = 1
  else
     t_size(i,2) = size(data,dim=i)
  end if
end do

allocate(tlist(&
    t_size(1,2),t_size(2,2),&
    t_size(3,2),t_size(4,2),&
    t_size(5,2)))

istart=1
do i=1,n_data
  call random_number(r)
  j = istart + floor((n_data+1-istart)*r)
  if(present(shuffle_list)) shuffle_list(i) = j
  idx_s(dim) = i
  idx_e(dim) = i
  jdx_s(dim) = j
  jdx_e(dim) = j
  tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2),&
       t_size(5,1):t_size(5,2)) = data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4),&
       idx_s(5):idx_e(5))
  data(&
       idx_s(1):idx_e(1),&
       idx_s(2):idx_e(2),&
       idx_s(3):idx_e(3),&
       idx_s(4):idx_e(4),&
       idx_s(5):idx_e(5)) = data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4),&
       jdx_s(5):jdx_e(5))
  data(&
       jdx_s(1):jdx_e(1),&
       jdx_s(2):jdx_e(2),&
       jdx_s(3):jdx_e(3),&
       jdx_s(4):jdx_e(4),&
       jdx_s(5):jdx_e(5)) = tlist(&
       t_size(1,1):t_size(1,2),&
       t_size(2,1):t_size(2,2),&
       t_size(3,1):t_size(3,2),&
       t_size(4,1):t_size(4,2),&
       t_size(5,1):t_size(5,2))

  rtmp1 = label(i)
  label(i) = label(j)
  label(j) = rtmp1

end do

end subroutine shuffle_5Drdata_1Drlist
!!!#####################################################

!!!#####################################################
!!! split an array along a dimension into two
!!!#####################################################
subroutine split_5Drdata(data,left,right,dim,&
  left_size,right_size,&
  shuffle,seed)
implicit none
real(real12), dimension(:,:,:,:,:), intent(in) :: data
real(real12), allocatable, dimension(:,:,:,:,:), intent(out) :: left, right
integer, intent(in) :: dim
real(real12), optional, intent(in) :: left_size, right_size
logical, optional, intent(in) :: shuffle
integer, optional, intent(in) :: seed

integer :: seed_, left_num_, right_num_
logical :: shuffle_
integer :: i, j
integer :: num_redos
real(real12) :: rtmp1
integer, allocatable, dimension(:) :: indices_l, indices_r
real(real12), allocatable, dimension(:) :: tlist
real(real12), allocatable, dimension(:,:,:,:,:) :: data_copy

type idx_type
  integer, allocatable, dimension(:) :: loc
end type idx_type
type(idx_type), dimension(5) :: idx


!! determine number of elements for left and right split
if(.not.present(left_size).and..not.present(right_size))then
  stop "ERROR: neither left_size nor right_size provided to split. Expected at least one."
elseif(present(left_size).and..not.present(right_size))then
  left_num_  = nint(left_size*size(data,dim))
  right_num_ = size(data,dim) - left_num_
elseif(.not.present(left_size).and.present(right_size))then
  right_num_ = nint(right_size*size(data,dim))
  left_num_  = size(data,dim) - right_num_
else
  left_num_  = nint(left_size*size(data,dim))
  right_num_ = nint(right_size*size(data,dim))
  if(left_num_ + right_num_ .ne. size(data,dim)) &
       right_num_ = size(data,dim) - left_num_
end if


!! initialies optional arguments
if(present(shuffle))then
  shuffle_ = shuffle
else
  shuffle_ = .false.
end if

if(present(seed))then
  seed_ = seed
else
  call system_clock(count=seed_)
end if

!! copy input data
data_copy = data
if(shuffle_) call shuffle_5Drdata(data_copy,dim,seed_)

!! get list of indices for right split
num_redos = 0
allocate(tlist(right_num_))
call random_number(tlist)
indices_r = floor(tlist*size(data,dim)) + 1
i = 1
indices_r_loop: do 
  if(i.ge.right_num_) exit indices_r_loop
  i = i + 1
  if(any(indices_r(:i-1).eq.indices_r(i)))then
     indices_r(i:right_num_-num_redos-1) = indices_r(i+1:right_num_-num_redos)
     call random_number(rtmp1)
     indices_r(right_num_) = floor(rtmp1*size(data,dim)) + 1
     i = i - 1
  end if
end do indices_r_loop

!! generate right split
do i=1,5
  if(i.eq.dim)then
     idx(i)%loc = indices_r
  else
     idx(i)%loc = (/ ( j, j=1,size(data,i) ) /)
  end if
end do
right = data_copy(idx(1)%loc,idx(2)%loc,idx(3)%loc,idx(4)%loc,idx(5)%loc)

!! get list of indices for left split
indices_l_loop: do i=1,size(data,dim)
  if(any(indices_r.eq.i)) cycle indices_l_loop
  if(allocated(indices_l)) then
     indices_l = [indices_l(:), i]
  else
     indices_l = [i]
  end if
end do indices_l_loop

!! generate left split
idx(dim)%loc = indices_l
left = data_copy(idx(1)%loc,idx(2)%loc,idx(3)%loc,idx(4)%loc,idx(5)%loc)

end subroutine split_5Drdata
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine split_3Didata_1Dilist(data,label,left_data,right_data,&
    left_label,right_label,dim,&
    left_size,right_size,&
    shuffle,seed,split_list)
 implicit none
 integer, dimension(:,:,:), intent(in) :: data
 integer, dimension(:), intent(in) :: label
 integer, allocatable, dimension(:,:,:), intent(out) :: left_data, right_data
 integer, allocatable, dimension(:), intent(out) :: left_label, right_label
 integer, intent(in) :: dim
 real(real12), optional, intent(in) :: left_size, right_size
 logical, optional, intent(in) :: shuffle
 integer, optional, intent(in) :: seed
 integer, optional, dimension(size(data,dim)), intent(out) :: split_list

 integer :: seed_, left_num_, right_num_
 logical :: shuffle_
 integer :: i, j
 integer :: num_redos
 real(real12) :: rtmp1
 integer, allocatable, dimension(:) :: indices_l, indices_r
 real(real12), allocatable, dimension(:) :: tlist
 integer, allocatable, dimension(:) :: label_copy
 integer, allocatable, dimension(:,:,:) :: data_copy

 type idx_type
    integer, allocatable, dimension(:) :: loc
 end type idx_type
 type(idx_type), dimension(3) :: idx


 !! determine number of elements for left and right split
 if(.not.present(left_size).and..not.present(right_size))then
    stop "ERROR: neither left_size nor right_size provided to split.&
         &Expected at least one."
 elseif(present(left_size).and..not.present(right_size))then
    left_num_  = nint(left_size*size(data,dim))
    right_num_ = size(data,dim) - left_num_
 elseif(.not.present(left_size).and.present(right_size))then
    right_num_ = nint(right_size*size(data,dim))
    left_num_  = size(data,dim) - right_num_
 else
    left_num_  = nint(left_size*size(data,dim))
    right_num_ = nint(right_size*size(data,dim))
    if(left_num_ + right_num_ .ne. size(data,dim)) &
         right_num_ = size(data,dim) - left_num_
 end if

 !! initialies optional arguments
 if(present(shuffle))then
    shuffle_ = shuffle
 else
    shuffle_ = .false.
 end if

 if(present(seed))then
    seed_ = seed
 else
    call system_clock(count=seed_)
 end if

 !! copy input data
 data_copy = data
 label_copy = label
 if(shuffle_) call shuffle_3Didata_1Dilist(data_copy,label_copy,dim,seed_)

 !! get list of indices for right split
 num_redos = 0
 allocate(tlist(right_num_))
 call random_number(tlist)
 indices_r = floor(tlist*size(data,dim)) + 1
 i = 1
 indices_r_loop: do 
    if(i.ge.right_num_) exit indices_r_loop
    i = i + 1
    if(any(indices_r(:i-1).eq.indices_r(i)))then
       indices_r(i:right_num_-num_redos-1) = &
            indices_r(i+1:right_num_-num_redos)
       call random_number(rtmp1)
       indices_r(right_num_) = floor(rtmp1*size(data,dim)) + 1
       i = i - 1
    end if
 end do indices_r_loop

 !! generate right split
 do i=1,3
    if(i.eq.dim)then
       idx(i)%loc = indices_r
    else
       idx(i)%loc = (/ ( j, j=1,size(data,i) ) /)
    end if
 end do
 right_data = data_copy(&
      idx(1)%loc,idx(2)%loc,idx(3)%loc)
 right_label = label_copy(indices_r)

 !! get list of indices for left split
 if(present(split_list)) split_list = 2
 indices_l_loop: do i=1,size(data,dim)
    if(any(indices_r.eq.i)) cycle indices_l_loop
    if(allocated(indices_l)) then
       indices_l = [indices_l(:), i]
    else
       indices_l = [i]
    end if
    if(present(split_list)) split_list(i) = 1
 end do indices_l_loop

 !! generate left split
 idx(dim)%loc = indices_l
 left_data = data_copy(&
      idx(1)%loc,idx(2)%loc,idx(3)%loc)
 left_label = label_copy(indices_l)

end subroutine split_3Didata_1Dilist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine split_3Didata_1Drlist(data,label,left_data,right_data,&
    left_label,right_label,dim,&
    left_size,right_size,&
    shuffle,seed,split_list)
 implicit none
 integer, dimension(:,:,:), intent(in) :: data
 real(real12), dimension(:), intent(in) :: label
 integer, allocatable, dimension(:,:,:), intent(out) :: left_data, right_data
 real(real12), allocatable, dimension(:), intent(out) :: left_label, right_label
 integer, intent(in) :: dim
 real(real12), optional, intent(in) :: left_size, right_size
 logical, optional, intent(in) :: shuffle
 integer, optional, intent(in) :: seed
 integer, optional, dimension(size(data,dim)), intent(out) :: split_list

 integer :: seed_, left_num_, right_num_
 logical :: shuffle_
 integer :: i, j
 integer :: num_redos
 real(real12) :: rtmp1
 integer, allocatable, dimension(:) :: indices_l, indices_r
 real(real12), allocatable, dimension(:) :: tlist
 real(real12), allocatable, dimension(:) :: label_copy
 integer, allocatable, dimension(:,:,:) :: data_copy

 type idx_type
    integer, allocatable, dimension(:) :: loc
 end type idx_type
 type(idx_type), dimension(3) :: idx


 !! determine number of elements for left and right split
 if(.not.present(left_size).and..not.present(right_size))then
    stop "ERROR: neither left_size nor right_size provided to split.&
         &Expected at least one."
 elseif(present(left_size).and..not.present(right_size))then
    left_num_  = nint(left_size*size(data,dim))
    right_num_ = size(data,dim) - left_num_
 elseif(.not.present(left_size).and.present(right_size))then
    right_num_ = nint(right_size*size(data,dim))
    left_num_  = size(data,dim) - right_num_
 else
    left_num_  = nint(left_size*size(data,dim))
    right_num_ = nint(right_size*size(data,dim))
    if(left_num_ + right_num_ .ne. size(data,dim)) &
         right_num_ = size(data,dim) - left_num_
 end if

 !! initialies optional arguments
 if(present(shuffle))then
    shuffle_ = shuffle
 else
    shuffle_ = .false.
 end if

 if(present(seed))then
    seed_ = seed
 else
    call system_clock(count=seed_)
 end if

 !! copy input data
 data_copy = data
 label_copy = label
 if(shuffle_) call shuffle_3Didata_1Drlist(data_copy,label_copy,dim,seed_)

 !! get list of indices for right split
 num_redos = 0
 allocate(tlist(right_num_))
 call random_number(tlist)
 indices_r = floor(tlist*size(data,dim)) + 1
 i = 1
 indices_r_loop: do 
    if(i.ge.right_num_) exit indices_r_loop
    i = i + 1
    if(any(indices_r(:i-1).eq.indices_r(i)))then
       indices_r(i:right_num_-num_redos-1) = &
            indices_r(i+1:right_num_-num_redos)
       call random_number(rtmp1)
       indices_r(right_num_) = floor(rtmp1*size(data,dim)) + 1
       i = i - 1
    end if
 end do indices_r_loop

 !! generate right split
 do i=1,3
    if(i.eq.dim)then
       idx(i)%loc = indices_r
    else
       idx(i)%loc = (/ ( j, j=1,size(data,i) ) /)
    end if
 end do
 right_data = data_copy(&
      idx(1)%loc,idx(2)%loc,idx(3)%loc)
 right_label = label_copy(indices_r)

 !! get list of indices for left split
 if(present(split_list)) split_list = 2
 indices_l_loop: do i=1,size(data,dim)
    if(any(indices_r.eq.i)) cycle indices_l_loop
    if(allocated(indices_l)) then
       indices_l = [indices_l(:), i]
    else
       indices_l = [i]
    end if
    if(present(split_list)) split_list(i) = 1
 end do indices_l_loop

 !! generate left split
 idx(dim)%loc = indices_l
 left_data = data_copy(&
      idx(1)%loc,idx(2)%loc,idx(3)%loc)
 left_label = label_copy(indices_l)

end subroutine split_3Didata_1Drlist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
subroutine split_5Drdata_1Drlist(data,label,left_data,right_data,&
  left_label,right_label,dim,&
  left_size,right_size,&
  shuffle,seed,split_list)
implicit none
real(real12), dimension(:,:,:,:,:), intent(in) :: data
real(real12), dimension(:), intent(in) :: label
real(real12), allocatable, dimension(:,:,:,:,:), intent(out) :: left_data, right_data
real(real12), allocatable, dimension(:), intent(out) :: left_label, right_label
integer, intent(in) :: dim
real(real12), optional, intent(in) :: left_size, right_size
logical, optional, intent(in) :: shuffle
integer, optional, intent(in) :: seed
integer, optional, dimension(size(data,dim)), intent(out) :: split_list

integer :: seed_, left_num_, right_num_
logical :: shuffle_
integer :: i, j
integer :: num_redos
real(real12) :: rtmp1
integer, allocatable, dimension(:) :: indices_l, indices_r
real(real12), allocatable, dimension(:) :: tlist
real(real12), allocatable, dimension(:) :: label_copy
real(real12), allocatable, dimension(:,:,:,:,:) :: data_copy

type idx_type
  integer, allocatable, dimension(:) :: loc
end type idx_type
type(idx_type), dimension(5) :: idx


!! determine number of elements for left and right split
if(.not.present(left_size).and..not.present(right_size))then
  stop "ERROR: neither left_size nor right_size provided to split.&
       &Expected at least one."
elseif(present(left_size).and..not.present(right_size))then
  left_num_  = nint(left_size*size(data,dim))
  right_num_ = size(data,dim) - left_num_
elseif(.not.present(left_size).and.present(right_size))then
  right_num_ = nint(right_size*size(data,dim))
  left_num_  = size(data,dim) - right_num_
else
  left_num_  = nint(left_size*size(data,dim))
  right_num_ = nint(right_size*size(data,dim))
  if(left_num_ + right_num_ .ne. size(data,dim)) &
       right_num_ = size(data,dim) - left_num_
end if

!! initialies optional arguments
if(present(shuffle))then
  shuffle_ = shuffle
else
  shuffle_ = .false.
end if

if(present(seed))then
  seed_ = seed
else
  call system_clock(count=seed_)
end if

!! copy input data
data_copy = data
label_copy = label
if(shuffle_) call shuffle_5Drdata_1Drlist(data_copy,label_copy,dim,seed_)

!! get list of indices for right split
num_redos = 0
allocate(tlist(right_num_))
call random_number(tlist)
indices_r = floor(tlist*size(data,dim)) + 1
i = 1
indices_r_loop: do 
  if(i.ge.right_num_) exit indices_r_loop
  i = i + 1
  if(any(indices_r(:i-1).eq.indices_r(i)))then
     indices_r(i:right_num_-num_redos-1) = &
          indices_r(i+1:right_num_-num_redos)
     call random_number(rtmp1)
     indices_r(right_num_) = floor(rtmp1*size(data,dim)) + 1
     i = i - 1
  end if
end do indices_r_loop

!! generate right split
do i=1,5
  if(i.eq.dim)then
     idx(i)%loc = indices_r
  else
     idx(i)%loc = (/ ( j, j=1,size(data,i) ) /)
  end if
end do
right_data = data_copy(&
    idx(1)%loc,idx(2)%loc,idx(3)%loc,idx(4)%loc,idx(5)%loc)
right_label = label_copy(indices_r)

!! get list of indices for left split
if(present(split_list)) split_list = 2
indices_l_loop: do i=1,size(data,dim)
  if(any(indices_r.eq.i)) cycle indices_l_loop
  if(allocated(indices_l)) then
     indices_l = [indices_l(:), i]
  else
     indices_l = [i]
  end if
  if(present(split_list)) split_list(i) = 1
end do indices_l_loop

!! generate left split
idx(dim)%loc = indices_l
left_data = data_copy(&
    idx(1)%loc,idx(2)%loc,idx(3)%loc,idx(4)%loc,idx(5)%loc)
left_label = label_copy(indices_l)

end subroutine split_5Drdata_1Drlist
!!!####################e#################################


!!!#############################################################################
!!!#############################################################################
!!!  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
!!!#############################################################################
!!!#############################################################################



!!!########################################################################
!!! return width of padding from kernel/filter size
!!!########################################################################
  pure function get_padding_half(width) result(output)
    implicit none
    integer, intent(in) :: width
    integer :: output
    
    output = ( width - (1 - mod(width,2)) - 1 ) / 2
  end function get_padding_half
!!!########################################################################


!!!########################################################################
!!! return width of padding from kernel/filter size
!!!########################################################################
  subroutine set_padding(pad, kernel_size, padding_method, verbose)
    use misc, only: to_lower
    implicit none
    integer, intent(out) :: pad
    integer, intent(in) :: kernel_size
    character(*), intent(inout) :: padding_method
    integer, optional, intent(in) :: verbose
    
    integer :: t_verbose = 0


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(verbose)) t_verbose = verbose

    !!---------------------------------------------------------------------
    !! padding method options
    !!---------------------------------------------------------------------
    !! none  = alt. name for 'valid'
    !! zero  = alt. name for 'same'
    !! symmetric = alt.name for 'replication'
    !! valid = no padding
    !! same  = maintain spatial dimensions
    !!         ... (i.e. padding added = (kernel_size - 1)/2)
    !!         ... defaults to zeros in the padding
    !! full  = enough padding for filter to slide over every possible position
    !!         ... (i.e. padding added = (kernel_size - 1)
    !! circular = maintain spatial dimensions
    !!            ... wraps data around for padding (periodic)
    !! reflection = maintains spatial dimensions
    !!              ... reflect data (about boundary index)
    !! replication = maintains spatial dimensions
    !!               ... reflect data (boundary included)
100 select case(to_lower(trim(padding_method)))
    case("none")
       padding_method = "valid"
       goto 100
    case("zero")
       padding_method = "same"
       goto 100
    case("half")
       padding_method = "same"
       goto 100
    case("symmetric")
       padding_method = "replication"
       goto 100
    case("valid", "vali")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'valid' (no padding)"
       pad = 0
       return
    case("same")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'same' (pad with zeros)"
    case("circular")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'same' (circular padding)"
    case("full")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'full' (all possible positions)"
       pad = kernel_size - 1
       return
    case("reflection")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'reflection' (reflect on boundary)"
    case("replication")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'replication' (reflect after boundary)"
    case default
       write(0,*) "ERROR: padding type '"//padding_method//"' not known"
       stop 1
    end select

    pad = get_padding_half(kernel_size)

  end subroutine set_padding
!!!########################################################################


!!!########################################################################
!!! pad dataset
!!!########################################################################
  subroutine pad_data(data, data_padded, &
       kernel_size, padding_method, &
       sample_dim, channel_dim, constant)
    implicit none
    !real(real12), allocatable, dimension(:,:), intent(inout) :: data
    real(real12), dimension(..), intent(in) :: data
    real(real12), allocatable, dimension(..), intent(out) :: data_padded
    integer, dimension(..), intent(in) :: kernel_size
    character(*), intent(inout) :: padding_method
    real(real12), optional, intent(in) :: constant

    integer, optional, intent(in) :: sample_dim, channel_dim
    
    integer :: i, j, idim
    integer :: num_samples, num_channels, ndim, ndata_dim
    integer :: sample_dim_ = 0, channel_dim_ = 0
    real(real12) :: constant_ = 0._real12
    integer, dimension(2) :: bound_store
    integer, allocatable, dimension(:) :: padding
    integer, allocatable, dimension(:,:) :: trgt_bound, dest_bound
    integer, allocatable, dimension(:,:) :: tmp_trgt_bound, tmp_dest_bound
    !real(real12), allocatable, dimension(:,:) :: data_copy


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(constant)) constant_ = constant
    if(present(sample_dim)) sample_dim_ = sample_dim
    if(present(channel_dim)) channel_dim_ = channel_dim

    ndim = rank(data)
#if defined(GFORTRAN)
    if(ndim.ne.rank(data_padded)) then
       stop "ERROR: data and data_padded are not the same rank"
    end if
#else
    select rank(data_padded)
    rank(1)
       if(ndim.ne.1)  stop "ERROR: data and data_padded are not the same rank"
    rank(2)
       if(ndim.ne.2)  stop "ERROR: data and data_padded are not the same rank"
    rank(3)
       if(ndim.ne.3)  stop "ERROR: data and data_padded are not the same rank"
    rank(4)
       if(ndim.ne.4)  stop "ERROR: data and data_padded are not the same rank"
    rank(5)
       if(ndim.ne.5)  stop "ERROR: data and data_padded are not the same rank"
    end select
#endif
    ndata_dim = ndim
    if(sample_dim_.gt.0)  ndata_dim = ndata_dim - 1
    if(channel_dim_.gt.0) ndata_dim = ndata_dim - 1

    select rank(data)
    rank(1)
       if(sample_dim_.gt.0) num_samples = size(data,sample_dim_)
       if(channel_dim_.gt.0) num_channels = size(data,channel_dim_)
    rank(2)
       if(sample_dim_.gt.0) num_samples = size(data,sample_dim_)
       if(channel_dim_.gt.0) num_channels = size(data,channel_dim_)
    rank(3)
       if(sample_dim_.gt.0) num_samples = size(data,sample_dim_)
       if(channel_dim_.gt.0) num_channels = size(data,channel_dim_)
    rank(4)
       if(sample_dim_.gt.0) num_samples = size(data,sample_dim_)
       if(channel_dim_.gt.0) num_channels = size(data,channel_dim_)
    rank(5)
       if(sample_dim_.gt.0) num_samples = size(data,sample_dim_)
       if(channel_dim_.gt.0) num_channels = size(data,channel_dim_)
    rank default
       stop "ERROR: cannot handle data with this rank"
    end select
    

!!!-----------------------------------------------------------------------------
!!! handle padding type name
!!!-----------------------------------------------------------------------------
    !! none  = alt. name for 'valid'
    !! zero  = alt. name for 'same'
    !! symmetric = alt.name for 'replication'
    !! valid = no padding
    !! same  = maintain spatial dimensions
    !!         ... (i.e. padding added = (kernel_size - 1)/2)
    !!         ... defaults to zeros in the padding
    !! full  = enough padding for filter to slide over every possible position
    !!         ... (i.e. padding added = (kernel_size - 1)
    !! circular = maintain spatial dimensions
    !!            ... wraps data around for padding (periodic)
    !! reflection = maintains spatial dimensions
    !!              ... reflect data (about boundary index)
    !! replication = maintains spatial dimensions
    !!               ... reflect data (boundary included)
    select rank(kernel_size)
    rank(0)
       allocate(padding(ndata_dim))
       do i=1,ndata_dim
          call set_padding(padding(i), kernel_size, padding_method, verbose=0)
       end do
    rank(1)
       if(size(kernel_size).eq.1.and.ndata_dim.gt.1)then
          allocate(padding(ndata_dim))
          do i=1,ndata_dim
             call set_padding(padding(i), kernel_size(1), padding_method, verbose=0)
          end do
       else
          if(sample_dim_.eq.0.and.channel_dim_.eq.0.and.&
               size(kernel_size).ne.ndim)then
             write(0,*) "kernel dimension:", size(kernel_size)
             write(0,*) "data rank:", ndim
             write(0,*) "ERROR: kernel_size length not equal to rank of data"
             stop 1
          elseif(sample_dim_.gt.0.and.channel_dim_.gt.0.and.&
               size(kernel_size).ne.ndim-2)then
             write(0,*) "kernel dimension:", size(kernel_size)
             write(0,*) "data rank:", ndim-2
             write(0,*) "ERROR: kernel_size length not equal to rank of data-2"
             stop 1
          elseif(xor(sample_dim_.gt.0,channel_dim_.gt.0).and.&
               size(kernel_size).ne.ndim-1)then
             write(0,*) "kernel dimension:", size(kernel_size)
             write(0,*) "data rank:", ndim-1
             write(0,*) "ERROR: kernel_size length not equal to rank of data-1"
             stop 1
          else
             allocate(padding(size(kernel_size)))
          end if
          do i=1,size(kernel_size)
             call set_padding(padding(i), kernel_size(i), padding_method, verbose=0)
          end do
       end if
    end select


!!!-----------------------------------------------------------------------------
!!! allocate data set
!!! ... if appropriate, add padding
!!!-----------------------------------------------------------------------------
    select case(padding_method)
    case("same")
    case("full")
    case("zero")
    case default
       if(abs(constant_).gt.1.E-8) &
            write(*,*) "WARNING: constant is ignored for selected padding method"
    end select

    
    allocate(dest_bound(2,ndim))
    allocate(trgt_bound(2,ndim))
    i = 0
    do idim=1,ndim
       trgt_bound(:,idim) = [ lbound(data,dim=idim), ubound(data,dim=idim) ]
       dest_bound(:,idim) = trgt_bound(:,idim)
       if(idim.eq.sample_dim_.or.idim.eq.channel_dim_) cycle
       i = i + 1
       dest_bound(:,idim) = dest_bound(:,idim) + [ -padding(i), padding(i) ]
    end do

    select rank(data_padded)
    rank(1)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1)), source = constant_)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(1)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1))
       end select
    rank(2)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1), &
            dest_bound(1,2):dest_bound(2,2)), source = constant_)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(2)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1),trgt_bound(1,2):trgt_bound(2,2)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1),trgt_bound(1,2):trgt_bound(2,2))
       end select
    rank(3)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1),&
            dest_bound(1,2):dest_bound(2,2),&
            dest_bound(1,3):dest_bound(2,3)), source = constant_)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(3)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3))
       end select
    rank(4)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1),&
            dest_bound(1,2):dest_bound(2,2),&
            dest_bound(1,3):dest_bound(2,3),&
            dest_bound(1,4):dest_bound(2,4)), source = constant_)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(4)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4))
       end select
    rank(5)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1),&
            dest_bound(1,2):dest_bound(2,2),&
            dest_bound(1,3):dest_bound(2,3),&
            dest_bound(1,4):dest_bound(2,4),&
            dest_bound(1,5):dest_bound(2,5)), source = constant_)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(5)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4), &
               trgt_bound(1,5):trgt_bound(2,5)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4), &
               trgt_bound(1,5):trgt_bound(2,5))
       end select
    end select


!!!-----------------------------------------------------------------------------
!!! return if constant -- or no -- padding
!!!-----------------------------------------------------------------------------
    select case(padding_method)
    case ("same")
       return
    case("full")
       return
    case("zero")
       return
    case("valid", "vali")
       return
    end select


!!!-----------------------------------------------------------------------------
!!! insert padding
!!!-----------------------------------------------------------------------------
    i = 0
    do idim=1,ndim
       if(idim.eq.sample_dim_.or.idim.eq.channel_dim_) cycle
       i = i + 1
       tmp_dest_bound = dest_bound
       tmp_trgt_bound = dest_bound
       tmp_dest_bound(:,idim) = [ dest_bound(1,idim), trgt_bound(1,idim) - 1 ]
       select case(padding_method)
       case ("circular")
          tmp_trgt_bound(:,idim) = [ trgt_bound(2,idim) - padding(i) + 1, trgt_bound(2,idim) ]
       case("reflection")
          tmp_trgt_bound(:,idim) = [ trgt_bound(1,idim) + 1, trgt_bound(1,idim) + padding(i) ]
       case("replication")
          tmp_trgt_bound(:,idim) = [ trgt_bound(1,idim), trgt_bound(1,idim) + padding(i) - 1 ]
       end select
       do j = 1, 2
          select rank(data_padded)
          rank(1)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1)) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1))
          rank(2)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2) ) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2) )
          rank(3)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3) ) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3) )
          rank(4)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3), &
                  tmp_dest_bound(1,4):tmp_dest_bound(2,4) ) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3), &
                  tmp_trgt_bound(1,4):tmp_trgt_bound(2,4) )
          rank(5)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3), &
                  tmp_dest_bound(1,4):tmp_dest_bound(2,4), &
                  tmp_dest_bound(1,5):tmp_dest_bound(2,5) ) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3), &
                  tmp_trgt_bound(1,4):tmp_trgt_bound(2,4), &
                  tmp_trgt_bound(1,5):tmp_trgt_bound(2,5) )
          end select

          if(j.eq.2) exit
          bound_store(:) = tmp_dest_bound(:,idim)
          select case(padding_method)
          case ("circular")
             tmp_dest_bound(:,idim) = tmp_trgt_bound(:,idim) + padding(i)
             tmp_trgt_bound(:,idim) = bound_store(:) + padding(i)
             !tmp_dest_bound(:,idim) = [ ubound(data,idim) + 1, ubound(data_copy,idim) ]
             !tmp_trgt_bound(1,idim) = [ lbound(data,idim), lbound(data,idim) + padding(i) - 1 ]
          case("reflection")
             tmp_dest_bound(:,idim) = tmp_trgt_bound(:,idim) + size(data,idim) - 1
             tmp_trgt_bound(:,idim) = bound_store(:) + size(data,idim) - 1
             !tmp_dest_bound(:,idim) = [ ubound(data,idim) + 1, ubound(data_copy,idim) ]
             !tmp_trgt_bound(1,idim) = [ ubound(data,idim) - padding(i), ubound(data,idim) - 1 ]
          case("replication")
             tmp_dest_bound(:,idim) = tmp_trgt_bound(:,idim) + size(data,idim)
             tmp_trgt_bound(:,idim) = bound_store(:) + size(data,idim)
             !tmp_dest_bound(:,idim) = [ ubound(data,idim) + 1, ubound(data_copy,idim) ]
             !tmp_trgt_bound(1,idim) = [ ubound(data,idim) - padding(i) + 1, ubound(data,idim) ]
          end select
       end do
    end do

  end subroutine pad_data
!!!########################################################################

end module misc_ml