!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor and Francis Huw Davies
!!! Code part of the ARTEMIS group (Hepplestone research group).
!!! Think Hepplestone, think HRG.
!!!#############################################################################
!!! module contains various miscellaneous functions and subroutines.
!!! module includes the following functions and subroutines:
!!! shuffle          (randomly shuffle a 2D array along one dimension)
!!!##################
!!! Icount           (counts words on line)
!!! grep             (finds 1st line containing the pattern)
!!! to_upper         (converts all characters in string to upper case)
!!! to_lower         (converts all characters in string to lower case)
!!!#############################################################################
module misc
  use constants, only: real12
  implicit none


  interface shuffle
     procedure shuffle_1Dlist, &
          shuffle_2Ddata, shuffle_3Didata, shuffle_3Drdata, &
          shuffle_4Ddata, shuffle_5Ddata, &
          shuffle_3Didata_1Dilist, shuffle_3Didata_1Drlist, &
          shuffle_4Ddata_1Dlist, shuffle_5Ddata_1Dilist, shuffle_5Ddata_1Drlist
  end interface shuffle

  interface split
     procedure &
          split_3Didata_1Dilist, split_3Didata_1Drlist, &
          split_5Drdata, &
          split_5Drdata_1Drlist
  end interface split

!!!updated 2023/11/08

contains
!!!#####################################################
!!! shuffle an array along one dimension
!!!#####################################################
  subroutine shuffle_1Dlist(list,seed)
    implicit none
    integer :: iseed, istart, num_data
    integer :: itmp1, i, j
    real(real12) :: r
    integer, optional, intent(in) :: seed
    integer, dimension(:), intent(inout) :: list
    
    if(present(seed)) iseed = seed

    num_data = size(list,dim=1)
    call random_seed(iseed)
    istart=1
    do i=1,num_data
       call random_number(r)
       j = istart + floor((num_data+1-istart)*r)
       if(i.eq.j) cycle
       itmp1   = list(j)
       list(j) = list(i)
       list(i) = itmp1
    end do

  end subroutine shuffle_1Dlist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine shuffle_2Ddata(arr,dim,seed)
    implicit none
    integer :: iseed,istart
    integer :: i,j,n_data,iother
    integer :: i1s,i2s,i1e,i2e,j1s,j2s,j1e,j2e
    real(real12) :: r
    real(real12), allocatable, dimension(:,:) :: tlist

    integer, intent(in) :: dim
    real(real12), dimension(:,:), intent(inout) :: arr

    integer, optional, intent(in) :: seed

    if(present(seed)) iseed = seed

    call random_seed(iseed)
    n_data = size(arr,dim=dim)
    if(dim.eq.1)then
       iother = 2
       i2s=1;i2e=size(arr,dim=iother)
       j2s=1;j2e=size(arr,dim=iother)
    else
       iother = 1
       i1s=1;i1e=size(arr,dim=iother)
       j1s=1;j1e=size(arr,dim=iother)
    end if
    istart=1
    allocate(tlist(1,size(arr,dim=iother)))
    !! why cycling over k?
    !! comment out
    !do k=1,2
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
       tlist(1:1,:) = arr(i1s:i1e,i2s:i2e)
       arr(i1s:i1e,i2s:i2e) = arr(j1s:j1e,j2s:j2e)
       arr(j1s:j1e,j2s:j2e) = tlist(1:1,:)
    end do
    !end do

  end subroutine shuffle_2Ddata
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine shuffle_3Drdata(arr,dim,seed)
    implicit none
    integer :: iseed,istart
    integer :: i,j,n_data
    real(real12) :: r
    integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
    integer, dimension(3,2) :: t_size
    real(real12), allocatable, dimension(:,:,:) :: tlist

    integer, intent(in) :: dim
    real(real12), dimension(:,:,:), intent(inout) :: arr

    integer, optional, intent(in) :: seed

    if(present(seed)) iseed = seed

    call random_seed(iseed)
    n_data = size(arr,dim=dim)
    do i=1,3
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(arr,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(arr,dim=i)
       if(i.eq.dim) then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(arr,dim=i)
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
            t_size(3,1):t_size(3,2)) = arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3))
       arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3)) = arr(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2),&
            jdx_s(3):jdx_e(3))
       arr(&
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
    subroutine shuffle_3Didata(arr,dim,seed)
      implicit none
      integer :: iseed,istart
      integer :: i,j,n_data
      real(real12) :: r
      integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
      integer, dimension(3,2) :: t_size
      integer, allocatable, dimension(:,:,:) :: tlist
  
      integer, intent(in) :: dim
      integer, dimension(:,:,:), intent(inout) :: arr
  
      integer, optional, intent(in) :: seed
  
      if(present(seed)) iseed = seed
  
      call random_seed(iseed)
      n_data = size(arr,dim=dim)
      do i=1,3
         t_size(i,1) = 1
         jdx_s(i) = 1
         jdx_e(i) = size(arr,dim=i)
         idx_s(i) = 1
         idx_e(i) = size(arr,dim=i)
         if(i.eq.dim) then
            t_size(i,2) = 1
         else
            t_size(i,2) = size(arr,dim=i)
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
              t_size(3,1):t_size(3,2)) = arr(&
              idx_s(1):idx_e(1),&
              idx_s(2):idx_e(2),&
              idx_s(3):idx_e(3))
         arr(&
              idx_s(1):idx_e(1),&
              idx_s(2):idx_e(2),&
              idx_s(3):idx_e(3)) = arr(&
              jdx_s(1):jdx_e(1),&
              jdx_s(2):jdx_e(2),&
              jdx_s(3):jdx_e(3))
         arr(&
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
  subroutine shuffle_4Ddata(arr,dim,seed)
    implicit none
    integer :: iseed,istart
    integer :: i,j,n_data
    real(real12) :: r
    integer, dimension(4) :: idx_s,idx_e,jdx_s,jdx_e
    integer, dimension(4,2) :: t_size
    real(real12), allocatable, dimension(:,:,:,:) :: tlist

    integer, intent(in) :: dim
    real(real12), dimension(:,:,:,:), intent(inout) :: arr

    integer, optional, intent(in) :: seed

    if(present(seed)) iseed = seed

    call random_seed(iseed)
    n_data = size(arr,dim=dim)
    do i=1,4
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(arr,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(arr,dim=i)
       if(i.eq.dim) then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(arr,dim=i)      
       end if
    end do

    allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2),t_size(4,2)))

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
            t_size(4,1):t_size(4,2)) = arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4))
       arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4)) = arr(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2),&
            jdx_s(3):jdx_e(3),&
            jdx_s(4):jdx_e(4))
       arr(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2),&
            jdx_s(3):jdx_e(3),&
            jdx_s(4):jdx_e(4)) = tlist(&
            t_size(1,1):t_size(1,2),&
            t_size(2,1):t_size(2,2),&
            t_size(3,1):t_size(3,2),&
            t_size(4,1):t_size(4,2))
    end do

  end subroutine shuffle_4Ddata
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine shuffle_5Ddata(arr,dim,seed)
    implicit none
    integer :: iseed,istart
    integer :: i,j,n_data
    real(real12) :: r
    integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
    integer, dimension(5,2) :: t_size
    real(real12), allocatable, dimension(:,:,:,:,:) :: tlist

    integer, intent(in) :: dim
    real(real12), dimension(:,:,:,:,:), intent(inout) :: arr

    integer, optional, intent(in) :: seed

    if(present(seed)) iseed = seed

    call random_seed(iseed)
    n_data = size(arr,dim=dim)
    do i=1,5
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(arr,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(arr,dim=i)
       if(i.eq.dim) then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(arr,dim=i)
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
       idx_s(dim) = i
       idx_e(dim) = i
       jdx_s(dim) = j
       jdx_e(dim) = j
       tlist(&
            t_size(1,1):t_size(1,2),&
            t_size(2,1):t_size(2,2),&
            t_size(3,1):t_size(3,2),&
            t_size(4,1):t_size(4,2),&
            t_size(5,1):t_size(5,2)) = arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4),&
            idx_s(5):idx_e(5))
       arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4),&
            idx_s(5):idx_e(5)) = arr(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2),&
            jdx_s(3):jdx_e(3),&
            jdx_s(4):jdx_e(4),&
            jdx_s(5):jdx_e(5))
       arr(&
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

  end subroutine shuffle_5Ddata
  !!!-----------------------------------------------------
  !!!-----------------------------------------------------
    subroutine shuffle_3Didata_1Dilist(arr,label,dim,seed)
      implicit none
      integer :: iseed,istart
      integer :: i,j,n_data
      integer :: itmp1
      real(real12) :: r
      integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
      integer, dimension(3,2) :: t_size
      integer, allocatable, dimension(:,:,:) :: tlist
  
      integer, intent(in) :: dim
      integer, dimension(:,:,:), intent(inout) :: arr
      integer, dimension(:), intent(inout) :: label
  
      integer, optional, intent(in) :: seed
  
      if(present(seed)) iseed = seed
  
      call random_seed(iseed)
      n_data = size(arr,dim=dim)
      do i=1,3
         t_size(i,1) = 1
         jdx_s(i) = 1
         jdx_e(i) = size(arr,dim=i)
         idx_s(i) = 1
         idx_e(i) = size(arr,dim=i)
         if(i.eq.dim) then
            t_size(i,2) = 1
         else
            t_size(i,2) = size(arr,dim=i)
         end if
      end do
  
      allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))
  
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
              t_size(3,1):t_size(3,2)) = arr(&
              idx_s(1):idx_e(1),&
              idx_s(2):idx_e(2),&
              idx_s(3):idx_e(3))
         arr(&
              idx_s(1):idx_e(1),&
              idx_s(2):idx_e(2),&
              idx_s(3):idx_e(3)) = arr(&
              jdx_s(1):jdx_e(1),&
              jdx_s(2):jdx_e(2),&
              jdx_s(3):jdx_e(3))
         arr(&
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
      subroutine shuffle_3Didata_1Drlist(arr,label,dim,seed)
        implicit none
        integer :: iseed,istart
        integer :: i,j,n_data
        integer :: itmp1
        real(real12) :: r
        integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
        integer, dimension(3,2) :: t_size
        integer, allocatable, dimension(:,:,:) :: tlist
    
        integer, intent(in) :: dim
        integer, dimension(:,:,:), intent(inout) :: arr
        real(real12), dimension(:), intent(inout) :: label
    
        integer, optional, intent(in) :: seed
    
        if(present(seed)) iseed = seed
    
        call random_seed(iseed)
        n_data = size(arr,dim=dim)
        do i=1,3
           t_size(i,1) = 1
           jdx_s(i) = 1
           jdx_e(i) = size(arr,dim=i)
           idx_s(i) = 1
           idx_e(i) = size(arr,dim=i)
           if(i.eq.dim) then
              t_size(i,2) = 1
           else
              t_size(i,2) = size(arr,dim=i)
           end if
        end do
    
        allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))
    
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
                t_size(3,1):t_size(3,2)) = arr(&
                idx_s(1):idx_e(1),&
                idx_s(2):idx_e(2),&
                idx_s(3):idx_e(3))
           arr(&
                idx_s(1):idx_e(1),&
                idx_s(2):idx_e(2),&
                idx_s(3):idx_e(3)) = arr(&
                jdx_s(1):jdx_e(1),&
                jdx_s(2):jdx_e(2),&
                jdx_s(3):jdx_e(3))
           arr(&
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
    
      end subroutine shuffle_3Didata_1Drlist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine shuffle_4Ddata_1Dlist(arr,label,dim,seed)
    implicit none
    integer :: iseed,istart
    integer :: i,j,n_data
    integer :: itmp1
    real(real12) :: r
    integer, dimension(4) :: idx_s,idx_e,jdx_s,jdx_e
    integer, dimension(4,2) :: t_size
    real(real12), allocatable, dimension(:,:,:,:) :: tlist

    integer, intent(in) :: dim
    real(real12), dimension(:,:,:,:), intent(inout) :: arr
    integer, dimension(:), intent(inout) :: label

    integer, optional, intent(in) :: seed

    if(present(seed)) iseed = seed

    call random_seed(iseed)
    n_data = size(arr,dim=dim)
    do i=1,4
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(arr,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(arr,dim=i)
       if(i.eq.dim) then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(arr,dim=i)
       end if
    end do

    allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2),t_size(4,2)))

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
            t_size(4,1):t_size(4,2)) = arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4))
       arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4)) = arr(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2),&
            jdx_s(3):jdx_e(3),&
            jdx_s(4):jdx_e(4))
       arr(&
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

  end subroutine shuffle_4Ddata_1Dlist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine shuffle_5Ddata_1Dilist(arr,label,dim,seed)
    implicit none
    integer :: iseed,istart
    integer :: i,j,n_data
    integer :: itmp1
    real(real12) :: r
    integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
    integer, dimension(5,2) :: t_size
    real(real12), allocatable, dimension(:,:,:,:,:) :: tlist

    integer, intent(in) :: dim
    real(real12), dimension(:,:,:,:,:), intent(inout) :: arr
    integer, dimension(:), intent(inout) :: label

    integer, optional, intent(in) :: seed

    if(present(seed)) iseed = seed

    call random_seed(iseed)
    n_data = size(arr,dim=dim)
    do i=1,5
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(arr,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(arr,dim=i)
       if(i.eq.dim) then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(arr,dim=i)
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
       idx_s(dim) = i
       idx_e(dim) = i
       jdx_s(dim) = j
       jdx_e(dim) = j
       tlist(&
            t_size(1,1):t_size(1,2),&
            t_size(2,1):t_size(2,2),&
            t_size(3,1):t_size(3,2),&
            t_size(4,1):t_size(4,2),&
            t_size(5,1):t_size(5,2)) = arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4),&
            idx_s(5):idx_e(5))
       arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4),&
            idx_s(5):idx_e(5)) = arr(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2),&
            jdx_s(3):jdx_e(3),&
            jdx_s(4):jdx_e(4),&
            jdx_s(5):jdx_e(5))
       arr(&
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

  end subroutine shuffle_5Ddata_1Dilist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine shuffle_5Ddata_1Drlist(arr,label,dim,seed,shuffle_list)
    implicit none
    integer :: iseed,istart
    integer :: i,j,n_data
    real(real12) :: rtmp1
    real(real12) :: r
    integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
    integer, dimension(5,2) :: t_size
    real(real12), allocatable, dimension(:,:,:,:,:) :: tlist

    integer, intent(in) :: dim
    real(real12), dimension(:,:,:,:,:), intent(inout) :: arr
    real(real12), dimension(:), intent(inout) :: label

    integer, optional, intent(in) :: seed
    integer, optional, dimension(size(arr,dim)), intent(out) :: shuffle_list

    if(present(seed)) iseed = seed

    call random_seed(iseed)
    n_data = size(arr,dim=dim)
    do i=1,5
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(arr,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(arr,dim=i)
       if(i.eq.dim) then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(arr,dim=i)
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
            t_size(5,1):t_size(5,2)) = arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4),&
            idx_s(5):idx_e(5))
       arr(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2),&
            idx_s(3):idx_e(3),&
            idx_s(4):idx_e(4),&
            idx_s(5):idx_e(5)) = arr(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2),&
            jdx_s(3):jdx_e(3),&
            jdx_s(4):jdx_e(4),&
            jdx_s(5):jdx_e(5))
       arr(&
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

  end subroutine shuffle_5Ddata_1Drlist
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

    integer :: t_seed, t_left_num, t_right_num
    logical :: t_shuffle
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
       t_left_num  = nint(left_size*size(data,dim))
       t_right_num = size(data,dim) - t_left_num
    elseif(present(left_size).and..not.present(right_size))then
       t_right_num = nint(right_size*size(data,dim))
       t_left_num  = size(data,dim) - t_right_num
    else
       t_left_num  = nint(left_size*size(data,dim))
       t_right_num = nint(right_size*size(data,dim))
       if(t_left_num + t_right_num .ne. size(data,dim)) &
            t_right_num = size(data,dim) - t_left_num
    end if


    !! initialies optional arguments
    if(present(shuffle))then
       t_shuffle = shuffle
    else
       t_shuffle = .false.
    end if

    if(present(seed))then
       t_seed = seed
    else
       call system_clock(count=t_seed)
    end if

    !! copy input data
    data_copy = data
    if(t_shuffle) call shuffle_5Ddata(data_copy,dim,t_seed)
    
    !! get list of indices for right split
    num_redos = 0
    allocate(tlist(t_right_num))
    call random_number(tlist)
    indices_r = floor(tlist*size(data,dim)) + 1
    i = 1
    indices_r_loop: do 
       if(i.ge.t_right_num) exit indices_r_loop
       i = i + 1
       if(any(indices_r(:i-1).eq.indices_r(i)))then
          indices_r(i:t_right_num-num_redos-1) = indices_r(i+1:t_right_num-num_redos)
          call random_number(rtmp1)
          indices_r(t_right_num) = floor(rtmp1*size(data,dim)) + 1
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
    subroutine split_3Didata_1Dilist(data,list,left_data,right_data,&
         left_list,right_list,dim,&
         left_size,right_size,&
         shuffle,seed,split_list)
      implicit none
      integer, dimension(:,:,:), intent(in) :: data
      integer, dimension(:), intent(in) :: list
      integer, allocatable, dimension(:,:,:), intent(out) :: left_data, right_data
      integer, allocatable, dimension(:), intent(out) :: left_list, right_list
      integer, intent(in) :: dim
      real(real12), optional, intent(in) :: left_size, right_size
      logical, optional, intent(in) :: shuffle
      integer, optional, intent(in) :: seed
      integer, optional, dimension(size(data,dim)), intent(out) :: split_list
  
      integer :: t_seed, t_left_num, t_right_num
      logical :: t_shuffle
      integer :: i, j
      integer :: num_redos
      real(real12) :: rtmp1
      integer, allocatable, dimension(:) :: indices_l, indices_r
      real(real12), allocatable, dimension(:) :: tlist
      integer, allocatable, dimension(:) :: list_copy
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
         t_left_num  = nint(left_size*size(data,dim))
         t_right_num = size(data,dim) - t_left_num
      elseif(present(left_size).and..not.present(right_size))then
         t_right_num = nint(right_size*size(data,dim))
         t_left_num  = size(data,dim) - t_right_num
      else
         t_left_num  = nint(left_size*size(data,dim))
         t_right_num = nint(right_size*size(data,dim))
         if(t_left_num + t_right_num .ne. size(data,dim)) &
              t_right_num = size(data,dim) - t_left_num
      end if
  
      !! initialies optional arguments
      if(present(shuffle))then
         t_shuffle = shuffle
      else
         t_shuffle = .false.
      end if
  
      if(present(seed))then
         t_seed = seed
      else
         call system_clock(count=t_seed)
      end if
  
      !! copy input data
      data_copy = data
      list_copy = list
      if(t_shuffle) call shuffle_3Didata_1Dilist(data_copy,list_copy,dim,t_seed)
  
      !! get list of indices for right split
      num_redos = 0
      allocate(tlist(t_right_num))
      call random_number(tlist)
      indices_r = floor(tlist*size(data,dim)) + 1
      i = 1
      indices_r_loop: do 
         if(i.ge.t_right_num) exit indices_r_loop
         i = i + 1
         if(any(indices_r(:i-1).eq.indices_r(i)))then
            indices_r(i:t_right_num-num_redos-1) = &
                 indices_r(i+1:t_right_num-num_redos)
            call random_number(rtmp1)
            indices_r(t_right_num) = floor(rtmp1*size(data,dim)) + 1
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
      right_list = list_copy(indices_r)
  
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
      left_list = list_copy(indices_l)
  
    end subroutine split_3Didata_1Dilist
  !!!-----------------------------------------------------
  !!!-----------------------------------------------------
    subroutine split_3Didata_1Drlist(data,list,left_data,right_data,&
         left_list,right_list,dim,&
         left_size,right_size,&
         shuffle,seed,split_list)
      implicit none
      integer, dimension(:,:,:), intent(in) :: data
      real(real12), dimension(:), intent(in) :: list
      integer, allocatable, dimension(:,:,:), intent(out) :: left_data, right_data
      real(real12), allocatable, dimension(:), intent(out) :: left_list, right_list
      integer, intent(in) :: dim
      real(real12), optional, intent(in) :: left_size, right_size
      logical, optional, intent(in) :: shuffle
      integer, optional, intent(in) :: seed
      integer, optional, dimension(size(data,dim)), intent(out) :: split_list
  
      integer :: t_seed, t_left_num, t_right_num
      logical :: t_shuffle
      integer :: i, j
      integer :: num_redos
      real(real12) :: rtmp1
      integer, allocatable, dimension(:) :: indices_l, indices_r
      real(real12), allocatable, dimension(:) :: tlist
      real(real12), allocatable, dimension(:) :: list_copy
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
         t_left_num  = nint(left_size*size(data,dim))
         t_right_num = size(data,dim) - t_left_num
      elseif(present(left_size).and..not.present(right_size))then
         t_right_num = nint(right_size*size(data,dim))
         t_left_num  = size(data,dim) - t_right_num
      else
         t_left_num  = nint(left_size*size(data,dim))
         t_right_num = nint(right_size*size(data,dim))
         if(t_left_num + t_right_num .ne. size(data,dim)) &
              t_right_num = size(data,dim) - t_left_num
      end if
  
      !! initialies optional arguments
      if(present(shuffle))then
         t_shuffle = shuffle
      else
         t_shuffle = .false.
      end if
  
      if(present(seed))then
         t_seed = seed
      else
         call system_clock(count=t_seed)
      end if
  
      !! copy input data
      data_copy = data
      list_copy = list
      if(t_shuffle) call shuffle_3Didata_1Drlist(data_copy,list_copy,dim,t_seed)
  
      !! get list of indices for right split
      num_redos = 0
      allocate(tlist(t_right_num))
      call random_number(tlist)
      indices_r = floor(tlist*size(data,dim)) + 1
      i = 1
      indices_r_loop: do 
         if(i.ge.t_right_num) exit indices_r_loop
         i = i + 1
         if(any(indices_r(:i-1).eq.indices_r(i)))then
            indices_r(i:t_right_num-num_redos-1) = &
                 indices_r(i+1:t_right_num-num_redos)
            call random_number(rtmp1)
            indices_r(t_right_num) = floor(rtmp1*size(data,dim)) + 1
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
      right_list = list_copy(indices_r)
  
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
      left_list = list_copy(indices_l)
  
    end subroutine split_3Didata_1Drlist
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine split_5Drdata_1Drlist(data,list,left_data,right_data,&
       left_list,right_list,dim,&
       left_size,right_size,&
       shuffle,seed,split_list)
    implicit none
    real(real12), dimension(:,:,:,:,:), intent(in) :: data
    real(real12), dimension(:), intent(in) :: list
    real(real12), allocatable, dimension(:,:,:,:,:), intent(out) :: left_data, right_data
    real(real12), allocatable, dimension(:), intent(out) :: left_list, right_list
    integer, intent(in) :: dim
    real(real12), optional, intent(in) :: left_size, right_size
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: seed
    integer, optional, dimension(size(data,dim)), intent(out) :: split_list

    integer :: t_seed, t_left_num, t_right_num
    logical :: t_shuffle
    integer :: i, j
    integer :: num_redos
    real(real12) :: rtmp1
    integer, allocatable, dimension(:) :: indices_l, indices_r
    real(real12), allocatable, dimension(:) :: tlist
    real(real12), allocatable, dimension(:) :: list_copy
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
       t_left_num  = nint(left_size*size(data,dim))
       t_right_num = size(data,dim) - t_left_num
    elseif(present(left_size).and..not.present(right_size))then
       t_right_num = nint(right_size*size(data,dim))
       t_left_num  = size(data,dim) - t_right_num
    else
       t_left_num  = nint(left_size*size(data,dim))
       t_right_num = nint(right_size*size(data,dim))
       if(t_left_num + t_right_num .ne. size(data,dim)) &
            t_right_num = size(data,dim) - t_left_num
    end if

    !! initialies optional arguments
    if(present(shuffle))then
       t_shuffle = shuffle
    else
       t_shuffle = .false.
    end if

    if(present(seed))then
       t_seed = seed
    else
       call system_clock(count=t_seed)
    end if

    !! copy input data
    data_copy = data
    list_copy = list
    if(t_shuffle) call shuffle_5Ddata_1Drlist(data_copy,list_copy,dim,t_seed)

    !! get list of indices for right split
    num_redos = 0
    allocate(tlist(t_right_num))
    call random_number(tlist)
    indices_r = floor(tlist*size(data,dim)) + 1
    i = 1
    indices_r_loop: do 
       if(i.ge.t_right_num) exit indices_r_loop
       i = i + 1
       if(any(indices_r(:i-1).eq.indices_r(i)))then
          indices_r(i:t_right_num-num_redos-1) = &
               indices_r(i+1:t_right_num-num_redos)
          call random_number(rtmp1)
          indices_r(t_right_num) = floor(rtmp1*size(data,dim)) + 1
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
    right_list = list_copy(indices_r)

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
    left_list = list_copy(indices_l)

  end subroutine split_5Drdata_1Drlist
!!!####################e#################################


!!!#############################################################################
!!!#############################################################################
!!!  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
!!!#############################################################################
!!!#############################################################################


!!!#####################################################
!!! counts the number of words on a line
!!!#####################################################
  integer function Icount(full_line,tmpchar)
    implicit none
    character(*), intent(in) :: full_line
    character(*), optional, intent(in) :: tmpchar

    character(len=:), allocatable :: fs
    integer ::items,pos,k,length
    items=0
    pos=1

    length=1
    if(present(tmpchar)) length=len(trim(tmpchar))
    allocate(character(len=length) :: fs)
    if(present(tmpchar)) then
       fs=trim(tmpchar)
    else
       fs=" "
    end if

    loop: do
       k=verify(full_line(pos:),fs)
       if (k.eq.0) exit loop
       items=items+1
       pos=k+pos-1
       k=scan(full_line(pos:),fs)
       if (k.eq.0) exit loop
       pos=k+pos-1
    end do loop
    Icount=items
  end function Icount
!!!#####################################################


!!!#####################################################
!!! grep 
!!!#####################################################
!!! searches a file untill it finds the mattching patern
  subroutine grep(unit,input,lstart)
    implicit none
    integer, intent(in) :: unit
    character(*), intent(in) :: input
    logical, optional, intent(in) :: lstart

    integer :: Reason
    character(1024) :: buffer
    !  character(1024), intent(out), optional :: linechar
    if(present(lstart))then
       if(lstart) rewind(unit)
    else
       rewind(unit)
    end if

    greploop: do
       read(unit,'(A100)',iostat=Reason) buffer
       if(Reason.lt.0) return
       if(index(trim(buffer),trim(input)).ne.0) exit greploop
    end do greploop
  end subroutine grep
!!!#####################################################


!!!#####################################################
!!! converts all characters in string to upper case
!!!#####################################################
  pure function to_upper(buffer) result(upper)
    implicit none
    character(*), intent(in) :: buffer
    character(len=:),allocatable :: upper

    integer :: i,j


    allocate(character(len=len(buffer)) :: upper)
    do i=1,len(buffer)
       j=iachar(buffer(i:i))
       if(j.ge.iachar("a").and.j.le.iachar("z"))then
          upper(i:i)=achar(j-32)
       else
          upper(i:i)=buffer(i:i)
       end if
    end do

    return
  end function to_upper
!!!#####################################################


!!!#####################################################
!!! converts all characters in string to lower case
!!!#####################################################
  pure function to_lower(buffer) result(lower)
    implicit none
    character(*), intent(in) :: buffer
    character(len=:),allocatable :: lower

    integer :: i,j


    allocate(character(len=len(buffer)) :: lower)
    do i=1,len(buffer)
       j=iachar(buffer(i:i))
       if(j.ge.iachar("A").and.j.le.iachar("Z"))then
          lower(i:i)=achar(j+32)
       else
          lower(i:i)=buffer(i:i)
       end if
    end do

    return
  end function to_lower
!!!#####################################################

end module misc
