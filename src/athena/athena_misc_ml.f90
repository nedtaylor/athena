module athena__misc_ml
  !! Module containing miscellaneous machine learning procedures
  !!
  !! This module contains various procedures that are useful for machine
  !! learning tasks. These include shuffling data, splitting data into
  !! train and test sets, and padding data.
  use coreutils, only: real32, stop_program
  implicit none


  private

  public :: shuffle, split
  public :: set_padding, pad_data


  interface shuffle
     !! Shuffle an array along one dimension
     !!
     !! This procedure shuffles an array along one dimension. The array
     !! can be of any rank, but the dimension along which to shuffle must
     !! be specified. An optional index array can also be shuffled.
     procedure shuffle_1Dilist, &
          shuffle_2Drdata, shuffle_3Didata, shuffle_3Drdata, &
          shuffle_4Drdata, shuffle_5Drdata, &
          shuffle_2Drdata_1Drlist, &
          shuffle_3Didata_1Dilist, shuffle_3Didata_1Drlist, &
          shuffle_4Drdata_1Dilist, shuffle_5Drdata_1Dilist, &
          shuffle_5Drdata_1Drlist
  end interface shuffle

  interface split
     !! Split an array into train and test sets
     !!
     !! This procedure splits an array into two sets along one dimension.
     !! The array can be of any rank, but the dimension along which to
     !! split must be specified. An optional index array can also be split.
     !! The size of the left and right splits can also be specified. The
     !! data can be shuffled before splitting.
     procedure split_2Drdata_1Drlist, &
          split_3Didata_1Dilist, split_3Didata_1Drlist, &
          split_5Drdata, &
          split_5Drdata_1Drlist
  end interface split



contains
!###############################################################################
  subroutine shuffle_1Dilist(data,seed)
    !! Shuffle a 1D array along one dimension
    implicit none

    ! Arguments
    integer, dimension(:), intent(inout) :: data
    !! 1D array to be shuffled
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: itmp1, i, j
    !! Loop indices
    integer :: istart, num_data, seed_size
    !! Start index, number of data points, seed size
    real(real32) :: r
    !! Random number
    integer, allocatable, dimension(:) :: iseed
    !! Random seed


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Shuffle the data
    !---------------------------------------------------------------------------
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
!-------------------------------------------------------------------------------
  subroutine shuffle_2Drdata(data,dim,seed)
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(inout) :: data
    !! 2D array to be shuffled
    integer, optional, intent(in) :: seed
    !! Random seed
    integer, intent(in) :: dim
    !! Dimension along which to shuffle

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data,iother
    !! Loop indices, number of data points, other dimension
    integer :: i1s,i2s,i1e,i2e,j1s,j2s,j1e,j2e
    !! Start and end indices
    real(real32) :: r
    !! Random number
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    real(real32), allocatable, dimension(:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Shuffle the data
    !---------------------------------------------------------------------------
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
!-------------------------------------------------------------------------------
  subroutine shuffle_3Drdata(data,dim,seed)
    !! Shuffle a 3D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:,:), intent(inout) :: data
    !! 3D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    real(real32) :: r
    !! Random number
    integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(3,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    real(real32), allocatable, dimension(:,:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
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
       if(i.eq.dim)then
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
!-------------------------------------------------------------------------------
  subroutine shuffle_3Didata(data,dim,seed)
    !! Shuffle a 3D array along one dimension
    implicit none

    ! Arguments
    integer, dimension(:,:,:), intent(inout) :: data
    !! 3D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    real(real32) :: r
    !! Random number
    integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(3,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    integer, allocatable, dimension(:,:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,3
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(data,dim=i)
       end if
    end do
    allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))


    ! Shuffle the data
    !---------------------------------------------------------------------------
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
!-------------------------------------------------------------------------------
  subroutine shuffle_4Drdata(data,dim,seed)
    !! Shuffle a 4D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:,:,:), intent(inout) :: data
    !! 4D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    real(real32) :: r
    !! Random number
    integer, dimension(4) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(4,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    real(real32), allocatable, dimension(:,:,:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,4
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(data,dim=i)
       end if
    end do
    allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2),t_size(4,2)))


    ! Shuffle the data
    !---------------------------------------------------------------------------
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
!-------------------------------------------------------------------------------
  subroutine shuffle_5Drdata(data,dim,seed)
    !! Shuffle a 5D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:,:,:,:), intent(inout) :: data
    !! 5D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    real(real32) :: r
    !! Random number
    integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(5,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    real(real32), allocatable, dimension(:,:,:,:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,5
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(data,dim=i)
       end if
    end do
    allocate(tlist(&
         t_size(1,2),t_size(2,2),&
         t_size(3,2),t_size(4,2),&
         t_size(5,2)))


    ! Shuffle the data
    !---------------------------------------------------------------------------
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
!-------------------------------------------------------------------------------
  subroutine shuffle_2Drdata_1Drlist(data,label,dim,seed,shuffle_list)
    !! Shuffle a 2D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(inout) :: data
    !! 2D array to be shuffled
    real(real32), dimension(:), intent(inout) :: label
    !! 1D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed
    integer, optional, dimension(size(data,dim)), intent(out) :: shuffle_list
    !! Index array

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    real(real32) :: rtmp1
    !! Temporary real
    real(real32) :: r
    !! Random number
    integer, dimension(2) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(2,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    real(real32), allocatable, dimension(:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,2
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(data,dim=i)
       end if
    end do

    allocate(tlist(t_size(1,2),t_size(2,2)))

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
            t_size(2,1):t_size(2,2)) = data(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2))
       data(&
            idx_s(1):idx_e(1),&
            idx_s(2):idx_e(2)) = data(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2))
       data(&
            jdx_s(1):jdx_e(1),&
            jdx_s(2):jdx_e(2)) = tlist(&
            t_size(1,1):t_size(1,2),&
            t_size(2,1):t_size(2,2))

       rtmp1 = label(i)
       label(i) = label(j)
       label(j) = rtmp1
    end do

  end subroutine shuffle_2Drdata_1Drlist
!-------------------------------------------------------------------------------
  subroutine shuffle_3Didata_1Dilist(data,label,dim,seed)
    !! Shuffle a 3D array along one dimension
    implicit none

    ! Arguments
    integer, dimension(:,:,:), intent(inout) :: data
    !! 3D array to be shuffled
    integer, dimension(:), intent(inout) :: label
    !! 1D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    integer :: itmp1
    !! Temporary integer
    real(real32) :: r
    !! Random number
    integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(3,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    integer, allocatable, dimension(:,:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,3
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(data,dim=i)
       end if
    end do
    allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))


    ! Shuffle the data
    !---------------------------------------------------------------------------
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
!-------------------------------------------------------------------------------
  subroutine shuffle_3Didata_1Drlist(data,label,dim,seed)
    !! Shuffle a 3D array along one dimension
    implicit none

    ! Arguments
    integer, dimension(:,:,:), intent(inout) :: data
    !! 3D array to be shuffled
    real(real32), dimension(:), intent(inout) :: label
    !! 1D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    integer :: itmp1
    !! Temporary integer
    real(real32) :: r
    !! Random number
    integer, dimension(3) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(3,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    integer, allocatable, dimension(:,:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,3
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(data,dim=i)
       end if
    end do
    allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2)))


    ! Shuffle the data
    !---------------------------------------------------------------------------
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

  end subroutine shuffle_3Didata_1Drlist
!-------------------------------------------------------------------------------
  subroutine shuffle_4Drdata_1Dilist(data,label,dim,seed)
    !! Shuffle a 4D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:,:,:), intent(inout) :: data
    !! 4D array to be shuffled
    integer, dimension(:), intent(inout) :: label
    !! 1D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: istart, seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    integer :: itmp1
    !! Temporary integer
    real(real32) :: r
    !! Random number
    integer, dimension(4) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(4,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    real(real32), allocatable, dimension(:,:,:,:) :: tlist
    !! Temporary list



    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,4
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(data,dim=i)
       end if
    end do
    allocate(tlist(t_size(1,2),t_size(2,2),t_size(3,2),t_size(4,2)))


    ! Shuffle the data
    !---------------------------------------------------------------------------
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
!-------------------------------------------------------------------------------
  subroutine shuffle_5Drdata_1Dilist(data,label,dim,seed)
    !! Shuffle a 5D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:,:,:,:), intent(inout) :: data
    !! 5D array to be shuffled
    integer, dimension(:), intent(inout) :: label
    !! 1D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    integer :: itmp1
    !! Temporary integer
    real(real32) :: r
    !! Random number
    integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(5,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    real(real32), allocatable, dimension(:,:,:,:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,5
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
          t_size(i,2) = 1
       else
          t_size(i,2) = size(data,dim=i)
       end if
    end do
    allocate(tlist(&
         t_size(1,2),t_size(2,2),&
         t_size(3,2),t_size(4,2),&
         t_size(5,2)))


    ! Shuffle the data
    !---------------------------------------------------------------------------
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
!-------------------------------------------------------------------------------
  subroutine shuffle_5Drdata_1Drlist(data,label,dim,seed,shuffle_list)
    !! Shuffle a 5D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:,:,:,:), intent(inout) :: data
    !! 5D array to be shuffled
    real(real32), dimension(:), intent(inout) :: label
    !! 1D array to be shuffled
    integer, intent(in) :: dim
    !! Dimension along which to shuffle
    integer, optional, intent(in) :: seed
    !! Random seed
    integer, optional, dimension(size(data,dim)), intent(out) :: shuffle_list
    !! Index array

    ! Local variables
    integer :: istart,seed_size
    !! Start index, seed size
    integer :: i,j,n_data
    !! Loop indices, number of data points
    real(real32) :: rtmp1
    !! Temporary real
    real(real32) :: r
    !! Random number
    integer, dimension(5) :: idx_s,idx_e,jdx_s,jdx_e
    !! Start and end indices
    integer, dimension(5,2) :: t_size
    !! Temporary size
    integer, allocatable, dimension(:) :: iseed
    !! Random seed
    real(real32), allocatable, dimension(:,:,:,:,:) :: tlist
    !! Temporary list


    ! Set or get random seed
    !---------------------------------------------------------------------------
    call random_seed(size=seed_size)
    allocate(iseed(seed_size))
    if(present(seed))then
       iseed = seed
       call random_seed(put=iseed)
    else
       call random_seed(get=iseed)
    end if


    ! Get the size of the data
    !---------------------------------------------------------------------------
    n_data = size(data,dim=dim)
    do i=1,5
       t_size(i,1) = 1
       jdx_s(i) = 1
       jdx_e(i) = size(data,dim=i)
       idx_s(i) = 1
       idx_e(i) = size(data,dim=i)
       if(i.eq.dim)then
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
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine split_5Drdata( &
       data, left, right, dim, &
       left_size, right_size, &
       shuffle, seed &
  )
    !! Split a 5D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:,:,:,:), intent(in) :: data
    !! 5D array to be split
    real(real32), allocatable, dimension(:,:,:,:,:), intent(out) :: left, right
    !! 5D arrays to store the left and right splits
    integer, intent(in) :: dim
    !! Dimension along which to split
    real(real32), optional, intent(in) :: left_size, right_size
    !! Size of the left and right splits
    logical, optional, intent(in) :: shuffle
    !! Shuffle the data before splitting
    integer, optional, intent(in) :: seed
    !! Random seed

    ! Local variables
    integer :: seed_, left_num_, right_num_
    !! Random seed, number of elements in left and right splits
    logical :: shuffle_
    !! Shuffle flag
    integer :: i, j
    !! Loop indices
    integer :: num_redos
    !! Number of redos
    real(real32) :: rtmp1
    !! Temporary real
    integer, allocatable, dimension(:) :: indices_l, indices_r
    !! Index arrays
    real(real32), allocatable, dimension(:) :: tlist
    !! Temporary list
    real(real32), allocatable, dimension(:,:,:,:,:) :: data_copy
    !! Copy of the input data

    type :: idx_type
       !! Type for index array
       integer, allocatable, dimension(:) :: loc
       !! Index array
    end type idx_type
    type(idx_type), dimension(5) :: idx
    !! Index array


    ! Determine number of elements for left and right split
    !---------------------------------------------------------------------------
    if(.not.present(left_size).and..not.present(right_size))then
       call stop_program("neither left_size nor right_size provided to split. &
            &Expected at least one." &
       )
       return
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


    ! Initialies optional arguments
    !---------------------------------------------------------------------------
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


    ! Copy input data
    !---------------------------------------------------------------------------
    data_copy = data
    if(shuffle_) call shuffle_5Drdata(data_copy,dim,seed_)


    ! Get list of indices for right split
    !---------------------------------------------------------------------------
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


    ! Generate right split
    !---------------------------------------------------------------------------
    do i=1,5
       if(i.eq.dim)then
          idx(i)%loc = indices_r
       else
          idx(i)%loc = (/ ( j, j=1,size(data,i) ) /)
       end if
    end do
    right = data_copy(idx(1)%loc,idx(2)%loc,idx(3)%loc,idx(4)%loc,idx(5)%loc)


    ! Get list of indices for left split
    !---------------------------------------------------------------------------
    indices_l_loop: do i=1,size(data,dim)
       if(any(indices_r.eq.i)) cycle indices_l_loop
       if(allocated(indices_l))then
          indices_l = [indices_l(:), i]
       else
          indices_l = [i]
       end if
    end do indices_l_loop


    ! Generate left split
    !---------------------------------------------------------------------------
    idx(dim)%loc = indices_l
    left = data_copy(idx(1)%loc,idx(2)%loc,idx(3)%loc,idx(4)%loc,idx(5)%loc)

  end subroutine split_5Drdata
!-------------------------------------------------------------------------------
  subroutine split_2Drdata_1Drlist( &
       data, label, left_data, right_data, &
       left_label, right_label, dim, &
       left_size, right_size, &
       shuffle, seed, split_list &
  )
    !! Split a 2D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: data
    !! 2D array to be split
    real(real32), dimension(:), intent(in) :: label
    !! 1D array to be split
    real(real32), allocatable, dimension(:,:), intent(out) :: &
         left_data, right_data
    !! 2D arrays to store the left and right splits
    real(real32), allocatable, dimension(:), intent(out) :: &
         left_label, right_label
    !! 1D arrays to store the left and right splits
    integer, intent(in) :: dim
    !! Dimension along which to split
    real(real32), optional, intent(in) :: left_size, right_size
    !! Size of the left and right splits
    logical, optional, intent(in) :: shuffle
    !! Shuffle the data before splitting
    integer, optional, intent(in) :: seed
    !! Random seed
    integer, optional, dimension(size(data,dim)), intent(out) :: split_list
    !! Index array

    ! Local variables
    integer :: seed_, left_num_, right_num_
    !! Random seed, number of elements in left and right splits
    logical :: shuffle_
    !! Shuffle flag
    integer :: i, j
    !! Loop indices
    integer :: num_redos
    !! Number of redos
    real(real32) :: rtmp1
    !! Temporary real
    integer, allocatable, dimension(:) :: indices_l, indices_r
    !! Index arrays
    real(real32), allocatable, dimension(:) :: tlist
    !! Temporary list
    real(real32), allocatable, dimension(:) :: label_copy
    !! Copy of the input label
    real(real32), allocatable, dimension(:,:) :: data_copy
    !! Copy of the input data

    type :: idx_type
       !! Type for index array
       integer, allocatable, dimension(:) :: loc
       !! Index array
    end type idx_type
    type(idx_type), dimension(5) :: idx
    !! Index array


    ! Determine number of elements for left and right split
    !---------------------------------------------------------------------------
    if(.not.present(left_size).and..not.present(right_size))then
       call stop_program("neither left_size nor right_size provided to split. &
            &Expected at least one." &
       )
       return
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

    ! Initialies optional arguments
    !---------------------------------------------------------------------------
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


    ! Copy input data
    !---------------------------------------------------------------------------
    data_copy = data
    label_copy = label
    if(shuffle_) call shuffle_2Drdata_1Drlist(data_copy,label_copy,dim,seed_)


    ! Get list of indices for right split
    !---------------------------------------------------------------------------
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


    ! Generate right split
    !---------------------------------------------------------------------------
    do i=1,2
       if(i.eq.dim)then
          idx(i)%loc = indices_r
       else
          idx(i)%loc = (/ ( j, j=1,size(data,i) ) /)
       end if
    end do
    right_data  = data_copy(idx(1)%loc,idx(2)%loc)
    right_label = label_copy(indices_r)


    ! Get list of indices for left split
    !---------------------------------------------------------------------------
    if(present(split_list)) split_list = 2
    indices_l_loop: do i=1,size(data,dim)
       if(any(indices_r.eq.i)) cycle indices_l_loop
       if(allocated(indices_l))then
          indices_l = [indices_l(:), i]
       else
          indices_l = [i]
       end if
       if(present(split_list)) split_list(i) = 1
    end do indices_l_loop


    ! Generate left split
    !---------------------------------------------------------------------------
    idx(dim)%loc = indices_l
    left_data = data_copy(idx(1)%loc,idx(2)%loc)
    left_label = label_copy(indices_l)

  end subroutine split_2Drdata_1Drlist
!-------------------------------------------------------------------------------
  subroutine split_3Didata_1Dilist( &
       data, label, left_data, right_data, &
       left_label, right_label, dim, &
       left_size, right_size, &
       shuffle, seed, split_list &
  )
    implicit none

    ! Arguments
    integer, dimension(:,:,:), intent(in) :: data
    !! 3D array to be split
    integer, dimension(:), intent(in) :: label
    !! 1D array to be split
    integer, allocatable, dimension(:,:,:), intent(out) :: left_data, right_data
    !! 3D arrays to store the left and right splits
    integer, allocatable, dimension(:), intent(out) :: left_label, right_label
    !! 1D arrays to store the left and right splits
    integer, intent(in) :: dim
    !! Dimension along which to split
    real(real32), optional, intent(in) :: left_size, right_size
    !! Size of the left and right splits
    logical, optional, intent(in) :: shuffle
    !! Shuffle the data before splitting
    integer, optional, intent(in) :: seed
    !! Random seed
    integer, optional, dimension(size(data,dim)), intent(out) :: split_list
    !! Index array

    ! Local variables
    integer :: seed_, left_num_, right_num_
    !! Random seed, number of elements in left and right splits
    logical :: shuffle_
    !! Shuffle flag
    integer :: i, j
    !! Loop indices
    integer :: num_redos
    !! Number of redos
    real(real32) :: rtmp1
    !! Temporary real
    integer, allocatable, dimension(:) :: indices_l, indices_r
    !! Index arrays
    real(real32), allocatable, dimension(:) :: tlist
    !! Temporary list
    integer, allocatable, dimension(:) :: label_copy
    !! Copy of the input label
    integer, allocatable, dimension(:,:,:) :: data_copy
    !! Copy of the input data

    type :: idx_type
       !! Type for index array
       integer, allocatable, dimension(:) :: loc
       !! Index array
    end type idx_type
    type(idx_type), dimension(3) :: idx
    !! Index array


    ! Determine number of elements for left and right split
    !---------------------------------------------------------------------------
    if(.not.present(left_size).and..not.present(right_size))then
       call stop_program("neither left_size nor right_size provided to split. &
            &Expected at least one." &
       )
       return
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

    ! Initialies optional arguments
    !---------------------------------------------------------------------------
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


    ! Copy input data
    !---------------------------------------------------------------------------
    data_copy = data
    label_copy = label
    if(shuffle_) call shuffle_3Didata_1Dilist(data_copy,label_copy,dim,seed_)


    ! Get list of indices for right split
    !---------------------------------------------------------------------------
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


    ! Generate right split
    !---------------------------------------------------------------------------
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


    ! Get list of indices for left split
    !---------------------------------------------------------------------------
    if(present(split_list)) split_list = 2
    indices_l_loop: do i=1,size(data,dim)
       if(any(indices_r.eq.i)) cycle indices_l_loop
       if(allocated(indices_l))then
          indices_l = [indices_l(:), i]
       else
          indices_l = [i]
       end if
       if(present(split_list)) split_list(i) = 1
    end do indices_l_loop


    ! Generate left split
    !---------------------------------------------------------------------------
    idx(dim)%loc = indices_l
    left_data = data_copy(&
         idx(1)%loc,idx(2)%loc,idx(3)%loc)
    left_label = label_copy(indices_l)

  end subroutine split_3Didata_1Dilist
!-------------------------------------------------------------------------------
  subroutine split_3Didata_1Drlist( &
       data, label, left_data, right_data, &
       left_label, right_label, dim, &
       left_size, right_size, &
       shuffle, seed, split_list &
  )
    !! Split a 3D array along one dimension
    implicit none

    ! Arguments
    integer, dimension(:,:,:), intent(in) :: data
    !! 3D array to be split
    real(real32), dimension(:), intent(in) :: label
    !! 1D array to be split
    integer, allocatable, dimension(:,:,:), intent(out) :: left_data, right_data
    !! 3D arrays to store the left and right splits
    real(real32), allocatable, dimension(:), intent(out) :: &
         left_label, right_label
    !! 1D arrays to store the left and right splits
    integer, intent(in) :: dim
    !! Dimension along which to split
    real(real32), optional, intent(in) :: left_size, right_size
    !! Size of the left and right splits
    logical, optional, intent(in) :: shuffle
    !! Shuffle the data before splitting
    integer, optional, intent(in) :: seed
    !! Random seed
    integer, optional, dimension(size(data,dim)), intent(out) :: split_list
    !! Index array

    ! Local variables
    integer :: seed_, left_num_, right_num_
    !! Random seed, number of elements in left and right splits
    logical :: shuffle_
    !! Shuffle flag
    integer :: i, j
    !! Loop indices
    integer :: num_redos
    !! Number of redos
    real(real32) :: rtmp1
    !! Temporary real
    integer, allocatable, dimension(:) :: indices_l, indices_r
    !! Index arrays
    real(real32), allocatable, dimension(:) :: tlist
    !! Temporary list
    real(real32), allocatable, dimension(:) :: label_copy
    !! Copy of the input label
    integer, allocatable, dimension(:,:,:) :: data_copy
    !! Copy of the input data

    type :: idx_type
       !! Type for index array
       integer, allocatable, dimension(:) :: loc
       !! Index array
    end type idx_type
    type(idx_type), dimension(3) :: idx
    !! Index array


    ! Determine number of elements for left and right split
    !---------------------------------------------------------------------------
    if(.not.present(left_size).and..not.present(right_size))then
       call stop_program("neither left_size nor right_size provided to split. &
            &Expected at least one." &
       )
       return
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

    ! Initialies optional arguments
    !---------------------------------------------------------------------------
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


    ! Copy input data
    !---------------------------------------------------------------------------
    data_copy = data
    label_copy = label
    if(shuffle_) call shuffle_3Didata_1Drlist(data_copy,label_copy,dim,seed_)


    ! Get list of indices for right split
    !---------------------------------------------------------------------------
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


    ! Generate right split
    !---------------------------------------------------------------------------
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


    ! Get list of indices for left split
    !---------------------------------------------------------------------------
    if(present(split_list)) split_list = 2
    indices_l_loop: do i=1,size(data,dim)
       if(any(indices_r.eq.i)) cycle indices_l_loop
       if(allocated(indices_l))then
          indices_l = [indices_l(:), i]
       else
          indices_l = [i]
       end if
       if(present(split_list)) split_list(i) = 1
    end do indices_l_loop


    ! Generate left split
    !---------------------------------------------------------------------------
    idx(dim)%loc = indices_l
    left_data = data_copy(&
         idx(1)%loc,idx(2)%loc,idx(3)%loc)
    left_label = label_copy(indices_l)

  end subroutine split_3Didata_1Drlist
!-------------------------------------------------------------------------------
  subroutine split_5Drdata_1Drlist( &
       data, label, left_data, right_data, &
       left_label, right_label, dim, &
       left_size, right_size, &
       shuffle, seed, split_list &
  )
    !! Split a 5D array along one dimension
    implicit none

    ! Arguments
    real(real32), dimension(:,:,:,:,:), intent(in) :: data
    !! 5D array to be split
    real(real32), dimension(:), intent(in) :: label
    !! 1D array to be split
    real(real32), allocatable, dimension(:,:,:,:,:), intent(out) :: &
         left_data, right_data
    !! 5D arrays to store the left and right splits
    real(real32), allocatable, dimension(:), intent(out) :: &
         left_label, right_label
    !! 1D arrays to store the left and right splits
    integer, intent(in) :: dim
    !! Dimension along which to split
    real(real32), optional, intent(in) :: left_size, right_size
    !! Size of the left and right splits
    logical, optional, intent(in) :: shuffle
    !! Shuffle the data before splitting
    integer, optional, intent(in) :: seed
    !! Random seed
    integer, optional, dimension(size(data,dim)), intent(out) :: split_list
    !! Index array

    ! Local variables
    integer :: seed_, left_num_, right_num_
    !! Random seed, number of elements in left and right splits
    logical :: shuffle_
    !! Shuffle flag
    integer :: i, j
    !! Loop indices
    integer :: num_redos
    !! Number of redos
    real(real32) :: rtmp1
    !! Temporary real
    integer, allocatable, dimension(:) :: indices_l, indices_r
    !! Index arrays
    real(real32), allocatable, dimension(:) :: tlist
    !! Temporary list
    real(real32), allocatable, dimension(:) :: label_copy
    !! Copy of the input label
    real(real32), allocatable, dimension(:,:,:,:,:) :: data_copy
    !! Copy of the input data

    type :: idx_type
       !! Type for index array
       integer, allocatable, dimension(:) :: loc
       !! Index array
    end type idx_type
    type(idx_type), dimension(5) :: idx
    !! Index array


    ! Determine number of elements for left and right split
    !---------------------------------------------------------------------------
    if(.not.present(left_size).and..not.present(right_size))then
       call stop_program("neither left_size nor right_size provided to split. &
            &Expected at least one." &
       )
       return
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


    ! Initialies optional arguments
    !---------------------------------------------------------------------------
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


    ! Copy input data
    !---------------------------------------------------------------------------
    data_copy = data
    label_copy = label
    if(shuffle_) call shuffle_5Drdata_1Drlist(data_copy,label_copy,dim,seed_)


    ! Get list of indices for right split
    !---------------------------------------------------------------------------
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


    ! Generate right split
    !---------------------------------------------------------------------------
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


    ! Get list of indices for left split
    !---------------------------------------------------------------------------
    if(present(split_list)) split_list = 2
    indices_l_loop: do i=1,size(data,dim)
       if(any(indices_r.eq.i)) cycle indices_l_loop
       if(allocated(indices_l))then
          indices_l = [indices_l(:), i]
       else
          indices_l = [i]
       end if
       if(present(split_list)) split_list(i) = 1
    end do indices_l_loop


    ! Generate left split
    !---------------------------------------------------------------------------
    idx(dim)%loc = indices_l
    left_data = data_copy(&
         idx(1)%loc,idx(2)%loc,idx(3)%loc,idx(4)%loc,idx(5)%loc)
    left_label = label_copy(indices_l)

  end subroutine split_5Drdata_1Drlist
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!



!###############################################################################
  pure function get_padding_half(width) result(output)
    !! Function to return half the padding width
    implicit none

    ! Arguments
    integer, intent(in) :: width
    !! Width of kernel/filter
    integer :: output
    !! Half the padding width

    output = ( width - (1 - mod(width,2)) - 1 ) / 2
  end function get_padding_half
!###############################################################################


!###############################################################################
  subroutine set_padding(pad, kernel_size, padding_method, verbose)
    !! Set padding for convolutional layers
    use coreutils, only: to_lower
    implicit none

    ! Arguments
    integer, intent(out) :: pad
    !! Padding width
    integer, intent(in) :: kernel_size
    !! Width of kernel/filter
    character(*), intent(inout) :: padding_method
    !! Padding method
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t_verbose = 0
    !! Temporary verbosity level
    character(256) :: err_msg
    !! Error message


    !---------------------------------------------------------------------------
    ! initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) t_verbose = verbose


    !---------------------------------------------------------------------------
    ! Padding method options
    !---------------------------------------------------------------------------
    ! none  = alt. name for 'valid'
    ! zero  = alt. name for 'same'
    ! symmetric = alt.name for 'replication'
    ! valid = no padding
    ! same  = maintain spatial dimensions
    !         ... (i.e. padding added = (kernel_size - 1)/2)
    !         ... defaults to zeros in the padding
    ! full  = enough padding for filter to slide over every possible position
    !         ... (i.e. padding added = (kernel_size - 1)
    ! circular = maintain spatial dimensions
    !            ... wraps data around for padding (periodic)
    ! reflection = maintains spatial dimensions
    !              ... reflect data (about boundary index)
    ! replication = maintains spatial dimensions
    !               ... reflect data (boundary included)
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
       if(t_verbose.gt.0) write(*,*) &
            "Padding type: 'full' (all possible positions)"
       pad = kernel_size - 1
       return
    case("reflection")
       if(t_verbose.gt.0) write(*,*) &
            "Padding type: 'reflection' (reflect on boundary)"
    case("replication")
       if(t_verbose.gt.0) write(*,*) &
            "Padding type: 'replication' (reflect after boundary)"
    case default
       write(err_msg,'("padding type ''",A,"'' not known")') padding_method
       call stop_program(err_msg)
       return
    end select

    pad = get_padding_half(kernel_size)

  end subroutine set_padding
!###############################################################################


!###############################################################################
  subroutine pad_data( &
       data, data_padded, &
       kernel_size, padding_method, &
       sample_dim, channel_dim, constant &
  )
    !! Pad data for convolutional layers
    implicit none

    ! Arguments
    real(real32), dimension(..), intent(in) :: data
    !! Data to be padded
    real(real32), allocatable, dimension(..), intent(out) :: data_padded
    !! Padded data
    integer, dimension(..), intent(in) :: kernel_size
    !! Width of kernel/filter
    character(*), intent(inout) :: padding_method
    !! Padding method
    real(real32), optional, intent(in) :: constant
    !! Constant value for padding
    integer, optional, intent(in) :: sample_dim, channel_dim
    !! Dimensions along which to pad

    ! Local variables
    integer :: i, j, idim
    !! Loop indices
    integer :: num_samples, num_channels, ndim, ndata_dim
    !! Number of samples, channels, dimensions
    integer :: sample_dim_ = 0, channel_dim_ = 0
    !! Sample and channel dimensions
    real(real32) :: constant_ = 0._real32
    !! Constant value for padding
    integer, dimension(2) :: bound_store
    !! Store boundary indices
    integer, allocatable, dimension(:) :: padding
    !! Padding width
    integer, allocatable, dimension(:,:) :: trgt_bound, dest_bound
    !! Target and destination boundaries
    integer, allocatable, dimension(:,:) :: tmp_trgt_bound, tmp_dest_bound
    !! Temporary target and destination boundaries

    character(256) :: err_msg
    !! Error message


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(constant)) constant_ = constant
    if(present(sample_dim)) sample_dim_ = sample_dim
    if(present(channel_dim)) channel_dim_ = channel_dim

    ndim = rank(data)
#if defined(GFORTRAN)
    if(ndim.ne.rank(data_padded))then
       call stop_program("data and data_padded are not the same rank")
       return
    end if
#else
    select rank(data_padded)
    rank(1)
       if(ndim.ne.1)then
          call stop_program("data and data_padded are not the same rank")
          return
       end if
    rank(2)
       if(ndim.ne.2)then
          call stop_program("data and data_padded are not the same rank")
          return
       end if
    rank(3)
       if(ndim.ne.3)then
          call stop_program("data and data_padded are not the same rank")
          return
       end if
    rank(4)
       if(ndim.ne.4)then
          call stop_program("data and data_padded are not the same rank")
          return
       end if
    rank(5)
       if(ndim.ne.5)then
          call stop_program("data and data_padded are not the same rank")
          return
       end if
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
       call stop_program("cannot handle data with this rank")
       return
    end select


    !---------------------------------------------------------------------------
    ! Handle padding type name
    !---------------------------------------------------------------------------
    ! none  = alt. name for 'valid'
    ! zero  = alt. name for 'same'
    ! symmetric = alt.name for 'replication'
    ! valid = no padding
    ! same  = maintain spatial dimensions
    !         ... (i.e. padding added = (kernel_size - 1)/2)
    !         ... defaults to zeros in the padding
    ! full  = enough padding for filter to slide over every possible position
    !         ... (i.e. padding added = (kernel_size - 1)
    ! circular = maintain spatial dimensions
    !            ... wraps data around for padding (periodic)
    ! reflection = maintains spatial dimensions
    !              ... reflect data (about boundary index)
    ! replication = maintains spatial dimensions
    !               ... reflect data (boundary included)
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
             call set_padding( &
                  padding(i), &
                  kernel_size(1), &
                  padding_method, &
                  verbose = 0 &
             )
          end do
       else
          if(sample_dim_.eq.0.and.channel_dim_.eq.0.and.&
               size(kernel_size).ne.ndim)then
             write(err_msg,'("&
                  &kernel_size length not equal to rank of data",A,"&
                  &kernel dimension: ",I0,A,"&
                  &data rank: ",I0)' &
             ) &
                  achar(13) // achar(10), size(kernel_size), &
                  achar(13) // achar(10), ndim
             call stop_program(err_msg)
             return
          elseif(sample_dim_.gt.0.and.channel_dim_.gt.0.and.&
               size(kernel_size).ne.ndim-2)then
             write(err_msg,'("&
                  &kernel_size length not equal to rank of data-2",A,"&
                  &kernel dimension: ",I0,A,"&
                  &data rank: ",I0)' &
             ) &
                  achar(13) // achar(10), size(kernel_size), &
                  achar(13) // achar(10), ndim-2
             call stop_program(err_msg)
             return
          elseif((sample_dim_.gt.0.or.channel_dim_.gt.0).and.&
               .not.(sample_dim_.gt.0.and.channel_dim_.gt.0).and.&
               size(kernel_size).ne.ndim-1)then
             write(err_msg,'("&
                  &kernel_size length not equal to rank of data-1",A,"&
                  &kernel dimension: ",I0,A,"&
                  &data rank: ",I0)' &
             ) &
                  achar(13) // achar(10), size(kernel_size), &
                  achar(13) // achar(10), ndim-1
             call stop_program(err_msg)
             return
          else
             allocate(padding(size(kernel_size)))
          end if
          do i=1,size(kernel_size)
             call set_padding( &
                  padding(i), kernel_size(i), padding_method, verbose=0 &
             )
          end do
       end if
    end select


    !---------------------------------------------------------------------------
    ! Allocate data set
    ! ... if appropriate, add padding
    !---------------------------------------------------------------------------
    select case(padding_method)
    case("same")
    case("full")
    case("zero")
    case default
       if(abs(constant_).gt.1.E-8)then
          write(*,*) "WARNING: constant is ignored for selected padding method"
       end if
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

       ! Copy input data
       !------------------------------------------------------------------------
       select rank(data)
       rank(1)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1) &
          ) = data( &
               trgt_bound(1,1):trgt_bound(2,1) &
          )
       end select
    rank(2)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1), &
            dest_bound(1,2):dest_bound(2,2)), source = constant_)

       ! Copy input data
       !------------------------------------------------------------------------
       select rank(data)
       rank(2)
          data_padded( &
               trgt_bound(1,1) : trgt_bound(2,1), &
               trgt_bound(1,2) : trgt_bound(2,2) &
          ) = data( &
               trgt_bound(1,1) : trgt_bound(2,1), &
               trgt_bound(1,2) : trgt_bound(2,2) &
          )
       end select
    rank(3)
       allocate( &
            data_padded(&
                 dest_bound(1,1):dest_bound(2,1),&
                 dest_bound(1,2):dest_bound(2,2),&
                 dest_bound(1,3):dest_bound(2,3) &
            ), source = constant_ &
       )

       ! Copy input data
       !------------------------------------------------------------------------
       select rank(data)
       rank(3)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3) &
          ) = data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3) &
          )
       end select
    rank(4)
       allocate( &
            data_padded( &
                 dest_bound(1,1):dest_bound(2,1), &
                 dest_bound(1,2):dest_bound(2,2), &
                 dest_bound(1,3):dest_bound(2,3), &
                 dest_bound(1,4):dest_bound(2,4) &
            ), source = constant_ &
       )

       ! Copy input data
       !------------------------------------------------------------------------
       select rank(data)
       rank(4)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4) &
          ) = data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4) &
          )
       end select
    rank(5)
       allocate( &
            data_padded(&
                 dest_bound(1,1):dest_bound(2,1), &
                 dest_bound(1,2):dest_bound(2,2), &
                 dest_bound(1,3):dest_bound(2,3), &
                 dest_bound(1,4):dest_bound(2,4), &
                 dest_bound(1,5):dest_bound(2,5) &
            ), source = constant_ &
       )

       ! Copy input data
       !------------------------------------------------------------------------
       select rank(data)
       rank(5)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4), &
               trgt_bound(1,5):trgt_bound(2,5) &
          ) = data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4), &
               trgt_bound(1,5):trgt_bound(2,5) &
          )
       end select
    end select


    !---------------------------------------------------------------------------
    ! Return if constant -- or no -- padding
    !---------------------------------------------------------------------------
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


    !---------------------------------------------------------------------------
    ! Insert padding
    !---------------------------------------------------------------------------
    i = 0
    do idim=1,ndim
       if(idim.eq.sample_dim_.or.idim.eq.channel_dim_) cycle
       i = i + 1
       tmp_dest_bound = dest_bound
       tmp_trgt_bound = dest_bound
       tmp_dest_bound(:,idim) = [ dest_bound(1,idim), trgt_bound(1,idim) - 1 ]
       select case(padding_method)
       case ("circular")
          tmp_trgt_bound(:,idim) = &
               [ trgt_bound(2,idim) - padding(i) + 1, trgt_bound(2,idim) ]
       case("reflection")
          tmp_trgt_bound(:,idim) = &
               [ trgt_bound(1,idim) + 1, trgt_bound(1,idim) + padding(i) ]
       case("replication")
          tmp_trgt_bound(:,idim) = &
               [ trgt_bound(1,idim), trgt_bound(1,idim) + padding(i) - 1 ]
       end select
       do j = 1, 2
          select rank(data_padded)
          rank(1)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1) &
             ) = data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1) &
             )
          rank(2)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2) &
             ) = data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2) &
             )
          rank(3)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3) &
             ) = data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3) &
             )
          rank(4)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3), &
                  tmp_dest_bound(1,4):tmp_dest_bound(2,4) &
             ) = data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3), &
                  tmp_trgt_bound(1,4):tmp_trgt_bound(2,4) &
             )
          rank(5)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3), &
                  tmp_dest_bound(1,4):tmp_dest_bound(2,4), &
                  tmp_dest_bound(1,5):tmp_dest_bound(2,5) &
             ) = data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3), &
                  tmp_trgt_bound(1,4):tmp_trgt_bound(2,4), &
                  tmp_trgt_bound(1,5):tmp_trgt_bound(2,5) &
             )
          end select

          if(j.eq.2) exit
          bound_store(:) = tmp_dest_bound(:,idim)
          select case(padding_method)
          case ("circular")
             tmp_dest_bound(:,idim) = tmp_trgt_bound(:,idim) + padding(i)
             tmp_trgt_bound(:,idim) = bound_store(:) + padding(i)
          case("reflection")
             tmp_dest_bound(:,idim) = &
                  tmp_trgt_bound(:,idim) + size(data,idim) - 1
             tmp_trgt_bound(:,idim) = bound_store(:) + size(data,idim) - 1
          case("replication")
             tmp_dest_bound(:,idim) = tmp_trgt_bound(:,idim) + size(data,idim)
             tmp_trgt_bound(:,idim) = bound_store(:) + size(data,idim)
          end select
       end do
    end do

  end subroutine pad_data
!###############################################################################

end module athena__misc_ml
