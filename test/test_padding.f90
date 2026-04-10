program test_padding
  use coreutils, only: real32
  use athena, only: pad_data
  use athena__misc_ml, only: set_padding
  implicit none

  real, allocatable, dimension(:) :: input_data1d_ncs, padded_data1d_ncs
  real, allocatable, dimension(:,:) :: input_data1d_nc, padded_data1d_nc
  real, allocatable, dimension(:,:,:) :: input_data1d, padded_data1d
  real, allocatable, dimension(:,:,:,:) :: input_data2d, padded_data2d
  real, allocatable, dimension(:,:,:,:,:) :: input_data3d, padded_data3d
  real(real32), allocatable :: edge_padded_1d(:)
  real(real32), allocatable :: edge_padded_2d(:,:)
  real(real32), allocatable :: padded_raw4d(:,:,:,:)
  real(real32), allocatable :: padded_raw5d(:,:,:,:,:)
  logical :: success = .true.

  integer :: i
  integer :: pad
  integer, parameter :: kernel_size = 3
  integer, parameter :: width = 8
  integer :: half
  integer :: out_width(10)
  real :: pad_value(10)
  character(len=20) :: padding_methods(10)
  real(real32) :: edge_data_1d(4)
  real(real32) :: edge_data_2d(2,2)
  real(real32) :: raw4d(2,2,2,2)
  real(real32) :: raw5d(2,2,2,2,2)


!-------------------------------------------------------------------------------
! initialise all data
!-------------------------------------------------------------------------------
  allocate(input_data1d_ncs(width))
  allocate(input_data1d_nc(width,1))
  allocate(input_data1d(width,1,1))
  allocate(input_data2d(width,width,1,1))
  allocate(input_data3d(width,width,width,1,1))
  input_data1d_ncs = 1.0
  input_data1d_nc = 1.0
  input_data1d = 1.0
  input_data2d = 1.0
  input_data3d = 1.0
  edge_data_1d = [1._real32, 2._real32, 3._real32, 4._real32]
  edge_data_2d(:,1) = [1._real32, 2._real32]
  edge_data_2d(:,2) = [3._real32, 4._real32]
  raw4d = 1._real32
  raw5d = 1._real32

  !! initialise padding methods
  padding_methods(1)  = 'none'
  padding_methods(2)  = 'valid'
  padding_methods(3)  = 'zero'
  padding_methods(4)  = 'half'
  padding_methods(5)  = 'same'
  padding_methods(6)  = 'full'
  padding_methods(7)  = 'circular'
  padding_methods(8)  = 'reflection'
  padding_methods(9)  = 'replication'
  padding_methods(10) = 'symmetric'

  !! calculate expected output widths and pad values
  half = kernel_size/2
  out_width(1)  = width
  out_width(2)  = width
  out_width(3)  = width + 2 * half
  out_width(4)  = width + 2 * half
  out_width(5)  = width + 2 * half
  out_width(6)  = width + 2 * (kernel_size - 1)
  out_width(7)  = width + 2 * half
  out_width(8)  = width + 2 * half
  out_width(9)  = width + 2 * half
  out_width(10) = width + 2 * half

  !! initialise pad values
  pad_value(1)  = 1.0
  pad_value(2)  = 1.0
  pad_value(3)  = 1.0
  pad_value(4)  = 1.0
  pad_value(5)  = 1.0
  pad_value(6)  = 1.0
  pad_value(7)  = 1.0
  pad_value(8)  = 1.0
  pad_value(9)  = 1.0
  pad_value(10) = 1.0


!-------------------------------------------------------------------------------
! test kernel rank
!-------------------------------------------------------------------------------
  call pad_data(input_data1d, padded_data1d, &
       [kernel_size], padding_method = padding_methods(3), &
       sample_dim = 3, channel_dim = 2)
  call pad_data(input_data1d, padded_data1d, &
       [kernel_size], padding_method = padding_methods(3), &
       sample_dim = 3, channel_dim = 0)
  call pad_data(input_data2d, padded_data2d, &
       [kernel_size, kernel_size], padding_method = padding_methods(3), &
       sample_dim = 4, channel_dim = 3)


!-------------------------------------------------------------------------------
! test set_padding aliases and error handling
!-------------------------------------------------------------------------------
  padding_methods(4) = 'half'
  call set_padding(pad, kernel_size, padding_methods(4), verbose=0)
  if(pad.ne.1 .or. trim(padding_methods(4)).ne.'same')then
     success = .false.
     write(0,*) 'set_padding did not normalise half padding correctly'
  end if

  padding_methods(10) = 'symmetric'
  call set_padding(pad, kernel_size, padding_methods(10), verbose=0)
  if(pad.ne.1 .or. trim(padding_methods(10)).ne.'replication')then
     success = .false.
     write(0,*) 'set_padding did not normalise symmetric padding correctly'
  end if

  padding_methods(6) = 'full'
  call set_padding(pad, kernel_size, padding_methods(6), verbose=0)
  if(pad.ne.2)then
     success = .false.
     write(0,*) 'set_padding returned the wrong full padding width'
  end if



!-------------------------------------------------------------------------------
! test padding methods
!-------------------------------------------------------------------------------
  do i = 1, size(padding_methods)
     !! 1D input data (no channel or sample dim)
     !!-------------------------------------------------------------------------
     call pad_data(input_data1d_ncs, padded_data1d_ncs, &
          kernel_size, padding_method = padding_methods(i), &
          sample_dim = 0, channel_dim = 0, constant = 1.0)

     if(any(shape(padded_data1d_ncs).ne.&
          [out_width(i)]))then
        success = .false.
        write(0,'("padded data for method ",A,&
             &" shape should be [",I0,"]")') &
             trim(padding_methods(i)), &
             out_width(i)
     end if
     if(out_width(i) .gt. width .and. &
          any(abs(padded_data1d_ncs(width+1:width+(out_width(i)-width)/2) - &
               pad_value(i)).gt.1.E-6))then
        success = .false.
        write(0,'("padded data for method ",A," should be ",F4.1,F4.1)') &
             trim(padding_methods(i)), pad_value(i), &
             padded_data1d_ncs(width+(out_width(i)-width)/2)
     end if

     !! 1D input data (no sample dim)
     !!-------------------------------------------------------------------------
     call pad_data(input_data1d_nc, padded_data1d_nc, &
          kernel_size, padding_method = padding_methods(i), &
          sample_dim = 0, channel_dim = 2, constant = 1.0)

     if(any(shape(padded_data1d_nc).ne.&
          [out_width(i),1]))then
        success = .false.
        write(0,'("padded data for method ",A,&
             &" shape should be [",I0,",1]")') &
             trim(padding_methods(i)), &
             out_width(i)
     end if
     if(out_width(i) .gt. width .and. &
          any(abs(padded_data1d_nc(width+1:width+(out_width(i)-width)/2,1) - &
               pad_value(i)).gt.1.E-6))then
        success = .false.
        write(0,'("padded data for method ",A," should be ",F4.1,F4.1)') &
             trim(padding_methods(i)), pad_value(i), &
             padded_data1d_nc(width+(out_width(i)-width)/2,1)
     end if

     !! 1D input data (no channel dim)
     !!-------------------------------------------------------------------------
     call pad_data(input_data1d_nc, padded_data1d_nc, &
          kernel_size, padding_method = padding_methods(i), &
          sample_dim = 2, channel_dim = 0, constant = 1.0)

     if(any(shape(padded_data1d_nc).ne.&
          [out_width(i),1]))then
        success = .false.
        write(0,'("padded data for method ",A,&
             &" shape should be [",I0,",1]")') &
             trim(padding_methods(i)), &
             out_width(i)
     end if
     if(out_width(i) .gt. width .and. &
          any(abs(padded_data1d_nc(width+1:width+(out_width(i)-width)/2,1) - &
               pad_value(i)).gt.1.E-6))then
        success = .false.
        write(0,'("padded data for method ",A," should be ",F4.1,F4.1)') &
             trim(padding_methods(i)), pad_value(i), &
             padded_data1d_nc(width+(out_width(i)-width)/2,1)
     end if

     !! 1D input data
     !!-------------------------------------------------------------------------
     call pad_data(input_data1d, padded_data1d, &
          kernel_size, padding_method = padding_methods(i), &
          sample_dim = 3, channel_dim = 2, constant = 1.0)

     if(any(shape(padded_data1d).ne.&
          [out_width(i),1,1]))then
        success = .false.
        write(0,'("padded data for method ",A,&
             &" shape should be [",I0,",1,1]")') &
             trim(padding_methods(i)), &
             out_width(i)
     end if
     if(out_width(i) .gt. width .and. &
          any(abs(padded_data1d(width+1:width+(out_width(i)-width)/2,1,1) - &
               pad_value(i)).gt.1.E-6))then
        success = .false.
        write(0,'("padded data for method ",A," should be ",F4.1,F4.1)') &
             trim(padding_methods(i)), pad_value(i), &
             padded_data1d(width+(out_width(i)-width)/2,1,1)
     end if

     !! 2D input data
     !!-------------------------------------------------------------------------
     call pad_data(input_data2d, padded_data2d, &
          kernel_size, padding_method = padding_methods(i), &
          sample_dim = 4, channel_dim = 3, constant = 1.0)

     if(any(shape(padded_data2d).ne.&
          [out_width(i),out_width(i),1,1]))then
        success = .false.
        write(0,'("padded data for method ",A,&
             &" shape should be [",I0,",",I0,",1,1]")') &
             trim(padding_methods(i)), &
             out_width(i),out_width(i)
     end if
     if(out_width(i) .gt. width .and. &
          any(abs(padded_data2d(width+1:width+(out_width(i)-width)/2,1,1,1) - &
               pad_value(i)).gt.1.E-6))then
        success = .false.
        write(0,'("padded data for method ",A," should be ",F4.1,F4.1)') &
             trim(padding_methods(i)), pad_value(i), &
             padded_data2d(width+(out_width(i)-width)/2,1,1,1)
     end if

     !! 3D input data
     !!-------------------------------------------------------------------------
     call pad_data(input_data3d, padded_data3d, &
          kernel_size, padding_method = padding_methods(i), &
          sample_dim = 5, channel_dim = 4, constant = 1.0)

     if(any(shape(padded_data3d).ne.&
          [out_width(i),out_width(i),out_width(i),1,1]))then
        success = .false.
        write(0,'("padded data for method ",A,&
             &" shape should be [",I0,",",I0,",",I0,",1,1]")') &
             trim(padding_methods(i)), &
             out_width(i),out_width(i),out_width(i)
     end if
     if(out_width(i) .gt. width .and. &
          any(abs(&
               padded_data3d(width+1:width+(out_width(i)-width)/2,1,1,1,1) - &
               pad_value(i)).gt.1.E-6))then
        success = .false.
        write(0,'("padded data for method ",A," should be ",F4.1,F4.1)') &
             trim(padding_methods(i)), pad_value(i), &
             padded_data3d(width+(out_width(i)-width)/2,1,1,1,1)
     end if
  end do


!-------------------------------------------------------------------------------
! test explicit edge padding values and raw higher-rank paths
!-------------------------------------------------------------------------------
  padding_methods(7) = 'circular'
  call pad_data(edge_data_1d, edge_padded_1d, kernel_size, padding_methods(7), &
       sample_dim = 0, channel_dim = 0)
  if(any(abs(edge_padded_1d - [4._real32, 1._real32, 2._real32, 3._real32, &
       4._real32, 1._real32]).gt.1.E-6_real32))then
     success = .false.
     write(0,*) 'circular 1D padding produced unexpected values'
  end if

  padding_methods(8) = 'reflection'
  call pad_data(edge_data_1d, edge_padded_1d, kernel_size, padding_methods(8), &
       sample_dim = 0, channel_dim = 0)
  if(any(abs(edge_padded_1d - [2._real32, 1._real32, 2._real32, 3._real32, &
       4._real32, 3._real32]).gt.1.E-6_real32))then
     success = .false.
     write(0,*) 'reflection 1D padding produced unexpected values'
  end if

  padding_methods(9) = 'replication'
  call pad_data(edge_data_1d, edge_padded_1d, kernel_size, padding_methods(9), &
       sample_dim = 0, channel_dim = 0)
  if(any(abs(edge_padded_1d - [1._real32, 1._real32, 2._real32, 3._real32, &
       4._real32, 4._real32]).gt.1.E-6_real32))then
     success = .false.
     write(0,*) 'replication 1D padding produced unexpected values'
  end if

  padding_methods(7) = 'circular'
  call pad_data(edge_data_2d, edge_padded_2d, [kernel_size, kernel_size], &
       padding_methods(7), sample_dim = 0, channel_dim = 0)
  if(any(shape(edge_padded_2d).ne.[4, 4]))then
     success = .false.
     write(0,*) 'raw 2D circular padding returned the wrong shape'
  elseif(any(abs(edge_padded_2d(1:2,1:2) - edge_data_2d).gt.1.E-6_real32))then
     success = .false.
     write(0,*) 'raw 2D circular padding lost the interior data'
  elseif(abs(edge_padded_2d(lbound(edge_padded_2d,1), &
       lbound(edge_padded_2d,2)) - 4._real32).gt.1.E-6_real32)then
     success = .false.
     write(0,*) 'raw 2D circular padding did not wrap the corner value'
  end if

  padding_methods(5) = 'same'
  call pad_data(raw4d, padded_raw4d, kernel_size, padding_methods(5), &
       sample_dim = 0, channel_dim = 0)
  if(any(shape(padded_raw4d).ne.[4, 4, 4, 4]))then
     success = .false.
     write(0,*) 'raw 4D padding returned the wrong shape'
  elseif(any(abs(padded_raw4d(1:2,1:2,1:2,1:2) - raw4d).gt.1.E-6_real32))then
     success = .false.
     write(0,*) 'raw 4D padding lost the interior data'
  end if

  call pad_data(raw5d, padded_raw5d, kernel_size, padding_methods(5), &
       sample_dim = 0, channel_dim = 0)
  if(any(shape(padded_raw5d).ne.[4, 4, 4, 4, 4]))then
     success = .false.
     write(0,*) 'raw 5D padding returned the wrong shape'
  elseif(any(abs( &
       padded_raw5d(1:2,1:2,1:2,1:2,1:2) - raw5d &
  ).gt.1.E-6_real32))then
     success = .false.
     write(0,*) 'raw 5D padding lost the interior data'
  end if



!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_padding passed all tests'
  else
     write(0,*) 'test_padding failed one or more tests'
     stop 1
  end if

end program test_padding
