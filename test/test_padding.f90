program test_padding
  use athena, only: &
       pad_data
  implicit none

  real, allocatable, dimension(:) :: input_data1d_ncs, padded_data1d_ncs
  real, allocatable, dimension(:,:) :: input_data1d_nc, padded_data1d_nc
  real, allocatable, dimension(:,:,:) :: input_data1d, padded_data1d
  real, allocatable, dimension(:,:,:,:) :: input_data2d, padded_data2d
  real, allocatable, dimension(:,:,:,:,:) :: input_data3d, padded_data3d
  logical :: success = .true.

  integer :: i
  integer, parameter :: kernel_size = 3
  integer, parameter :: width = 8
  integer :: half
  integer :: out_width(10)
  real :: pad_value(10)
  character(len=20) :: padding_methods(10)


!!!-----------------------------------------------------------------------------
!!! initialise all data
!!!-----------------------------------------------------------------------------
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


!!!-----------------------------------------------------------------------------
!!! test kernel rank
!!!-----------------------------------------------------------------------------
  call pad_data(input_data1d, padded_data1d, &
       [kernel_size], padding_method = padding_methods(3), &
       sample_dim = 3, channel_dim = 2)
  call pad_data(input_data1d, padded_data1d, &
       [kernel_size], padding_method = padding_methods(3), &
       sample_dim = 3, channel_dim = 0)
  call pad_data(input_data2d, padded_data2d, &
       [kernel_size, kernel_size], padding_method = padding_methods(3), &
       sample_dim = 4, channel_dim = 3)


!!!-----------------------------------------------------------------------------
!!! test padding methods
!!!-----------------------------------------------------------------------------
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


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_padding passed all tests'
  else
     write(0,*) 'test_padding failed one or more tests'
     stop 1
  end if

end program test_padding