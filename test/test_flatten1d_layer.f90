program test_flatten1d_layer
  use constants, only: real12
  use flatten1d_layer, only: flatten1d_layer_type
  implicit none

  type(flatten1d_layer_type) :: flatten_layer
  integer, parameter :: batch_size = 1, width = 8, num_channels = 3
  real(real12), allocatable, dimension(:,:,:) :: input_data
  real(real12), allocatable, dimension(:,:) :: output, gradient
  logical :: success = .true.
  real, parameter :: tol = 1e-6

  integer :: i, j, output_width

  integer :: seed_size = 1
  integer, allocatable, dimension(:) :: seed


!!!-----------------------------------------------------------------------------
!!! Initialize random number generator with a seed
!!!-----------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=0)
  call random_seed(put = seed)


!!!-----------------------------------------------------------------------------
!!! set up layer
!!!-----------------------------------------------------------------------------
  flatten_layer = flatten1d_layer_type( &
       input_shape = [width, num_channels], batch_size = batch_size)

  !! check layer type
  if(.not. flatten_layer%name .eq. 'flatten1d')then
     success = .false.
     write(0,*) 'flatten1d layer has wrong name'
  end if

  !! check input shape
  if(any(flatten_layer%input_shape .ne. [width, num_channels]))then
     success = .false.
     write(0,*) 'flatten1d layer has wrong input_shape'
  end if

  !! check output shape
  if(any(flatten_layer%output_shape .ne. [width*num_channels]))then
     success = .false.
     write(0,*) 'flatten1d layer has wrong output_shape'
  end if

  !! check batch size
  if(flatten_layer%batch_size .ne. 1)then
     success = .false.
     write(0,*) 'flatten1d layer has wrong batch size'
  end if


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  allocate(input_data(width, num_channels, batch_size), source = 0.0)
  call random_number(input_data)

  !! run forward pass
  call flatten_layer%forward(input_data)
  call flatten_layer%get_output(output)

  !! check outputs have expected value
  if(any(abs(pack(input_data(:,:,1),mask=.true.) - output(:,1)).gt.tol))then
     success = .false.
     write(0,*) 'flatten1d layer forward pass incorrect'
   end if


!!!-----------------------------------------------------------------------------
!!! test backward pass and check expected output
!!!-----------------------------------------------------------------------------
  !! run backward pass
  allocate(gradient, source = output)
  call flatten_layer%backward(input_data, gradient)

  !! check gradient has expected value
  if(any(abs(flatten_layer%di - input_data).gt.tol))then
    success = .false.
    write(0,*) 'flatten1d layer backward pass incorrect'
  end if


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_flatten1d_layer passed all tests'
  else
     write(0,*) 'test_flatten1d_layer failed one or more tests'
     stop 1
  end if

end program test_flatten1d_layer