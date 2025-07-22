program test_flatten_layer
  use athena__constants, only: real32
  use athena__flatten_layer, only: flatten_layer_type, read_flatten_layer
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: array3d_type, array4d_type, array5d_type
  implicit none

  type(flatten_layer_type) :: flatten_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: batch_size = 1, width = 8, num_channels = 3
  integer :: unit
  real(real32), allocatable, dimension(:,:,:) :: input_data3d
  real(real32), allocatable, dimension(:,:,:,:) :: input_data4d
  real(real32), allocatable, dimension(:,:,:,:,:) :: input_data5d
  real(real32), allocatable, dimension(:,:) :: output, gradient
  logical :: success = .true.
  real, parameter :: tol = 1e-6

  integer :: i, j, output_width

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed


!!!-----------------------------------------------------------------------------
!!! Initialise random number generator with a seed
!!!-----------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=0)
  call random_seed(put = seed)


!!!-----------------------------------------------------------------------------
!!! set up layer 2D
!!!-----------------------------------------------------------------------------
  flatten_layer = flatten_layer_type( &
       input_shape = [width, num_channels], batch_size = batch_size)

  !! check layer type
  if(.not. flatten_layer%name .eq. 'flatten')then
     success = .false.
     write(0,*) 'flatten layer has wrong name'
  end if

  !! check input shape
  if(any(flatten_layer%input_shape .ne. [width, num_channels]))then
     success = .false.
     write(0,*) 'flatten layer (2D) has wrong input_shape'
  end if

  !! check output shape
  if(any(flatten_layer%output_shape .ne. [width*num_channels]))then
     success = .false.
     write(0,*) 'flatten layer (2D) has wrong output shape'
  end if

  !! check batch size
  if(flatten_layer%batch_size .ne. 1)then
     success = .false.
     write(0,*) 'flatten layer (2D) has wrong batch size'
  end if


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output for 2D
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  allocate(input_data3d(width, num_channels, batch_size), source = 0.0)
  call random_number(input_data3d)

  !! run forward pass
  call flatten_layer%forward(input_data3d)
  call flatten_layer%get_output(output)

  !! check outputs have expected value
  if(any(abs(pack(input_data3d(:,:,1),mask=.true.) - output(:,1)).gt.tol))then
     success = .false.
     write(0,*) 'flatten layer (2D) forward pass incorrect'
  end if


!!!-----------------------------------------------------------------------------
!!! test backward pass and check expected output for 2D
!!!-----------------------------------------------------------------------------
  !! run backward pass
  allocate(gradient, source = output)
  call flatten_layer%backward(input_data3d, gradient)

  !! check gradient has expected value
  select type(di => flatten_layer%di(1,1))
  type is (array3d_type)
     if(any(abs(di%val_ptr - input_data3d).gt.tol))then
        success = .false.
        write(0,*) 'flatten layer (2D) backward pass incorrect'
     end if
  class default
     success = .false.
     write(0,*) 'flatten layer (2D) has not set di type correctly'
  end select





!!!-----------------------------------------------------------------------------
!!! set up layer 3D
!!!-----------------------------------------------------------------------------
  flatten_layer = flatten_layer_type( &
       input_shape = [width, width, num_channels], batch_size = batch_size)

  !! check input shape
  if(any(flatten_layer%input_shape .ne. [width, width, num_channels]))then
     success = .false.
     write(0,*) 'flatten layer (3D) has wrong input_shape'
  end if

  !! check output shape
  if(any(flatten_layer%output_shape .ne. [width*width*num_channels]))then
     success = .false.
     write(0,*) 'flatten layer (3D) has wrong output shape'
  end if

  !! check batch size
  if(flatten_layer%batch_size .ne. 1)then
     success = .false.
     write(0,*) 'flatten layer (3D) has wrong batch size'
  end if


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output for 3D
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  allocate(input_data4d(width, width, num_channels, batch_size), source = 0.0)
  call random_number(input_data4d)

  !! run forward pass
  call flatten_layer%forward(input_data4d)
  call flatten_layer%get_output(output)

  !! check outputs have expected value
  if(any(abs(pack(input_data4d(:,:,:,1),mask=.true.) - output(:,1)).gt.tol))then
     success = .false.
     write(0,*) 'flatten layer (3D) forward pass incorrect'
  end if


!!!-----------------------------------------------------------------------------
!!! test backward pass and check expected output for 3D
!!!-----------------------------------------------------------------------------
  !! run backward pass
  deallocate(gradient)
  allocate(gradient, source = output)
  call flatten_layer%backward(input_data4d, gradient)

  !! check gradient has expected value
  select type(di => flatten_layer%di(1,1))
  type is (array4d_type)
     if(any(abs(di%val_ptr - input_data4d).gt.tol))then
        success = .false.
        write(0,*) 'flatten layer (3D) backward pass incorrect'
     end if
  class default
     success = .false.
     write(0,*) 'flatten layer (3D) has not set di type correctly'
  end select






!!!-----------------------------------------------------------------------------
!!! set up layer 4D
!!!-----------------------------------------------------------------------------
  flatten_layer = flatten_layer_type( &
       input_shape = [width, width, width, num_channels], &
       batch_size = batch_size &
  )

  !! check input shape
  if(any( &
       flatten_layer%input_shape .ne. &
       [width, width, width, num_channels] &
  ))then
     success = .false.
     write(0,*) 'flatten layer (4D) has wrong input_shape'
  end if

  !! check output shape
  if(any( &
       flatten_layer%output_shape .ne. &
       [width*width*width*num_channels] &
  ))then
     success = .false.
     write(0,*) 'flatten layer (4D) has wrong output shape'
  end if

  !! check batch size
  if(flatten_layer%batch_size .ne. 1)then
     success = .false.
     write(0,*) 'flatten layer (4D) has wrong batch size'
  end if


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output for 4D
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  allocate(input_data5d( &
       width, width, width, num_channels, batch_size &
  ), source = 0.0 )
  call random_number(input_data4d)

  !! run forward pass
  call flatten_layer%forward(input_data5d)
  call flatten_layer%get_output(output)

  !! check outputs have expected value
  if( &
       any( &
            abs( pack( &
                 input_data5d(:,:,:,:,1), &
                 mask=.true. &
            ) - output(:,1) ) .gt. tol &
       ) &
  )then
     success = .false.
     write(0,*) 'flatten layer (4D) forward pass incorrect'
  end if


!!!-----------------------------------------------------------------------------
!!! test backward pass and check expected output for 4D
!!!-----------------------------------------------------------------------------
  !! run backward pass
  deallocate(gradient)
  allocate(gradient, source = output)
  call flatten_layer%backward(input_data5d, gradient)

  !! check gradient has expected value
  select type(di => flatten_layer%di(1,1))
  type is (array5d_type)
     if(any(abs(di%val_ptr - input_data5d).gt.tol))then
        success = .false.
        write(0,*) 'flatten layer (4D) backward pass incorrect'
     end if
  class default
     success = .false.
     write(0,*) 'flatten layer (4D) has not set di type correctly'
  end select






!!!-----------------------------------------------------------------------------
!!! Test file I/O operations
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_flatten_layer.tmp', &
       status='replace', action='write')
  
  ! Write layer to file
  write(unit,'("FLATTEN")')
  call flatten_layer%print_to_unit(unit)
  write(unit,'("END FLATTEN")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_flatten_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_flatten_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (flatten_layer_type)
     if (.not. read_layer%name .eq. 'flatten') then
        success = .false.
        write(0,*) 'read flatten layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not flatten_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_flatten_layer.tmp', status='old')
  close(unit, status='delete')


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_flatten_layer passed all tests'
  else
     write(0,*) 'test_flatten_layer failed one or more tests'
     stop 1
  end if

end program test_flatten_layer
