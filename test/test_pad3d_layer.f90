program test_pad3d_layer
  !! Unit tests for the pad3d layer module
  use athena__constants, only: real32
  use athena__pad3d_layer, only: pad3d_layer_type, read_pad3d_layer
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: array5d_type
  implicit none

  type(pad3d_layer_type) :: pad3d_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: batch_size = 2
  integer, parameter :: width = 3
  integer, parameter :: height = 4
  integer, parameter :: depth = 2
  integer, parameter :: channels = 2
  logical :: success = .true.
  real(real32), parameter :: tol = 1.0e-6_real32

  ! Test data
  real(real32), allocatable, dimension(:,:) :: input_2d, output_2d
  real(real32), allocatable, dimension(:,:) :: gradient_2d
  real(real32), allocatable, dimension(:,:,:,:,:) :: input_5d, output_5d
  real(real32), allocatable, dimension(:,:,:,:,:) :: gradient_5d

  integer :: i, j, k, l, m
  integer :: unit_num = 10
  integer :: expected_width, expected_height, expected_depth

  ! Random seed setup
  integer :: seed_size
  integer, allocatable, dimension(:) :: seed


!!!-----------------------------------------------------------------------------
!!! Initialise random number generator with a seed
!!!-----------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=42)
  call random_seed(put = seed)


!!!-----------------------------------------------------------------------------
!!! Test 3D padding layer setup with zero padding
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing 3D padding layer setup with zero padding..."
  
  pad3d_layer = pad3d_layer_type( &
       padding = [1, 2, 1], &
       method = "zero", &
       input_shape = [width, height, depth, channels], &
       batch_size = batch_size &
  )

  ! Check layer properties
  if (.not. pad3d_layer%name .eq. 'pad3d') then
     success = .false.
     write(0,*) 'pad3d layer has wrong name'
  end if

  if (.not. pad3d_layer%type .eq. 'pad') then
     success = .false.
     write(0,*) 'pad3d layer has wrong type'
  end if

  if (any(pad3d_layer%input_shape .ne. [width, height, depth, channels])) then
     success = .false.
     write(0,*) 'pad3d layer has wrong input_shape'
  end if

  expected_width = width + 2 * 1   ! padding[1] on both sides
  expected_height = height + 2 * 2 ! padding[2] on both sides
  expected_depth = depth + 2 * 1   ! padding[3] on both sides
  if (any(pad3d_layer%output_shape .ne. &
          [expected_width, expected_height, expected_depth, channels])) then
     success = .false.
     write(0,*) 'pad3d layer has wrong output_shape'
     write(0,*) 'Expected:', &
          [expected_width, expected_height, expected_depth, channels]
     write(0,*) 'Got:', pad3d_layer%output_shape
  end if

  if (pad3d_layer%input_rank .ne. 4) then
     success = .false.
     write(0,*) 'pad3d layer has wrong input_rank'
  end if

  if (pad3d_layer%output_rank .ne. 4) then
     success = .false.
     write(0,*) 'pad3d layer has wrong output_rank'
  end if

  if (pad3d_layer%batch_size .ne. batch_size) then
     success = .false.
     write(0,*) 'pad3d layer has wrong batch_size'
  end if


!!!-----------------------------------------------------------------------------
!!! Test 2D input forward pass with zero padding (flattened 3D)
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing 2D input forward pass with zero padding..."

  ! Initialize test input (flattened 4D to 1D for simplicity)  
  allocate(input_2d(width * height * depth * channels, batch_size))
  do i = 1, width * height * depth
     input_2d(i,1) = real(i)
     input_2d(i,2) = real(i+width*height*depth)
  end do
  ! Replicate for each channel
  do i = 2, channels
     input_2d(width * height * depth * (i - 1) + 1: &
              width * height * depth * i, 1) = &
          input_2d(:width*height*depth,1) + (i - 1) * width * height * depth
     input_2d(width * height * depth * (i - 1) + 1: &
              width * height * depth * i, 2) = &
          input_2d(:width*height*depth,2) + (i - 1) * width * height * depth
  end do

  ! Run forward pass
  call pad3d_layer%forward(input_2d)
  call pad3d_layer%get_output(output_2d)

  ! Check output dimensions
  if (size(output_2d, 1) .ne. &
      expected_width * expected_height * expected_depth * channels .or. &
      size(output_2d, 2) .ne. batch_size) then
     success = .false.
     write(0,*) 'pad3d layer forward output has wrong dimensions'
     write(0,*) 'Expected shape:', &
          [expected_width * expected_height * expected_depth * channels, &
           batch_size]
     write(0,*) 'Got shape:', shape(output_2d)
  end if

  deallocate(input_2d, output_2d)


!!!-----------------------------------------------------------------------------
!!! Test 5D input forward pass with zero padding
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing 5D input forward pass with zero padding..."

  ! Initialize test input
  allocate(input_5d(width, height, depth, channels, batch_size))
  call random_number(input_5d)

  ! Run forward pass
  call pad3d_layer%forward(input_5d)
  call pad3d_layer%get_output(output_5d)

  ! Check output dimensions
  if (size(output_5d, 1) .ne. expected_width .or. &
      size(output_5d, 2) .ne. expected_height .or. &
      size(output_5d, 3) .ne. expected_depth .or. &
      size(output_5d, 4) .ne. channels .or. &
      size(output_5d, 5) .ne. batch_size) then
     success = .false.
     write(0,*) 'pad3d layer 5D forward output has wrong dimensions'
     write(0,*) 'Expected shape:', &
          [expected_width, expected_height, expected_depth, &
           channels, batch_size]
     write(0,*) 'Got shape:', shape(output_5d)
  end if

  ! Check zero padding on width dimension
  if (any(abs(output_5d(1,:,:,:,:)) .gt. tol) .or. &
      any(abs(output_5d(expected_width,:,:,:,:)) .gt. tol)) then
     success = .false.
     write(0,*) 'pad3d layer width zero padding incorrect'
  end if

  ! Check zero padding on height dimension
  if (any(abs(output_5d(:,1:2,:,:,:)) .gt. tol) .or. &
      any(abs(output_5d(:,expected_height-1:expected_height,:,:,:)) &
          .gt. tol)) then
     success = .false.
     write(0,*) 'pad3d layer height zero padding incorrect'
  end if

  ! Check zero padding on depth dimension
  if (any(abs(output_5d(:,:,1,:,:)) .gt. tol) .or. &
      any(abs(output_5d(:,:,expected_depth,:,:)) .gt. tol)) then
     success = .false.
     write(0,*) 'pad3d layer depth zero padding incorrect'
  end if

  ! Check that middle elements match input
  if (any(abs(output_5d(2:expected_width-1, &
                        3:expected_height-2, &
                        2:expected_depth-1, :, :) - &
              input_5d) .gt. tol)) then
     success = .false.
     write(0,*) 'pad3d layer 5D forward pass incorrect for middle elements'
  end if


!!!-----------------------------------------------------------------------------
!!! Test backward pass
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing backward pass..."

  ! Initialize gradient
  allocate(gradient_5d(expected_width, expected_height, expected_depth, &
                       channels, batch_size))
  gradient_5d = 1.0_real32

  ! Run backward pass
  call pad3d_layer%backward(input_5d, gradient_5d)

  ! Check that gradient is correctly trimmed back to input size
  select type(di => pad3d_layer%di(1,1))
  type is (array5d_type)
     if (any(shape(di%val_ptr) .ne. &
             [width, height, depth, channels, batch_size])) then
        success = .false.
        write(0,*) 'pad3d layer backward gradient has wrong dimensions'
        write(0,*) 'Expected shape:', &
             [width, height, depth, channels, batch_size]
        write(0,*) 'Got shape:', shape(di%val_ptr)
     end if

     ! For zero padding, gradient in the middle should equal input gradient
     if (any(abs(di%val_ptr - gradient_5d(2:expected_width-1, &
                                     3:expected_height-2, &
                                     2:expected_depth-1, :, :)) .gt. tol)) then
        success = .false.
        write(0,*) 'pad3d layer backward pass incorrect'
     end if
  class default
     success = .false.
     write(0,*) 'pad3d layer has not set di type correctly'
  end select

  deallocate(input_5d, output_5d, gradient_5d)


!!!-----------------------------------------------------------------------------
!!! Test different padding methods
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing different padding methods..."

  test_methods_block: block
  character(20), dimension(4) :: padding_methods = [ &
       "zero      ", "same      ", "valid     ", "full      " ]

  do i = 1, size(padding_methods)
     pad3d_layer = pad3d_layer_type( &
          padding = [1, 1, 1], &
          method = trim(padding_methods(i)), &
          input_shape = [width, height, depth, channels], &
          batch_size = 1 &
     )

     ! Test forward pass doesn't crash
     allocate(input_5d(width, height, depth, 1, 1))
     call random_number(input_5d)
     
     call pad3d_layer%forward(input_5d)
     call pad3d_layer%get_output(output_5d)

     if (.not. allocated(output_5d)) then
        success = .false.
        write(0,*) 'output not allocated for padding method: ', &
             trim(padding_methods(i))
     end if

     deallocate(input_5d, output_5d)
  end do
  end block test_methods_block


!!!-----------------------------------------------------------------------------
!!! Test different padding sizes
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing different padding sizes..."

  test_sizes_block: block
  integer, dimension(3, 3) :: test_paddings
  test_paddings(1,:) = [0, 0, 0]
  test_paddings(2,:) = [1, 1, 2]
  test_paddings(3,:) = [2, 1, 0]

  do i = 1, size(test_paddings, 1)
     pad3d_layer = pad3d_layer_type( &
          padding = test_paddings(i,:), &
          method = "zero", &
          input_shape = [width, height, depth, channels], &
          batch_size = 1 &
     )

     expected_width = width + 2 * test_paddings(i,1)
     expected_height = height + 2 * test_paddings(i,2)
     expected_depth = depth + 2 * test_paddings(i,3)

     ! Check output shape
     if (any(pad3d_layer%output_shape .ne. &
             [expected_width, expected_height, expected_depth, channels])) then
        success = .false.
        write(0,*) 'pad3d layer output shape incorrect for padding size:', &
             test_paddings(i,:)
        write(0,*) 'Expected:', &
             [expected_width, expected_height, expected_depth, channels]
        write(0,*) 'Got:', pad3d_layer%output_shape
     end if

     ! Test forward pass
     allocate(input_5d(width, height, depth, 1, 1))
     input_5d = 1.0_real32
     
     call pad3d_layer%forward(input_5d)
     call pad3d_layer%get_output(output_5d)

     ! For zero padding, check that padding is actually zero
     if (test_paddings(i,1) > 0) then
        if (any(abs(output_5d(1:test_paddings(i,1),:,:,:,:)) .gt. tol) .or. &
            any(abs(output_5d(expected_width-test_paddings(i,1)+1: &
                              expected_width,:,:,:,:)) .gt. tol)) then
           success = .false.
           write(0,*) 'pad3d layer width zero padding incorrect for size:', &
                test_paddings(i,1)
        end if
     end if

     if (test_paddings(i,2) > 0) then
        if (any(abs(output_5d(:,1:test_paddings(i,2),:,:,:)) .gt. tol) .or. &
            any(abs(output_5d(:,expected_height-test_paddings(i,2)+1: &
                              expected_height,:,:,:)) .gt. tol)) then
           success = .false.
           write(0,*) 'pad3d layer height zero padding incorrect for size:', &
                test_paddings(i,2)
        end if
     end if

     if (test_paddings(i,3) > 0) then
        if (any(abs(output_5d(:,:,1:test_paddings(i,3),:,:)) .gt. tol) .or. &
            any(abs(output_5d(:,:,expected_depth-test_paddings(i,3)+1: &
                              expected_depth,:,:)) .gt. tol)) then
           success = .false.
           write(0,*) 'pad3d layer depth zero padding incorrect for size:', &
                test_paddings(i,3)
        end if
     end if

     deallocate(input_5d, output_5d)
  end do
  end block test_sizes_block


!!!-----------------------------------------------------------------------------
!!! Test asymmetric padding
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing asymmetric padding..."

  pad3d_layer = pad3d_layer_type( &
       padding = [1, 2, 0], &
       method = "zero", &
       input_shape = [width, height, depth, channels], &
       batch_size = 1 &
  )

  ! Test with a known input pattern
  allocate(input_5d(width, height, depth, 1, 1))
  do i = 1, width
     do j = 1, height
        do k = 1, depth
           input_5d(i, j, k, 1, 1) = real(i * 100 + j * 10 + k, real32)
        end do
     end do
  end do

  call pad3d_layer%forward(input_5d)
  call pad3d_layer%get_output(output_5d)

  ! Check that the center part matches the input
  expected_width = width + 2
  expected_height = height + 4
  expected_depth = depth + 0
  if (any(abs(output_5d(2:expected_width-1, 3:expected_height-2, &
                        1:expected_depth, 1, 1) - &
              input_5d(:, :, :, 1, 1)) .gt. tol)) then
     success = .false.
     write(0,*) 'pad3d layer asymmetric padding center incorrect'
  end if

  deallocate(input_5d, output_5d)


!!!-----------------------------------------------------------------------------
!!! Test batch size modification
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing batch size modification..."

  pad3d_layer = pad3d_layer_type( &
       padding = [1, 1, 1], &
       method = "zero", &
       input_shape = [width, height, depth, channels] &
  )

  call pad3d_layer%set_batch_size(batch_size)

  if (pad3d_layer%batch_size .ne. batch_size) then
     success = .false.
     write(0,*) 'pad3d layer set_batch_size failed'
  end if


!!!-----------------------------------------------------------------------------
!!! Test file I/O operations
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(unit=unit_num, file='test_pad3d_layer.tmp', &
       status='replace', action='write')
  
  ! Write layer to file
  write(unit_num,'("PAD3D")')
  call pad3d_layer%print_to_unit(unit_num)
  write(unit_num,'("END PAD3D")')
  close(unit_num)

  ! Read layer from file
  open(unit=unit_num, file='test_pad3d_layer.tmp', &
       status='old', action='read')
  read(unit_num,*) ! Skip first line
  read_layer = read_pad3d_layer(unit_num)
  close(unit_num)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (pad3d_layer_type)
     if (.not. read_layer%name .eq. 'pad3d') then
        success = .false.
        write(0,*) 'read pad3d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not pad3d_layer_type'
  end select

  ! Clean up temporary file
  open(unit=unit_num, file='test_pad3d_layer.tmp', status='old')
  close(unit_num, status='delete')


!!!-----------------------------------------------------------------------------
!!! Test edge cases
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing edge cases..."

  ! Test with minimal dimensions
  pad3d_layer = pad3d_layer_type( &
       padding = [0, 0, 0], &
       method = "zero", &
       input_shape = [1, 1, 1, 1], &
       batch_size = 1 &
  )

  allocate(input_5d(1, 1, 1, 1, 1))
  input_5d = 42.0_real32

  call pad3d_layer%forward(input_5d)
  call pad3d_layer%get_output(output_5d)

  if (any(abs(output_5d - 42.0_real32) .gt. tol)) then
     success = .false.
     write(0,*) 'pad3d layer edge case (no padding) incorrect'
  end if

  deallocate(input_5d, output_5d)


!!!-----------------------------------------------------------------------------
!!! Check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if (success) then
     write(*,*) 'test_pad3d_layer passed all tests'
  else
     write(0,*) 'test_pad3d_layer failed one or more tests'
     stop 1
  end if

end program test_pad3d_layer
