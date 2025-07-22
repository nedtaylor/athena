program test_pad2d_layer
  !! Unit tests for the pad2d layer module
  use athena__constants, only: real32
  use athena__pad2d_layer, only: pad2d_layer_type, read_pad2d_layer
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: array4d_type
  implicit none

  type(pad2d_layer_type) :: pad2d_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: batch_size = 2
  integer, parameter :: width = 4
  integer, parameter :: height = 3
  integer, parameter :: channels = 2
  logical :: success = .true.
  real(real32), parameter :: tol = 1.0e-6_real32

  ! Test data
  real(real32), allocatable, dimension(:,:) :: input_2d, output_2d
  real(real32), allocatable, dimension(:,:) :: gradient_2d
  real(real32), allocatable, dimension(:,:,:,:) :: input_4d, output_4d
  real(real32), allocatable, dimension(:,:,:,:) :: gradient_4d

  integer :: i, j, k, l
  integer :: unit
  integer :: expected_width, expected_height

  ! Random seed setup
  integer :: seed_size
  integer, allocatable, dimension(:) :: seed


!-------------------------------------------------------------------------------
! Initialise random number generator with a seed
!-------------------------------------------------------------------------------
  call random_seed(size = seed_size)
  allocate(seed(seed_size), source=42)
  call random_seed(put = seed)


!-------------------------------------------------------------------------------
! Test 2D padding layer setup with zero padding
!-------------------------------------------------------------------------------
  write(*,*) "Testing 2D padding layer setup with zero padding..."
  
  pad2d_layer = pad2d_layer_type( &
       padding = [1, 2], &
       method = "zero", &
       input_shape = [width, height, channels], &
       batch_size = batch_size &
  )

  ! Check layer properties
  if (.not. pad2d_layer%name .eq. 'pad2d') then
     success = .false.
     write(0,*) 'pad2d layer has wrong name'
  end if

  if (.not. pad2d_layer%type .eq. 'pad') then
     success = .false.
     write(0,*) 'pad2d layer has wrong type'
  end if

  if (any(pad2d_layer%input_shape .ne. [width, height, channels])) then
     success = .false.
     write(0,*) 'pad2d layer has wrong input_shape'
  end if

  expected_width = width + 2 * 1   ! padding[1] on both sides
  expected_height = height + 2 * 2 ! padding[2] on both sides
  if (any(pad2d_layer%output_shape .ne. &
          [expected_width, expected_height, channels])) then
     success = .false.
     write(0,*) 'pad2d layer has wrong output_shape'
     write(0,*) 'Expected:', [expected_width, expected_height, channels]
     write(0,*) 'Got:', pad2d_layer%output_shape
  end if

  if (pad2d_layer%input_rank .ne. 3) then
     success = .false.
     write(0,*) 'pad2d layer has wrong input_rank'
  end if

  if (pad2d_layer%output_rank .ne. 3) then
     success = .false.
     write(0,*) 'pad2d layer has wrong output_rank'
  end if

  if (pad2d_layer%batch_size .ne. batch_size) then
     success = .false.
     write(0,*) 'pad2d layer has wrong batch_size'
  end if


!-------------------------------------------------------------------------------
! Test 2D input forward pass with zero padding
!-------------------------------------------------------------------------------
  write(*,*) "Testing 2D input forward pass with zero padding..."

  ! Initialize test input (flattened 3D to 1D for simplicity)
  allocate(input_2d(width * height * channels, batch_size))
  do i = 1, width * height
     input_2d(i,1) = real(i)
     input_2d(i,2) = real(i+width*height)
  end do
  ! Replicate for each channel
  do i = 2, channels
     input_2d(width * height * (i - 1) + 1:width * height * i, 1) = &
          input_2d(:width*height,1) + (i - 1) * width * height
     input_2d(width * height * (i - 1) + 1:width * height * i, 2) = &
          input_2d(:width*height,2) + (i - 1) * width * height
  end do

  ! Run forward pass
  call pad2d_layer%forward(input_2d)
  call pad2d_layer%get_output(output_2d)

  ! Check output dimensions
  if (size(output_2d, 1) .ne. expected_width * expected_height * channels .or. &
      size(output_2d, 2) .ne. batch_size) then
     success = .false.
     write(0,*) 'pad2d layer forward output has wrong dimensions'
     write(0,*) 'Expected shape:', &
          [expected_width * expected_height * channels, batch_size]
     write(0,*) 'Got shape:', shape(output_2d)
  end if

  deallocate(input_2d, output_2d)


!-------------------------------------------------------------------------------
! Test 4D input forward pass with zero padding
!-------------------------------------------------------------------------------
  write(*,*) "Testing 4D input forward pass with zero padding..."

  ! Initialize test input
  allocate(input_4d(width, height, channels, batch_size))
  call random_number(input_4d)

  ! Run forward pass
  call pad2d_layer%forward(input_4d)
  call pad2d_layer%get_output(output_4d)

  ! Check output dimensions
  if (size(output_4d, 1) .ne. expected_width .or. &
      size(output_4d, 2) .ne. expected_height .or. &
      size(output_4d, 3) .ne. channels .or. &
      size(output_4d, 4) .ne. batch_size) then
     success = .false.
     write(0,*) 'pad2d layer 4D forward output has wrong dimensions'
     write(0,*) 'Expected shape:', &
          [expected_width, expected_height, channels, batch_size]
     write(0,*) 'Got shape:', shape(output_4d)
  end if

  ! Check zero padding on width dimension (first and last columns)
  if (any(abs(output_4d(1,:,:,:)) .gt. tol) .or. &
      any(abs(output_4d(expected_width,:,:,:)) .gt. tol)) then
     success = .false.
     write(0,*) 'pad2d layer width zero padding incorrect'
  end if

  ! Check zero padding on height dimension (first 2 and last 2 rows)
  if (any(abs(output_4d(:,1:2,:,:)) .gt. tol) .or. &
      any(abs(output_4d(:,expected_height-1:expected_height,:,:)) &
          .gt. tol)) then
     success = .false.
     write(0,*) 'pad2d layer height zero padding incorrect'
  end if

  ! Check that middle elements match input
  if (any(abs(output_4d(2:expected_width-1, 3:expected_height-2, :, :) - &
              input_4d) .gt. tol)) then
     success = .false.
     write(0,*) 'pad2d layer 4D forward pass incorrect for middle elements'
  end if


!-------------------------------------------------------------------------------
! Test backward pass
!-------------------------------------------------------------------------------
  write(*,*) "Testing backward pass..."

  ! Initialize gradient
  allocate(gradient_4d(expected_width, expected_height, channels, batch_size))
  gradient_4d = 1.0_real32

  ! Run backward pass
  call pad2d_layer%backward(input_4d, gradient_4d)

  ! Check that gradient is correctly trimmed back to input size
  select type(di => pad2d_layer%di(1,1))
  type is (array4d_type)
     if (any(shape(di%val_ptr) .ne. [width, height, channels, batch_size])) then
        success = .false.
        write(0,*) 'pad2d layer backward gradient has wrong dimensions'
        write(0,*) 'Expected shape:', [width, height, channels, batch_size]
        write(0,*) 'Got shape:', di%shape
     end if

     ! For zero padding, gradient in the middle should equal input gradient
     if (any(abs(di%val_ptr - gradient_4d(2:expected_width-1, &
                                     3:expected_height-2, :, :)) .gt. tol)) then
        success = .false.
        write(0,*) 'pad2d layer backward pass incorrect'
     end if
  class default
     success = .false.
     write(0,*) 'pad2d layer has not set di type correctly'
  end select

  deallocate(input_4d, output_4d, gradient_4d)


!-------------------------------------------------------------------------------
! Test different padding methods
!-------------------------------------------------------------------------------
  write(*,*) "Testing different padding methods..."

  test_methods_block: block
  character(20), dimension(4) :: padding_methods = [ &
       "zero      ", "same      ", "valid     ", "full      " ]

  do i = 1, size(padding_methods)
     pad2d_layer = pad2d_layer_type( &
          padding = [1, 1], &
          method = trim(padding_methods(i)), &
          input_shape = [width, height, channels], &
          batch_size = 1 &
     )

     ! Test forward pass doesn't crash
     allocate(input_4d(width, height, 1, 1))
     call random_number(input_4d)
     
     call pad2d_layer%forward(input_4d)
     call pad2d_layer%get_output(output_4d)

     if (.not. allocated(output_4d)) then
        success = .false.
        write(0,*) 'output not allocated for padding method: ', &
             trim(padding_methods(i))
     end if

     deallocate(input_4d, output_4d)
  end do
  end block test_methods_block


!!!-----------------------------------------------------------------------------
! Test comprehensive padding method functionality
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing comprehensive padding method functionality..."

  comprehensive_methods_block: block
    real(real32), allocatable, dimension(:,:,:,:) :: &
         input_simple, output_simple, output_expected, &
         input_back, output_back, gradient_out, gradient_expected
    integer, parameter :: simple_width = 4, simple_height = 4
    integer, parameter :: simple_channels = 1
    integer, parameter :: pad_h = 2, pad_w = 2
    
    allocate(input_back(simple_width, simple_height, simple_channels, 1))
    allocate(gradient_expected(simple_width, simple_height, simple_channels, 1))
    allocate(gradient_out(simple_width + 2 * pad_w, simple_height + 2 * pad_h, simple_channels, 1))
    input_back(:,:,1,1) = reshape( &
         [1.0_real32, 5.0_real32,  9.0_real32, 13.0_real32, &
          2.0_real32, 6.0_real32, 10.0_real32, 14.0_real32, &
          3.0_real32, 7.0_real32, 11.0_real32, 15.0_real32, &
          4.0_real32, 8.0_real32, 12.0_real32, 16.0_real32], &
         [simple_width, simple_height])
    gradient_out(:,:,1,1) = reshape( &
         [0.1_real32, 0.2_real32, 0.3_real32, 0.4_real32, &
          0.5_real32, 0.6_real32, 0.7_real32, 0.8_real32, &
          0.9_real32, 1.0_real32, 1.1_real32, 1.2_real32, &
          1.3_real32, 1.4_real32, 1.5_real32, 1.6_real32, &
          1.7_real32, 1.8_real32, 1.9_real32, 2.0_real32, &
          2.1_real32, 2.2_real32, 2.3_real32, 2.4_real32, &
          2.5_real32, 2.6_real32, 2.7_real32, 2.8_real32, &
          2.9_real32, 3.0_real32, 3.1_real32, 3.2_real32, &
          3.3_real32, 3.4_real32, 3.5_real32, 3.6_real32, &
          3.7_real32, 3.8_real32, 3.9_real32, 4.0_real32, &
          4.1_real32, 4.2_real32, 4.3_real32, 4.4_real32, &
          4.5_real32, 4.6_real32, 4.7_real32, 4.8_real32, &
          4.9_real32, 5.0_real32, 5.1_real32, 5.2_real32, &
          5.3_real32, 5.4_real32, 5.5_real32, 5.6_real32, &
          5.7_real32, 5.8_real32, 5.9_real32, 6.0_real32, &
          6.1_real32, 6.2_real32, 6.3_real32, 6.4_real32 ], &
          [simple_width + 2 * pad_w, simple_height + 2 * pad_h])
    allocate(output_expected(simple_width + 2 * pad_w, simple_height + 2 * pad_h, simple_channels, 1))

    ! Create simple test data: 4x4 matrix  
    allocate(input_simple(simple_width, simple_height, simple_channels, 1))
    input_simple(:,:,1,1) = reshape( &
         [1.0_real32, 5.0_real32,  9.0_real32, 13.0_real32, &
          2.0_real32, 6.0_real32, 10.0_real32, 14.0_real32, &
          3.0_real32, 7.0_real32, 11.0_real32, 15.0_real32, &
          4.0_real32, 8.0_real32, 12.0_real32, 16.0_real32], &
         [simple_width, simple_height])



    ! Test zero/constant padding
    write(*,*) "  Testing zero/constant padding..."
    pad2d_layer = pad2d_layer_type( &
         padding = [pad_w, pad_h], &
         method = "zero", &
         input_shape = [simple_width, simple_height, simple_channels], &
         batch_size = 1 &
    )
    call pad2d_layer%forward(input_simple)
    call pad2d_layer%get_output(output_simple)

    output_expected = 0._real32
    output_expected(3:6,3:6,1,1) = input_simple(:,:,1,1)
    if(any(abs(output_simple - output_expected) .gt. tol))then
       success = .false.
       write(0,*) 'Zero padding method failed'
       write(0,*) 'Expected: ', output_expected(:,:,1,1)
       write(0,*) 'Got:', output_simple(:,:,1,1)
    end if

    gradient_expected = 0._real32
    gradient_expected(:,:,1,1) = gradient_out(3:6,3:6,1,1)
    call pad2d_layer%backward(input_back, gradient_out)
    select type(di => pad2d_layer%di(1,1))
    type is(array4d_type)
       ! For zero padding, gradients should just be extracted from middle
       if(any(abs(di%val_ptr - gradient_expected) .gt. tol))then
          success = .false.
          write(0,*) 'Zero padding backward pass failed'
          write(0,*) 'Expected: ', gradient_expected(:,:,1,1)
          write(0,*) 'Got:', di%val_ptr(:,:,1,1)
       end if
    end select
    deallocate(output_simple)

    ! Test replication/replicate padding  
    write(*,*) "  Testing replication padding..."
    pad2d_layer = pad2d_layer_type( &
         padding = [pad_w, pad_h], &
         method = "replicate", &
         input_shape = [simple_width, simple_height, simple_channels], &
         batch_size = 1 &
    )
    call pad2d_layer%forward(input_simple)
    call pad2d_layer%get_output(output_simple)
    
    output_expected(3:6,3:6,1,1) = input_simple(:,:,1,1)
    output_expected(1:2,1:2,1,1) = input_simple(1,1,1,1)
    output_expected(7:8,7:8,1,1) = input_simple(simple_width,simple_height,1,1)
    if(any(abs(output_simple - output_expected) .gt. tol))then
       success = .false.
       write(0,*) 'Replication padding method failed'
       write(0,*) 'Expected: ', output_expected(:,:,1,1)
       write(0,*) 'Got:', output_simple(:,:,1,1)
    end if

    call pad2d_layer%backward(input_simple, gradient_out)
    gradient_expected(:,:,1,1) = gradient_out(3:6,3:6,1,1)
    gradient_expected(1,1,1,1) = gradient_expected(1,1,1,1) + sum(gradient_out(1:2,1:2,1,1))
    gradient_expected(1,4,1,1) = gradient_expected(1,4,1,1) + sum(gradient_out(1:2,7:8,1,1))
    gradient_expected(4,1,1,1) = gradient_expected(4,1,1,1) + sum(gradient_out(7:8,1:2,1,1))
    gradient_expected(4,4,1,1) = gradient_expected(4,4,1,1) + sum(gradient_out(7:8,7:8,1,1))
    gradient_expected(1,:,1,1) = gradient_expected(1,:,1,1) + sum(gradient_out(1:2,3:6,1,1),dim=1)
    gradient_expected(4,:,1,1) = gradient_expected(4,:,1,1) + sum(gradient_out(7:8,3:6,1,1),dim=1)
    gradient_expected(:,1,1,1) = gradient_expected(:,1,1,1) + sum(gradient_out(3:6,1:2,1,1),dim=2)
    gradient_expected(:,4,1,1) = gradient_expected(:,4,1,1) + sum(gradient_out(3:6,7:8,1,1),dim=2)
    select type(di => pad2d_layer%di(1,1))
    type is(array4d_type)
       if(any(abs(di%val_ptr - gradient_expected) .gt. tol))then
          success = .false.
          write(0,*) 'Replication padding backward pass failed'
          write(0,*) 'Expected: ', gradient_expected(:,1,1,1)
          write(0,*) 'Got:', di%val_ptr(:,1,1,1)
       end if
    end select
    deallocate(output_simple, gradient_expected)
    
    ! Test reflection padding
    write(*,*) "  Testing reflection padding..."
    pad2d_layer = pad2d_layer_type( &
         padding = [pad_w, pad_h], &
         method = "reflect", &
         input_shape = [simple_width, simple_height, simple_channels], &
         batch_size = 1 &
    )
    call pad2d_layer%forward(input_simple)
    call pad2d_layer%get_output(output_simple)
    
    output_expected(3:6,3:6,1,1) = input_simple(:,:,1,1)
    output_expected(2:1:-1,2:1:-1,1,1) = input_simple(2:3,2:3,1,1)
    output_expected(7:8,7:8,1,1) = input_simple(3:2:-1,3:2:-1,1,1)
    if(any(abs(output_simple - output_expected) .gt. tol))then
       success = .false.
       write(0,*) 'Reflection padding method failed'
       write(0,*) 'Expected: ', output_expected(:,:,1,1)
       write(0,*) 'Got:', output_simple(:,:,1,1)
    end if

    allocate(gradient_expected(simple_width, simple_height, simple_channels, 1))
    call pad2d_layer%backward(input_simple, gradient_out)
    gradient_expected(:,:,1,1) = gradient_out(3:6,3:6,1,1)
    gradient_expected(2:3,2:3,1,1) = gradient_expected(2:3,2:3,1,1) + gradient_out(2:1:-1,2:1:-1,1,1)
    gradient_expected(3:2:-1,3:2:-1,1,1) = gradient_expected(3:2:-1,3:2:-1,1,1) + gradient_out(7:8,7:8,1,1)
    gradient_expected(2:3,3:2:-1,1,1) = gradient_expected(2:3,3:2:-1,1,1) + gradient_out(2:1:-1,7:8,1,1)
    gradient_expected(3:2:-1,2:3,1,1) = gradient_expected(3:2:-1,2:3,1,1) + gradient_out(7:8,2:1:-1,1,1)

    gradient_expected(2:3,:,1,1) = gradient_expected(2:3,:,1,1) + gradient_out(2:1:-1,3:6,1,1)
    gradient_expected(3:2:-1,:,1,1) = gradient_expected(3:2:-1,:,1,1) + gradient_out(7:8,3:6,1,1)
    gradient_expected(:,2:3,1,1) = gradient_expected(:,2:3,1,1) + gradient_out(3:6,2:1:-1,1,1)
    gradient_expected(:,3:2:-1,1,1) = gradient_expected(:,3:2:-1,1,1) + gradient_out(3:6,7:8,1,1)
    select type(di => pad2d_layer%di(1,1))
    type is(array4d_type)
       if(any(abs(di%val_ptr - gradient_expected) .gt. tol))then
          success = .false.
          write(0,*) 'Reflection padding backward pass failed'
          write(0,*) 'Expected: ', gradient_expected(:,:,1,1)
          write(0,*) 'Got:', di%val_ptr(:,:,1,1)
       end if
    end select
    deallocate(output_simple, gradient_expected)
    

    deallocate(input_simple)
  end block comprehensive_methods_block



!-------------------------------------------------------------------------------
! Test different padding sizes
!-------------------------------------------------------------------------------
  write(*,*) "Testing different padding sizes..."

  test_sizes_block: block
  integer, dimension(3, 2) :: test_paddings
  test_paddings(1,:) = [0, 0]
  test_paddings(2,:) = [1, 2]
  test_paddings(3,:) = [2, 1]

  do i = 1, size(test_paddings, 1)
     pad2d_layer = pad2d_layer_type( &
          padding = test_paddings(i,:), &
          method = "zero", &
          input_shape = [width, height, channels], &
          batch_size = 1 &
     )

     expected_width = width + 2 * test_paddings(i,1)
     expected_height = height + 2 * test_paddings(i,2)

     ! Check output shape
     if (any(pad2d_layer%output_shape .ne. &
             [expected_width, expected_height, channels])) then
        success = .false.
        write(0,*) 'pad2d layer output shape incorrect for padding size:', &
             test_paddings(i,:)
        write(0,*) 'Expected:', [expected_width, expected_height, channels]
        write(0,*) 'Got:', pad2d_layer%output_shape
     end if

     ! Test forward pass
     allocate(input_4d(width, height, 1, 1))
     input_4d = 1.0_real32
     
     call pad2d_layer%forward(input_4d)
     call pad2d_layer%get_output(output_4d)

     ! For zero padding, check that padding is actually zero
     if (test_paddings(i,1) > 0) then
        if (any(abs(output_4d(1:test_paddings(i,1),:,:,:)) .gt. tol) .or. &
            any(abs(output_4d(expected_width-test_paddings(i,1)+1: &
                              expected_width,:,:,:)) .gt. tol)) then
           success = .false.
           write(0,*) 'pad2d layer width zero padding incorrect for size:', &
                test_paddings(i,1)
        end if
     end if

     if (test_paddings(i,2) > 0) then
        if (any(abs(output_4d(:,1:test_paddings(i,2),:,:)) .gt. tol) .or. &
            any(abs(output_4d(:,expected_height-test_paddings(i,2)+1: &
                              expected_height,:,:)) .gt. tol)) then
           success = .false.
           write(0,*) 'pad2d layer height zero padding incorrect for size:', &
                test_paddings(i,2)
        end if
     end if

     deallocate(input_4d, output_4d)
  end do
  end block test_sizes_block


!-------------------------------------------------------------------------------
! Test symmetric padding
!-------------------------------------------------------------------------------
  write(*,*) "Testing symmetric padding..."

  pad2d_layer = pad2d_layer_type( &
       padding = [1, 1], &
       method = "zero", &
       input_shape = [width, height, channels], &
       batch_size = 1 &
  )

  ! Test with a known input pattern
  allocate(input_4d(width, height, 1, 1))
  do i = 1, width
     do j = 1, height
        input_4d(i, j, 1, 1) = real(i * 10 + j, real32)
     end do
  end do

  call pad2d_layer%forward(input_4d)
  call pad2d_layer%get_output(output_4d)

  ! Check that the center part matches the input
  expected_width = width + 2
  expected_height = height + 2
  if (any(abs(output_4d(2:expected_width-1, 2:expected_height-1, 1, 1) - &
              input_4d(:, :, 1, 1)) .gt. tol)) then
     success = .false.
     write(0,*) 'pad2d layer symmetric padding center incorrect'
  end if

  deallocate(input_4d, output_4d)


!-------------------------------------------------------------------------------
! Test batch size modification
!-------------------------------------------------------------------------------
  write(*,*) "Testing batch size modification..."

  pad2d_layer = pad2d_layer_type( &
       padding = [1, 1], &
       method = "zero", &
       input_shape = [width, height, channels] &
  )

  call pad2d_layer%set_batch_size(batch_size)

  if (pad2d_layer%batch_size .ne. batch_size) then
     success = .false.
     write(0,*) 'pad2d layer set_batch_size failed'
  end if


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_pad2d_layer.tmp', &
       status='replace', action='write')
  
  ! Write layer to file
  write(unit,'("PAD2D")')
  call pad2d_layer%print_to_unit(unit)
  write(unit,'("END PAD2D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_pad2d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_pad2d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (pad2d_layer_type)
     if (.not. read_layer%name .eq. 'pad2d') then
        success = .false.
        write(0,*) 'read pad2d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not pad2d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_pad2d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if (success) then
     write(*,*) 'test_pad2d_layer passed all tests'
  else
     write(0,*) 'test_pad2d_layer failed one or more tests'
     stop 1
  end if

end program test_pad2d_layer
