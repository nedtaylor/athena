program test_pad1d_layer
  !! Unit tests for the pad1d layer module
  use athena__constants, only: real32
  use athena__pad1d_layer, only: pad1d_layer_type, read_pad1d_layer
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: array3d_type
  implicit none

  type(pad1d_layer_type) :: pad1d_layer
  class(base_layer_type), allocatable :: read_layer
  integer, parameter :: batch_size = 2
  integer, parameter :: width = 6
  integer, parameter :: channels = 3
  logical :: success = .true.
  real(real32), parameter :: tol = 1.0e-6_real32

  ! Test data
  real(real32), allocatable, dimension(:,:) :: input_2d, output_2d
  real(real32), allocatable, dimension(:,:) :: gradient_2d
  real(real32), allocatable, dimension(:,:,:) :: input_3d, output_3d
  real(real32), allocatable, dimension(:,:,:) :: gradient_3d

  integer :: i, j, k
  integer :: unit_num = 10
  integer :: expected_width

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
!!! Test 1D padding layer setup with zero padding
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing 1D padding layer setup with zero padding..."
  
  pad1d_layer = pad1d_layer_type( &
       padding = [2], &
       method = "zero", &
       input_shape = [width, channels], &
       batch_size = batch_size &
  )

  ! Check layer properties
  if (.not. pad1d_layer%name .eq. 'pad1d') then
     success = .false.
     write(0,*) 'pad1d layer has wrong name'
  end if

  if (.not. pad1d_layer%type .eq. 'pad') then
     success = .false.
     write(0,*) 'pad1d layer has wrong type'
  end if

  if (any(pad1d_layer%input_shape .ne. [width])) then
     success = .false.
     write(0,*) 'pad1d layer has wrong input_shape'
  end if

  expected_width = width + 2 * 2 ! padding on both sides
  if (any(pad1d_layer%output_shape .ne. [expected_width, channels])) then
     success = .false.
     write(0,*) 'pad1d layer has wrong output_shape'
     write(0,*) 'Expected:', [expected_width, channels]
     write(0,*) 'Got:', pad1d_layer%output_shape
  end if

  if (pad1d_layer%input_rank .ne. 2) then
     success = .false.
     write(0,*) 'pad1d layer has wrong input_rank'
  end if

  if (pad1d_layer%output_rank .ne. 2) then
     success = .false.
     write(0,*) 'pad1d layer has wrong output_rank'
  end if

  if (pad1d_layer%batch_size .ne. batch_size) then
     success = .false.
     write(0,*) 'pad1d layer has wrong batch_size'
  end if


!!!-----------------------------------------------------------------------------
!!! Test 2D input forward pass with zero padding
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing 2D input forward pass with zero padding..."

  ! Initialize test input
  allocate(input_2d(width * channels, batch_size))
  input_2d(:width,1) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  input_2d(:width,2) = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  do i = 2, channels, 1
     input_2d(width * (i - 1) + 1:width * i, 1) = input_2d(:width,1) + (i - 1) * width
     input_2d(width * (i - 1) + 1:width * i, 2) = input_2d(:width,2) + (i - 1) * width
  end do

  ! Run forward pass
  call pad1d_layer%forward(input_2d)
  call pad1d_layer%get_output(output_2d)

  ! Check output dimensions
  if (size(output_2d, 1) .ne. expected_width * channels .or. &
      size(output_2d, 2) .ne. batch_size) then
     success = .false.
     write(0,*) 'pad1d layer forward output has wrong dimensions'
     write(0,*) 'Expected shape:', [expected_width, batch_size]
     write(0,*) 'Got shape:', shape(output_2d)
  end if

  ! Check zero padding (first 2 and last 2 elements should be zero)
  if (any(abs(output_2d(1:2,:)) .gt. tol) .or. &
      any(abs(output_2d(expected_width-1:expected_width,:)) .gt. tol)) then
     success = .false.
     write(0,*) 'pad1d layer zero padding incorrect'
     write(0,*) 'First elements:', output_2d(1:2,1)
     write(0,*) 'Last elements:', output_2d(expected_width-1:expected_width,1)
  end if

  ! Check that middle elements match input
  if (any(abs(output_2d(3:8,1) - input_2d(:width,1)) .gt. tol) .or. &
      any(abs(output_2d(3:8,2) - input_2d(:width,2)) .gt. tol)) then
     success = .false.
     write(0,*) 'pad1d layer forward pass incorrect for middle elements'
     write(0,*) 'Expected middle:', input_2d(:,1)
     write(0,*) 'Got middle:', output_2d(3:8,1)
  end if

  deallocate(input_2d, output_2d)


!!!-----------------------------------------------------------------------------
!!! Test 3D input forward pass with zero padding
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing 3D input forward pass with zero padding..."

  ! Initialize test input
  allocate(input_3d(width, channels, batch_size))
  call random_number(input_3d)

  ! Run forward pass
  call pad1d_layer%forward(input_3d)
  call pad1d_layer%get_output(output_3d)

  ! Check output dimensions
  if (size(output_3d, 1) .ne. expected_width .or. &
      size(output_3d, 2) .ne. channels .or. &
      size(output_3d, 3) .ne. batch_size) then
     success = .false.
     write(0,*) 'pad1d layer 3D forward output has wrong dimensions'
  end if

  ! Check zero padding
  if (any(abs(output_3d(1:2,:,:)) .gt. tol) .or. &
      any(abs(output_3d(expected_width-1:expected_width,:,:)) .gt. tol)) then
     success = .false.
     write(0,*) 'pad1d layer 3D zero padding incorrect'
  end if

  ! Check that middle elements match input
  if (any(abs(output_3d(3:8,:,:) - input_3d) .gt. tol)) then
     success = .false.
     write(0,*) 'pad1d layer 3D forward pass incorrect for middle elements'
  end if


!!!-----------------------------------------------------------------------------
!!! Test backward pass
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing backward pass..."

  ! Initialize gradient
  allocate(gradient_3d(expected_width, channels, batch_size))
  gradient_3d = 1.0_real32

  ! Run backward pass
  call pad1d_layer%backward(input_3d, gradient_3d)

  ! Check that gradient is correctly trimmed back to input size
  select type(di => pad1d_layer%di(1,1))
  type is (array3d_type)
     if (any(shape(di%val_ptr) .ne. [width, channels, batch_size])) then
        success = .false.
        write(0,*) 'pad1d layer backward gradient has wrong dimensions'
        write(0,*) 'Expected shape:', [width, channels, batch_size]
        write(0,*) 'Got shape:', shape(di%val_ptr)
     end if

     ! For zero padding, gradient in the middle should equal input gradient
     if (any(abs(di%val_ptr - gradient_3d(3:8,:,:)) .gt. tol)) then
        success = .false.
        write(0,*) 'pad1d layer backward pass incorrect'
     end if
  class default
     success = .false.
     write(0,*) 'pad1d layer has not set di type correctly'
  end select

  deallocate(input_3d, output_3d, gradient_3d)


!!!-----------------------------------------------------------------------------
!!! Test different padding methods
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing different padding methods..."

  test_methods_block: block
  character(20), dimension(4) :: padding_methods = [ &
       "zero      ", "same      ", "valid     ", "full      " ]
  integer :: expected_widths(4)
  integer :: j

  expected_widths(1) = width + 4  ! zero padding: +2 on each side
  expected_widths(2) = width + 4  ! same padding: +2 on each side
  expected_widths(3) = width      ! valid padding: no padding
  expected_widths(4) = width + 4  ! full padding: +2 on each side

  do i = 1, size(padding_methods)
     pad1d_layer = pad1d_layer_type( &
          padding = [2], &
          method = trim(padding_methods(i)), &
          input_shape = [width, channels], &
          batch_size = 1 &
     )

     ! Check output shape for each method
     if (padding_methods(i) .ne. "valid      ") then
        if (any(pad1d_layer%output_shape .ne. [expected_widths(i)])) then
           success = .false.
           write(0,*) 'pad1d layer output shape incorrect for method: ', &
                trim(padding_methods(i))
           write(0,*) 'Expected:', [expected_widths(i)]
           write(0,*) 'Got:', pad1d_layer%output_shape
        end if
     end if

     ! Test forward pass doesn't crash
     allocate(input_2d(width * channels, 1))
     input_2d(:width,1) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
     do j = 2, channels, 1
        input_2d(width * (j - 1) + 1:width * j, 1) = input_2d(:width,1) + (j - 1) * width
     end do
     
     call pad1d_layer%forward(input_2d)
     call pad1d_layer%get_output(output_2d)

     if (.not. allocated(output_2d)) then
        success = .false.
        write(0,*) 'output not allocated for padding method: ', &
             trim(padding_methods(i))
     end if

     deallocate(input_2d, output_2d)
  end do
  end block test_methods_block


!!!-----------------------------------------------------------------------------
!!! Test different padding sizes
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing different padding sizes..."

  test_sizes_block: block
  integer, dimension(3) :: test_paddings = [0, 1, 3]
  integer :: expected_widths_pad(3)

  expected_widths_pad = width + 2 * test_paddings

  do i = 1, size(test_paddings)
     pad1d_layer = pad1d_layer_type( &
          padding = [test_paddings(i)], &
          method = "zero", &
          input_shape = [width, channels], &
          batch_size = 1 &
     )

     ! Check output shape
     if (any(pad1d_layer%output_shape .ne. [expected_widths_pad(i)])) then
        success = .false.
        write(0,*) 'pad1d layer output shape incorrect for padding size:', &
             test_paddings(i)
        write(0,*) 'Expected:', [expected_widths_pad(i)]
        write(0,*) 'Got:', pad1d_layer%output_shape
     end if

     ! Test forward pass
     allocate(input_2d(width * channels, 1))
     input_2d = 1.0_real32
     
     call pad1d_layer%forward(input_2d)
     call pad1d_layer%get_output(output_2d)

     ! For zero padding, check that padding is actually zero
     if (test_paddings(i) > 0) then
        if (any(abs(output_2d(1:test_paddings(i),:)) .gt. tol) .or. &
            any(abs(output_2d(expected_widths_pad(i)-test_paddings(i)+1: &
                              expected_widths_pad(i),:)) .gt. tol)) then
           success = .false.
           write(0,*) 'pad1d layer zero padding incorrect for size:', &
                test_paddings(i)
        end if
     end if

     deallocate(input_2d, output_2d)
  end do
  end block test_sizes_block


!!!-----------------------------------------------------------------------------
!!! Test batch size modification
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing batch size modification..."

  pad1d_layer = pad1d_layer_type( &
       padding = [1], &
       method = "zero", &
       input_shape = [width, channels] &
  )

  call pad1d_layer%set_batch_size(batch_size)

  if (pad1d_layer%batch_size .ne. batch_size) then
     success = .false.
     write(0,*) 'pad1d layer set_batch_size failed'
  end if


!!!-----------------------------------------------------------------------------
!!! Test file I/O operations
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(unit=unit_num, file='test_pad1d_layer.tmp', &
       status='replace', action='write')
  
  ! Write layer to file
  write(unit_num,'("PAD1D")')
  call pad1d_layer%print_to_unit(unit_num)
  write(unit_num,'("END PAD1D")')
  close(unit_num)

  ! Read layer from file
  open(unit=unit_num, file='test_pad1d_layer.tmp', &
       status='old', action='read')
  read(unit_num,*) ! Skip first line
  read_layer = read_pad1d_layer(unit_num)
  close(unit_num)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (pad1d_layer_type)
     if (.not. read_layer%name .eq. 'pad1d') then
        success = .false.
        write(0,*) 'read pad1d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not pad1d_layer_type'
  end select

  ! Clean up temporary file
  open(unit=unit_num, file='test_pad1d_layer.tmp', status='old')
  close(unit_num, status='delete')


!!!-----------------------------------------------------------------------------
!!! Check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if (success) then
     write(*,*) 'test_pad1d_layer passed all tests'
  else
     write(0,*) 'test_pad1d_layer failed one or more tests'
     stop 1
  end if

end program test_pad1d_layer
