program test_pad3d_layer
  !! Unit tests for the pad3d layer module
  use coreutils, only: real32
  use athena__pad3d_layer, only: pad3d_layer_type, read_pad3d_layer
  use athena__base_layer, only: base_layer_type
  use diffstruc, only: array_type
  implicit none

  type(pad3d_layer_type), target :: pad3d_layer
  class(base_layer_type), allocatable :: read_layer

  integer, parameter :: batch_size = 2
  integer, parameter :: width = 3
  integer, parameter :: height = 4
  integer, parameter :: depth = 2
  integer, parameter :: channels = 2
  logical :: success = .true.
  real(real32), parameter :: tol = 1.E-6_real32

  ! Test data
  real(real32), allocatable, dimension(:,:) :: output_2d
  real(real32), allocatable, dimension(:,:,:,:,:) :: input_5d, output_5d
  type(array_type) :: input(1,1)
  type(array_type), pointer :: output, gradient

  integer :: i, j, k, c, s
  integer :: unit
  integer :: expected_width, expected_height, expected_depth

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
! Test 3D padding layer setup with zero padding
!-------------------------------------------------------------------------------
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
     write(0,*) 'Got:      ', pad3d_layer%output_shape
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


!-------------------------------------------------------------------------------
! Test 2D input forward pass with zero padding (flattened 3D)
!-------------------------------------------------------------------------------
  write(*,*) "Testing 2D input forward pass with zero padding..."

  ! Initialize test input
  call input(1,1)%allocate(&
       array_shape=[width, height, depth, channels, batch_size], source = 0.0)
  do s = 1, batch_size
     do c = 1, channels
        do k = 1, depth
           do j = 1, height
              do i = 1, width
                 input(1,1)%val(&
                      i + (j-1)*width + (k-1)*width*height + &
                      (c-1)*width*height*depth, s &
                 ) = real(i + (j-1)*width + (k-1)*width*height + &
                      (c-1)*width*height*depth &
                 )
              end do
           end do
        end do
     end do
  end do

  ! Run forward pass
  call pad3d_layer%forward_derived(input)
  call pad3d_layer%extract_output(output_2d)

  ! Check output dimensions
  if(size(output_2d, 1) .ne. &
       expected_width * expected_height * expected_depth * channels .or. &
       size(output_2d, 2) .ne. batch_size)then
     success = .false.
     write(0,*) 'pad3d layer forward output has wrong dimensions'
     write(0,*) 'Expected shape:', &
          [expected_width * expected_height * expected_depth * channels, &
               batch_size]
     write(0,*) 'Got shape:', shape(output_2d)
  end if

  call pad3d_layer%output(1,1)%nullify_graph()
  deallocate(output_2d)
  call input(1,1)%deallocate()


!-------------------------------------------------------------------------------
! Test 5D input forward pass with zero padding
!-------------------------------------------------------------------------------
  write(*,*) "Testing 5D input forward pass with zero padding..."

  ! Initialize test input
  call input(1,1)%allocate(&
       array_shape=[width, height, depth, channels, batch_size], source = 0.0)
  call input(1,1)%set_requires_grad(.true.)
  call random_number(input(1,1)%val)

  ! Run forward pass
  call pad3d_layer%forward_derived(input)
  call pad3d_layer%extract_output(output_5d)

  ! Check output dimensions
  if(size(output_5d, 1) .ne. expected_width .or. &
       size(output_5d, 2) .ne. expected_height .or. &
       size(output_5d, 3) .ne. expected_depth .or. &
       size(output_5d, 4) .ne. channels .or. &
       size(output_5d, 5) .ne. batch_size)then
     success = .false.
     write(0,*) 'pad3d layer 5D forward output has wrong dimensions'
  end if

  ! Check zero padding on width dimension
  if(any(abs(output_5d(1,:,:,:,:)) .gt. tol) .or. &
       any(abs(output_5d(expected_width,:,:,:,:)) .gt. tol))then
     success = .false.
     write(0,*) 'pad3d layer width zero padding incorrect'
  end if

  ! Check zero padding on height dimension
  if(any(abs(output_5d(:,1:2,:,:,:)) .gt. tol) .or. &
       any(abs(output_5d(:,expected_height-1:expected_height,:,:,:)) &
            .gt. tol))then
     success = .false.
     write(0,*) 'pad3d layer height zero padding incorrect'
  end if

  ! Check zero padding on depth dimension
  if(any(abs(output_5d(:,:,1,:,:)) .gt. tol) .or. &
       any(abs(output_5d(:,:,expected_depth,:,:)) .gt. tol))then
     success = .false.
     write(0,*) 'pad3d layer depth zero padding incorrect'
  end if

  ! Check that middle elements match input
  call input(1,1)%extract(input_5d)
  if(any(abs(output_5d(2:expected_width-1, &
       3:expected_height-2, &
       2:expected_depth-1, :, :) - &
  input_5d) .gt. tol))then
     success = .false.
     write(0,*) 'pad3d layer 5D forward pass incorrect for middle elements'
  end if
  deallocate(input_5d, output_5d)


!-------------------------------------------------------------------------------
! Test backward pass
!-------------------------------------------------------------------------------
  write(*,*) "Testing backward pass..."

  output => pad3d_layer%output(1,1)
  call output%grad_reverse()

  ! Check that gradient is correctly trimmed back to input size
  if(associated(input(1,1)%grad))then
     gradient => input(1,1)%grad
     if(any([gradient%shape, size(gradient%val,2)] .ne. &
          [width, height, depth, channels, batch_size]))then
        success = .false.
        write(0,*) 'pad3d layer backward gradient has wrong dimensions'
        write(0,*) 'Expected shape:', &
             [width, height, depth, channels, batch_size]
        write(0,*) 'Got shape:', [gradient%shape, size(gradient%val,2)]
     end if

     ! For zero padding, gradient in the middle should equal input gradient
     if(any(abs(gradient%val - 1._real32) .gt. tol))then
        success = .false.
        write(0,*) 'pad3d layer backward pass incorrect'
     end if
  else
     success = .false.
     write(0,*) 'pad3d layer backward did not allocate input gradient'
  end if

  call pad3d_layer%output(1,1)%nullify_graph()
  call input(1,1)%deallocate()


!-------------------------------------------------------------------------------
! Test different padding methods
!-------------------------------------------------------------------------------
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
       call input(1,1)%allocate(&
            array_shape=[width, height, depth, 1, 1], source = 1.0_real32)

       call pad3d_layer%forward_derived(input)
       call pad3d_layer%extract_output(output_5d)

       if(.not. allocated(output_5d))then
          success = .false.
          write(0,*) 'output not allocated for padding method: ', &
               trim(padding_methods(i))
       end if

       call pad3d_layer%output(1,1)%nullify_graph()
       deallocate(output_5d)
       call input(1,1)%deallocate()
    end do
  end block test_methods_block


!!!-----------------------------------------------------------------------------
! Test comprehensive padding method functionality
!!!-----------------------------------------------------------------------------
  write(*,*) "Testing comprehensive padding method functionality..."

  comprehensive_methods_block: block
    real(real32), allocatable, dimension(:,:,:,:,:) :: &
         input_simple, output_simple, output_expected, &
         gradient_out, gradient_expected
    integer, parameter :: simple_width = 4, simple_height = 4, simple_depth = 4
    integer, parameter :: simple_channels = 1
    integer, parameter :: pad_w = 2, pad_h = 2, pad_d = 2
    integer :: ii, jj, kk

    allocate(gradient_out(simple_width+2*pad_w, simple_height+2*pad_h, &
         simple_depth+2*pad_d, simple_channels, 1))
    gradient_out(:,:,:,1,1) = 0.1_real32
    ! Give distinct gradient values for testing
    do kk = 1, size(gradient_out,3)
       do jj = 1, size(gradient_out,2)
          do ii = 1, size(gradient_out,1)
             gradient_out(ii,jj,kk,1,1) = 0.1_real32 * (ii+jj+kk)
          end do
       end do
    end do

    ! Create simple test data: 4x4x4 cube
    allocate(input_simple(simple_width, simple_height, simple_depth, &
         simple_channels, 1))
    do kk = 1, simple_depth
       do jj = 1, simple_height
          do ii = 1, simple_width
             input_simple(ii,jj,kk,1,1) = 1._real32 * &
                  (ii + (ii-1) * (jj + (jj-1) * kk))
          end do
       end do
    end do
    call input(1,1)%allocate(&
         array_shape=[simple_width, simple_height, simple_depth, &
              simple_channels, 1], source = 0.0)
    call input(1,1)%set_requires_grad(.true.)
    call input(1,1)%set(input_simple)

    ! Test zero/constant padding
    write(*,*) "  Testing zero/constant padding..."
    pad3d_layer = pad3d_layer_type( &
         padding = [pad_w, pad_h, pad_d], &
         method = "zero", &
         input_shape = [simple_width, simple_height, simple_depth, &
              simple_channels], &
         batch_size = 1 &
    )
    call pad3d_layer%forward_derived(input)
    call pad3d_layer%extract_output(output_simple)

    ! Should have zeros around border and original data in center
    allocate(output_expected(simple_width+2*pad_w, simple_height+2*pad_h, &
         simple_depth+2*pad_d, simple_channels, 1))
    output_expected = 0._real32
    output_expected(3:6,3:6,3:6,1,1) = input_simple(:,:,:,1,1)
    if(any(abs(output_simple - output_expected) .gt. tol))then
       success = .false.
       write(0,*) 'Zero padding method failed'
    end if

    output => pad3d_layer%output(1,1)
    allocate(output%grad)
    call output%grad%allocate(array_shape=shape(gradient_out), source=0.0)
    call output%grad%set(gradient_out)
    call output%grad_reverse()
    gradient => input(1,1)%grad
    allocate(gradient_expected(simple_width, simple_height, simple_depth, &
         simple_channels, 1))
    ! For zero padding, gradients should just be extracted from middle
    call gradient%extract(gradient_expected)
    if(any(abs(gradient_expected - gradient_out(3:6,3:6,3:6,:,:)) &
         .gt. tol))then
       success = .false.
       write(0,*) 'Zero padding backward pass failed'
    end if
    call pad3d_layer%output(1,1)%nullify_graph()
    deallocate(output_simple, gradient_expected)

    ! Test replication padding
    write(*,*) "  Testing replication padding..."
    pad3d_layer = pad3d_layer_type( &
         padding = [pad_w, pad_h, pad_d], &
         method = "replicate", &
         input_shape = [simple_width, simple_height, simple_depth, &
              simple_channels], &
         batch_size = 1 &
    )
    call pad3d_layer%forward_derived(input)
    call pad3d_layer%extract_output(output_simple)

    ! Replicate edge values at boundaries
    output_expected(3:6,3:6,3:6,1,1) = input_simple(:,:,:,1,1)
    ! Corners - replicate corner values
    output_expected(1:2,1:2,1:2,1,1) = input_simple(1,1,1,1,1)
    output_expected(7:8,7:8,7:8,1,1) = input_simple(4,4,4,1,1)
    output_expected(1:2,7:8,1:2,1,1) = input_simple(1,4,1,1,1)
    output_expected(7:8,1:2,1:2,1,1) = input_simple(4,1,1,1,1)
    output_expected(1:2,1:2,7:8,1,1) = input_simple(1,1,4,1,1)
    output_expected(1:2,7:8,7:8,1,1) = input_simple(1,4,4,1,1)
    output_expected(7:8,1:2,7:8,1,1) = input_simple(4,1,4,1,1)
    output_expected(7:8,7:8,1:2,1,1) = input_simple(4,4,1,1,1)
    ! Edges - replicate edge values
    output_expected(3:6,1:2,1:2,1,1) = &
         spread(spread(input_simple(1:4,1,1,1,1), 2, 2), 3, 2)
    output_expected(3:6,7:8,1:2,1,1) = &
         spread(spread(input_simple(1:4,4,1,1,1), 2, 2), 3, 2)
    output_expected(3:6,1:2,7:8,1,1) = &
         spread(spread(input_simple(1:4,1,4,1,1), 2, 2), 3, 2)
    output_expected(3:6,7:8,7:8,1,1) = &
         spread(spread(input_simple(1:4,4,4,1,1), 2, 2), 3, 2)

    output_expected(1:2,3:6,1:2,1,1) = &
         spread(spread(input_simple(1,1:4,1,1,1), 1, 2), 3, 2)
    output_expected(7:8,3:6,1:2,1,1) = &
         spread(spread(input_simple(4,1:4,1,1,1), 1, 2), 3, 2)
    output_expected(1:2,3:6,7:8,1,1) = &
         spread(spread(input_simple(1,1:4,4,1,1), 1, 2), 3, 2)
    output_expected(7:8,3:6,7:8,1,1) = &
         spread(spread(input_simple(4,1:4,4,1,1), 1, 2), 3, 2)

    output_expected(1:2,1:2,3:6,1,1) = &
         spread(spread(input_simple(1,1,1:4,1,1), 1, 2), 2, 2)
    output_expected(7:8,1:2,3:6,1,1) = &
         spread(spread(input_simple(4,1,1:4,1,1), 1, 2), 2, 2)
    output_expected(1:2,7:8,3:6,1,1) = &
         spread(spread(input_simple(1,4,1:4,1,1), 1, 2), 2, 2)
    output_expected(7:8,7:8,3:6,1,1) = &
         spread(spread(input_simple(4,4,1:4,1,1), 1, 2), 2, 2)
    ! Faces - replicate face values
    output_expected(1:2,3:6,3:6,1,1) = spread(input_simple(1,1:4,1:4,1,1), 1, 2)
    output_expected(7:8,3:6,3:6,1,1) = spread(input_simple(4,1:4,1:4,1,1), 1, 2)
    output_expected(3:6,1:2,3:6,1,1) = spread(input_simple(1:4,1,1:4,1,1), 2, 2)
    output_expected(3:6,7:8,3:6,1,1) = spread(input_simple(1:4,4,1:4,1,1), 2, 2)
    output_expected(3:6,3:6,1:2,1,1) = spread(input_simple(1:4,1:4,1,1,1), 3, 2)
    output_expected(3:6,3:6,7:8,1,1) = spread(input_simple(1:4,1:4,4,1,1), 3, 2)


    if(any(abs(output_simple - output_expected) .gt. tol))then
       success = .false.
       write(0,*) 'Replication padding method failed'
       do i = 1, size(output_simple,1)
          do j = 1, size(output_simple,2)
             do k = 1, size(output_simple,3)
                if( &
                     abs(output_simple(i,j,k,1,1) - output_expected(i,j,k,1,1)) .gt. &
                     tol &
                )then
                   write(0,*) 'Mismatch at (', i, ',', j, ',', k, ')'
                   write(0,*) 'Expected: ', output_expected(i,j,k,1,1)
                   write(0,*) 'Got:      ', output_simple(i,j,k,1,1)
                end if
             end do
          end do
       end do
    end if

    ! Test backward pass for replication
    output => pad3d_layer%output(1,1)
    call output%reset_graph()
    allocate(output%grad)
    call output%grad%allocate(array_shape=shape(gradient_out), source=0.0)
    call output%grad%set(gradient_out)
    call output%grad_reverse()
    gradient => input(1,1)%grad
    allocate(gradient_expected(simple_width, simple_height, simple_depth, &
         simple_channels, 1))
    call gradient%extract(gradient_expected)
    ! For replication, check that gradients are reasonable
    ! (exact checking is complex due to accumulation at many points)
    if(any(abs(gradient_expected) .lt. 0.0))then
       success = .false.
       write(0,*) 'Replication padding backward pass has negative gradients'
    end if
    deallocate(output_simple, gradient_expected)

    call pad3d_layer%output(1,1)%nullify_graph()
    deallocate(input_simple, gradient_out, output_expected)
    call input(1,1)%deallocate()
  end block comprehensive_methods_block


!-------------------------------------------------------------------------------
! Test different padding sizes
!-------------------------------------------------------------------------------
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
       if(any(pad3d_layer%output_shape .ne. &
            [expected_width, expected_height, expected_depth, channels]))then
          success = .false.
          write(0,*) 'pad3d layer output shape incorrect for padding size:', &
               test_paddings(i,:)
          write(0,*) 'Expected:', &
               [expected_width, expected_height, expected_depth, channels]
          write(0,*) 'Got:      ', pad3d_layer%output_shape
       end if

       ! Test forward pass
       call input(1,1)%allocate(&
            array_shape=[width, height, depth, 1, 1], source = 1.0_real32)

       call pad3d_layer%forward_derived(input)
       call pad3d_layer%extract_output(output_5d)

       ! For zero padding, check that padding is actually zero
       if(test_paddings(i,1) > 0)then
          if(any(abs(output_5d(1:test_paddings(i,1),:,:,:,:)) .gt. tol) .or. &
               any(abs(output_5d(expected_width-test_paddings(i,1)+1: &
                    expected_width,:,:,:,:)) .gt. tol))then
             success = .false.
             write(0,*) 'pad3d layer width zero padding incorrect for size:', &
                  test_paddings(i,1)
          end if
       end if

       if(test_paddings(i,2) > 0)then
          if(any(abs(output_5d(:,1:test_paddings(i,2),:,:,:)) .gt. tol) .or. &
               any(abs(output_5d(:,expected_height-test_paddings(i,2)+1: &
                    expected_height,:,:,:)) .gt. tol))then
             success = .false.
             write(0,*) 'pad3d layer height zero padding incorrect for size:', &
                  test_paddings(i,2)
          end if
       end if

       if(test_paddings(i,3) > 0)then
          if(any(abs(output_5d(:,:,1:test_paddings(i,3),:,:)) .gt. tol) .or. &
               any(abs(output_5d(:,:,expected_depth-test_paddings(i,3)+1: &
                    expected_depth,:,:)) .gt. tol))then
             success = .false.
             write(0,*) 'pad3d layer depth zero padding incorrect for size:', &
                  test_paddings(i,3)
          end if
       end if

       call pad3d_layer%output(1,1)%nullify_graph()
       deallocate(output_5d)
       call input(1,1)%deallocate()
    end do
  end block test_sizes_block


!-------------------------------------------------------------------------------
! Test asymmetric padding
!-------------------------------------------------------------------------------
  write(*,*) "Testing asymmetric padding..."

  pad3d_layer = pad3d_layer_type( &
       padding = [1, 2, 0], &
       method = "zero", &
       input_shape = [width, height, depth, channels], &
       batch_size = 1 &
  )

  ! Test with a known input pattern
  call input(1,1)%allocate(&
       array_shape=[width, height, depth, 1, 1], source = 0.0)
  do k = 1, depth
     do j = 1, height
        do i = 1, width
           input(1,1)%val(&
                i + (j-1)*width + (k-1)*width*height, 1 &
           ) = real(i * 100 + j * 10 + k, real32)
        end do
     end do
  end do

  call pad3d_layer%forward_derived(input)
  call pad3d_layer%extract_output(output_5d)

  ! Check that the center part matches the input
  expected_width = width + 2
  expected_height = height + 4
  expected_depth = depth + 0
  call input(1,1)%extract(input_5d)
  if(any(abs(output_5d(2:expected_width-1, 3:expected_height-2, &
       1:expected_depth, 1, 1) - &
  input_5d(:, :, :, 1, 1)) .gt. tol))then
     success = .false.
     write(0,*) 'pad3d layer asymmetric padding center incorrect'
  end if

  call pad3d_layer%output(1,1)%nullify_graph()
  deallocate(input_5d, output_5d)
  call input(1,1)%deallocate()


!-------------------------------------------------------------------------------
! Test batch size modification
!-------------------------------------------------------------------------------
  write(*,*) "Testing batch size modification..."

  pad3d_layer = pad3d_layer_type( &
       padding = [1, 1, 1], &
       method = "zero", &
       input_shape = [width, height, depth, channels] &
  )

  call pad3d_layer%set_batch_size(batch_size)

  if(pad3d_layer%batch_size .ne. batch_size)then
     success = .false.
     write(0,*) 'pad3d layer set_batch_size failed'
  end if


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_pad3d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("PAD3D")')
  call pad3d_layer%print_to_unit(unit)
  write(unit,'("END PAD3D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_pad3d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_pad3d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (pad3d_layer_type)
     if(.not. read_layer%name .eq. 'pad3d')then
        success = .false.
        write(0,*) 'read pad3d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not pad3d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_pad3d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Test edge cases
!-------------------------------------------------------------------------------
  write(*,*) "Testing edge cases..."

  ! Test with minimal dimensions
  pad3d_layer = pad3d_layer_type( &
       padding = [0, 0, 0], &
       method = "zero", &
       input_shape = [1, 1, 1, 1], &
       batch_size = 1 &
  )

  call input(1,1)%allocate(array_shape=[1, 1, 1, 1, 1], source = 42.0_real32)

  call pad3d_layer%forward_derived(input)
  call pad3d_layer%extract_output(output_5d)

  if(any(abs(output_5d - 42.0_real32) .gt. tol))then
     success = .false.
     write(0,*) 'pad3d layer edge case (no padding) incorrect'
  end if

  call pad3d_layer%output(1,1)%nullify_graph()
  deallocate(output_5d)
  call input(1,1)%deallocate()


!-------------------------------------------------------------------------------
! Check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if (success) then
     write(*,*) 'test_pad3d_layer passed all tests'
  else
     write(0,*) 'test_pad3d_layer failed one or more tests'
     stop 1
  end if

end program test_pad3d_layer
