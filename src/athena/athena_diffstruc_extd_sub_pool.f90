submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_pool
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function avgpool1d(input, pool_size, stride) result(output)
    !! 1D average pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, m, s
    integer :: stride_idx, idx
    integer, dimension(3) :: output_shape

    output_shape = [ &
         (input%shape(1) - pool_size) / stride + 1, &
         input%shape(2), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)
    do concurrent(&
         s = 1:output_shape(3), &
         m = 1:output_shape(2), &
         i = 1:output_shape(1))
       stride_idx = (i - 1) * stride + (m - 1) * input%shape(1)
       idx = i + (m - 1) * output_shape(1)
       output%val(idx, s) = sum( &
            input%val( stride_idx + 1 : stride_idx + pool_size, s ) &
       ) / pool_size
    end do
    allocate(output%adj_ja(1,2))
    output%adj_ja(1,1) = pool_size
    output%adj_ja(1,2) = stride

    output%get_partial_left => get_partial_avgpool1d
    output%get_partial_left_val => get_partial_avgpool1d_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'avgpool'
       output%left_operand => input
    end if

  end function avgpool1d
!-------------------------------------------------------------------------------
  function get_partial_avgpool1d(this, upstream_grad) result(output)
    !! Get the partial derivative for average pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = &
         [ this%left_operand%shape, size(this%val, dim=2) ] &
    )
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_avgpool1d
!-------------------------------------------------------------------------------
  pure subroutine get_partial_avgpool1d_val(this, upstream_grad, output)
    !! Optimised backward pass for 1D average pooling
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, m, s, p
    integer :: base_idx, out_idx, input_h
    real(real32) :: pool_norm, grad_val
    integer, dimension(3) :: input_shape
    integer, dimension(1) :: pool_size, stride

    ! Unpack parameters
    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    pool_size(1) = this%adj_ja(1,1)
    stride(1) = this%adj_ja(1,2)
    input_h = input_shape(1)

    output = 0._real32

    pool_norm = 1.0_real32 / real(pool_size(1), real32)

    ! Parallelised over batch and spatial/channel dimensions
    do concurrent(s = 1:input_shape(3), m = 1:this%shape(2), &
         i = 1:this%shape(1))

       ! Compute indices once
       base_idx = (i - 1) * stride(1) + (m - 1) * input_h
       out_idx = i + (m - 1) * this%shape(1)
       grad_val = upstream_grad(out_idx, s) * pool_norm

       ! Distribute gradient over pooling window
       do p = 0, pool_size(1) - 1
          output(base_idx + p + 1, s) = output(base_idx + p + 1, s) + grad_val
       end do
    end do

  end subroutine get_partial_avgpool1d_val
!###############################################################################


!###############################################################################
  module function avgpool2d(input, pool_size, stride) result(output)
    !! 2D average pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, dimension(2), intent(in) :: pool_size
    integer, dimension(2), intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, m, s, i_step, j_step
    integer :: stride_idx, idx, multiplier
    integer :: channel_size_in, channel_size_out
    real(real32) :: pool_sum, pool_norm
    integer, dimension(4) :: output_shape

    output_shape = [ &
         (input%shape(1) - pool_size(1)) / stride(1) + 1, &
         (input%shape(2) - pool_size(2)) / stride(2) + 1, &
         input%shape(3), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)
    pool_norm = 1.0_real32 / real(pool_size(1) * pool_size(2), real32)

    ! Pre-compute as integers
    channel_size_in = input%shape(1) * input%shape(2)
    channel_size_out = output_shape(1) * output_shape(2)

    do concurrent(&
         s = 1:output_shape(4), &
         m = 1:output_shape(3), &
         j = 1:output_shape(2), &
         i = 1:output_shape(1))

       ! Compute indices once
       stride_idx = (i-1)*stride(1) + &
            ((j-1)*stride(2)) * input%shape(1) + &
            (m-1) * channel_size_in
       idx = i + (j - 1) * output_shape(1) + (m - 1) * channel_size_out

       pool_sum = 0._real32
       do j_step = 0, pool_size(2)-1
          do i_step = 0, pool_size(1)-1
             pool_sum = pool_sum + &
                  input%val(stride_idx + i_step + j_step * input%shape(1) + 1, s)
          end do
       end do
       output%val(idx, s) = pool_sum * pool_norm
    end do
    allocate(output%adj_ja(2,2))
    output%adj_ja(:,1) = pool_size
    output%adj_ja(:,2) = stride

    output%get_partial_left => get_partial_avgpool2d
    output%get_partial_left_val => get_partial_avgpool2d_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'avgpool'
       output%left_operand => input
    end if

  end function avgpool2d
!-------------------------------------------------------------------------------
  function get_partial_avgpool2d(this, upstream_grad) result(output)
    !! Get the partial derivative for average pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = &
         [ this%left_operand%shape, size(this%val, dim=2) ] &
    )
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_avgpool2d
!-------------------------------------------------------------------------------
  pure subroutine get_partial_avgpool2d_val(this, upstream_grad, output)
    !! Optimised backward pass for 2D average pooling
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, m, s
    integer :: i_step, j_step
    integer :: base_idx, in_idx, out_idx, input_h
    integer :: channel_size_in, channel_size_out
    real(real32) :: pool_norm, grad_val
    integer, dimension(4) :: input_shape
    integer, dimension(2) :: pool_size, stride

    ! Unpack parameters
    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    pool_size = this%adj_ja(:,1)
    stride    = this%adj_ja(:,2)
    input_h = input_shape(1)
    channel_size_in = input_h * input_shape(2)
    channel_size_out = this%shape(1) * this%shape(2)

    output = 0._real32

    pool_norm = 1.0_real32 / real(pool_size(1) * pool_size(2), real32)

    do concurrent( &
         s = 1:input_shape(4), &
         m = 1:this%shape(3), &
         j = 1:this%shape(2), &
         i = 1:this%shape(1))

       ! Compute indices once
       base_idx = (i-1) * stride(1) + ((j-1) * stride(2)) * input_h + &
            (m-1) * channel_size_in
       out_idx = i + (j-1) * this%shape(1) + (m-1) * channel_size_out
       grad_val = upstream_grad(out_idx, s) * pool_norm

       ! Distribute gradient over pooling window
       do j_step = 0, pool_size(2) - 1
          do i_step = 0, pool_size(1) - 1
             in_idx = base_idx + i_step + j_step * input_h + 1
             output(in_idx, s) = output(in_idx, s) + grad_val
          end do
       end do
    end do

  end subroutine get_partial_avgpool2d_val
!###############################################################################


!###############################################################################
  module function avgpool3d(input, pool_size, stride) result(output)
    !! 3D average pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, dimension(3), intent(in) :: pool_size
    integer, dimension(3), intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: i_step, j_step, k_step
    integer :: stride_idx, idx
    integer :: channel_size_in, channel_size_out
    real(real32) :: pool_sum, pool_norm
    integer, dimension(5) :: output_shape

    ! output_shape = [H_out, W_out, D_out, C, B]
    output_shape = [ &
         (input%shape(1) - pool_size(1)) / stride(1) + 1, &
         (input%shape(2) - pool_size(2)) / stride(2) + 1, &
         (input%shape(3) - pool_size(3)) / stride(3) + 1, &
         input%shape(4), &
         size(input%val, dim=2) ]

    output => input%create_result(array_shape = output_shape)
    pool_norm = 1.0_real32 / real(product(pool_size), real32)

    ! Pre-compute as integers
    channel_size_in = input%shape(1) * input%shape(2) * input%shape(3)
    channel_size_out = output_shape(1) * output_shape(2) * output_shape(3)

    do concurrent( &
         s = 1:output_shape(5), &
         m = 1:output_shape(4), &
         k = 1:output_shape(3), &
         j = 1:output_shape(2), &
         i = 1:output_shape(1))

       ! Compute indices once
       stride_idx = ((i-1)*stride(1)) + &
            ((j-1)*stride(2)) * input%shape(1) + &
            ((k-1)*stride(3)) * input%shape(1) * input%shape(2) + &
            (m-1) * channel_size_in
       idx = i + (j-1) * output_shape(1) + &
            (k-1) * output_shape(1)*output_shape(2) + &
            (m-1) * channel_size_out

       pool_sum = 0._real32
       do k_step = 0, pool_size(3)-1
          do j_step = 0, pool_size(2)-1
             do i_step = 0, pool_size(1)-1
                pool_sum = pool_sum + input%val(stride_idx + i_step + &
                     j_step * input%shape(1) + &
                     k_step * input%shape(1) * input%shape(2) + 1, s)
             end do
          end do
       end do

       output%val(idx, s) = pool_sum * pool_norm
    end do

    allocate(output%adj_ja(3,2))
    output%adj_ja(:,1) = pool_size
    output%adj_ja(:,2) = stride

    output%get_partial_left => get_partial_avgpool3d
    output%get_partial_left_val => get_partial_avgpool3d_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'avgpool3d'
       output%left_operand => input
    end if

  end function avgpool3d
!-------------------------------------------------------------------------------
  function get_partial_avgpool3d(this, upstream_grad) result(output)
    !! Get the partial derivative for 3D average pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = &
         [ this%left_operand%shape, size(this%val, dim=2) ] &
    )
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_avgpool3d
!-------------------------------------------------------------------------------
  pure subroutine get_partial_avgpool3d_val(this, upstream_grad, output)
    !! Optimised backward pass for 3D average pooling
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: i_step, j_step, k_step
    integer :: base_idx, in_idx, out_idx, input_h, input_hw
    integer :: channel_size_in, channel_size_out
    real(real32) :: pool_norm, grad_val
    integer, dimension(5) :: input_shape
    integer, dimension(3) :: pool_size, stride

    ! Unpack parameters
    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    pool_size = this%adj_ja(:,1)
    stride    = this%adj_ja(:,2)
    input_h = input_shape(1)
    input_hw = input_h * input_shape(2)
    channel_size_in = input_hw * input_shape(3)
    channel_size_out = this%shape(1) * this%shape(2) * this%shape(3)

    output = 0._real32

    pool_norm = 1.0_real32 / real(product(pool_size), real32)

    do concurrent( &
         s = 1:input_shape(5), &
         m = 1:this%shape(4), &
         k = 1:this%shape(3), &
         j = 1:this%shape(2), &
         i = 1:this%shape(1))

       ! Compute indices once
       base_idx = (i-1)*stride(1) + ((j-1)*stride(2)) * input_h + &
            ((k-1)*stride(3)) * input_hw + (m-1) * channel_size_in
       out_idx = i + (j-1) * this%shape(1) + &
            (k-1) * this%shape(1)*this%shape(2) + &
            (m-1) * channel_size_out
       grad_val = upstream_grad(out_idx, s) * pool_norm

       ! Distribute gradient over pooling window
       do k_step = 0, pool_size(3)-1
          do j_step = 0, pool_size(2)-1
             do i_step = 0, pool_size(1)-1
                in_idx = base_idx + i_step + j_step * input_h + &
                     k_step * input_hw + 1
                output(in_idx, s) = output(in_idx, s) + grad_val
             end do
          end do
       end do
    end do

  end subroutine get_partial_avgpool3d_val
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function maxpool1d(input, pool_size, stride) result(output)
    !! 1D max pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, m, s
    integer :: stride_idx, idx
    integer, dimension(3) :: output_shape

    output_shape = [ &
         (input%shape(1) - pool_size) / stride + 1, &
         input%shape(2), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)
    do concurrent(&
         s = 1:output_shape(3), &
         m = 1:output_shape(2), &
         i = 1:output_shape(1))
       stride_idx = (i - 1) * stride + (m - 1) * input%shape(1)
       idx = i + (m - 1) * output_shape(1)
       output%val(idx, s) = maxval( &
            input%val( stride_idx + 1 : stride_idx + pool_size, s ) &
       )
    end do
    allocate(output%adj_ja(1,2))
    output%adj_ja(1,1) = pool_size
    output%adj_ja(1,2) = stride

    output%get_partial_left => get_partial_maxpool1d
    output%get_partial_left_val => get_partial_maxpool1d_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'maxpool'
       output%left_operand => input
    end if

  end function maxpool1d
!-------------------------------------------------------------------------------
  function get_partial_maxpool1d(this, upstream_grad) result(output)
    !! Get the partial derivative for max pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = &
         [ this%left_operand%shape, size(this%val, dim=2) ] &
    )
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_maxpool1d
!-------------------------------------------------------------------------------
  pure subroutine get_partial_maxpool1d_val(this, upstream_grad, output)
    !! Optimised backward pass for 1D max pooling
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, m, s, p
    integer :: base_idx, max_idx, out_idx, input_h
    real(real32) :: pool_max, grad_val
    integer, dimension(3) :: input_shape
    integer, dimension(1) :: pool_size, stride

    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    pool_size(1) = this%adj_ja(1,1)
    stride(1) = this%adj_ja(1,2)
    input_h = input_shape(1)

    output = 0._real32

    do concurrent(s = 1:input_shape(3), m = 1:this%shape(2), &
         i = 1:this%shape(1))

       ! Compute indices once
       base_idx = (i - 1) * stride(1) + (m - 1) * input_h
       out_idx = i + (m - 1) * this%shape(1)
       grad_val = upstream_grad(out_idx, s)

       ! Find max value location - initialise with first element
       max_idx = base_idx + 1
       pool_max = this%left_operand%val(max_idx, s)

       ! Search remaining elements for max
       do p = 1, pool_size(1) - 1
          if(this%left_operand%val(base_idx + p + 1, s) .gt. pool_max)then
             pool_max = this%left_operand%val(base_idx + p + 1, s)
             max_idx = base_idx + p + 1
          end if
       end do

       ! Assign gradient to max location
       output(max_idx, s) = output(max_idx, s) + grad_val
    end do

  end subroutine get_partial_maxpool1d_val
!###############################################################################


!###############################################################################
  module function maxpool2d(input, pool_size, stride) result(output)
    !! 2D max pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, dimension(2), intent(in) :: pool_size
    integer, dimension(2), intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, m, s, i_step, j_step
    integer :: base_idx, stride_idx, idx, input_h
    real(real32) :: pool_max, val_tmp
    integer :: channel_size_in, channel_size_out
    integer, dimension(4) :: output_shape

    output_shape = [ &
         (input%shape(1) - pool_size(1)) / stride(1) + 1, &
         (input%shape(2) - pool_size(2)) / stride(2) + 1, &
         input%shape(3), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)

    ! Pre-compute as integers to avoid type conversion in loop
    input_h = input%shape(1)
    channel_size_in = input_h * input%shape(2)
    channel_size_out = output_shape(1) * output_shape(2)

    do concurrent(&
         s = 1:output_shape(4), &
         m = 1:output_shape(3), &
         j = 1:output_shape(2), &
         i = 1:output_shape(1))

       ! Compute indices once per output position
       base_idx = (i-1)*stride(1) + ((j-1)*stride(2)) * input_h + &
            (m-1) * channel_size_in
       idx = i + (j - 1) * output_shape(1) + (m - 1) * channel_size_out

       ! Find max value - initialise with first element for better performance
       stride_idx = base_idx + 1
       pool_max = input%val(stride_idx, s)

       ! Continue with remaining elements
       do j_step = 0, pool_size(2)-1
          do i_step = 0, pool_size(1)-1
             if(i_step .eq. 0 .and. j_step .eq. 0) cycle  ! Already processed
             stride_idx = base_idx + i_step + j_step * input_h + 1
             if(input%val(stride_idx, s) .gt. pool_max) &
                  pool_max = input%val(stride_idx, s)
          end do
       end do

       output%val(idx, s) = pool_max
    end do

    allocate(output%adj_ja(2,2))
    output%adj_ja(:,1) = pool_size
    output%adj_ja(:,2) = stride

    output%get_partial_left => get_partial_maxpool2d
    output%get_partial_left_val => get_partial_maxpool2d_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'maxpool'
       output%left_operand => input
    end if

  end function maxpool2d
!-------------------------------------------------------------------------------
  function get_partial_maxpool2d(this, upstream_grad) result(output)
    !! Get the partial derivative for max pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = &
         [ this%left_operand%shape, size(this%val, dim=2) ] &
    )
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_maxpool2d
!-------------------------------------------------------------------------------
  pure subroutine get_partial_maxpool2d_val(this, upstream_grad, output)
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, m, s
    integer :: i_step, j_step
    integer :: base_idx, in_idx, out_idx, max_idx, input_h
    real(real32) :: pool_max, val_tmp, grad_val
    integer :: channel_size_in, channel_size_out
    integer, dimension(4) :: input_shape
    integer, dimension(2) :: pool_size, stride

    ! Unpack parameters
    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    pool_size = this%adj_ja(:,1)
    stride    = this%adj_ja(:,2)
    input_h = input_shape(1)
    channel_size_in = input_h * input_shape(2)
    channel_size_out = this%shape(1) * this%shape(2)

    output = 0._real32

    ! Parallelised over batch and spatial/channel dimensions
    do concurrent(s = 1:input_shape(4), m = 1:this%shape(3), &
         j = 1:this%shape(2), i = 1:this%shape(1))

       ! Compute indices once
       base_idx = (i-1) * stride(1) + ((j-1) * stride(2)) * input_h + &
            (m-1) * channel_size_in
       out_idx = i + (j-1) * this%shape(1) + (m-1) * channel_size_out
       grad_val = upstream_grad(out_idx, s)

       ! Find max value location - initialise with first element
       max_idx = base_idx + 1
       pool_max = this%left_operand%val(max_idx, s)

       ! Search remaining elements for max
       do j_step = 0, pool_size(2) - 1
          do i_step = 0, pool_size(1) - 1
             if(i_step .eq. 0 .and. j_step .eq. 0) cycle  ! Already processed
             in_idx = base_idx + i_step + j_step * input_h + 1
             val_tmp = this%left_operand%val(in_idx, s)

             if(val_tmp .gt. pool_max)then
                pool_max = val_tmp
                max_idx = in_idx
             end if
          end do
       end do

       ! Assign gradient to max location
       output(max_idx, s) = output(max_idx, s) + grad_val
    end do

  end subroutine get_partial_maxpool2d_val
!###############################################################################


!###############################################################################
  module function maxpool3d(input, pool_size, stride) result(output)
    !! 3D max pooling operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    integer, dimension(3), intent(in) :: pool_size
    integer, dimension(3), intent(in) :: stride
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: i_step, j_step, k_step
    integer :: stride_idx, idx
    integer :: channel_size_in, channel_size_out
    real(real32) :: pool_max
    integer, dimension(5) :: output_shape

    ! output_shape = [H_out, W_out, D_out, C, B]
    output_shape = [ &
         (input%shape(1) - pool_size(1)) / stride(1) + 1, &
         (input%shape(2) - pool_size(2)) / stride(2) + 1, &
         (input%shape(3) - pool_size(3)) / stride(3) + 1, &
         input%shape(4), &
         size(input%val, dim=2) ]

    output => input%create_result(array_shape = output_shape)

    ! Pre-compute as integers
    channel_size_in = input%shape(1) * input%shape(2) * input%shape(3)
    channel_size_out = output_shape(1) * output_shape(2) * output_shape(3)

    do concurrent( &
         s = 1:output_shape(5), &
         m = 1:output_shape(4), &
         k = 1:output_shape(3), &
         j = 1:output_shape(2), &
         i = 1:output_shape(1))

       ! Compute indices once per output position
       stride_idx = ((i-1)*stride(1)) + &
            ((j-1)*stride(2)) * input%shape(1) + &
            ((k-1)*stride(3)) * input%shape(1) * input%shape(2) + &
            (m-1) * channel_size_in + 1
       idx = i + (j-1) * output_shape(1) + &
            (k-1) * output_shape(1)*output_shape(2) + &
            (m-1) * channel_size_out

       ! Find max value - initialise with first element
       pool_max = input%val(stride_idx, s)

       do k_step = 0, pool_size(3)-1
          do j_step = 0, pool_size(2)-1
             do i_step = 0, pool_size(1)-1
                if(i_step .eq. 0 .and. j_step .eq. 0 .and. k_step .eq. 0) cycle
                if( &
                     input%val( &
                          stride_idx + i_step + &
                          j_step * input%shape(1) + &
                          k_step * input%shape(1) * input%shape(2), s &
                     ) .gt. pool_max &
                )then
                   pool_max = input%val(stride_idx + i_step + &
                        j_step * input%shape(1) + &
                        k_step * input%shape(1) * input%shape(2), s)
                end if
             end do
          end do
       end do

       output%val(idx, s) = pool_max
    end do

    allocate(output%adj_ja(3,2))
    output%adj_ja(:,1) = pool_size
    output%adj_ja(:,2) = stride

    output%get_partial_left => get_partial_maxpool3d
    output%get_partial_left_val => get_partial_maxpool3d_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'maxpool3d'
       output%left_operand => input
    end if

  end function maxpool3d
!-------------------------------------------------------------------------------
  function get_partial_maxpool3d(this, upstream_grad) result(output)
    !! Get the partial derivative for 3D max pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = &
         [ this%left_operand%shape, size(this%val, dim=2) ] &
    )
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_maxpool3d
!-------------------------------------------------------------------------------
  pure subroutine get_partial_maxpool3d_val(this, upstream_grad, output)
    !! Optimised backward pass for 3D max pooling
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, k, m, s
    integer :: i_step, j_step, k_step
    integer :: base_idx, in_idx, out_idx, max_idx
    integer :: input_h, input_hw
    integer :: channel_size_in, channel_size_out
    real(real32) :: pool_max, val_tmp, grad_val
    integer, dimension(5) :: input_shape
    integer, dimension(3) :: pool_size, stride

    ! Unpack parameters
    input_shape = [ this%left_operand%shape, size(this%val, dim=2) ]
    pool_size = this%adj_ja(:,1)
    stride    = this%adj_ja(:,2)
    input_h = input_shape(1)
    input_hw = input_h * input_shape(2)
    channel_size_in = input_hw * input_shape(3)
    channel_size_out = this%shape(1) * this%shape(2) * this%shape(3)

    output = 0._real32

    ! Parallelised over batch and spatial/channel dimensions
    do concurrent(s = 1:input_shape(5), m = 1:this%shape(4), &
         k = 1:this%shape(3), j = 1:this%shape(2), i = 1:this%shape(1))

       ! Compute indices once
       base_idx = (i-1)*stride(1) + ((j-1)*stride(2)) * input_h + &
            ((k-1)*stride(3)) * input_hw + (m-1) * channel_size_in
       out_idx = i + (j-1) * this%shape(1) + &
            (k-1) * this%shape(1)*this%shape(2) + &
            (m-1) * channel_size_out
       grad_val = upstream_grad(out_idx, s)

       ! Find max value location - initialise with first element
       max_idx = base_idx + 1
       pool_max = this%left_operand%val(max_idx, s)

       ! Search remaining elements for max
       do k_step = 0, pool_size(3)-1
          do j_step = 0, pool_size(2)-1
             do i_step = 0, pool_size(1)-1
                if(i_step .eq. 0 .and. j_step .eq. 0 .and. k_step .eq. 0) cycle
                in_idx = base_idx + i_step + j_step * input_h + &
                     k_step * input_hw + 1
                val_tmp = this%left_operand%val(in_idx, s)

                if(val_tmp .gt. pool_max)then
                   pool_max = val_tmp
                   max_idx = in_idx
                end if
             end do
          end do
       end do

       ! Assign gradient to max location
       output(max_idx, s) = output(max_idx, s) + grad_val
    end do

  end subroutine get_partial_maxpool3d_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_pool
