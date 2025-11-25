submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_conv
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function conv1d(input, kernel, stride, dilation) result(output)
    !! 1D convolution operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, intent(in) :: stride
    integer, intent(in) :: dilation
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, k, c_in, c_out, s
    integer :: i_in, k_idx
    integer :: input_h, kernel_h, output_h, num_channels, num_filters
    real(real32) :: conv_sum
    integer, dimension(3) :: output_shape

    ! Extract dimensions
    ! input: [H_in, C_in, B]
    ! kernel: [K, C_in, C_out]
    input_h = input%shape(1)
    num_channels = input%shape(2)
    kernel_h = kernel%shape(1)
    num_filters = kernel%shape(3)

    ! Calculate output dimensions
    output_h = (input_h - dilation*(kernel_h - 1) - 1) / &
         stride + 1
    output_shape = [output_h, num_filters, size(input%val, dim=2)]

    output => input%create_result(array_shape = output_shape)
    output%val = 0._real32

    ! Perform convolution
    do concurrent(s = 1:output_shape(3), c_out = 1:num_filters, &
         i = 1:output_h)
       conv_sum = 0._real32
       do c_in = 1, num_channels
          do k = 1, kernel_h
             i_in = ( i - 1 ) * stride + ( k - 1 ) * dilation + 1
             if(i_in .ge. 1 .and. i_in .le. input_h)then
                k_idx = k + ( c_in - 1 ) * kernel_h + &
                     ( c_out - 1 ) * kernel_h * num_channels
                conv_sum = conv_sum + &
                     input%val(i_in + ( c_in - 1 ) * input_h, s) * &
                     kernel%val(k_idx, 1)
             end if
          end do
       end do
       output%val(i + (c_out-1)*output_h, s) = conv_sum
    end do

    ! Store parameters for backward pass
    allocate(output%indices(2))
    output%indices(1) = num_channels
    output%indices(2) = num_filters
    allocate(output%adj_ja(1,3))
    output%adj_ja(1,1) = stride
    output%adj_ja(1,2) = dilation
    output%adj_ja(1,3) = kernel_h

    output%get_partial_left => get_partial_conv1d_input
    output%get_partial_right => get_partial_conv1d_kernel
    output%get_partial_left_val => get_partial_conv1d_input_val
    output%get_partial_right_val => get_partial_conv1d_kernel_val
    if(input%requires_grad .or. kernel%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'conv1d'
       output%left_operand => input
       output%right_operand => kernel
    end if

  end function conv1d
!-------------------------------------------------------------------------------
  function get_partial_conv1d_input(this, upstream_grad) result(output)
    !! Get partial derivative wrt input for 1D convolution
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%left_operand%shape, &
         size(this%left_operand%val, dim=2) ])
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_conv1d_input
!-------------------------------------------------------------------------------
  function get_partial_conv1d_kernel(this, upstream_grad) result(output)
    !! Get partial derivative wrt kernel for 1D convolution
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%right_operand%shape, 1 ])
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_conv1d_kernel
!-------------------------------------------------------------------------------
  subroutine get_partial_conv1d_input_val(this, upstream_grad, output)
    !! Get partial derivative wrt input for 1D convolution (subroutine version)
    implicit none

    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, k, c_in, c_out, s
    integer :: i_in, k_idx, out_idx
    integer :: input_h, kernel_h, output_h, num_channels, num_filters
    integer :: stride, dilation
    real(real32) :: grad_val, kernel_val
    class(array_type), pointer :: input, kernel

    input => this%left_operand
    kernel => this%right_operand

    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1,1)
    dilation = this%adj_ja(1,2)
    kernel_h = this%adj_ja(1,3)

    input_h = input%shape(1)
    output_h = this%shape(1)

    output = 0._real32

    ! Parallelized over batch, channels, and output positions
    do concurrent(s = 1:size(upstream_grad, dim=2), c_in = 1:num_channels, &
         i = 1:output_h, c_out = 1:num_filters)
       out_idx = i + (c_out-1)*output_h
       grad_val = upstream_grad(out_idx, s)

       if(abs(grad_val) > 1.e-30_real32)then
          do k = 1, kernel_h
             i_in = ( i - 1 ) * stride + ( k - 1 ) * dilation + 1
             if(i_in .ge. 1 .and. i_in .le. input_h)then
                k_idx = k + ( c_in - 1 ) * kernel_h + &
                     ( c_out - 1 ) * kernel_h * num_channels
                kernel_val = kernel%val(k_idx, 1)
                output(i_in + ( c_in - 1 ) * input_h, s) = &
                     output(i_in + ( c_in - 1 ) * input_h, s) + &
                     grad_val * kernel_val
             end if
          end do
       end if
    end do

  end subroutine get_partial_conv1d_input_val
!-------------------------------------------------------------------------------
  subroutine get_partial_conv1d_kernel_val(this, upstream_grad, output)
    !! Get partial derivative wrt kernel for 1D convolution (subroutine version)
    implicit none

    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, k, c_in, c_out, s
    integer :: i_in, k_idx, out_idx
    integer :: input_h, kernel_h, output_h, num_channels, num_filters
    integer :: stride, dilation
    real(real32) :: grad_sum
    class(array_type), pointer :: input, kernel

    input => this%left_operand
    kernel => this%right_operand

    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1,1)
    dilation = this%adj_ja(1,2)
    kernel_h = this%adj_ja(1,3)

    input_h = input%shape(1)
    output_h = this%shape(1)

    output = 0._real32

    ! Parallelized over filters, channels, and kernel positions
    do concurrent(c_out = 1:num_filters, c_in = 1:num_channels, k = 1:kernel_h)
       k_idx = k + ( c_in - 1 ) * kernel_h + &
            ( c_out - 1 ) * kernel_h * num_channels

       grad_sum = 0._real32
       do s = 1, size(upstream_grad, dim=2)
          do i = 1, output_h
             i_in = ( i - 1 ) * stride + ( k - 1 ) * dilation + 1
             if(i_in .ge. 1 .and. i_in .le. input_h)then
                out_idx = i + ( c_out - 1 ) * output_h
                grad_sum = grad_sum + upstream_grad(out_idx, s) * &
                     input%val(i_in + ( c_in - 1 ) * input_h, s)
             end if
          end do
       end do
       output(k_idx, 1) = grad_sum
    end do

  end subroutine get_partial_conv1d_kernel_val
!###############################################################################


!###############################################################################
  module function conv2d(input, kernel, stride, dilation) result(output)
    !! 2D convolution operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, dimension(2), intent(in) :: stride
    integer, dimension(2), intent(in) :: dilation
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: i_in, j_in, k_idx, out_idx, in_idx, in_base_idx, k_base_idx
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer :: channel_size_in, channel_size_out, kernel_channel_size
    integer :: dil_kernel_h_m1, dil_kernel_w_m1
    real(real32) :: conv_sum
    integer, dimension(4) :: output_shape

    ! Extract dimensions
    ! input: [H_in, W_in, C_in, B]
    ! kernel: [K_h, K_w, C_in, C_out]
    input_h = input%shape(1)
    input_w = input%shape(2)
    num_channels = input%shape(3)
    kernel_h = kernel%shape(1)
    kernel_w = kernel%shape(2)
    num_filters = kernel%shape(4)

    ! Pre-compute common values
    channel_size_in = input_h * input_w
    kernel_channel_size = kernel_h * kernel_w
    dil_kernel_h_m1 = dilation(1) * (kernel_h - 1)
    dil_kernel_w_m1 = dilation(2) * (kernel_w - 1)

    ! Calculate output dimensions
    output_h = (input_h - dil_kernel_h_m1 - 1) / stride(1) + 1
    output_w = (input_w - dil_kernel_w_m1 - 1) / stride(2) + 1
    output_shape = [output_h, output_w, num_filters, &
         size(input%val, dim=2)]

    output => input%create_result(array_shape = output_shape)
    output%val = 0._real32

    channel_size_out = output_h * output_w

    ! Perform convolution
    do concurrent(s = 1:output_shape(4), c_out = 1:num_filters, &
         j = 1:output_w, i = 1:output_h)
       conv_sum = 0._real32
       do c_in = 1, num_channels
          in_base_idx = (c_in - 1) * channel_size_in
          k_base_idx = (c_in - 1) * kernel_channel_size + &
               (c_out - 1) * kernel_channel_size * num_channels
          do kj = 1, kernel_w
             j_in = (j - 1) * stride(2) + (kj - 1) * dilation(2) + 1
             if(j_in .ge. 1 .and. j_in .le. input_w)then
                do ki = 1, kernel_h
                   i_in = (i - 1) * stride(1) + (ki - 1) * dilation(1) + 1
                   if(i_in .ge. 1 .and. i_in .le. input_h)then
                      in_idx = i_in + (j_in - 1) * input_h + in_base_idx
                      k_idx = ki + (kj - 1) * kernel_h + k_base_idx
                      conv_sum = conv_sum + input%val(in_idx, s) * &
                           kernel%val(k_idx, 1)
                   end if
                end do
             end if
          end do
       end do
       out_idx = i + (j - 1) * output_h + (c_out - 1) * channel_size_out
       output%val(out_idx, s) = conv_sum
    end do

    ! Store parameters for backward pass
    allocate(output%indices(2))
    output%indices(1) = num_channels
    output%indices(2) = num_filters
    allocate(output%adj_ja(2,3))
    output%adj_ja(1:2,1) = stride
    output%adj_ja(1:2,2) = dilation
    output%adj_ja(1,3) = kernel_h
    output%adj_ja(2,3) = kernel_w


    output%get_partial_left => get_partial_conv2d_input
    output%get_partial_right => get_partial_conv2d_kernel
    output%get_partial_left_val => get_partial_conv2d_input_val
    output%get_partial_right_val => get_partial_conv2d_kernel_val
    if(input%requires_grad .or. kernel%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'conv2d'
       output%left_operand => input
       output%right_operand => kernel
    end if

  end function conv2d
!-------------------------------------------------------------------------------
  function get_partial_conv2d_input(this, upstream_grad) result(output)
    !! Get partial derivative wrt input for 2D convolution
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%left_operand%shape, &
         size(this%left_operand%val, dim=2) ])
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_conv2d_input
!-------------------------------------------------------------------------------
  function get_partial_conv2d_kernel(this, upstream_grad) result(output)
    !! Get partial derivative wrt kernel for 2D convolution
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%right_operand%shape, 1 ])
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_conv2d_kernel
!-------------------------------------------------------------------------------
  subroutine get_partial_conv2d_input_val(this, upstream_grad, output)
    !! Get partial derivative wrt input for 2D convolution (subroutine version)
    implicit none

    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: i_in, j_in, k_idx, out_idx, in_idx
    integer :: in_base_idx, k_base_idx, kernel_channel_size
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer, dimension(2) :: stride, dilation
    integer :: channel_size_in, channel_size_out
    real(real32) :: grad_val, kernel_val
    class(array_type), pointer :: input, kernel

    input => this%left_operand
    kernel => this%right_operand

    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1:2,1)
    dilation = this%adj_ja(1:2,2)
    kernel_h = this%adj_ja(1,3)
    kernel_w = this%adj_ja(2,3)

    input_h = input%shape(1)
    input_w = input%shape(2)
    output_h = this%shape(1)
    output_w = this%shape(2)
    channel_size_in  = input_h * input_w
    channel_size_out = output_h * output_w
    kernel_channel_size = kernel_h * kernel_w

    output = 0._real32

    ! Parallelised over batch, output channels, output spatial dims
    do concurrent(s = 1:size(upstream_grad, dim=2), c_out = 1:num_filters, &
         j = 1:output_w, i = 1:output_h)
       out_idx = i + (j-1)*output_h + (c_out-1)*channel_size_out
       grad_val = upstream_grad(out_idx, s)

       if(abs(grad_val) .gt. 1.e-30_real32)then
          do c_in = 1, num_channels
             in_base_idx = (c_in - 1) * channel_size_in
             k_base_idx = (c_in - 1) * kernel_channel_size + &
                  (c_out - 1) * kernel_channel_size * num_channels

             do kj = 1, kernel_w
                j_in = (j - 1) * stride(2) + (kj - 1) * dilation(2) + 1
                if(j_in .ge. 1 .and. j_in .le. input_w)then
                   do ki = 1, kernel_h
                      i_in = (i - 1) * stride(1) + (ki - 1) * dilation(1) + 1
                      if(i_in .ge. 1 .and. i_in .le. input_h)then
                         in_idx = i_in + (j_in - 1) * input_h + in_base_idx
                         k_idx = (kernel_h - ki + 1) + &
                              (kernel_w - kj) * kernel_h + k_base_idx
                         kernel_val = kernel%val(k_idx, 1)
                         output(in_idx, s) = output(in_idx, s) + &
                              grad_val * kernel_val
                      end if
                   end do
                end if
             end do
          end do
       end if
    end do

  end subroutine get_partial_conv2d_input_val
!-------------------------------------------------------------------------------
  subroutine get_partial_conv2d_kernel_val(this, upstream_grad, output)
    !! Get partial derivative wrt kernel for 2D convolution (subroutine version)
    implicit none

    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: i_in, j_in, k_idx, out_idx, in_idx
    integer :: in_base_idx, out_base_idx, k_base_idx, kernel_channel_size
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer, dimension(2) :: stride, dilation
    integer :: channel_size_in, channel_size_out
    real(real32) :: grad_sum

    class(array_type), pointer :: input, kernel

    input => this%left_operand
    kernel => this%right_operand

    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1:2,1)
    dilation = this%adj_ja(1:2,2)
    kernel_h = this%adj_ja(1,3)
    kernel_w = this%adj_ja(2,3)

    input_h = input%shape(1)
    input_w = input%shape(2)
    output_h = this%shape(1)
    output_w = this%shape(2)
    channel_size_in  = input_h * input_w
    channel_size_out = output_h * output_w
    kernel_channel_size = kernel_h * kernel_w

    output = 0._real32

    ! Parallelised over filters, channels, and kernel dimensions
    do concurrent(c_out = 1:num_filters, c_in = 1:num_channels, &
         kj = 1:kernel_w, ki = 1:kernel_h)
       out_base_idx = (c_out - 1) * channel_size_out
       in_base_idx = (c_in - 1) * channel_size_in
       k_base_idx = (c_in - 1) * kernel_channel_size + &
            (c_out - 1) * kernel_channel_size * num_channels
       k_idx = ki + (kj - 1) * kernel_h + k_base_idx

       grad_sum = 0._real32
       do s = 1, size(upstream_grad, dim=2)
          do j = 1, output_w
             j_in = (j - 1) * stride(2) + (kj - 1) * dilation(2) + 1
             if(j_in .ge. 1 .and. j_in .le. input_w)then
                do i = 1, output_h
                   i_in = (i - 1) * stride(1) + (ki - 1) * dilation(1) + 1
                   if(i_in .ge. 1 .and. i_in .le. input_h)then
                      in_idx  = i_in + (j_in - 1) * input_h + in_base_idx
                      out_idx = i + (j - 1) * output_h + out_base_idx
                      grad_sum = grad_sum + &
                           upstream_grad(out_idx, s) * input%val(in_idx, s)
                   end if
                end do
             end if
          end do
       end do
       output(k_idx, 1) = grad_sum
    end do

  end subroutine get_partial_conv2d_kernel_val
!###############################################################################


!###############################################################################
  module function conv3d(input, kernel, stride, dilation) result(output)
    !! 3D convolution operation
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, dimension(3), intent(in) :: stride
    integer, dimension(3), intent(in) :: dilation
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, k, ki, kj, kk, c_in, c_out, s
    integer :: i_in, j_in, k_in, k_idx, out_idx, in_idx
    integer :: input_h, input_w, input_d, kernel_h, kernel_w, kernel_d
    integer :: output_h, output_w, output_d
    integer :: num_channels, num_filters
    integer :: channel_size_in, channel_size_out
    real(real32) :: conv_sum
    integer, dimension(5) :: output_shape

    ! Extract dimensions
    ! input: [H_in, W_in, D_in, C_in, B]
    ! kernel: [K_h, K_w, K_d, C_in, C_out]
    input_h = input%shape(1)
    input_w = input%shape(2)
    input_d = input%shape(3)
    num_channels = input%shape(4)
    kernel_h = kernel%shape(1)
    kernel_w = kernel%shape(2)
    kernel_d = kernel%shape(3)
    num_filters = kernel%shape(5)

    ! Calculate output dimensions
    output_h = (input_h - dilation(1)*(kernel_h - 1) - 1) / &
         stride(1) + 1
    output_w = (input_w - dilation(2)*(kernel_w - 1) - 1) / &
         stride(2) + 1
    output_d = (input_d - dilation(3)*(kernel_d - 1) - 1) / &
         stride(3) + 1
    output_shape = [output_h, output_w, output_d, num_filters, &
         size(input%val, dim=2)]

    output => input%create_result(array_shape = output_shape)
    output%val = 0._real32

    channel_size_in = input_h * input_w * input_d
    channel_size_out = output_h * output_w * output_d

    ! Perform convolution - optimised with do concurrent
    do concurrent(s = 1:output_shape(5), c_out = 1:num_filters, &
         k = 1:output_d, j = 1:output_w, i = 1:output_h)

       conv_sum = 0._real32
       do c_in = 1, num_channels
          do kk = 1, kernel_d
             k_in = ( k - 1 ) * stride(3) + ( kk - 1 ) * dilation(3) + 1
             if(k_in .ge. 1 .and. k_in .le. input_d)then
                do kj = 1, kernel_w
                   j_in = ( j - 1 ) * stride(2) + (kj - 1) * dilation(2) + 1
                   if(j_in .ge. 1 .and. j_in .le. input_w)then
                      do ki = 1, kernel_h
                         i_in = ( i - 1 ) * stride(1) + &
                              ( ki - 1 ) * dilation(1) + 1
                         if(i_in .ge. 1 .and. i_in .le. input_h)then
                            in_idx = i_in + ( j_in - 1 ) * input_h + &
                                 ( k_in - 1 ) * input_h * input_w + &
                                 ( c_in - 1 ) * channel_size_in
                            k_idx = ki + ( kj - 1 ) * kernel_h + &
                                 ( kk - 1 ) * kernel_h * kernel_w + &
                                 ( c_in - 1 ) * kernel_h * kernel_w * &
                                 kernel_d + &
                                 ( c_out - 1 ) * kernel_h * kernel_w * &
                                 kernel_d * num_channels
                            conv_sum = conv_sum + input%val(in_idx, s) * &
                                 kernel%val(k_idx, 1)
                         end if
                      end do
                   end if
                end do
             end if
          end do
       end do
       out_idx = i + ( j - 1 ) * output_h + &
            ( k - 1 ) * output_h * output_w + &
            ( c_out - 1 ) * channel_size_out
       output%val(out_idx, s) = conv_sum
    end do

    ! Store parameters for backward pass
    allocate(output%indices(2))
    output%indices(1) = num_channels
    output%indices(2) = num_filters
    allocate(output%adj_ja(3,3))
    output%adj_ja(1:3,1) = stride
    output%adj_ja(1:3,2) = dilation
    output%adj_ja(1,3) = kernel_h
    output%adj_ja(2,3) = kernel_w
    output%adj_ja(3,3) = kernel_d

    output%get_partial_left => get_partial_conv3d_input
    output%get_partial_right => get_partial_conv3d_kernel
    output%get_partial_left_val => get_partial_conv3d_input_val
    output%get_partial_right_val => get_partial_conv3d_kernel_val
    if(input%requires_grad .or. kernel%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'conv3d'
       output%left_operand => input
       output%right_operand => kernel
    end if

  end function conv3d
!-------------------------------------------------------------------------------
  function get_partial_conv3d_input(this, upstream_grad) result(output)
    !! Get partial derivative wrt input for 3D convolution
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%left_operand%shape, &
         size(this%left_operand%val, dim=2) ])
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_conv3d_input
!-------------------------------------------------------------------------------
  function get_partial_conv3d_kernel(this, upstream_grad) result(output)
    !! Get partial derivative wrt kernel for 3D convolution
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%right_operand%shape, 1 ])
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_conv3d_kernel
!-------------------------------------------------------------------------------
  subroutine get_partial_conv3d_input_val(this, upstream_grad, output)
    !! Get partial derivative wrt input for 3D convolution (subroutine version)
    implicit none

    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, k, ki, kj, kk, c_in, c_out, s
    integer :: i_in, j_in, k_in, k_idx, out_idx, in_idx
    integer :: input_h, input_w, input_d, kernel_h, kernel_w, kernel_d
    integer :: output_h, output_w, output_d
    integer :: num_channels, num_filters
    integer :: channel_size_in, channel_size_out
    integer, dimension(3) :: stride, dilation
    real(real32) :: grad_val, kernel_val
    class(array_type), pointer :: input, kernel

    input => this%left_operand
    kernel => this%right_operand

    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1:3,1)
    dilation = this%adj_ja(1:3,2)
    kernel_h = this%adj_ja(1,3)
    kernel_w = this%adj_ja(2,3)
    kernel_d = this%adj_ja(3,3)

    input_h = input%shape(1)
    input_w = input%shape(2)
    input_d = input%shape(3)
    output_h = this%shape(1)
    output_w = this%shape(2)
    output_d = this%shape(3)

    output = 0._real32

    channel_size_in = input_h * input_w * input_d
    channel_size_out = output_h * output_w * output_d

    do s = 1, size(upstream_grad, dim=2)
       do c_in = 1, num_channels
          do k = 1, output_d
             do j = 1, output_w
                do i = 1, output_h
                   do c_out = 1, num_filters
                      out_idx = i + ( j - 1 ) * output_h + &
                           ( k - 1 ) * output_h * output_w + &
                           ( c_out - 1 ) * channel_size_out
                      grad_val = upstream_grad(out_idx, s)

                      do kk = 1, kernel_d
                         k_in = ( k - 1 ) * stride(3) + &
                              ( kk - 1 ) * dilation(3) + 1
                         if( k_in .ge. 1 .and. k_in .le. input_d )then
                            do kj = 1, kernel_w
                               j_in = ( j - 1 ) * stride(2) + &
                                    ( kj - 1 ) * dilation(2) + 1
                               if( j_in .ge. 1 .and. j_in .le. input_w )then
                                  do ki = 1, kernel_h
                                     i_in = ( i - 1 ) * stride(1) + &
                                          ( ki - 1 ) * dilation(1) + 1
                                     if( i_in .ge. 1 .and. &
                                          i_in .le. input_h )then
                                        in_idx = i_in + &
                                             ( j_in - 1 ) * input_h + &
                                             ( k_in - 1 ) * input_h * input_w + &
                                             ( c_in - 1 ) * channel_size_in
                                        k_idx = ki + ( kj - 1 ) * kernel_h + &
                                             ( kk - 1 ) * kernel_h * kernel_w + &
                                             ( c_in - 1 ) * kernel_h * &
                                             kernel_w * kernel_d + &
                                             ( c_out - 1 ) * kernel_h * &
                                             kernel_w * kernel_d * num_channels
                                        kernel_val = kernel%val(k_idx, 1)
                                        output(in_idx, s) = &
                                             output(in_idx, s) + &
                                             grad_val * kernel_val
                                     end if
                                  end do
                               end if
                            end do
                         end if
                      end do
                   end do
                end do
             end do
          end do
       end do
    end do

  end subroutine get_partial_conv3d_input_val
!-------------------------------------------------------------------------------
  subroutine get_partial_conv3d_kernel_val(this, upstream_grad, output)
    !! Get partial derivative wrt kernel for 3D convolution (subroutine version)
    implicit none

    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, k, ki, kj, kk, c_in, c_out, s
    integer :: i_in, j_in, k_in, k_idx, out_idx, in_idx
    integer :: input_h, input_w, input_d, kernel_h, kernel_w, kernel_d
    integer :: output_h, output_w, output_d
    integer :: num_channels, num_filters
    integer :: channel_size_in, channel_size_out
    integer, dimension(3) :: stride, dilation
    real(real32) :: grad_sum
    class(array_type), pointer :: input, kernel

    input => this%left_operand
    kernel => this%right_operand

    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1:3,1)
    dilation = this%adj_ja(1:3,2)
    kernel_h = this%adj_ja(1,3)
    kernel_w = this%adj_ja(2,3)
    kernel_d = this%adj_ja(3,3)

    input_h = input%shape(1)
    input_w = input%shape(2)
    input_d = input%shape(3)
    output_h = this%shape(1)
    output_w = this%shape(2)
    output_d = this%shape(3)

    output = 0._real32

    channel_size_in = input_h * input_w * input_d
    channel_size_out = output_h * output_w * output_d

    do c_out = 1, num_filters
       do c_in = 1, num_channels
          do kk = 1, kernel_d
             do kj = 1, kernel_w
                do ki = 1, kernel_h
                   k_idx = ki + (kj-1)*kernel_h + &
                        (kk-1)*kernel_h*kernel_w + &
                        (c_in-1)*kernel_h*kernel_w*kernel_d + &
                        (c_out-1)*kernel_h*kernel_w*kernel_d*num_channels

                   grad_sum = 0._real32
                   do s = 1, size(upstream_grad, dim=2)
                      do k = 1, output_d
                         k_in = (k-1)*stride(3) + (kk-1)*dilation(3) + 1
                         if(k_in >= 1 .and. k_in <= input_d)then
                            do j = 1, output_w
                               j_in = (j-1)*stride(2) + &
                                    (kj-1)*dilation(2) + 1
                               if(j_in >= 1 .and. j_in <= input_w)then
                                  do i = 1, output_h
                                     i_in = (i-1)*stride(1) + &
                                          (ki-1)*dilation(1) + 1
                                     if(i_in >= 1 .and. i_in <= input_h)then
                                        in_idx = i_in + (j_in-1)*input_h + &
                                             (k_in-1)*input_h*input_w + &
                                             (c_in-1)*channel_size_in
                                        out_idx = i + (j-1)*output_h + &
                                             (k-1)*output_h*output_w + &
                                             (c_out-1)*channel_size_out
                                        grad_sum = grad_sum + &
                                             upstream_grad(out_idx, s) * &
                                             input%val(in_idx, s)
                                     end if
                                  end do
                               end if
                            end do
                         end if
                      end do
                   end do
                   output(k_idx, 1) = grad_sum
                end do
             end do
          end do
       end do
    end do

  end subroutine get_partial_conv3d_kernel_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_conv
