submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_conv
  !! Submodule containing implementations for extended diffstruc array operations

contains

!-------------------------------------------------------------------------------
  pure function build_tap_map(input_n, kernel_n, stride, dilation, output_n) &
       result(map)
    !! Map an input position and kernel tap to the output position it feeds.
    !! A zero entry means the tap does not contribute to this input position.
    integer, intent(in) :: input_n, kernel_n, stride, dilation, output_n
    integer, dimension(input_n, kernel_n) :: map
    integer :: pos, tap, out_pos, numerator

    do tap = 1, kernel_n
       do pos = 1, input_n
          if(stride == 1 .and. dilation == 1) then
             out_pos = pos - (tap - 1)
          else
             numerator = pos - 1 - (tap - 1) * dilation
             if(numerator >= 0 .and. mod(numerator, stride) == 0) then
                out_pos = numerator / stride + 1
             else
                out_pos = 0
             end if
          end if
          if(out_pos < 1 .or. out_pos > output_n) out_pos = 0
          map(pos,tap) = out_pos
       end do
    end do
  end function build_tap_map
!############################################################################---

!###############################################################################
  pure subroutine conv2d_im2col(input_val, col, s, input_h, input_w, &
       num_channels, kernel_h, kernel_w, stride_h, stride_w, dil_h, dil_w, &
       output_h, output_w)
    !! Build one sample's patch matrix in the kernel's storage order.
    !!
    !! col(k, hw) = input at the tap k feeding output position hw, with
    !! k = (c_in, k_h, k_w) matching kernel%val.
    real(real32), dimension(:,:), intent(in) :: input_val
    real(real32), dimension(:), intent(out) :: col
    integer, intent(in) :: s, input_h, input_w, num_channels
    integer, intent(in) :: kernel_h, kernel_w, stride_h, stride_w, dil_h, dil_w
    integer, intent(in) :: output_h, output_w
    integer :: hw, ii, jj, cin, ki, kj, k, iin, jin

    do hw = 1, output_h * output_w
       ii = mod(hw - 1, output_h) + 1
       jj = (hw - 1) / output_h + 1
       do cin = 1, num_channels
          do kj = 1, kernel_w
             jin = (jj - 1) * stride_w + (kj - 1) * dil_w + 1
             do ki = 1, kernel_h
                iin = (ii - 1) * stride_h + (ki - 1) * dil_h + 1
                k = ki + (kj - 1) * kernel_h + (cin - 1) * kernel_h * kernel_w
                col(k + (hw - 1) * num_channels * kernel_h * kernel_w) = &
                     input_val(iin + (jin - 1) * input_h + &
                          (cin - 1) * input_h * input_w, s)
             end do
          end do
       end do
    end do
  end subroutine conv2d_im2col
!###############################################################################

!###############################################################################
  pure subroutine conv2d_col2im_add(dcol, xgrad, s, input_h, input_w, &
       num_channels, kernel_h, kernel_w, stride_h, stride_w, dil_h, dil_w, &
       output_h, output_w)
    !! Scatter-add one sample's patch gradients back to the input.
    real(real32), dimension(:), intent(in) :: dcol
    real(real32), dimension(:,:), intent(inout) :: xgrad
    integer, intent(in) :: s, input_h, input_w, num_channels
    integer, intent(in) :: kernel_h, kernel_w, stride_h, stride_w, dil_h, dil_w
    integer, intent(in) :: output_h, output_w
    integer :: hw, ii, jj, cin, ki, kj, k, iin, jin

    do hw = 1, output_h * output_w
       ii = mod(hw - 1, output_h) + 1
       jj = (hw - 1) / output_h + 1
       do cin = 1, num_channels
          do kj = 1, kernel_w
             jin = (jj - 1) * stride_w + (kj - 1) * dil_w + 1
             do ki = 1, kernel_h
                iin = (ii - 1) * stride_h + (ki - 1) * dil_h + 1
                if(iin < 1 .or. iin > input_h) cycle
                if(jin < 1 .or. jin > input_w) cycle
                k = ki + (kj - 1) * kernel_h + (cin - 1) * kernel_h * kernel_w
                xgrad(iin + (jin - 1) * input_h + &
                     (cin - 1) * input_h * input_w, s) = &
                     xgrad(iin + (jin - 1) * input_h + &
                          (cin - 1) * input_h * input_w, s) + &
                     dcol(k + (hw - 1) * num_channels * kernel_h * kernel_w)
             end do
          end do
       end do
    end do
  end subroutine conv2d_col2im_add
!###############################################################################

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
    real(real32), dimension(:), allocatable :: kernel_copy
    real(real32), pointer, contiguous :: in_val(:,:), out_val(:,:)
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

    allocate(kernel_copy(size(kernel%val,1)))
    kernel_copy(:) = kernel%val(:,1)
    in_val => input%val
    out_val => output%val

    ! Reuse one input patch for all output filters.
    do concurrent(s = 1:output_shape(3))
       block
         real(real32), dimension(num_channels, kernel_h) :: patch
         integer :: ii, cin, cout, kk
         integer :: iin
         real(real32) :: acc

         do ii = 1, output_h
            do cin = 1, num_channels
               do kk = 1, kernel_h
                  iin = (ii - 1) * stride + (kk - 1) * dilation + 1
                  patch(cin,kk) = &
                       input%val(iin + (cin - 1) * input_h, s)
               end do
            end do
            do cout = 1, num_filters
               acc = 0._real32
               do cin = 1, num_channels
                  do kk = 1, kernel_h
                     k_idx = kk + (cin - 1) * kernel_h + &
                          (cout - 1) * kernel_h * num_channels
                     acc = acc + patch(cin,kk) * kernel_copy(k_idx)
                  end do
               end do
               output%val(ii + (cout-1)*output_h, s) = acc
            end do
         end do
       end block
    end do
    deallocate(kernel_copy)

    ! Store parameters for backward pass
    allocate(output%indices(2))
    output%indices(1) = num_channels
    output%indices(2) = num_filters
    allocate(output%adj_ja(1,3))
    output%adj_ja(1,1) = stride
    output%adj_ja(1,2) = dilation
    output%adj_ja(1,3) = kernel_h

    output%value_partial_ok = .true.
    output%get_partial_left => get_partial_conv1d_input
    output%get_partial_right => get_partial_conv1d_kernel
    output%get_partial_left_val => get_partial_conv1d_input_val
    output%get_partial_right_val => get_partial_conv1d_kernel_val
    if(input%requires_grad .or. kernel%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'conv1d'
       output%left_operand => canonical(input)
       output%right_operand => canonical(kernel)
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
  pure subroutine get_partial_conv1d_input_val(this, upstream_grad, output)
    !! Get partial derivative wrt input for 1D convolution (subroutine version)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, k, c_in, c_out, s
    integer :: i_in, ii, k_idx, out_idx, in_idx
    integer :: input_h, kernel_h, output_h, num_channels, num_filters
    integer :: stride, dilation, numerator
    real(real32) :: acc, kernel_val
    integer, dimension(:,:), allocatable :: i_map


    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1,1)
    dilation = this%adj_ja(1,2)
    kernel_h = this%adj_ja(1,3)

    input_h = this%left_operand%shape(1)
    output_h = this%shape(1)

    i_map = build_tap_map(input_h, kernel_h, stride, dilation, output_h)

    do s = 1, size(upstream_grad, dim=2)
       do c_in = 1, num_channels
          do i_in = 1, input_h
             acc = 0._real32
             do c_out = 1, num_filters
                do k = 1, kernel_h
                   ii = i_map(i_in,k)
                   if(ii == 0) cycle
                   out_idx = ii + (c_out - 1) * output_h
                   k_idx = k + (c_in - 1) * kernel_h + &
                        (c_out - 1) * kernel_h * num_channels
                   kernel_val = this%right_operand%val(k_idx, 1)
                   acc = acc + upstream_grad(out_idx, s) * kernel_val
                end do
             end do
             in_idx = i_in + (c_in - 1) * input_h
             output(in_idx, s) = acc
          end do
       end do
    end do
    deallocate(i_map)

  end subroutine get_partial_conv1d_input_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_conv1d_kernel_val(this, upstream_grad, output)
    !! Get partial derivative wrt kernel for 1D convolution (subroutine version)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, k, c_in, c_out, s
    integer :: i_in, k_idx, out_idx
    integer :: input_h, kernel_h, output_h, num_channels, num_filters
    integer :: stride, dilation
    real(real32) :: grad_sum
    real(real32), dimension(:), allocatable :: dw
    real(real32), dimension(:,:), allocatable :: patch


    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1,1)
    dilation = this%adj_ja(1,2)
    kernel_h = this%adj_ja(1,3)

    input_h = this%left_operand%shape(1)
    output_h = this%shape(1)

    allocate(dw(size(output,1)))
    allocate(patch(num_channels, kernel_h))
    dw = 0._real32

    do s = 1, size(upstream_grad, dim=2)
       do i = 1, output_h
          do c_in = 1, num_channels
             do k = 1, kernel_h
                i_in = (i - 1) * stride + (k - 1) * dilation + 1
                patch(c_in,k) = this%left_operand%val( &
                     i_in + (c_in - 1) * input_h, s)
             end do
          end do

          do c_out = 1, num_filters
             out_idx = i + (c_out - 1) * output_h
             grad_sum = upstream_grad(out_idx, s)
             do c_in = 1, num_channels
                do k = 1, kernel_h
                   k_idx = k + (c_in - 1) * kernel_h + &
                        (c_out - 1) * kernel_h * num_channels
                   dw(k_idx) = dw(k_idx) + grad_sum * patch(c_in,k)
                end do
             end do
          end do
       end do
    end do

    output(:,1) = dw
    deallocate(dw, patch)

  end subroutine get_partial_conv1d_kernel_val
!###############################################################################


!###############################################################################
  module function conv2d(input, kernel, stride, dilation) result(output)
    !! 2D convolution operation
    !!
    !! Each sample's input patch matrix is built once and multiplied by the
    !! kernel with a single dense matrix product:
    !!
    !!   out[hw, c_out] = sum_k patch[k, hw] * kernel[k, c_out]
    !!
    !! with k = (c_in, k_h, k_w) in the kernel's storage order. This replaces
    !! the scalar filter loop with a BLAS3 call while keeping the value/graph
    !! layout unchanged.
    implicit none

    ! Arguments
    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, dimension(2), intent(in) :: stride
    integer, dimension(2), intent(in) :: dilation
    type(array_type), pointer :: output

    ! Local variables
    integer :: s
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer :: channel_size_in, kernel_size, num_positions
    integer :: dil_kernel_h_m1, dil_kernel_w_m1
    real(real32), dimension(:), allocatable :: col
    integer, dimension(4) :: output_shape

    ! input: [H_in, W_in, C_in, B]; kernel: [K_h, K_w, C_in, C_out]
    input_h = input%shape(1)
    input_w = input%shape(2)
    num_channels = input%shape(3)
    kernel_h = kernel%shape(1)
    kernel_w = kernel%shape(2)
    num_filters = kernel%shape(4)

    channel_size_in = input_h * input_w
    kernel_size = kernel_h * kernel_w
    dil_kernel_h_m1 = dilation(1) * (kernel_h - 1)
    dil_kernel_w_m1 = dilation(2) * (kernel_w - 1)
    output_h = (input_h - dil_kernel_h_m1 - 1) / stride(1) + 1
    output_w = (input_w - dil_kernel_w_m1 - 1) / stride(2) + 1
    num_positions = output_h * output_w
    output_shape = [output_h, output_w, num_filters, size(input%val, dim=2)]

    output => input%create_result(array_shape = output_shape)

    allocate(col(num_channels * kernel_size * num_positions))

    do s = 1, size(input%val, dim=2)
       call conv2d_im2col(input%val, col, s, input_h, input_w, num_channels, &
            kernel_h, kernel_w, stride(1), stride(2), dilation(1), dilation(2), &
            output_h, output_w)
       call gemm('T', 'N', num_positions, num_filters, &
            num_channels*kernel_size, col, num_channels*kernel_size, &
            kernel%val(:,1), num_channels*kernel_size, 0.0_real32, &
            output%val(1:num_positions*num_filters,s), num_positions)
    end do
    deallocate(col)

    ! Store parameters for the partials
    allocate(output%indices(2))
    output%indices(1) = num_channels
    output%indices(2) = num_filters
    allocate(output%adj_ja(2,3))
    output%adj_ja(1:2,1) = stride
    output%adj_ja(1:2,2) = dilation
    output%adj_ja(1,3) = kernel_h
    output%adj_ja(2,3) = kernel_w

    output%value_partial_ok = .true.
    output%get_partial_left => get_partial_conv2d_input
    output%get_partial_right => get_partial_conv2d_kernel
    output%get_partial_left_val => get_partial_conv2d_input_val
    output%get_partial_right_val => get_partial_conv2d_kernel_val
    output%get_partial_right_val_sum => get_partial_conv2d_kernel_val_sum
    if(input%requires_grad .or. kernel%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'conv2d'
       output%left_operand => canonical(input)
       output%right_operand => canonical(kernel)
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
    !! Gradient with respect to the input: one dense product per sample
    !! followed by a scatter-add of the patch gradients (the transpose of the
    !! forward im2col).
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: s
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer :: kernel_size, num_positions, channel_size_in
    integer, dimension(2) :: stride, dilation
    real(real32), dimension(:), allocatable :: dcol

    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1:2,1)
    dilation = this%adj_ja(1:2,2)
    kernel_h = this%adj_ja(1,3)
    kernel_w = this%adj_ja(2,3)

    input_h = this%left_operand%shape(1)
    input_w = this%left_operand%shape(2)
    output_h = this%shape(1)
    output_w = this%shape(2)
    num_positions = output_h * output_w
    kernel_size = kernel_h * kernel_w
    channel_size_in = input_h * input_w

    allocate(dcol(num_channels*kernel_size * num_positions))

    output = 0.0_real32
    do s = 1, size(upstream_grad, dim=2)
       call gemm('N', 'T', num_channels*kernel_size, num_positions, &
            num_filters, this%right_operand%val(:,1), &
            num_channels*kernel_size, &
            upstream_grad(1:num_positions*num_filters,s), num_positions, &
            0.0_real32, dcol, num_channels*kernel_size)
       call conv2d_col2im_add(dcol, output, s, input_h, input_w, num_channels, &
            kernel_h, kernel_w, stride(1), stride(2), dilation(1), dilation(2), &
            output_h, output_w)
    end do
    deallocate(dcol)
  end subroutine get_partial_conv2d_input_val
!-------------------------------------------------------------------------------
  subroutine get_partial_conv2d_kernel_val(this, upstream_grad, output)
    !! Gradient with respect to the kernel for one buffered call.
    !!
    !! The reduced form is evaluated once across the batch and stored in the
    !! first column; remaining columns are left untouched, matching the layout
    !! the reverse engine supplies for a non-sample-dependent operand.
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    call get_partial_conv2d_kernel_val_sum(this, upstream_grad, output(:,1))
  end subroutine get_partial_conv2d_kernel_val
!-------------------------------------------------------------------------------
  subroutine get_partial_conv2d_kernel_val_sum(this, upstream_grad, output)
    !! Batch-summed gradient with respect to the kernel.
    !!
    !! dW[k, c_out] += sum_{s, hw} patch_s[k, hw] * upstream_s[hw, c_out],
    !! accumulated with one dense product per sample straight into the
    !! caller's accumulator.
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    ! Local variables
    integer :: s
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer :: kernel_size, num_positions
    integer, dimension(2) :: stride, dilation
    real(real32), dimension(:), allocatable :: col

    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1:2,1)
    dilation = this%adj_ja(1:2,2)
    kernel_h = this%adj_ja(1,3)
    kernel_w = this%adj_ja(2,3)

    input_h = this%left_operand%shape(1)
    input_w = this%left_operand%shape(2)
    output_h = this%shape(1)
    output_w = this%shape(2)
    num_positions = output_h * output_w
    kernel_size = kernel_h * kernel_w

    allocate(col(num_channels*kernel_size * num_positions))
    output = 0.0_real32

    do s = 1, size(upstream_grad, dim=2)
       call conv2d_im2col(this%left_operand%val, col, s, input_h, input_w, &
            num_channels, kernel_h, kernel_w, stride(1), stride(2), &
            dilation(1), dilation(2), output_h, output_w)
       call gemm('N', 'N', num_channels*kernel_size, num_filters, &
            num_positions, col, num_channels*kernel_size, &
            upstream_grad(1:num_positions*num_filters,s), num_positions, &
            1.0_real32, output, num_channels*kernel_size)
    end do
    deallocate(col)
  end subroutine get_partial_conv2d_kernel_val_sum
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
    integer :: in_base_idx, k_base_idx
    integer :: input_h, input_w, input_d, kernel_h, kernel_w, kernel_d
    integer :: output_h, output_w, output_d
    integer :: num_channels, num_filters
    integer :: channel_size_in, channel_size_out
    real(real32) :: conv_sum
    real(real32), dimension(:), allocatable :: kernel_copy
    real(real32), pointer, contiguous :: in_val(:,:), out_val(:,:)
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

    allocate(kernel_copy(size(kernel%val,1)))
    kernel_copy(:) = kernel%val(:,1)
    in_val => input%val
    out_val => output%val

    ! Reuse one input patch across all output filters.
    do concurrent(s = 1:output_shape(5))
       block
         real(real32), dimension(num_channels, kernel_h, kernel_w, &
              kernel_d) :: patch
         integer :: ii, jj, kk_, cin, cout, ki_, kj_, kk__
         integer :: iin, jin, kin
         real(real32) :: acc

         do kk__ = 1, output_d
            do jj = 1, output_w
               do ii = 1, output_h
                  do cin = 1, num_channels
                     in_base_idx = (cin - 1) * channel_size_in
                     do kk_ = 1, kernel_d
                        kin = (kk__ - 1) * stride(3) + &
                             (kk_ - 1) * dilation(3) + 1
                        do kj_ = 1, kernel_w
                           jin = (jj - 1) * stride(2) + &
                                (kj_ - 1) * dilation(2) + 1
                           do ki_ = 1, kernel_h
                              iin = (ii - 1) * stride(1) + &
                                   (ki_ - 1) * dilation(1) + 1
                              patch(cin,ki_,kj_,kk_) = input%val( &
                                   iin + (jin - 1) * input_h + &
                                   (kin - 1) * input_h * input_w + &
                                   in_base_idx, s)
                           end do
                        end do
                     end do
                  end do

                  do cout = 1, num_filters
                     acc = 0._real32
                     do cin = 1, num_channels
                        k_base_idx = (cin - 1) * kernel_h * kernel_w * &
                             kernel_d + (cout - 1) * kernel_h * kernel_w * &
                             kernel_d * num_channels
                        do kk_ = 1, kernel_d
                           do kj_ = 1, kernel_w
                              do ki_ = 1, kernel_h
                                 k_idx = ki_ + (kj_ - 1) * kernel_h + &
                                      (kk_ - 1) * kernel_h * kernel_w + &
                                      k_base_idx
                                 acc = acc + patch(cin,ki_,kj_,kk_) * &
                                      kernel_copy(k_idx)
                              end do
                           end do
                        end do
                     end do
                     out_idx = ii + (jj - 1) * output_h + &
                          (kk__ - 1) * output_h * output_w + &
                          (cout - 1) * channel_size_out
                     out_val(out_idx, s) = acc
                  end do
               end do
            end do
         end do
       end block
    end do
    deallocate(kernel_copy)

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

    output%value_partial_ok = .true.
    output%get_partial_left => get_partial_conv3d_input
    output%get_partial_right => get_partial_conv3d_kernel
    output%get_partial_left_val => get_partial_conv3d_input_val
    output%get_partial_right_val => get_partial_conv3d_kernel_val
    if(input%requires_grad .or. kernel%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'conv3d'
       output%left_operand => canonical(input)
       output%right_operand => canonical(kernel)
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
  pure subroutine get_partial_conv3d_input_val(this, upstream_grad, output)
    !! Get partial derivative wrt input for 3D convolution (subroutine version)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, k, ki, kj, kk, c_in, c_out, s
    integer :: i_in, j_in, k_in, ii, jj, kin, k_idx, out_idx, in_idx
    integer :: input_h, input_w, input_d, kernel_h, kernel_w, kernel_d
    integer :: output_h, output_w, output_d
    integer :: num_channels, num_filters
    integer :: channel_size_in, channel_size_out
    integer, dimension(3) :: stride, dilation
    integer :: numerator
    real(real32) :: acc, kernel_val
    integer, dimension(:,:), allocatable :: i_map, j_map, k_map


    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1:3,1)
    dilation = this%adj_ja(1:3,2)
    kernel_h = this%adj_ja(1,3)
    kernel_w = this%adj_ja(2,3)
    kernel_d = this%adj_ja(3,3)

    input_h = this%left_operand%shape(1)
    input_w = this%left_operand%shape(2)
    input_d = this%left_operand%shape(3)
    output_h = this%shape(1)
    output_w = this%shape(2)
    output_d = this%shape(3)

    channel_size_in = input_h * input_w * input_d
    channel_size_out = output_h * output_w * output_d

    i_map = build_tap_map(input_h, kernel_h, stride(1), dilation(1), output_h)
    j_map = build_tap_map(input_w, kernel_w, stride(2), dilation(2), output_w)
    k_map = build_tap_map(input_d, kernel_d, stride(3), dilation(3), output_d)

    do s = 1, size(upstream_grad, dim=2)
       do c_in = 1, num_channels
          do k_in = 1, input_d
             do j_in = 1, input_w
                do i_in = 1, input_h
                   acc = 0._real32
                   do c_out = 1, num_filters
                      do kk = 1, kernel_d
                         kin = k_map(k_in,kk)
                         if(kin == 0) cycle
                         do kj = 1, kernel_w
                            jj = j_map(j_in,kj)
                            if(jj == 0) cycle
                            do ki = 1, kernel_h
                               ii = i_map(i_in,ki)
                               if(ii == 0) cycle
                               out_idx = ii + (jj - 1) * output_h + &
                                    (kin - 1) * output_h * output_w + &
                                    (c_out - 1) * channel_size_out
                               k_idx = ki + (kj - 1) * kernel_h + &
                                    (kk - 1) * kernel_h * kernel_w + &
                                    (c_in - 1) * kernel_h * kernel_w * &
                                    kernel_d + &
                                    (c_out - 1) * kernel_h * kernel_w * &
                                    kernel_d * num_channels
                               kernel_val = this%right_operand%val(k_idx, 1)
                               acc = acc + upstream_grad(out_idx, s) * &
                                    kernel_val
                            end do
                         end do
                      end do
                   end do
                   in_idx = i_in + (j_in - 1) * input_h + &
                        (k_in - 1) * input_h * input_w + &
                        (c_in - 1) * channel_size_in
                   output(in_idx, s) = acc
                end do
             end do
          end do
       end do
    end do
    deallocate(i_map, j_map, k_map)

  end subroutine get_partial_conv3d_input_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_conv3d_kernel_val(this, upstream_grad, output)
    !! Get partial derivative wrt kernel for 3D convolution (subroutine version)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    ! Local variables
    integer :: i, j, k, ki, kj, kk, c_in, c_out, s
    integer :: i_in, j_in, k_in, k_idx, out_idx, in_idx
    integer :: in_base_idx, k_base_idx
    integer :: input_h, input_w, input_d, kernel_h, kernel_w, kernel_d
    integer :: output_h, output_w, output_d
    integer :: num_channels, num_filters
    integer :: channel_size_in, channel_size_out
    integer, dimension(3) :: stride, dilation
    real(real32) :: grad_sum
    real(real32), dimension(:), allocatable :: dw
    real(real32), dimension(:,:,:,:), allocatable :: patch


    ! Unpack parameters
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    stride = this%adj_ja(1:3,1)
    dilation = this%adj_ja(1:3,2)
    kernel_h = this%adj_ja(1,3)
    kernel_w = this%adj_ja(2,3)
    kernel_d = this%adj_ja(3,3)

    input_h = this%left_operand%shape(1)
    input_w = this%left_operand%shape(2)
    input_d = this%left_operand%shape(3)
    output_h = this%shape(1)
    output_w = this%shape(2)
    output_d = this%shape(3)

    channel_size_in = input_h * input_w * input_d
    channel_size_out = output_h * output_w * output_d

    allocate(dw(size(output,1)))
    allocate(patch(num_channels, kernel_h, kernel_w, kernel_d))
    dw = 0._real32

    do s = 1, size(upstream_grad, dim=2)
       do k = 1, output_d
          do j = 1, output_w
             do i = 1, output_h
                do c_in = 1, num_channels
                   in_base_idx = (c_in - 1) * channel_size_in
                   do kk = 1, kernel_d
                      k_in = (k - 1) * stride(3) + (kk - 1) * dilation(3) + 1
                      do kj = 1, kernel_w
                         j_in = (j - 1) * stride(2) + &
                              (kj - 1) * dilation(2) + 1
                         do ki = 1, kernel_h
                            i_in = (i - 1) * stride(1) + &
                                 (ki - 1) * dilation(1) + 1
                            in_idx = i_in + (j_in - 1) * input_h + &
                                 (k_in - 1) * input_h * input_w + in_base_idx
                            patch(c_in,ki,kj,kk) = &
                                 this%left_operand%val(in_idx, s)
                         end do
                      end do
                   end do
                end do

                do c_out = 1, num_filters
                   out_idx = i + (j - 1) * output_h + &
                        (k - 1) * output_h * output_w + &
                        (c_out - 1) * channel_size_out
                   grad_sum = upstream_grad(out_idx, s)
                   do c_in = 1, num_channels
                      k_base_idx = (c_in - 1) * kernel_h * kernel_w * &
                           kernel_d + (c_out - 1) * kernel_h * kernel_w * &
                           kernel_d * num_channels
                      do kk = 1, kernel_d
                         do kj = 1, kernel_w
                            do ki = 1, kernel_h
                               k_idx = ki + (kj - 1) * kernel_h + &
                                    (kk - 1) * kernel_h * kernel_w + k_base_idx
                               dw(k_idx) = dw(k_idx) + &
                                    grad_sum * patch(c_in,ki,kj,kk)
                            end do
                         end do
                      end do
                   end do
                end do
             end do
          end do
       end do
    end do

    output(:,1) = dw
    deallocate(dw, patch)

  end subroutine get_partial_conv3d_kernel_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_conv
