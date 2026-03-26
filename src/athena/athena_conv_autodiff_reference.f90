module athena__conv_autodiff_reference
  !! Reference convolution block used for benchmark and consistency checks.
  use coreutils, only: real32
  use diffstruc, only: array_type
  implicit none

  private

  public :: initialise_conv_autodiff_input
  public :: initialise_conv_autodiff_parameters
  public :: reference_conv_block_forward
  public :: reference_conv_block_backward
  public :: reference_conv2d_forward
  public :: reference_conv2d_backward
  public :: reference_conv2d_backward_original
  public :: reference_conv2d_original

contains

!###############################################################################
  subroutine initialise_conv_autodiff_input(input)
    implicit none

    real(real32), dimension(:,:,:,:), intent(out) :: input

    integer :: i, j, c, s

    do s = 1, size(input, dim=4)
       do c = 1, size(input, dim=3)
          do j = 1, size(input, dim=2)
             do i = 1, size(input, dim=1)
                input(i,j,c,s) = real( &
                     modulo(13 * i + 7 * j + 5 * c + 3 * s, 29), real32) / &
                29._real32 - 0.5_real32
             end do
          end do
       end do
    end do

  end subroutine initialise_conv_autodiff_input
!-------------------------------------------------------------------------------
  subroutine initialise_conv_autodiff_parameters( &
       kernel1, bias1, kernel2, bias2)
    implicit none

    real(real32), dimension(:,:,:,:), intent(out) :: kernel1
    real(real32), dimension(:), intent(out) :: bias1
    real(real32), dimension(:,:,:,:), intent(out) :: kernel2
    real(real32), dimension(:), intent(out) :: bias2

    integer :: i, j, c_in, c_out

    do c_out = 1, size(kernel1, dim=4)
       bias1(c_out) = 0.01_real32 * real(c_out - 1, real32)
       do c_in = 1, size(kernel1, dim=3)
          do j = 1, size(kernel1, dim=2)
             do i = 1, size(kernel1, dim=1)
                kernel1(i,j,c_in,c_out) = 0.02_real32 * real( &
                     modulo(i + 2 * j + 3 * c_in + 5 * c_out, 11) - 5, &
                     real32)
             end do
          end do
       end do
    end do

    do c_out = 1, size(kernel2, dim=4)
       bias2(c_out) = -0.015_real32 * real(c_out - 1, real32)
       do c_in = 1, size(kernel2, dim=3)
          do j = 1, size(kernel2, dim=2)
             do i = 1, size(kernel2, dim=1)
                kernel2(i,j,c_in,c_out) = 0.015_real32 * real( &
                     modulo(2 * i + 3 * j + 5 * c_in + 7 * c_out, 13) - 6, &
                     real32)
             end do
          end do
       end do
    end do

  end subroutine initialise_conv_autodiff_parameters
!-------------------------------------------------------------------------------
  subroutine reference_conv_block_forward( &
       input, kernel1, bias1, kernel2, bias2, conv1, act1, conv2, loss)
    implicit none

    real(real32), dimension(:,:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:,:), intent(in) :: kernel1
    real(real32), dimension(:), intent(in) :: bias1
    real(real32), dimension(:,:,:,:), intent(in) :: kernel2
    real(real32), dimension(:), intent(in) :: bias2
    real(real32), dimension(:,:,:,:), intent(out) :: conv1
    real(real32), dimension(:,:,:,:), intent(out) :: act1
    real(real32), dimension(:,:,:,:), intent(out) :: conv2
    real(real32), intent(out) :: loss

    call reference_conv2d_forward(input, kernel1, bias1, conv1)
    act1 = max(conv1, 0._real32)
    call reference_conv2d_forward(act1, kernel2, bias2, conv2)
    loss = sum(conv2) / real(size(conv2), real32)

  end subroutine reference_conv_block_forward
!-------------------------------------------------------------------------------
  subroutine reference_conv_block_backward( &
       input, kernel1, kernel2, conv1, conv2, grad_input, grad_kernel1, &
       grad_bias1, grad_kernel2, grad_bias2)
    implicit none

    real(real32), dimension(:,:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:,:), intent(in) :: kernel1
    real(real32), dimension(:,:,:,:), intent(in) :: kernel2
    real(real32), dimension(:,:,:,:), intent(in) :: conv1
    real(real32), dimension(:,:,:,:), intent(in) :: conv2
    real(real32), dimension(:,:,:,:), intent(out) :: grad_input
    real(real32), dimension(:,:,:,:), intent(out) :: grad_kernel1
    real(real32), dimension(:), intent(out) :: grad_bias1
    real(real32), dimension(:,:,:,:), intent(out) :: grad_kernel2
    real(real32), dimension(:), intent(out) :: grad_bias2

    real(real32), allocatable, dimension(:,:,:,:) :: grad_conv2
    real(real32), allocatable, dimension(:,:,:,:) :: grad_act1
    real(real32), allocatable, dimension(:,:,:,:) :: grad_conv1

    allocate(grad_conv2, mold = conv2)
    allocate(grad_act1, mold = conv1)
    allocate(grad_conv1, mold = conv1)

    grad_conv2 = 1._real32 / real(size(conv2), real32)
    call reference_conv2d_backward( &
         input = max(conv1, 0._real32), kernel = kernel2, &
         upstream_grad = grad_conv2, grad_input = grad_act1, &
         grad_kernel = grad_kernel2, grad_bias = grad_bias2)

    grad_conv1 = 0._real32
    where(conv1 .gt. 0._real32)
       grad_conv1 = grad_act1
    end where

    call reference_conv2d_backward( &
         input = input, kernel = kernel1, upstream_grad = grad_conv1, &
         grad_input = grad_input, grad_kernel = grad_kernel1, &
         grad_bias = grad_bias1)

    deallocate(grad_conv2)
    deallocate(grad_act1)
    deallocate(grad_conv1)

  end subroutine reference_conv_block_backward
!-------------------------------------------------------------------------------
  subroutine reference_conv2d_forward(input, kernel, bias, output)
    implicit none

    real(real32), dimension(:,:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:,:), intent(in) :: kernel
    real(real32), dimension(:), intent(in) :: bias
    real(real32), dimension(:,:,:,:), intent(out) :: output

    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: input_h, input_w, output_h, output_w
    real(real32) :: conv_sum

    input_h = size(input, dim=1)
    input_w = size(input, dim=2)
    output_h = size(output, dim=1)
    output_w = size(output, dim=2)

    do s = 1, size(input, dim=4)
       do c_out = 1, size(output, dim=3)
          do j = 1, output_w
             do i = 1, output_h
                conv_sum = 0._real32
                do c_in = 1, size(input, dim=3)
                   do kj = 1, size(kernel, dim=2)
                      if(j + kj - 1 .le. input_w)then
                         do ki = 1, size(kernel, dim=1)
                            if(i + ki - 1 .le. input_h)then
                               conv_sum = conv_sum + &
                                    input(i + ki - 1, j + kj - 1, c_in, s) * &
                                    kernel(ki, kj, c_in, c_out)
                            end if
                         end do
                      end if
                   end do
                end do
                output(i,j,c_out,s) = conv_sum + bias(c_out)
             end do
          end do
       end do
    end do

  end subroutine reference_conv2d_forward
!-------------------------------------------------------------------------------
  subroutine reference_conv2d_backward( &
       input, kernel, upstream_grad, grad_input, grad_kernel, grad_bias)
    implicit none

    real(real32), dimension(:,:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:,:), intent(in) :: kernel
    real(real32), dimension(:,:,:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:,:,:), intent(out) :: grad_input
    real(real32), dimension(:,:,:,:), intent(out) :: grad_kernel
    real(real32), dimension(:), intent(out) :: grad_bias

    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: i_in, j_in
    real(real32) :: grad_val

    grad_input = 0._real32
    grad_kernel = 0._real32
    grad_bias = 0._real32

    do s = 1, size(upstream_grad, dim=4)
       do c_out = 1, size(upstream_grad, dim=3)
          do j = 1, size(upstream_grad, dim=2)
             do i = 1, size(upstream_grad, dim=1)
                grad_val = upstream_grad(i,j,c_out,s)
                grad_bias(c_out) = grad_bias(c_out) + grad_val
                do c_in = 1, size(input, dim=3)
                   do kj = 1, size(kernel, dim=2)
                      j_in = j + kj - 1
                      if(j_in .le. size(input, dim=2))then
                         do ki = 1, size(kernel, dim=1)
                            i_in = i + ki - 1
                            if(i_in .le. size(input, dim=1))then
                               grad_kernel(ki,kj,c_in,c_out) = &
                                    grad_kernel(ki,kj,c_in,c_out) + &
                                    grad_val * input(i_in,j_in,c_in,s)
                               grad_input(i_in,j_in,c_in,s) = &
                                    grad_input(i_in,j_in,c_in,s) + &
                                    grad_val * kernel( &
                                         size(kernel, dim=1) - ki + 1, &
                                         size(kernel, dim=2) - kj + 1, c_in, c_out)
                            end if
                         end do
                      end if
                   end do
                end do
             end do
          end do
       end do
    end do

  end subroutine reference_conv2d_backward
!-------------------------------------------------------------------------------
  subroutine reference_conv2d_backward_original( &
       input, kernel, upstream_grad, grad_input, grad_kernel, grad_bias)
    !! Baseline split backward matching the original separate autodiff kernels.
    implicit none

    real(real32), dimension(:,:,:,:), intent(in) :: input
    real(real32), dimension(:,:,:,:), intent(in) :: kernel
    real(real32), dimension(:,:,:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:,:,:), intent(out) :: grad_input
    real(real32), dimension(:,:,:,:), intent(out) :: grad_kernel
    real(real32), dimension(:), intent(out) :: grad_bias

    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: i_in, j_in
    real(real32) :: grad_val, grad_sum

    grad_input = 0._real32
    grad_bias = 0._real32

    do s = 1, size(upstream_grad, dim=4)
       do c_out = 1, size(upstream_grad, dim=3)
          do j = 1, size(upstream_grad, dim=2)
             do i = 1, size(upstream_grad, dim=1)
                grad_val = upstream_grad(i,j,c_out,s)
                grad_bias(c_out) = grad_bias(c_out) + grad_val
                if(abs(grad_val) .gt. 1.e-30_real32)then
                   do c_in = 1, size(input, dim=3)
                      do kj = 1, size(kernel, dim=2)
                         j_in = j + kj - 1
                         if(j_in .le. size(input, dim=2))then
                            do ki = 1, size(kernel, dim=1)
                               i_in = i + ki - 1
                               if(i_in .le. size(input, dim=1))then
                                  grad_input(i_in,j_in,c_in,s) = &
                                       grad_input(i_in,j_in,c_in,s) + &
                                       grad_val * kernel( &
                                            size(kernel, dim=1) - ki + 1, &
                                            size(kernel, dim=2) - kj + 1, &
                                            c_in, c_out)
                               end if
                            end do
                         end if
                      end do
                   end do
                end if
             end do
          end do
       end do
    end do

    grad_kernel = 0._real32
    do c_out = 1, size(kernel, dim=4)
       do c_in = 1, size(kernel, dim=3)
          do kj = 1, size(kernel, dim=2)
             do ki = 1, size(kernel, dim=1)
                grad_sum = 0._real32
                do s = 1, size(upstream_grad, dim=4)
                   do j = 1, size(upstream_grad, dim=2)
                      j_in = j + kj - 1
                      if(j_in .le. size(input, dim=2))then
                         do i = 1, size(upstream_grad, dim=1)
                            i_in = i + ki - 1
                            if(i_in .le. size(input, dim=1))then
                               grad_sum = grad_sum + &
                                    upstream_grad(i,j,c_out,s) * &
                                    input(i_in,j_in,c_in,s)
                            end if
                         end do
                      end if
                   end do
                end do
                grad_kernel(ki,kj,c_in,c_out) = grad_sum
             end do
          end do
       end do
    end do

  end subroutine reference_conv2d_backward_original
!-------------------------------------------------------------------------------
  function reference_conv2d_original( &
       input, kernel, stride, dilation) result(output)
    !! Preserved original array_type-based conv2d autodiff implementation.
    implicit none

    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, dimension(2), intent(in) :: stride
    integer, dimension(2), intent(in) :: dilation
    type(array_type), pointer :: output

    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: i_in, j_in, k_idx, out_idx, in_idx
    integer :: in_base_idx, k_base_idx
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer :: channel_size_in, channel_size_out, kernel_channel_size
    integer :: dil_kernel_h_m1, dil_kernel_w_m1
    integer, dimension(4) :: output_shape
    real(real32) :: conv_sum

    input_h = input%shape(1)
    input_w = input%shape(2)
    num_channels = input%shape(3)
    kernel_h = kernel%shape(1)
    kernel_w = kernel%shape(2)
    num_filters = kernel%shape(4)

    channel_size_in = input_h * input_w
    kernel_channel_size = kernel_h * kernel_w
    dil_kernel_h_m1 = dilation(1) * (kernel_h - 1)
    dil_kernel_w_m1 = dilation(2) * (kernel_w - 1)

    output_h = (input_h - dil_kernel_h_m1 - 1) / stride(1) + 1
    output_w = (input_w - dil_kernel_w_m1 - 1) / stride(2) + 1
    output_shape = [output_h, output_w, num_filters, size(input%val, dim=2)]

    output => input%create_result(array_shape = output_shape)
    output%val = 0._real32

    channel_size_out = output_h * output_w

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

    allocate(output%indices(2))
    output%indices(1) = num_channels
    output%indices(2) = num_filters
    allocate(output%adj_ja(2,3))
    output%adj_ja(1:2,1) = stride
    output%adj_ja(1:2,2) = dilation
    output%adj_ja(1,3) = kernel_h
    output%adj_ja(2,3) = kernel_w
    output%get_partial_left_val => reference_conv2d_input_val_original
    output%get_partial_right_val => reference_conv2d_kernel_val_original
    if(input%requires_grad .or. kernel%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'conv2d_original'
       output%left_operand => input
       output%right_operand => kernel
    end if

  end function reference_conv2d_original
!-------------------------------------------------------------------------------
  pure subroutine reference_conv2d_input_val_original( &
       this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: i_in, j_in, k_idx, out_idx, in_idx
    integer :: in_base_idx, k_base_idx, kernel_channel_size
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer, dimension(2) :: stride, dilation
    integer :: channel_size_in, channel_size_out
    real(real32) :: grad_val, kernel_val

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
    channel_size_in = input_h * input_w
    channel_size_out = output_h * output_w
    kernel_channel_size = kernel_h * kernel_w

    output = 0._real32

    do concurrent(s = 1:size(upstream_grad, dim=2), c_out = 1:num_filters, &
         j = 1:output_w, i = 1:output_h)
       out_idx = i + (j - 1) * output_h + (c_out - 1) * channel_size_out
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
                         kernel_val = this%right_operand%val(k_idx, 1)
                         output(in_idx, s) = output(in_idx, s) + &
                              grad_val * kernel_val
                      end if
                   end do
                end if
             end do
          end do
       end if
    end do

  end subroutine reference_conv2d_input_val_original
!-------------------------------------------------------------------------------
  pure subroutine reference_conv2d_kernel_val_original( &
       this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j, ki, kj, c_in, c_out, s
    integer :: i_in, j_in, k_idx, out_idx, in_idx
    integer :: in_base_idx, out_base_idx, k_base_idx, kernel_channel_size
    integer :: input_h, input_w, kernel_h, kernel_w
    integer :: output_h, output_w, num_channels, num_filters
    integer, dimension(2) :: stride, dilation
    integer :: channel_size_in, channel_size_out
    real(real32) :: grad_sum

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
    channel_size_in = input_h * input_w
    channel_size_out = output_h * output_w
    kernel_channel_size = kernel_h * kernel_w

    output = 0._real32

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
                      in_idx = i_in + (j_in - 1) * input_h + in_base_idx
                      out_idx = i + (j - 1) * output_h + out_base_idx
                      grad_sum = grad_sum + &
                           upstream_grad(out_idx, s) * &
                           this%left_operand%val(in_idx, s)
                   end if
                end do
             end if
          end do
       end do
       output(k_idx, 1) = grad_sum
    end do

  end subroutine reference_conv2d_kernel_val_original
!###############################################################################

end module athena__conv_autodiff_reference
