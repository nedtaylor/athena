submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_conv
  !! Submodule containing implementations for extended diffstruc array operations

contains

  module function conv1d(input, kernel, stride, dilation) result(output)
    !! 1D convolution operation
    implicit none
    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, intent(in) :: stride, dilation
    type(array_type), pointer :: output

    output => conv_nd(input, kernel, 1, [stride], [dilation])
  end function conv1d
!############################################################################---

!###############################################################################
  module function conv2d(input, kernel, stride, dilation) result(output)
    !! 2D convolution operation
    implicit none
    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, dimension(2), intent(in) :: stride, dilation
    type(array_type), pointer :: output

    output => conv_nd(input, kernel, 2, stride, dilation)
  end function conv2d
!############################################################################---

!###############################################################################
  module function conv3d(input, kernel, stride, dilation) result(output)
    !! 3D convolution operation
    implicit none
    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, dimension(3), intent(in) :: stride, dilation
    type(array_type), pointer :: output

    output => conv_nd(input, kernel, 3, stride, dilation)
  end function conv3d
!############################################################################---

!###############################################################################
  function conv_nd(input, kernel, nd, stride, dilation) result(output)
    !! Shared forward path for 1D/2D/3D convolution.
    !!
    !! For each sample, the receptive-field patches are gathered into a
    !! K x positions matrix (K = C_in * kernel volume) and multiplied by the
    !! kernel with one dense product. All dimensions use the same value and
    !! gradient code; only the geometry differs.
    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    integer, intent(in) :: nd
    integer, dimension(nd), intent(in) :: stride, dilation
    type(array_type), pointer :: output

    integer :: d, s, nth
    integer :: in_dims(3), k_dims(3), out_dims(3), shape_(5)
    integer :: stride_(3), dilation_(3)
    integer :: num_channels, num_filters, kernel_volume
    integer :: num_positions, kernel_size
    integer, allocatable :: table(:,:)
    real(real32), dimension(:), allocatable :: col

    in_dims = 0
    k_dims = 0
    out_dims = 0
    stride_ = 0
    dilation_ = 0
    do d = 1, nd
       in_dims(d) = input%shape(d)
       k_dims(d) = kernel%shape(d)
       stride_(d) = stride(d)
       dilation_(d) = dilation(d)
       out_dims(d) = (in_dims(d) - dilation(d)*(k_dims(d)-1) - 1) / &
            stride(d) + 1
    end do
    num_channels = input%shape(nd+1)
    num_filters = kernel%shape(nd+2)
    kernel_volume = product(k_dims(1:nd))
    num_positions = product(out_dims(1:nd))
    kernel_size = num_channels * kernel_volume

    shape_(1:nd) = out_dims(1:nd)
    shape_(nd+1) = num_filters
    shape_(nd+2) = size(input%val, dim=2)
    output => input%create_result(array_shape = shape_(1:nd+2))

    call conv_build_table(table, in_dims, k_dims, out_dims, stride_, &
         dilation_, nd, num_channels, kernel_size, num_positions)
    ! Each sample owns its column of the output, the patch buffer and the
    ! matrix product, so the loop threads safely; the table and kernel are
    ! read-only. Threading is opt-in: the default context requests one thread,
    ! which is faster here because each matrix product is small and BLAS
    ! already uses the available cores for larger shapes.
    nth = 1
#ifdef _OPENMP
    nth = get_num_threads(context_of(input))
    if(nth <= 0) nth = 1
#endif
    !$omp parallel do schedule(static) private(col) num_threads(nth) &
    !$omp if(nth > 1 .and. size(input%val, dim=2) > 1)
    do s = 1, size(input%val, dim=2)
       allocate(col(kernel_size * num_positions))
       call conv_gather(input%val, col, table, s, kernel_size, num_positions)
       call gemm('T', 'N', num_positions, num_filters, kernel_size, &
            col, kernel_size, kernel%val(:,1), kernel_size, 0.0_real32, &
            output%val(1:num_positions*num_filters, s), num_positions)
       deallocate(col)
    end do
    !$omp end parallel do
    deallocate(table)

    allocate(output%indices(3))
    output%indices(1) = num_channels
    output%indices(2) = num_filters
    output%indices(3) = nd
    allocate(output%adj_ja(nd,3))
    output%adj_ja(:,1) = stride
    output%adj_ja(:,2) = dilation
    output%adj_ja(:,3) = k_dims(1:nd)

    output%value_partial_ok = .true.
    output%get_partial_left => get_partial_conv_input
    output%get_partial_right => get_partial_conv_kernel
    output%get_partial_left_val => get_partial_conv_input_val
    output%get_partial_right_val => get_partial_conv_kernel_val
    output%get_partial_right_val_sum => get_partial_conv_kernel_val_sum
    if(input%requires_grad .or. kernel%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'conv'//char(iachar('0') + nd)//'d'
       output%left_operand => canonical(input)
       output%right_operand => canonical(kernel)
    end if
  end function conv_nd
!############################################################################---

!###############################################################################
  subroutine conv_build_table(table, in_dims, k_dims, out_dims, stride, &
       dilation, nd, num_channels, kernel_size, num_positions)
    !! Map every (kernel tap, output position) to its input offset.
    !!
    !! A zero entry marks a tap outside the input (only possible with padding
    !! geometries that the layers do not produce). The table removes all index
    !! arithmetic from the gather/scatter loops.
    integer, allocatable, intent(out) :: table(:,:)
    integer, dimension(3), intent(in) :: in_dims, k_dims, out_dims
    integer, dimension(3), intent(in) :: stride, dilation
    integer, intent(in) :: nd, num_channels, kernel_size, num_positions

    integer :: d, hw, ci, tap, k, t, in_volume, kernel_volume
    integer :: opos(3), kpos(3), ipos(3), offset, base, idx

    allocate(table(kernel_size, num_positions))
    table = 0
    in_volume = product(in_dims(1:nd))
    kernel_volume = product(k_dims(1:nd))
    do hw = 1, num_positions
       t = hw - 1
       do d = 1, nd
          opos(d) = mod(t, out_dims(d)) + 1
          t = t / out_dims(d)
       end do
       do ci = 1, num_channels
          do tap = 1, kernel_volume
             t = tap - 1
             do d = 1, nd
                kpos(d) = mod(t, k_dims(d)) + 1
                t = t / k_dims(d)
             end do
             offset = 1
             base = 1
             do d = 1, nd
                ipos(d) = (opos(d)-1)*stride(d) + (kpos(d)-1)*dilation(d) + 1
                if(ipos(d) < 1 .or. ipos(d) > in_dims(d)) exit
                offset = offset + (ipos(d)-1)*base
                base = base * in_dims(d)
             end do
             k = tap + (ci-1)*kernel_volume
             if(d > nd) table(k,hw) = offset + (ci-1)*in_volume
          end do
       end do
    end do
  end subroutine conv_build_table
!############################################################################---

!###############################################################################
  pure subroutine conv_gather(input_val, col, table, s, kernel_size, num_positions)
    !! Gather one sample's receptive fields into the patch matrix.
    real(real32), dimension(:,:), intent(in) :: input_val
    real(real32), dimension(:), intent(out) :: col
    integer, dimension(:,:), intent(in) :: table
    integer, intent(in) :: s, kernel_size, num_positions
    integer :: hw, k, idx

    do hw = 1, num_positions
       do k = 1, kernel_size
          idx = table(k, hw)
          if(idx > 0) then
             col(k + (hw-1)*kernel_size) = input_val(idx, s)
          else
             col(k + (hw-1)*kernel_size) = 0.0_real32
          end if
       end do
    end do
  end subroutine conv_gather
!############################################################################---

!###############################################################################
  pure subroutine conv_scatter_add(dcol, xgrad, table, s, kernel_size, num_positions)
    !! Scatter-add one sample's patch gradients back to the input.
    real(real32), dimension(:), intent(in) :: dcol
    real(real32), dimension(:,:), intent(inout) :: xgrad
    integer, dimension(:,:), intent(in) :: table
    integer, intent(in) :: s, kernel_size, num_positions
    integer :: hw, k, idx

    do hw = 1, num_positions
       do k = 1, kernel_size
          idx = table(k, hw)
          if(idx > 0) then
             xgrad(idx, s) = xgrad(idx, s) + &
                  dcol(k + (hw-1)*kernel_size)
          end if
       end do
    end do
  end subroutine conv_scatter_add
!############################################################################---

!###############################################################################
  function get_partial_conv_input(this, upstream_grad) result(output)
    !! Graph-building partial with respect to the input.
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%left_operand%shape, &
         size(this%left_operand%val, dim=2) ])
    call this%get_partial_left_val(upstream_grad%val, output%val)
  end function get_partial_conv_input
!############################################################################---

!###############################################################################
  function get_partial_conv_kernel(this, upstream_grad) result(output)
    !! Graph-building partial with respect to the kernel.
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%right_operand%shape, 1 ])
    call this%get_partial_right_val(upstream_grad%val, output%val)
  end function get_partial_conv_kernel
!############################################################################---

!###############################################################################
  subroutine get_partial_conv_input_val(this, upstream_grad, output)
    !! Gradient with respect to the input: one dense product per sample plus
    !! scatter-add of the patch gradients.
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: nd, s, d
    integer :: in_dims(3), k_dims(3), out_dims(3), stride_(3), dilation_(3)
    integer :: num_channels, num_filters, kernel_volume, kernel_size
    integer :: num_positions
    integer, allocatable :: table(:,:)
    real(real32), dimension(:), allocatable :: dcol

    call conv_unpack(this, nd, in_dims, k_dims, out_dims, stride_, dilation_, &
         num_channels, num_filters, kernel_volume, kernel_size, num_positions)
    call conv_build_table(table, in_dims, k_dims, out_dims, stride_, &
         dilation_, nd, num_channels, kernel_size, num_positions)
    allocate(dcol(kernel_size * num_positions))

    output = 0.0_real32
    do s = 1, size(upstream_grad, dim=2)
       call gemm('N', 'T', kernel_size, num_positions, num_filters, &
            this%right_operand%val(:,1), kernel_size, &
            upstream_grad(1:num_positions*num_filters, s), num_positions, &
            0.0_real32, dcol, kernel_size)
       call conv_scatter_add(dcol, output, table, s, kernel_size, num_positions)
    end do
    deallocate(dcol, table)
  end subroutine get_partial_conv_input_val
!############################################################################---

!###############################################################################
  subroutine get_partial_conv_kernel_val(this, upstream_grad, output)
    !! Gradient with respect to the kernel for one buffered call.
    !!
    !! The batch-reduced result is written to the first column and every other
    !! column is zeroed, so a caller that sums columns still gets the correct
    !! value instead of relying on scratch contents.
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = 0.0_real32
    call get_partial_conv_kernel_val_sum(this, upstream_grad, output(:,1))
  end subroutine get_partial_conv_kernel_val
!############################################################################---

!###############################################################################
  subroutine get_partial_conv_kernel_val_sum(this, upstream_grad, output)
    !! Batch-summed gradient with respect to the kernel, written straight into
    !! the caller's accumulator.
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: nd, s
    integer :: in_dims(3), k_dims(3), out_dims(3), stride_(3), dilation_(3)
    integer :: num_channels, num_filters, kernel_volume, kernel_size
    integer :: num_positions
    integer, allocatable :: table(:,:)
    real(real32), dimension(:), allocatable :: col

    call conv_unpack(this, nd, in_dims, k_dims, out_dims, stride_, dilation_, &
         num_channels, num_filters, kernel_volume, kernel_size, num_positions)
    call conv_build_table(table, in_dims, k_dims, out_dims, stride_, &
         dilation_, nd, num_channels, kernel_size, num_positions)
    allocate(col(kernel_size * num_positions))
    output = 0.0_real32

    do s = 1, size(upstream_grad, dim=2)
       call conv_gather(this%left_operand%val, col, table, s, &
            kernel_size, num_positions)
       call gemm('N', 'N', kernel_size, num_filters, num_positions, &
            col, kernel_size, &
            upstream_grad(1:num_positions*num_filters, s), num_positions, &
            1.0_real32, output, kernel_size)
    end do
    deallocate(col, table)
  end subroutine get_partial_conv_kernel_val_sum
!############################################################################---

!###############################################################################
  pure subroutine conv_unpack(this, nd, in_dims, k_dims, out_dims, stride, &
       dilation, num_channels, num_filters, kernel_volume, kernel_size, &
       num_positions)
    !! Read the geometry of a convolution node back from its metadata.
    class(array_type), intent(in) :: this
    integer, intent(out) :: nd
    integer, dimension(3), intent(out) :: in_dims, k_dims, out_dims
    integer, dimension(3), intent(out) :: stride, dilation
    integer, intent(out) :: num_channels, num_filters, kernel_volume
    integer, intent(out) :: kernel_size, num_positions
    integer :: d

    nd = this%indices(3)
    num_channels = this%indices(1)
    num_filters = this%indices(2)
    in_dims = 1
    k_dims = 1
    out_dims = 1
    stride = 1
    dilation = 1
    do d = 1, nd
       in_dims(d) = this%left_operand%shape(d)
       out_dims(d) = this%shape(d)
       k_dims(d) = this%adj_ja(d,3)
       stride(d) = this%adj_ja(d,1)
       dilation(d) = this%adj_ja(d,2)
    end do
    kernel_volume = product(k_dims(1:nd))
    kernel_size = num_channels * kernel_volume
    num_positions = product(out_dims(1:nd))
  end subroutine conv_unpack
!############################################################################---

end submodule athena__diffstruc_extd_submodule_conv
