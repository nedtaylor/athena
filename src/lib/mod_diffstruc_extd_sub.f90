submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function add_array_ptr(a, idx1, idx2) result(c)
    !! Add two autodiff arrays
    implicit none

    ! Arguments
    type(array_ptr_type), dimension(:), intent(in) :: a
    integer, intent(in) :: idx1, idx2
    type(array_type), pointer :: c

    ! Local variables
    integer :: i

    c => a(1)%array(idx1, idx2) + a(2)%array(idx1, idx2)
    do i = 2, size(a)
       c => c + a(i)%array(idx1, idx2)
    end do
  end function add_array_ptr
!###############################################################################


!###############################################################################
  module function concat_array_ptr(a, idx1, idx2, dim) result(c)
    !! Concatenate two autodiff arrays along a specified dimension
    implicit none

    ! Arguments
    type(array_ptr_type), dimension(:), intent(in) :: a
    integer, intent(in) :: idx1, idx2, dim
    type(array_type), pointer :: c

    ! Local variables
    integer :: i

    allocate(c)
    c => a(1)%array(idx1, idx2) .concat. a(2)%array(idx1, idx2)
    do i = 3, size(a)
       c => c .concat. a(i)%array(idx1, idx2)
    end do
  end function concat_array_ptr
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function avgpool(input, pool_size, stride) result(output)
    !! Average pooling operation
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

    output_shape = [( &
         (input%shape(1) - pool_size) / stride + 1), &
         input%shape(2), &
         size(input%val, dim=2)]
    output => input%create_result(array_shape = output_shape)
    do concurrent(&
         s = 1:output_shape(3), &
         m = 1:output_shape(2), &
         i = 1:output_shape(1))
       stride_idx = (i - 1) * stride + 1 + (m - 1) * input%shape(1)
       idx = i + (m - 1) * output_shape(1)
       output%val(idx, s) = sum( &
            input%val( stride_idx : stride_idx + pool_size - 1, s ) &
       ) / pool_size
    end do
    allocate(output%adj_ja(1,2))
    output%adj_ja(1,1) = pool_size
    output%adj_ja(1,2) = stride

    output%get_partial_left => get_partial_avgpool
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'avgpool'
       output%left_operand = input
    end if

  end function avgpool
!-------------------------------------------------------------------------------
  function get_partial_avgpool(this, upstream_grad) result(output)
    !! Get the partial derivative for average pooling
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! Local variables
    integer :: i, m, s
    integer :: stride_idx, idx
    integer, dimension(3) :: input_shape

    input_shape = this%shape
    call output%allocate(array_shape = input_shape)
    do concurrent(&
         s = 1:this%shape(3), &
         m = 1:this%shape(2), &
         i = 1:this%shape(1))
       stride_idx = (i - 1) * this%adj_ja(1,2) + 1 + (m - 1) * input_shape(1)
       idx = i + (m - 1) * this%shape(1)
       output%val( stride_idx : stride_idx + this%adj_ja(1,2) - 1, s ) = &
            output%val( stride_idx : stride_idx + this%adj_ja(1,2) - 1, s ) + &
            upstream_grad%val(idx, s) / this%adj_ja(1,1)
    end do

  end function get_partial_avgpool
!###############################################################################

end submodule athena__diffstruc_extd_submodule
