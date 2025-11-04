submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule
  !! Submodule containing implementations for extended diffstruc array operations
  use coreutils, only: stop_program

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
  module function add_bias(input, bias, dim) result(output)
    !! Add bias to input array along specified dimension
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: bias
    integer, intent(in) :: dim
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, k, s, idx
    integer :: num_elements_pre, num_elements_post, num_dims

    num_dims = size(input%shape)
    if(dim .gt. num_dims) then
       call stop_program("Dimension for add_bias exceeds input dimensions")
       return
    elseif(size(bias%shape) .ne. 1)then
       call stop_program("Bias must be a 1D array")
       return
    end if
    output => input%create_result()
    num_elements_pre = 1
    do i = 1, num_dims
       if(i .lt. dim)then
          num_elements_pre = num_elements_pre * input%shape(i)
       end if
    end do
    num_elements_post = num_elements_pre * bias%shape(1)

    do s = 1, size(input%val, 2)
       do k = 1, num_elements_post
          do j = 1, bias%shape(1)
             idx = (j - 1) * num_elements_pre + (k - 1) * num_elements_post
             do i = 1, num_elements_pre
                output%val(idx + i, s) = input%val(idx + i, s) + bias%val(j,1)
             end do
          end do
       end do
    end do
    allocate(output%indices(1))
    output%indices(1) = dim

    output%get_partial_left => get_partial_add
    output%get_partial_right => get_partial_add_bias
    if(input%requires_grad .or. bias%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward .or. bias%is_forward
       output%operation = 'add_bias'
       output%left_operand => input
       output%right_operand => bias
    end if

  end function add_bias
!-------------------------------------------------------------------------------
  function get_partial_add(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad
  end function get_partial_add
!-------------------------------------------------------------------------------
  function get_partial_add_bias(this, upstream_grad) result(output)
    !! Get partial derivative with respect to bias operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    integer :: i, j, k, s, idx
    integer :: num_elements_pre, num_elements_post, num_dims

    num_dims = size(this%left_operand%shape)
    num_elements_pre = 1
    do i = 1, num_dims
       if(i .lt. this%indices(1))then
          num_elements_pre = num_elements_pre * this%left_operand%shape(i)
       end if
    end do
    num_elements_post = num_elements_pre * this%right_operand%shape(1)

    call output%allocate(array_shape = this%right_operand%shape)
    output%val = 0._real32
    do s = 1, size(upstream_grad%val, 2)
       do k = 1, num_elements_post
          do j = 1, this%right_operand%shape(1)
             idx = (j - 1) * num_elements_pre + (k - 1) * num_elements_post
             do i = 1, num_elements_pre
                output%val(j,1) = output%val(j,1) + upstream_grad%val(idx + i, s)
             end do
          end do
       end do
    end do

  end function get_partial_add_bias
!###############################################################################

end submodule athena__diffstruc_extd_submodule
