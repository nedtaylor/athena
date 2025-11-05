submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule
  !! Submodule containing implementations for extended diffstruc array operations
  use coreutils, only: stop_program
  use diffstruc, only: &
       operator(+), operator(-), operator(*), operator(.concat.), exp, sum

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
  module function add_bias(input, bias, dim, dim_act_on_shape) result(output)
    !! Add bias to input array along specified dimension
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: bias
    integer, intent(in) :: dim
    logical, intent(in), optional :: dim_act_on_shape
    type(array_type), pointer :: output

    ! Local variables
    integer :: i, j, k, s, idx, itmp1
    integer :: num_elements_pre, num_elements_post, num_dims
    logical :: dim_act_on_shape_

    if(present(dim_act_on_shape))then
       dim_act_on_shape_ = dim_act_on_shape
    else
       dim_act_on_shape_ = .false.
    end if

    output => input%create_result()
    allocate(output%indices(2))
    output%indices(1) = dim
    if(dim_act_on_shape_)then
       num_dims = size(input%shape)
       if(dim .gt. num_dims) then
          call stop_program("Dimension for add_bias exceeds input dimensions")
          return
       elseif(size(bias%shape) .ne. 1)then
          call stop_program("Bias must be a 1D array")
          return
       end if
       num_elements_pre = 1
       num_elements_post = 1
       do i = 1, num_dims
          if(i .lt. dim)then
             num_elements_pre = num_elements_pre * input%shape(i)
          elseif(i .gt. dim)then
             num_elements_post = num_elements_post * input%shape(i)
          end if
       end do

       itmp1 = num_elements_pre * input%shape(dim)
       do s = 1, size(input%val, 2)
          do k = 1, num_elements_post
             do j = 1, bias%shape(1)
                idx = (j - 1) * num_elements_pre + (k - 1) * itmp1
                do i = 1, num_elements_pre
                   output%val(idx + i, s) = input%val(idx + i, s) + bias%val(j,1)
                end do
             end do
          end do
       end do
       output%indices(2) = 1
    else
       call stop_program("add_bias: dim_act_on_shape=.false. not implemented yet")
       output%indices(2) = 0
    end if

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

    integer :: i, j, k, s, idx, itmp1
    integer :: num_elements_pre, num_elements_post, num_dims

    num_dims = size(this%left_operand%shape)
    num_elements_pre = 1
    num_elements_post = 1
    do i = 1, num_dims
       if(i .lt. this%indices(1))then
          num_elements_pre = num_elements_pre * this%left_operand%shape(i)
       elseif(i .gt. this%indices(1))then
          num_elements_post = num_elements_post * this%left_operand%shape(i)
       end if
    end do

    itmp1 = num_elements_pre * this%left_operand%shape(this%indices(1))
    call output%allocate(array_shape = [ this%right_operand%shape, 1 ])
    output%val = 0._real32
    do s = 1, size(upstream_grad%val, 2)
       do k = 1, num_elements_post
          do j = 1, this%right_operand%shape(1)
             idx = (j - 1) * num_elements_pre + (k - 1) * itmp1
             do i = 1, num_elements_pre
                output%val(j,1) = output%val(j,1) + upstream_grad%val(idx + i, s)
             end do
          end do
       end do
    end do

  end function get_partial_add_bias
!###############################################################################


!###############################################################################
  module function piecewise_array(input, min_val, max_val, intercept) result(output)
    !! Apply piecewise activation function to input array
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: input
    real(real32), intent(in) :: min_val
    real(real32), intent(in) :: max_val
    real(real32), intent(in) :: intercept
    type(array_type), pointer :: output
    type(array_type), pointer :: b_array

    output => input%create_result()
    where(input%val.le.min_val)
       output%val = 0._real32
    elsewhere(input%val.ge.max_val)
       output%val = 1._real32
    elsewhere
       output%val = input%val + intercept
    end where

    output%get_partial_left => get_partial_piecewise
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'piecewise'
       output%left_operand => input
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[3, 1])
    b_array%val(1,1) = min_val
    b_array%val(2,1) = max_val
    b_array%val(3,1) = intercept
    output%right_operand => b_array
    output%owns_left_operand = .true.

  end function piecewise_array
!-------------------------------------------------------------------------------
  function get_partial_piecewise(this, upstream_grad) result(output)
    !! Get partial derivative of piecewise activation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [this%shape, size(this%val, 2)])
    where(this%left_operand%val.gt.this%right_operand%val(1,1) .and. &
         this%left_operand%val.lt.this%right_operand%val(2,1) &
    )
       output%val = upstream_grad%val
    elsewhere
       output%val = 0._real32
    end where

  end function get_partial_piecewise
!###############################################################################


!###############################################################################
  module function softmax_array(input, dim) result(output)
    implicit none
    class(array_type), intent(in), target :: input
    integer, intent(in) :: dim
    type(array_type), pointer :: output

    integer :: i

    output => input%create_result()
    if(dim.eq.1)then
       do i = 1, size(input%val, 1)
          output%val(i, :) = exp(input%val(i, :) - maxval(input%val(i,:)))
          output%val(i, :) = output%val(i, :) / sum(output%val(i, :))
       end do
    elseif(dim.eq.2)then
       do i = 1, size(input%val, 2)
          output%val(:, i) = exp(input%val(:, i) - maxval(input%val(:, i)))
          output%val(:, i) = output%val(:, i) / sum(output%val(:, i))
       end do
    else
       call stop_program("softmax_array: Unsupported dimension")
    end if
    allocate(output%indices(1))
    output%indices(1) = dim

    output%get_partial_left => get_partial_softmax
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'softmax'
       output%left_operand => input
    end if

  end function softmax_array
!-------------------------------------------------------------------------------
  function get_partial_softmax(this, upstream_grad) result(output)
    !! Get partial derivative of softmax activation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ! Note: Full Jacobian is not computed for efficiency; this is a simplified version
    call output%allocate(array_shape = [this%shape, size(this%val, this%indices(1))])
    ptr => this * upstream_grad
    output = ptr - this * sum(ptr, dim=this%indices(1))

  end function get_partial_softmax
!###############################################################################

end submodule athena__diffstruc_extd_submodule
