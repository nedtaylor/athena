submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule
  !! Submodule containing implementations for extended diffstruc array operations
  use coreutils, only: stop_program
  use diffstruc, only: &
       operator(+), operator(-), operator(*), concat, exp, sum, merge

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
    do i = 3, size(a), 1
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

    c => concat(a(1)%array(idx1, idx2), a(2)%array(idx1, idx2), dim)
    do i = 3, size(a), 1
       c => concat(c, a(i)%array(idx1, idx2), dim)
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
    output%get_partial_left_val => get_partial_add_val
    output%get_partial_right_val => get_partial_add_bias_val
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
  pure subroutine get_partial_add_val(this, upstream_grad, output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    if(size(upstream_grad,2).ne.size(output,2))then
       if(size(output,1).eq.1)then
          output(1,1) = sum(upstream_grad)
       else
          output(:,1) = sum(upstream_grad, dim=2)
       end if
    else
       if(size(output,1).eq.1.and.size(output,1).ne.size(upstream_grad,1))then
          output(1,:) = sum(upstream_grad,1)
       else
          output = upstream_grad
       end if
    end if
  end subroutine get_partial_add_val
!-------------------------------------------------------------------------------
  function get_partial_add_bias(this, upstream_grad) result(output)
    !! Get partial derivative with respect to bias operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [ this%right_operand%shape, 1 ])
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_add_bias
!-------------------------------------------------------------------------------
  pure subroutine get_partial_add_bias_val(this, upstream_grad, output)
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

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
    output = 0._real32
    do s = 1, size(upstream_grad, 2)
       do k = 1, num_elements_post
          do j = 1, this%right_operand%shape(1)
             idx = (j - 1) * num_elements_pre + (k - 1) * itmp1
             do i = 1, num_elements_pre
                output(j,1) = output(j,1) + upstream_grad(idx + i, s)
             end do
          end do
       end do
    end do

  end subroutine get_partial_add_bias_val
!###############################################################################


!###############################################################################
  module function piecewise_array(input, gradient, limit) result(output)
    !! Apply piecewise activation function to input array
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: input
    real(real32), intent(in) :: gradient
    real(real32), intent(in) :: limit
    type(array_type), pointer :: output
    type(array_type), pointer :: b_array

    output => input%create_result()
    where(input%val.ge.limit)
       output%val = gradient * (input%val - limit) + limit
    elsewhere(input%val.le.-limit)
       output%val = gradient * (input%val + limit) - limit
    elsewhere
       output%val = input%val
    end where

    output%get_partial_left => get_partial_piecewise
    output%get_partial_left_val => get_partial_piecewise_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'piecewise'
       output%left_operand => input
       output%owns_left_operand = input%is_temporary
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[2, 1])
    b_array%val(1,1) = gradient
    b_array%val(2,1) = limit
    output%right_operand => b_array
    output%owns_right_operand = .true.

  end function piecewise_array
!-------------------------------------------------------------------------------
  function get_partial_piecewise(this, upstream_grad) result(output)
    !! Get partial derivative of piecewise activation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: ptr

    ptr => merge( &
         upstream_grad, &
         upstream_grad * this%right_operand%val(1,1), &
         this%left_operand%val.le.-this%right_operand%val(2,1) .or. &
         this%left_operand%val.ge.this%right_operand%val(2,1) &
    )
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_piecewise
!-------------------------------------------------------------------------------
  pure subroutine get_partial_piecewise_val(this, upstream_grad, output)
    !! Get partial derivative of piecewise activation (in-place version)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    where(this%left_operand%val.le.this%right_operand%val(2,1) .or. &
         this%left_operand%val.ge.-this%right_operand%val(2,1) &
    )
       output = upstream_grad
    elsewhere
       output = upstream_grad * this%right_operand%val(1,1)
    end where

  end subroutine get_partial_piecewise_val
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
    output%get_partial_left_val => get_partial_softmax_val
    output%get_partial_left_val_sum => get_partial_softmax_val_sum
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'softmax'
       output%left_operand => input
       output%owns_left_operand = input%is_temporary
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

    integer :: dim

    if(this%indices(1).eq.1)then
       dim = 2
    else
       dim = 1
    end if
    ! ptr => this * upstream_grad
    ! ptr => ptr - this * sum(ptr, dim=dim)
    ptr => softmax_reverse_array(this, upstream_grad, this%indices(1))
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_softmax
!-------------------------------------------------------------------------------
  pure subroutine get_partial_softmax_val(this, upstream_grad, output)
    !! Get partial derivative of softmax activation (in-place version)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: s, dim

    if(this%indices(1).eq.1)then
       dim = 2
    else
       dim = 1
    end if
    output = this%val * upstream_grad
    if(dim.eq.1)then
       do s = 1, size(this%val, 2)
          output(:, s) = output(:, s) - this%val(:, s) * sum(output(:, s))
       end do
    elseif(dim.eq.2)then
       do s = 1, size(this%val, 1)
          output(s, :) = output(s, :) - this%val(s, :) * sum(output(s, :))
       end do
    end if
  end subroutine get_partial_softmax_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_softmax_val_sum(this, upstream_grad, output)
    !! Get partial derivative of softmax activation (in-place version, summed over samples)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: s, dim
    real(real32) :: dot
    real(real32), allocatable :: tmp(:,:)

    if (this%indices(1) .eq. 1) then
       dim = 2
    else
       dim = 1
    end if

    tmp = this%val * upstream_grad

    if (dim == 1) then
       do s = 1, size(tmp,2)
          dot = sum(tmp(:,s))
          tmp(:,s) = tmp(:,s) - this%val(:,s) * dot
       end do
       output = sum(tmp, dim=2)
    else
       do s = 1, size(tmp,1)
          dot = sum(tmp(s,:))
          tmp(s,:) = tmp(s,:) - this%val(s,:) * dot
       end do
       output = sum(tmp, dim=1)
    end if
  end subroutine get_partial_softmax_val_sum
!###############################################################################


!###############################################################################
  module function swish_array(input, beta) result(output)
    !! Swish activation function
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: input
    real(real32), intent(in) :: beta
    type(array_type), pointer :: output
    type(array_type), pointer :: b_array

    output => input%create_result()
    output%val = input%val * (1._real32 / (1._real32 + exp(-beta * input%val)))

    output%get_partial_left => get_partial_swish
    output%get_partial_left_val => get_partial_swish_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'swish'
       output%left_operand => input
       output%owns_left_operand = input%is_temporary
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%is_scalar = .true.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1,1) = beta
    output%right_operand => b_array
    output%owns_right_operand = .true.

  end function swish_array
!-------------------------------------------------------------------------------
  function get_partial_swish(this, upstream_grad) result(output)
    !! Get partial derivative of swish activation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: ptr
    type(array_type), pointer :: exp_term

    exp_term => exp(this%right_operand%val(1,1) * this%left_operand)

    ptr => upstream_grad * exp_term * ( &
         this%right_operand%val(1,1) * this%left_operand + &
         exp_term + 1._real32 &
    ) / ( ( exp_term + 1._real32 )**2._real32 )

    call output%assign_and_deallocate_source(ptr)
  end function get_partial_swish
!-------------------------------------------------------------------------------
  pure subroutine get_partial_swish_val(this, upstream_grad, output)
    !! Get partial derivative of swish activation (in-place version)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    real(real32), dimension(size(this%val,1), size(this%val,2)) :: exp_term

    exp_term = exp(this%right_operand%val(1,1) * this%left_operand%val)
    output = upstream_grad * exp_term * ( &
         this%right_operand%val(1,1) * this%left_operand%val + &
         exp_term + 1._real32 &
    ) / ( ( exp_term + 1._real32 )**2._real32 )

  end subroutine get_partial_swish_val
!###############################################################################


!###############################################################################
  function softmax_reverse_array(softmax, gradient, dim) result(output)
    !! Softmax function for reverse mode autodiff
    implicit none
    class(array_type), intent(in), target :: softmax
    class(array_type), intent(in), target :: gradient
    integer, intent(in) :: dim
    type(array_type), pointer :: output

    integer :: i
    real(real32), dimension(size(softmax%val,1), size(softmax%val,2)) :: temp_val


    output => softmax%create_result()
    temp_val = gradient%val * softmax%val
    if(dim.eq.1)then
       do concurrent(i=1:size(softmax%val,1))
          temp_val(i, :) = temp_val(i, :) - softmax%val(i, :) * sum(temp_val(i, :))
       end do
    elseif(dim.eq.2)then
       do concurrent(i=1:size(softmax%val,2))
          temp_val(:, i) = temp_val(:, i) - softmax%val(:, i) * sum(temp_val(:, i))
       end do
    else
       call stop_program("softmax_reverse_array: Unsupported dimension")
    end if
    output%val = temp_val
    output%indices = [dim]

    output%get_partial_left => get_partial_softmax_reverse_left
    output%get_partial_left_val => get_partial_softmax_reverse_left_val
    output%get_partial_right => get_partial_softmax_reverse_right
    output%get_partial_right_val => get_partial_softmax_reverse_right_val
    if(softmax%requires_grad .or. gradient%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = softmax%is_forward .or. gradient%is_forward
       output%operation = 'softmax_reverse'
       output%left_operand => softmax
       output%right_operand => gradient
       output%owns_left_operand = softmax%is_temporary
       output%owns_right_operand = gradient%is_temporary
    end if

  end function softmax_reverse_array
!-------------------------------------------------------------------------------
  function get_partial_softmax_reverse_left(this, upstream_grad) result(output)
    !! Get partial derivative of softmax reverse operation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: sum_yg, sum_yu
    type(array_type), pointer :: ptr

    sum_yg => sum(this%left_operand * this%right_operand, dim=this%indices(1))
    sum_yu => sum(this%left_operand * upstream_grad, dim=this%indices(1))

    ptr => upstream_grad * (this%right_operand - sum_yg) - this%right_operand * sum_yu
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_softmax_reverse_left
!-------------------------------------------------------------------------------
  function get_partial_softmax_reverse_right(this, upstream_grad) result(output)
    !! Get partial derivative of softmax reverse operation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: ptr

    ptr => ( &
         upstream_grad - &
         sum(this%left_operand * upstream_grad, dim=this%indices(1)) &
    ) * this%left_operand
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_softmax_reverse_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_softmax_reverse_left_val(this, upstream_grad, output)
    !! Get partial derivative of softmax reverse operation (in-place version)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: dim, i
    real(real32), dimension(size(this%val,3-this%indices(1))) :: sum_yg
    real(real32), dimension(size(this%val,3-this%indices(1))) :: sum_yu

    dim = this%indices(1)
    sum_yg = sum(this%left_operand%val * this%right_operand%val, dim=dim)
    sum_yu = sum(this%left_operand%val * upstream_grad, dim=dim)

    if(dim.eq.1)then
       do concurrent(i=1:size(this%val,2))
          output(:, i) = &
               upstream_grad(:, i) * (this%right_operand%val(:, i) - sum_yg(i)) - &
               this%right_operand%val(:, i) * sum_yu(i)
       end do
    else
       do concurrent(i=1:size(this%val,1))
          output(i, :) = &
               upstream_grad(i, :) * (this%right_operand%val(i, :) - sum_yg(i)) - &
               this%right_operand%val(i, :) * sum_yu(i)
       end do
    end if

  end subroutine get_partial_softmax_reverse_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_softmax_reverse_right_val(this, upstream_grad, output)
    !! Get partial derivative of softmax reverse operation (in-place version)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: dim, i
    real(real32), dimension(size(this%val,3-this%indices(1))) :: sum_yu

    dim = this%indices(1)
    if(dim.eq.1)then
       sum_yu = sum(this%left_operand%val * upstream_grad, dim=dim)
       do concurrent(i=1:size(this%val,1))
          output(i, :) = upstream_grad(i, :) - sum_yu(i) * this%left_operand%val(i, :)
       end do
    else
       sum_yu = sum(this%left_operand%val * upstream_grad, dim=dim)
       do concurrent(i=1:size(this%val,2))
          output(:, i) = upstream_grad(:, i) - sum_yu(i) * this%left_operand%val(:, i)
       end do
    end if
  end subroutine get_partial_softmax_reverse_right_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule
