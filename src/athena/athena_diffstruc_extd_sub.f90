submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule
  !! Submodule containing implementations for extended diffstruc array operations
  use coreutils, only: stop_program
  use diffstruc, only: &
       operator(+), operator(-), operator(*), concat, exp, matmul, sum, tanh, &
       merge

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
       if(dim .gt. num_dims)then
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
    output%get_partial_right_val_sum => get_partial_add_bias_val_sum
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
!-------------------------------------------------------------------------------
  pure subroutine get_partial_add_bias_val_sum(this, upstream_grad, output)
    !! Sum-reduced partial derivative with respect to bias operand
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

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
                output(j) = output(j) + upstream_grad(idx + i, s)
             end do
          end do
       end do
    end do

  end subroutine get_partial_add_bias_val_sum
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

    integer :: s, i, nfeat, nsamp
    real(real32) :: dot

    output = 0.0_real32
    if (this%indices(1) .eq. 1) then
       nsamp = size(this%val,1)
       nfeat = size(this%val,2)
       do s = 1, nsamp
          ! compute g·y
          dot = 0.0_real32
          do i = 1, nfeat
             dot = dot + upstream_grad(s,i) * this%val(s,i)
          end do
          ! accumulate reduced gradient
          do concurrent( i = 1 : nfeat )
             output(i) = output(i) + this%val(s,i) * (upstream_grad(s,i) - dot)
          end do
       end do
    else
       nsamp = size(this%val,2)
       nfeat = size(this%val,1)
       do s = 1, nsamp
          dot = 0.0_real32
          do i = 1, nfeat
             dot = dot + upstream_grad(i,s) * this%val(i,s)
          end do
          do concurrent( i = 1 : nfeat )
             output(i) = output(i) + this%val(i,s) * (upstream_grad(i,s) - dot)
          end do
       end do
    end if
  end subroutine get_partial_softmax_val_sum
!###############################################################################


!###############################################################################
  module function full_relu(input, kernel, bias) result(output)
    implicit none

    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    type(array_type), intent(in), target :: bias
    type(array_type), pointer :: output

    integer :: num_inputs, num_outputs, num_samples, s
    logical :: requires_grad
    type(array_type), pointer :: packed_params
    real(real32), pointer :: weights(:,:)

    num_inputs = kernel%shape(2)
    num_outputs = kernel%shape(1)
    num_samples = size(input%val, 2)

    output => input%create_result(array_shape = [num_outputs, num_samples])
    weights(1:num_outputs, 1:num_inputs) => kernel%val(:,1)
    output%val = matmul(weights, input%val)
    do s = 1, num_samples
       output%val(:,s) = output%val(:,s) + bias%val(:,1)
    end do
    where(output%val .lt. 0._real32)
       output%val = 0._real32
    end where

    requires_grad = input%requires_grad .or. kernel%requires_grad .or. &
         bias%requires_grad
    if(requires_grad)then
       packed_params => pack_params_array(kernel, bias)
       output%get_partial_left => get_partial_full_relu_input
       output%get_partial_right => get_partial_full_relu_params
       output%get_partial_left_val => get_partial_full_relu_input_val
       output%get_partial_right_val => get_partial_full_relu_params_val
       output%get_partial_right_val_sum => get_partial_full_relu_params_val_sum
       output%get_partial_both_val => get_partial_full_relu_both_val
       output%requires_grad = .true.
       output%is_forward = input%is_forward .or. kernel%is_forward .or. &
            bias%is_forward
       output%operation = 'full_relu'
       output%left_operand => input
       output%right_operand => packed_params
       output%owns_left_operand = input%is_temporary
       output%owns_right_operand = .true.
    end if

  end function full_relu
!-------------------------------------------------------------------------------
  module function full_tanh(input, kernel, bias) result(output)
    implicit none

    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    type(array_type), intent(in), target :: bias
    type(array_type), pointer :: output

    integer :: num_inputs, num_outputs, num_samples, s
    logical :: requires_grad
    type(array_type), pointer :: packed_params
    real(real32), pointer :: weights(:,:)

    num_inputs = kernel%shape(2)
    num_outputs = kernel%shape(1)
    num_samples = size(input%val, 2)

    output => input%create_result(array_shape = [num_outputs, num_samples])
    weights(1:num_outputs, 1:num_inputs) => kernel%val(:,1)
    output%val = matmul(weights, input%val)
    do s = 1, num_samples
       output%val(:,s) = output%val(:,s) + bias%val(:,1)
    end do
    output%val = tanh(output%val)

    requires_grad = input%requires_grad .or. kernel%requires_grad .or. &
         bias%requires_grad
    if(requires_grad)then
       packed_params => pack_params_array(kernel, bias)
       output%get_partial_left => get_partial_full_tanh_input
       output%get_partial_right => get_partial_full_tanh_params
       output%get_partial_left_val => get_partial_full_tanh_input_val
       output%get_partial_right_val => get_partial_full_tanh_params_val
       output%get_partial_right_val_sum => get_partial_full_tanh_params_val_sum
       output%get_partial_both_val => get_partial_full_tanh_both_val
       output%requires_grad = .true.
       output%is_forward = input%is_forward .or. kernel%is_forward .or. &
            bias%is_forward
       output%operation = 'full_tanh'
       output%left_operand => input
       output%right_operand => packed_params
       output%owns_left_operand = input%is_temporary
       output%owns_right_operand = .true.
    end if

  end function full_tanh
!-------------------------------------------------------------------------------
  module function full_softmax(input, kernel, bias) result(output)
    implicit none

    type(array_type), intent(in), target :: input
    type(array_type), intent(in), target :: kernel
    type(array_type), intent(in), target :: bias
    type(array_type), pointer :: output

    integer :: num_inputs, num_outputs, num_samples, i, s
    logical :: requires_grad
    type(array_type), pointer :: packed_params
    real(real32), pointer :: weights(:,:)
    real(real32) :: sample_max, sample_sum

    num_inputs = kernel%shape(2)
    num_outputs = kernel%shape(1)
    num_samples = size(input%val, 2)

    output => input%create_result(array_shape = [num_outputs, num_samples])
    weights(1:num_outputs, 1:num_inputs) => kernel%val(:,1)
    output%val = matmul(weights, input%val)
    do s = 1, num_samples
       output%val(:,s) = output%val(:,s) + bias%val(:,1)
       sample_max = maxval(output%val(:,s))
       output%val(:,s) = exp(output%val(:,s) - sample_max)
       sample_sum = 0._real32
       do i = 1, num_outputs
          sample_sum = sample_sum + output%val(i,s)
       end do
       output%val(:,s) = output%val(:,s) / sample_sum
    end do

    requires_grad = input%requires_grad .or. kernel%requires_grad .or. &
         bias%requires_grad
    if(requires_grad)then
       packed_params => pack_params_array(kernel, bias)
       output%get_partial_left_val => get_partial_full_softmax_input_val
       output%get_partial_right_val => get_partial_full_softmax_params_val
       output%get_partial_right_val_sum => get_partial_full_softmax_params_val_sum
       output%get_partial_both_val => get_partial_full_softmax_both_val
       output%requires_grad = .true.
       output%is_forward = input%is_forward .or. kernel%is_forward .or. &
            bias%is_forward
       output%operation = 'full_softmax'
       output%left_operand => input
       output%right_operand => packed_params
       output%owns_left_operand = input%is_temporary
       output%owns_right_operand = .true.
    end if

  end function full_softmax
!-------------------------------------------------------------------------------
  function pack_params_array(kernel, bias) result(output)
    implicit none

    class(array_type), intent(in), target :: kernel
    class(array_type), intent(in), target :: bias
    type(array_type), pointer :: output

    integer :: split_idx, total_size

    split_idx = size(kernel%val, 1)
    total_size = split_idx + size(bias%val, 1)
    output => kernel%create_result(array_shape = [total_size, 1])
    output%is_sample_dependent = .false.
    output%val(1:split_idx,1) = kernel%val(:,1)
    output%val(split_idx+1:total_size,1) = bias%val(:,1)
    allocate(output%indices(1))
    output%indices(1) = split_idx
    output%get_partial_left => get_partial_packed_params_left
    output%get_partial_right => get_partial_packed_params_right
    output%get_partial_left_val => get_partial_packed_params_left_val
    output%get_partial_right_val => get_partial_packed_params_right_val
    if(kernel%requires_grad .or. bias%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = kernel%is_forward .or. bias%is_forward
       output%operation = 'pack_params'
       output%left_operand => kernel
       output%right_operand => bias
       output%owns_left_operand = kernel%is_temporary
       output%owns_right_operand = bias%is_temporary
    end if

  end function pack_params_array
!-------------------------------------------------------------------------------
  function get_partial_packed_params_left(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [this%shape, 1])
    output%is_sample_dependent = .false.
    output%val = 0._real32
    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_packed_params_left
!-------------------------------------------------------------------------------
  function get_partial_packed_params_right(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    call output%allocate(array_shape = [this%shape, 1])
    output%is_sample_dependent = .false.
    output%val = 0._real32
    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_packed_params_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_packed_params_left_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output(1:this%indices(1),1) = upstream_grad(:,1)

  end subroutine get_partial_packed_params_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_packed_params_right_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output(this%indices(1)+1:,1) = upstream_grad(:,1)

  end subroutine get_partial_packed_params_right_val
!-------------------------------------------------------------------------------
  function unpack_full_kernel_tangent(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(in) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type), pointer :: output

    integer :: split_idx, num_inputs, num_outputs

    split_idx = this%right_operand%indices(1)
    num_outputs = this%shape(1)
    num_inputs = this%left_operand%shape(1)
    output => this%create_result(array_shape = [num_outputs, num_inputs, 1])
    output%is_sample_dependent = .false.
    output%requires_grad = .false.
    output%val(:,1) = upstream_grad%val(1:split_idx,1)

  end function unpack_full_kernel_tangent
!-------------------------------------------------------------------------------
  function unpack_full_bias_tangent(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(in) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type), pointer :: output

    integer :: split_idx

    split_idx = this%right_operand%indices(1)
    output => this%create_result(array_shape = [this%shape(1), 1])
    output%is_sample_dependent = .false.
    output%requires_grad = .false.
    output%val(:,1) = upstream_grad%val(split_idx+1:,1)

  end function unpack_full_bias_tangent
!-------------------------------------------------------------------------------
  pure subroutine compute_full_relu_delta(this, upstream_grad, delta)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: delta

    delta = 0._real32
    where(this%val .gt. 0._real32)
       delta = upstream_grad
    end where

  end subroutine compute_full_relu_delta
!-------------------------------------------------------------------------------
  pure subroutine compute_full_tanh_delta(this, upstream_grad, delta)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: delta

    delta = upstream_grad * (1._real32 - this%val * this%val)

  end subroutine compute_full_tanh_delta
!-------------------------------------------------------------------------------
  function full_tanh_reverse_array(input) result(output)
    implicit none

    class(array_type), intent(in), target :: input
    type(array_type), pointer :: output

    output => input%create_result()
    output%val = 1._real32 - input%val * input%val
    output%get_partial_left => get_partial_full_tanh_reverse
    output%get_partial_left_val => get_partial_full_tanh_reverse_val
    if(input%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = input%is_forward
       output%operation = 'full_tanh_reverse'
       output%left_operand => input
       output%owns_left_operand = input%is_temporary
    end if

  end function full_tanh_reverse_array
!-------------------------------------------------------------------------------
  function get_partial_full_tanh_reverse(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    ptr => (-2._real32) * upstream_grad * this%left_operand
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_full_tanh_reverse
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_tanh_reverse_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = (-2._real32) * upstream_grad * this%left_operand%val

  end subroutine get_partial_full_tanh_reverse_val
!-------------------------------------------------------------------------------
  pure subroutine compute_full_softmax_delta(this, upstream_grad, delta)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: delta

    integer :: i, s
    real(real32) :: dot_product_val

    do s = 1, size(this%val, 2)
       dot_product_val = 0._real32
       do i = 1, size(this%val, 1)
          dot_product_val = dot_product_val + &
               upstream_grad(i,s) * this%val(i,s)
       end do
       do i = 1, size(this%val, 1)
          delta(i,s) = this%val(i,s) * (upstream_grad(i,s) - dot_product_val)
       end do
    end do

  end subroutine compute_full_softmax_delta
!-------------------------------------------------------------------------------
  pure subroutine compute_full_input_grad(this, delta, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: delta
    real(real32), dimension(:,:), intent(out) :: output

    integer :: split_idx, num_inputs, num_outputs
    real(real32), dimension(this%shape(1), this%left_operand%shape(1)) :: weights

    split_idx = this%right_operand%indices(1)
    num_outputs = this%shape(1)
    num_inputs = this%left_operand%shape(1)
    weights = reshape(this%right_operand%val(1:split_idx,1), [num_outputs, num_inputs])
    output = matmul(transpose(weights), delta)

  end subroutine compute_full_input_grad
!-------------------------------------------------------------------------------
  pure subroutine compute_full_param_grad(this, delta, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: delta
    real(real32), dimension(:), intent(out) :: output

    integer :: split_idx, num_inputs, num_outputs
    real(real32), dimension(this%shape(1), this%left_operand%shape(1)) :: kernel_grad

    split_idx = this%right_operand%indices(1)
    num_outputs = this%shape(1)
    num_inputs = this%left_operand%shape(1)
    kernel_grad = matmul(delta, transpose(this%left_operand%val))
    output(1:split_idx) = reshape(kernel_grad, [num_outputs * num_inputs])
    output(split_idx+1:) = sum(delta, dim = 2)

  end subroutine compute_full_param_grad
!-------------------------------------------------------------------------------
  function get_partial_full_relu_input(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: kernel
    type(array_type), pointer :: kernel_const
    type(array_type), pointer :: z_dir
    type(array_type), pointer :: ptr
    integer :: split_idx, num_inputs, num_outputs

    if(associated(this%right_operand%left_operand))then
       kernel => this%right_operand%left_operand
    else
       split_idx = this%right_operand%indices(1)
       num_outputs = this%shape(1)
       num_inputs = this%left_operand%shape(1)
       allocate(kernel_const)
       call kernel_const%allocate(array_shape = [num_outputs, num_inputs, 1], &
            source = 0._real32)
       kernel_const%is_sample_dependent = .false.
       kernel_const%requires_grad = .false.
       kernel_const%val(:,1) = this%right_operand%val(1:split_idx,1)
       kernel => kernel_const
    end if

    z_dir => matmul(kernel, upstream_grad)
    ptr => z_dir * (this%val .gt. 0._real32)
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_full_relu_input
!-------------------------------------------------------------------------------
  function get_partial_full_relu_params(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: kernel_dir
    type(array_type), pointer :: bias_dir
    type(array_type), pointer :: z_dir
    type(array_type), pointer :: ptr

    if((.not.upstream_grad%requires_grad) .and. &
         maxval(abs(upstream_grad%val)) .eq. 0._real32)then
       call output%allocate(array_shape = [this%shape, size(this%val, 2)])
       output%val = 0._real32
       return
    end if

    kernel_dir => unpack_full_kernel_tangent(this, upstream_grad)
    bias_dir => unpack_full_bias_tangent(this, upstream_grad)

    z_dir => matmul(kernel_dir, this%left_operand) + bias_dir
    ptr => z_dir * (this%val .gt. 0._real32)
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_full_relu_params
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_relu_input_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_relu_delta(this, upstream_grad, delta)
    call compute_full_input_grad(this, delta, output)

  end subroutine get_partial_full_relu_input_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_relu_params_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output(:,1) = 0._real32
    call get_partial_full_relu_params_val_sum(this, upstream_grad, output(:,1))

  end subroutine get_partial_full_relu_params_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_relu_params_val_sum(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_relu_delta(this, upstream_grad, delta)
    call compute_full_param_grad(this, delta, output)

  end subroutine get_partial_full_relu_params_val_sum
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_relu_both_val(this, upstream_grad, left_output, right_output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: left_output
    real(real32), dimension(:), intent(out) :: right_output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_relu_delta(this, upstream_grad, delta)
    call compute_full_input_grad(this, delta, left_output)
    call compute_full_param_grad(this, delta, right_output)

  end subroutine get_partial_full_relu_both_val
!-------------------------------------------------------------------------------
  function get_partial_full_tanh_input(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: kernel
    type(array_type), pointer :: kernel_const
    type(array_type), pointer :: bias
    type(array_type), pointer :: bias_const
    type(array_type), pointer :: z_dir
    type(array_type), pointer :: z_split
    type(array_type), pointer :: output_split
    type(array_type), pointer :: ptr
    integer :: split_idx, num_inputs, num_outputs

    if(associated(this%right_operand%left_operand))then
       kernel => this%right_operand%left_operand
    else
       split_idx = this%right_operand%indices(1)
       num_outputs = this%shape(1)
       num_inputs = this%left_operand%shape(1)
       allocate(kernel_const)
       call kernel_const%allocate(array_shape = [num_outputs, num_inputs, 1], &
            source = 0._real32)
       kernel_const%is_sample_dependent = .false.
       kernel_const%requires_grad = .false.
       kernel_const%val(:,1) = this%right_operand%val(1:split_idx,1)
       kernel => kernel_const
    end if

    if(associated(this%right_operand%right_operand))then
       bias => this%right_operand%right_operand
    else
       split_idx = this%right_operand%indices(1)
       allocate(bias_const)
       call bias_const%allocate(array_shape = [this%shape(1), 1], source = 0._real32)
       bias_const%is_sample_dependent = .false.
       bias_const%requires_grad = .false.
       bias_const%val(:,1) = this%right_operand%val(split_idx+1:,1)
       bias => bias_const
    end if

    z_dir => matmul(kernel, upstream_grad)
    z_split => matmul(kernel, this%left_operand) + bias
    output_split => tanh(z_split)
    z_split%is_temporary = .false.
    output_split%is_temporary = .false.
    allocate(ptr)
    ptr = output_split%get_partial_left(z_dir)
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_full_tanh_input
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_tanh_input_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_tanh_delta(this, upstream_grad, delta)
    call compute_full_input_grad(this, delta, output)

  end subroutine get_partial_full_tanh_input_val
!-------------------------------------------------------------------------------
  function get_partial_full_tanh_params(this, upstream_grad) result(output)
    implicit none

    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: kernel
    type(array_type), pointer :: kernel_const
    type(array_type), pointer :: bias
    type(array_type), pointer :: bias_const
    type(array_type), pointer :: kernel_dir
    type(array_type), pointer :: bias_dir
    type(array_type), pointer :: z_dir
    type(array_type), pointer :: z_split
    type(array_type), pointer :: output_split
    type(array_type), pointer :: ptr
    integer :: split_idx, num_inputs, num_outputs

    if((.not.upstream_grad%requires_grad) .and. &
         maxval(abs(upstream_grad%val)) .eq. 0._real32)then
       call output%allocate(array_shape = [this%shape, size(this%val, 2)])
       output%val = 0._real32
       return
    end if

    if(associated(this%right_operand%left_operand))then
       kernel => this%right_operand%left_operand
    else
       split_idx = this%right_operand%indices(1)
       num_outputs = this%shape(1)
       num_inputs = this%left_operand%shape(1)
       allocate(kernel_const)
       call kernel_const%allocate(array_shape = [num_outputs, num_inputs, 1], &
            source = 0._real32)
       kernel_const%is_sample_dependent = .false.
       kernel_const%requires_grad = .false.
       kernel_const%val(:,1) = this%right_operand%val(1:split_idx,1)
       kernel => kernel_const
    end if

    if(associated(this%right_operand%right_operand))then
       bias => this%right_operand%right_operand
    else
       split_idx = this%right_operand%indices(1)
       allocate(bias_const)
       call bias_const%allocate(array_shape = [this%shape(1), 1], source = 0._real32)
       bias_const%is_sample_dependent = .false.
       bias_const%requires_grad = .false.
       bias_const%val(:,1) = this%right_operand%val(split_idx+1:,1)
       bias => bias_const
    end if

    kernel_dir => unpack_full_kernel_tangent(this, upstream_grad)
    bias_dir => unpack_full_bias_tangent(this, upstream_grad)

    z_dir => matmul(kernel_dir, this%left_operand) + bias_dir
    z_split => matmul(kernel, this%left_operand) + bias
    output_split => tanh(z_split)
    z_split%is_temporary = .false.
    output_split%is_temporary = .false.
    allocate(ptr)
    ptr = output_split%get_partial_left(z_dir)
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_full_tanh_params
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_tanh_params_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output(:,1) = 0._real32
    call get_partial_full_tanh_params_val_sum(this, upstream_grad, output(:,1))

  end subroutine get_partial_full_tanh_params_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_tanh_params_val_sum(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_tanh_delta(this, upstream_grad, delta)
    call compute_full_param_grad(this, delta, output)

  end subroutine get_partial_full_tanh_params_val_sum
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_tanh_both_val(this, upstream_grad, left_output, right_output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: left_output
    real(real32), dimension(:), intent(out) :: right_output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_tanh_delta(this, upstream_grad, delta)
    call compute_full_input_grad(this, delta, left_output)
    call compute_full_param_grad(this, delta, right_output)

  end subroutine get_partial_full_tanh_both_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_softmax_input_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_softmax_delta(this, upstream_grad, delta)
    call compute_full_input_grad(this, delta, output)

  end subroutine get_partial_full_softmax_input_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_softmax_params_val(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output(:,1) = 0._real32
    call get_partial_full_softmax_params_val_sum(this, upstream_grad, output(:,1))

  end subroutine get_partial_full_softmax_params_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_softmax_params_val_sum(this, upstream_grad, output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_softmax_delta(this, upstream_grad, delta)
    call compute_full_param_grad(this, delta, output)

  end subroutine get_partial_full_softmax_params_val_sum
!-------------------------------------------------------------------------------
  pure subroutine get_partial_full_softmax_both_val(this, upstream_grad, left_output, right_output)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: left_output
    real(real32), dimension(:), intent(out) :: right_output

    real(real32), dimension(size(upstream_grad,1), size(upstream_grad,2)) :: delta

    call compute_full_softmax_delta(this, upstream_grad, delta)
    call compute_full_input_grad(this, delta, left_output)
    call compute_full_param_grad(this, delta, right_output)

  end subroutine get_partial_full_softmax_both_val

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
