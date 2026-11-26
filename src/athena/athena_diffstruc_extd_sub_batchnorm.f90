submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_batchnorm
  !! Submodule containing implementations for extended diffstruc array operations

contains

!-------------------------------------------------------------------------------
  pure integer function encode_real32(value) result(bits)
    !! Store a real value in the integer aux data of a context record.
    real(real32), intent(in) :: value
    bits = transfer(value, 1)
  end function encode_real32
!-------------------------------------------------------------------------------
  pure real(real32) function decode_real32(bits) result(value)
    !! Recover a real value stored in the integer aux data of a record.
    integer, intent(in) :: bits
    value = transfer(bits, 0.0_real32)
  end function decode_real32
!###############################################################################


!###############################################################################
  module function batchnorm_inference( &
       input, params, mean, variance, epsilon &
  ) result( output )
    implicit none
    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: params
    real(real32), dimension(:), intent(in) :: mean
    real(real32), dimension(:), intent(in) :: variance
    real(real32), intent(in) :: epsilon
    type(batchnorm_array_type), pointer :: output

    integer :: i, c, s
    integer :: num_elements, num_dims

    allocate(output)
    if(output%allocated) call output%deallocate()
    call output%allocate(array_shape = [ input%shape, size(input%val,2) ])
    output%epsilon = epsilon
    output%mean = mean
    output%variance = variance
    num_dims = size(input%shape)
    num_elements = product(input%shape(1:num_dims - 1))
    allocate(output%self%adj_ja(input%shape(num_dims) + 1, 2))
    do c = 1, input%shape(num_dims)
       output%self%adj_ja(c, 1) = encode_real32(mean(c))
       output%self%adj_ja(c, 2) = encode_real32(variance(c))
    end do
    output%self%adj_ja(input%shape(num_dims) + 1, 1) = encode_real32(epsilon)
    do concurrent(c = 1:input%shape(num_dims))
       do concurrent(s = 1:size(input%val,2), i = 1:num_elements)
          output%val(i + (c-1) * num_elements, s) = &
               params%val(c,1) * ( input%val(i + (c-1) * num_elements, s) - &
                    mean(c) ) / sqrt(variance(c) + output%epsilon) + &
               params%val(c+input%shape(num_dims),1)
       end do
    end do

  end function batchnorm_inference
!-------------------------------------------------------------------------------
  module function batchnorm( &
       input, params, momentum, mean, variance, epsilon &
  ) result( output )
    !! Batch normalisation operation
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: params
    real(real32), intent(in) :: momentum
    real(real32), dimension(:), intent(in) :: mean
    real(real32), dimension(:), intent(in) :: variance
    real(real32), intent(in) :: epsilon
    type(batchnorm_array_type), pointer :: output

    ! Local variables
    integer :: i, c, s
    integer :: num_elements, num_dims
    real(real32) :: mu, var, norm

    allocate(output)
    if(output%allocated) call output%deallocate()
    call output%allocate(array_shape = [ input%shape, size(input%val,2) ])
    output%epsilon = epsilon
    output%mean = mean
    output%variance = variance
    num_dims = size(input%shape)
    num_elements = product(input%shape(1:num_dims - 1))
    norm = real(num_elements * size(input%val,2), real32)
    allocate(output%self%adj_ja(input%shape(num_dims) + 1, 2))
    output%self%adj_ja(input%shape(num_dims) + 1, 1) = encode_real32(epsilon)
    do c = 1, input%shape(num_dims)
       mu = 0._real32
       var = 0._real32
       mu = sum(input%val((c-1) * num_elements+1:c*num_elements,:)) / norm
       var = sum( (input%val((c-1) * num_elements+1:c*num_elements,:) - mu) ** 2 ) / &
            norm

       if(momentum .gt. 1.E-8_real32)then
          output%mean(c) = momentum * mean(c) + (1._real32 - momentum) * mu
          output%variance(c) = momentum * variance(c) + (1._real32 - momentum) * var
       else
          output%mean(c) = mu
          output%variance(c) = var
       end if
       output%self%adj_ja(c, 1) = encode_real32(output%mean(c))
       output%self%adj_ja(c, 2) = encode_real32(output%variance(c))

       do concurrent(s = 1:size(input%val,2), i = 1:num_elements)
          output%val(i + (c-1) * num_elements, s) = &
               params%val(c,1) * ( input%val(i + (c-1) * num_elements, s) - mu ) / &
               sqrt(var + output%epsilon) + params%val(c+input%shape(num_dims),1)
       end do
    end do

    ! The context owns the node record, so graph metadata must be written to
    ! the record (output%self), not to the local handle views.
    output%self%value_partial_ok = .true.
    output%self%get_partial_left => get_partial_batchnorm_left
    output%self%get_partial_left_val => get_partial_batchnorm_left_val
    output%self%get_partial_right => get_partial_batchnorm_right
    output%self%get_partial_right_val => get_partial_batchnorm_right_val
    if(input%requires_grad .or. params%requires_grad)then
       output%self%requires_grad = .true.
       output%self%is_forward = input%is_forward .or. params%is_forward
       output%self%operation = 'batchnorm'
       output%self%left_operand => canonical(input)
       output%self%right_operand => canonical(params)
    end if

  end function batchnorm
!-------------------------------------------------------------------------------
  function get_partial_batchnorm_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    class(array_type), pointer :: input

    input => this%left_operand
    call output%allocate(array_shape = [ input%shape, size(upstream_grad%val,2) ])

    call this%get_partial_left_val(upstream_grad%val, output%val)

  end function get_partial_batchnorm_left
!-------------------------------------------------------------------------------
  function get_partial_batchnorm_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    class(array_type), pointer :: params

    params => this%right_operand
    call output%allocate(array_shape = [ params%shape, 1 ])

    call this%get_partial_right_val(upstream_grad%val, output%val)

  end function get_partial_batchnorm_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_batchnorm_left_val(this, upstream_grad, output)
    !! Get partial derivative wrt input for batchnorm (subroutine version)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, c, s, num_dims, num_elements
    real(real32), allocatable :: x_hat(:,:), dx_hat(:,:)
    real(real32) :: mu, var, eps, norm
    integer, dimension(size(this%shape)) :: input_shape

    input_shape = this%left_operand%shape

    eps = decode_real32(this%adj_ja(size(this%adj_ja,1), 1))
    num_dims = size(this%shape)
    num_elements = product(this%shape(1:num_dims - 1))

    output = 0._real32

    allocate(x_hat(num_elements, size(upstream_grad,2)))
    allocate(dx_hat(num_elements, size(upstream_grad,2)))
    norm = real( &
         product(input_shape(1:num_dims - 1)) * size(upstream_grad,2), &
         real32 &
    )

    do c = 1, input_shape(num_dims)
       mu = decode_real32(this%adj_ja(c, 1))
       var = decode_real32(this%adj_ja(c, 2))

       ! Normalised input
       x_hat = ( &
            this%left_operand%val((c-1)*num_elements+1:c*num_elements,:) - &
            mu &
       ) / sqrt(var + eps)

       ! Gradient of normalised input
       dx_hat = upstream_grad((c-1)*num_elements+1:c*num_elements,:) * &
            this%right_operand%val(c,1)

       ! Gradient wrt input
       do concurrent(s = 1:size(upstream_grad,2), i = 1:num_elements)
          output(i + (c-1)*num_elements,s) = &
               (1._real32 / (norm * sqrt(var + eps))) * &
               (norm * dx_hat(i,s) - sum(dx_hat) - x_hat(i,s) * sum(dx_hat * x_hat))
       end do
    end do

  end subroutine get_partial_batchnorm_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_batchnorm_right_val(this, upstream_grad, output)
    !! Get partial derivative wrt params for batchnorm (subroutine version)
    implicit none

    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: c, num_dims, num_elements
    real(real32), allocatable :: x_hat(:,:)
    real(real32) :: mu, var, eps
    integer, dimension(size(this%shape)) :: input_shape

    input_shape = this%left_operand%shape

    eps = decode_real32(this%adj_ja(size(this%adj_ja,1), 1))
    num_dims = size(this%shape)
    num_elements = product(this%shape(1:num_dims - 1))

    output = 0._real32

    allocate(x_hat(num_elements, size(upstream_grad,2)))

    do c = 1, input_shape(num_dims)
       mu = decode_real32(this%adj_ja(c, 1))
       var = decode_real32(this%adj_ja(c, 2))

       ! Normalised input
       x_hat(:,:) = ( &
            this%left_operand%val((c-1)*num_elements+1:c*num_elements,:) - mu &
       ) / sqrt(var + eps)

       output(c,1) = &
            sum(upstream_grad((c-1)*num_elements+1:c*num_elements,:) * x_hat)
       output(c + input_shape(num_dims),1) = &
            sum(upstream_grad((c-1)*num_elements+1:c*num_elements,:))

    end do

  end subroutine get_partial_batchnorm_right_val
!###############################################################################

end submodule athena__diffstruc_extd_submodule_batchnorm
