submodule (athena__diffstruc_extd) athena__diffstruc_extd_submodule_batchnorm
  !! Submodule containing implementations for extended diffstruc array operations

contains

!###############################################################################
  module function batchnorm_inference( &
       input, params, norm, mean, variance, epsilon &
  ) result( output )
    implicit none
    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: params
    real(real32), intent(in) :: norm
    real(real32), dimension(:), intent(in) :: mean
    real(real32), dimension(:), intent(in) :: variance
    real(real32), intent(in) :: epsilon
    type(batchnorm_array_type), pointer :: output

    integer :: i, c, s
    integer :: num_elements, num_dims

    call output%allocate(array_shape = input%shape)
    output%epsilon = epsilon
    output%mean = mean
    output%variance = variance
    num_dims = size(input%shape)
    num_elements = product(input%shape(1:num_dims - 1))
    do concurrent(c = 1:input%shape(num_dims))
       do concurrent(i=1:num_elements)
          output%val(i + (c-1) * num_elements:i + c * num_elements - 1, s) = &
               params%val(c,1) * ( &
                    input%val(i + (c-1) * num_elements:i + c * num_elements - 1, s) &
                    - mean(c) &
               ) / &
               sqrt(variance(c) + output%epsilon) + &
               params%val(c+input%shape(num_dims),1)
       end do
    end do

  end function batchnorm_inference
!-------------------------------------------------------------------------------
  module function batchnorm( &
       input, params, norm, momentum, mean, variance, epsilon &
  ) result( output )
    !! Batch normalization operation
    implicit none

    ! Arguments
    class(array_type), intent(in), target :: input
    class(array_type), intent(in), target :: params
    real(real32), intent(in) :: norm
    real(real32), intent(in) :: momentum
    real(real32), dimension(:), intent(in) :: mean
    real(real32), dimension(:), intent(in) :: variance
    real(real32), intent(in) :: epsilon
    type(batchnorm_array_type), pointer :: output

    ! Local variables
    integer :: i, c, s
    integer :: num_elements, num_dims
    real(real32) :: mu, var

    allocate(output)
    if(output%allocated) call output%deallocate()
    call output%allocate(array_shape = [ input%shape, size(input%val,2) ])
    output%epsilon = epsilon
    output%mean = mean
    output%variance = variance
    num_dims = size(input%shape)
    num_elements = product(input%shape(1:num_dims - 1))
    do concurrent(c = 1:input%shape(num_dims))
       mu = 0._real32
       var = 0._real32
       mu = sum(input%val((c-1) * num_elements+1:c*num_elements,:)) / norm
       var = sum( (input%val((c-1) * num_elements+1:c*num_elements,:) - mu) ** 2 ) / &
            norm

       if(momentum .gt. 1.E-8_real32) then
          output%mean(c) = momentum * mean(c) + (1._real32 - momentum) * mu
          output%variance(c) = momentum * variance(c) + (1._real32 - momentum) * var
       else
          output%mean(c) = mu
          output%variance(c) = var
       end if

       do concurrent(s = 1:size(input%val,2), i = 1:num_elements)
          output%val(i + (c-1) * num_elements, s) = &
               params%val(c,1) * ( input%val(i + (c-1) * num_elements, s) - mu ) / &
               sqrt(var + output%epsilon) + params%val(c+input%shape(num_dims),1)
       end do
    end do

    output%get_partial_left => get_partial_batchnorm_left
    output%get_partial_right => get_partial_batchnorm_right
    if(input%requires_grad .or. params%requires_grad) then
       output%requires_grad = .true.
       output%is_forward = input%is_forward .or. params%is_forward
       output%operation = 'batchnorm'
       output%left_operand => input
       output%right_operand => params
    end if

  end function batchnorm
!-------------------------------------------------------------------------------
  function get_partial_batchnorm_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    integer :: i, c, s, num_dims, num_elements
    real(real32), allocatable :: x_hat(:,:), dx_hat(:,:)
    real(real32) :: mu, var, eps, norm
    class(array_type), pointer :: input, params

    input => this%left_operand
    params => this%right_operand
    select type(this)
    type is (batchnorm_array_type)
       eps = this%epsilon
       num_dims = size(this%shape)
       num_elements = product(this%shape(1:num_dims - 1))

       call output%allocate(array_shape = [ input%shape, size(upstream_grad%val,2) ])
       output%val = 0._real32

       allocate(x_hat(num_elements, size(upstream_grad%val,2)))
       allocate(dx_hat(num_elements, size(upstream_grad%val,2)))
       norm = real( &
            product(input%shape(1:num_dims - 1)) * size(upstream_grad%val,2), &
            real32 &
       )

       do concurrent(c = 1:input%shape(num_dims))
          mu = this%mean(c)
          var = this%variance(c)

          ! Normalised input
          x_hat = (input%val((c-1)*num_elements+1:c*num_elements,:) - mu) / &
               sqrt(var + eps)

          ! Gradient of normalised input
          dx_hat = upstream_grad%val((c-1)*num_elements+1:c*num_elements,:) * &
               params%val(c,1)

          ! Gradient wrt input
          do concurrent(s = 1:size(upstream_grad%val,2), i = 1:num_elements)
             output%val(i + (c-1)*num_elements,s) = &
                  (1._real32 / (norm * sqrt(var + eps))) * &
                  (norm * dx_hat(i,s) - sum(dx_hat) - x_hat(i,s) * sum(dx_hat * x_hat))
          end do
       end do
    end select
  end function get_partial_batchnorm_left
!-------------------------------------------------------------------------------
  function get_partial_batchnorm_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    integer :: c, i, num_dims, num_elements
    real(real32), allocatable :: x_hat(:,:), dx_hat(:,:)
    real(real32) :: mu, var, eps
    class(array_type), pointer :: input, params

    input => this%left_operand
    params => this%right_operand
    select type(this)
    type is (batchnorm_array_type)
       eps = this%epsilon
       num_dims = size(this%shape)
       num_elements = product(this%shape(1:num_dims - 1))

       call output%allocate(array_shape = [ params%shape, 1 ])
       output%val = 0._real32

       allocate(x_hat(num_elements, size(upstream_grad%val,2)))

       do concurrent(c = 1:input%shape(num_dims))
          mu = this%mean(c)
          var = this%variance(c)

          ! Normalised input
          x_hat(:,:) = (input%val((c-1)*num_elements+1:c*num_elements,:) - mu) / &
               sqrt(var + eps)

          output%val(c,1) = &
               sum(upstream_grad%val((c-1)*num_elements+1:c*num_elements,:) * x_hat)
          output%val(c + input%shape(num_dims),1) = &
               sum(upstream_grad%val((c-1)*num_elements+1:c*num_elements,:))

       end do
    end select
  end function get_partial_batchnorm_right
!###############################################################################

end submodule athena__diffstruc_extd_submodule_batchnorm
