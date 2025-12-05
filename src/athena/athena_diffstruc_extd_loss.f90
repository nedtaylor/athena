submodule (athena__diffstruc_extd) athena__diffstruc_extd_loss_submodule
  !! Submodule containing implementations for extended diffstruc array operations
  use coreutils, only: stop_program
  use diffstruc, only: sign, merge, abs, operator(.le.)

contains

!###############################################################################
  module function huber_array(delta, gamma) result( output )
    !! Huber loss function
    implicit none
    class(array_type), intent(in), target :: delta
    real(real32), intent(in) :: gamma
    type(array_type), pointer :: output

    type(array_type), pointer :: b_array

    output => delta%create_result()
    where (abs(delta%val) .le. gamma)
       output%val = 0.5_real32 * (delta%val)**2._real32
    elsewhere
       output%val = gamma * (abs(delta%val) - 0.5_real32 * gamma)
    end where

    output%get_partial_left => get_partial_huber
    output%get_partial_left_val => get_partial_huber_val
    if(delta%requires_grad)then
       output%requires_grad = .true.
       output%is_forward = delta%is_forward
       output%operation = 'huber'
       output%left_operand => delta
       output%owns_left_operand = delta%is_temporary
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%is_scalar = .true.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1,1) = gamma
    output%right_operand => b_array
    output%owns_right_operand = .true.

  end function huber_array
!-------------------------------------------------------------------------------
  function get_partial_huber(this, upstream_grad) result(output)
    !! Get partial derivative of huber loss
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: ptr

    ptr => merge( &
         this%left_operand, &
         this%right_operand%val(1,1) * sign(1._real32, this%left_operand), &
         abs(this%left_operand) .le. this%right_operand%val(1,1) &
    )

    call output%assign_and_deallocate_source(ptr)
  end function get_partial_huber
!-------------------------------------------------------------------------------
  pure subroutine get_partial_huber_val(this, upstream_grad, output)
    !! Get partial derivative of huber loss (in-place version)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    where (abs(this%left_operand%val) .le. this%right_operand%val(1,1))
       output = this%left_operand%val
    elsewhere
       output = this%right_operand%val(1,1) * sign(1._real32, this%left_operand%val)
    end where

  end subroutine get_partial_huber_val
!###############################################################################

end submodule athena__diffstruc_extd_loss_submodule
