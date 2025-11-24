module athena__activation_swish
  !! Module containing implementation of the swish activation function
  !!
  !! This module implements the swish activation function: f(x) = x * sigmoid(β*x)
  !! where β is a learnable parameter (default β=1)
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(-), &
       operator(*), operator(/), exp
  use athena__misc_types, only: activation_type
  use athena__diffstruc_extd, only: swish
  implicit none

  private

  public :: swish_setup

  type, extends(activation_type) :: swish_type
     !! Type for swish activation function with overloaded procedures
     real(real32) :: beta = 1._real32
     !! Beta parameter for swish function
   contains
     procedure, pass(this) :: activate_array => swish_activate_array
     procedure, pass(this) :: activate_1d => swish_activate_1d
     !! Apply swish activation to 1D array
     procedure, pass(this) :: activate_2d => swish_activate_2d
     !! Apply swish activation to 2D array
     procedure, pass(this) :: activate_3d => swish_activate_3d
     !! Apply swish activation to 3D array
     procedure, pass(this) :: activate_4d => swish_activate_4d
     !! Apply swish activation to 4D array
     procedure, pass(this) :: activate_5d => swish_activate_5d
     !! Apply swish activation to 5D array
     procedure, pass(this) :: differentiate_1d => swish_differentiate_1d
     !! Differentiate swish activation for 1D array
     procedure, pass(this) :: differentiate_2d => swish_differentiate_2d
     !! Differentiate swish activation for 2D array
     procedure, pass(this) :: differentiate_3d => swish_differentiate_3d
     !! Differentiate swish activation for 3D array
     procedure, pass(this) :: differentiate_4d => swish_differentiate_4d
     !! Differentiate swish activation for 4D array
     procedure, pass(this) :: differentiate_5d => swish_differentiate_5d
     !! Differentiate swish activation for 5D array
  end type swish_type

  interface swish_setup
     !! Interface for setting up swish activation function
     procedure initialise
  end interface swish_setup

contains

!###############################################################################
  pure function initialise(threshold, scale, beta) result(swish_func)
    !! Initialise a swish activation function
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    real(real32), optional, intent(in) :: beta
    !! Optional beta parameter for swish function

    type(swish_type) :: swish_func
    !! Swish activation type

    swish_func%name = "swish"

    if(present(scale))then
       swish_func%scale = scale
    else
       swish_func%scale = 1._real32
    end if

    if(present(threshold))then
       swish_func%threshold = threshold
    else
       swish_func%threshold = -min(huge(1._real32),32._real32)
    end if

    if(present(beta))then
       swish_func%beta = beta
    else
       swish_func%beta = 1._real32
    end if
  end function initialise
!###############################################################################


!###############################################################################
  function swish_activate_array(this, val) result(output)
    !! Apply swish activation to 1D array
    !!
    !! Computes: f(x) = x * sigmoid(β*x) = x / (1 + exp(-β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Swish activation output

    ! Compute sigmoid(β*x)
    ! Compute swish: x * sigmoid(β*x)
    output => this%scale * swish(val, this%beta)
  end function swish_activate_array
  function swish_activate_1d(this, val) result(output)
    !! Apply swish activation to 1D array
    !!
    !! Computes: f(x) = x * sigmoid(β*x) = x / (1 + exp(-β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Swish activation output

    ! Local variables
    real(real32), dimension(size(val,dim=1)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute swish: x * sigmoid(β*x)
    output = this%scale * val * sigmoid_part
  end function swish_activate_1d
!###############################################################################


!###############################################################################
  function swish_activate_2d(this, val) result(output)
    !! Apply swish activation to 2D array
    !!
    !! Computes: f(x) = x * sigmoid(β*x) = x / (1 + exp(-β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Swish activation output

    ! Local variables
    real(real32), dimension(size(val,1),size(val,2)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute swish: x * sigmoid(β*x)
    output = this%scale * val * sigmoid_part
  end function swish_activate_2d
!###############################################################################


!###############################################################################
  function swish_activate_3d(this, val) result(output)
    !! Apply swish activation to 3D array
    !!
    !! Computes: f(x) = x * sigmoid(β*x) = x / (1 + exp(-β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Swish activation output

    ! Local variables
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute swish: x * sigmoid(β*x)
    output = this%scale * val * sigmoid_part
  end function swish_activate_3d
!###############################################################################


!###############################################################################
  function swish_activate_4d(this, val) result(output)
    !! Apply swish activation to 4D array
    !!
    !! Computes: f(x) = x * sigmoid(β*x) = x / (1 + exp(-β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Swish activation output

    ! Local variables
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute swish: x * sigmoid(β*x)
    output = this%scale * val * sigmoid_part
  end function swish_activate_4d
!###############################################################################


!###############################################################################
  function swish_activate_5d(this, val) result(output)
    !! Apply swish activation to 5D array
    !!
    !! Computes: f(x) = x * sigmoid(β*x) = x / (1 + exp(-β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Swish activation output

    ! Local variables
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute swish: x * sigmoid(β*x)
    output = this%scale * val * sigmoid_part
  end function swish_activate_5d
!###############################################################################


!###############################################################################
  function swish_differentiate_1d(this, val) result(output)
    !! Differentiate swish activation for 1D array
    !!
    !! Computes the derivative:
    !!      df/dx = sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values

    ! Local variables
    real(real32), dimension(size(val,dim=1)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute derivative: sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    output = this%scale * ( sigmoid_part + &
         this%beta * val * sigmoid_part * (1._real32 - sigmoid_part) )
  end function swish_differentiate_1d
!###############################################################################


!###############################################################################
  function swish_differentiate_2d(this, val) result(output)
    !! Differentiate swish activation for 2D array
    !!
    !! Computes the derivative:
    !!      df/dx = sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values

    ! Local variables
    real(real32), dimension(size(val,1),size(val,2)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute derivative: sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    output = this%scale * ( sigmoid_part + &
         this%beta * val * sigmoid_part * (1._real32 - sigmoid_part) )
  end function swish_differentiate_2d
!###############################################################################


!###############################################################################
  function swish_differentiate_3d(this, val) result(output)
    !! Differentiate swish activation for 3D array
    !!
    !! Computes the derivative:
    !!      df/dx = sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output values

    ! Local variables
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute derivative: sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    output = this%scale * ( sigmoid_part + &
         this%beta * val * sigmoid_part * (1._real32 - sigmoid_part) )
  end function swish_differentiate_3d
!###############################################################################


!###############################################################################
  function swish_differentiate_4d(this, val) result(output)
    !! Differentiate swish activation for 4D array
    !!
    !! Computes the derivative:
    !!      df/dx = sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output values

    ! Local variables
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute derivative: sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    output = this%scale * ( sigmoid_part + &
         this%beta * val * sigmoid_part * (1._real32 - sigmoid_part) )
  end function swish_differentiate_4d
!###############################################################################


!###############################################################################
  function swish_differentiate_5d(this, val) result(output)
    !! Differentiate swish activation for 5D array
    !!
    !! Computes the derivative:
    !!      df/dx = sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    implicit none

    ! Arguments
    class(swish_type), intent(in) :: this
    !! Swish activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values

    ! Local variables
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: sigmoid_part
    !! Sigmoid component

    ! Compute sigmoid(β*x)
    sigmoid_part = 1._real32 / (1._real32 + exp(-this%beta * val))

    ! Compute derivative: sigmoid(β*x) + x * β * sigmoid(β*x) * (1 - sigmoid(β*x))
    output = this%scale * ( sigmoid_part + &
         this%beta * val * sigmoid_part * (1._real32 - sigmoid_part) )
  end function swish_differentiate_5d
!###############################################################################

end module athena__activation_swish
