module athena__activation_swish
  !! Module containing implementation of the swish activation function
  !!
  !! This module implements the swish activation function: f(x) = x * sigmoid(β*x)
  !! where β is a learnable parameter (default β=1)
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(*)
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
     procedure, pass(this) :: activate => swish_activate
  end type swish_type

  interface swish_setup
     !! Interface for setting up swish activation function
     procedure initialise
  end interface swish_setup

contains

!###############################################################################
  pure function initialise(threshold, scale, beta)
    !! Initialise a swish activation function
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    real(real32), optional, intent(in) :: beta
    !! Optional beta parameter for swish function

    type(swish_type) :: initialise
    !! Swish activation type

    initialise%name = "swish"

    if(present(scale))then
       initialise%scale = scale
       initialise%apply_scaling = .true.
    else
       initialise%scale = 1._real32
    end if

    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = -min(huge(1._real32),32._real32)
    end if

    if(present(beta))then
       initialise%beta = beta
    else
       initialise%beta = 1._real32
    end if
  end function initialise
!###############################################################################


!###############################################################################
  function swish_activate(this, val) result(output)
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
    if(this%apply_scaling)then
       output => swish(val, this%beta) * this%scale
    else
       output => swish(val, this%beta)
    end if
  end function swish_activate
!###############################################################################

end module athena__activation_swish
