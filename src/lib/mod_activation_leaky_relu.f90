module athena__activation_leaky_relu
  !! Module containing implementation of the leaky ReLU activation function
  !!
  !! This module implements the Leaky Rectified Linear Unit function:
  !! f(x) = x if x > 0, 0.01x otherwise
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(*), max
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: leaky_relu_setup


  type, extends(activation_type) :: leaky_relu_type
   contains
     procedure, pass(this) :: activate => leaky_relu_activate
  end type leaky_relu_type

  interface leaky_relu_setup
     procedure initialise
  end interface leaky_relu_setup



contains

!###############################################################################
  pure function initialise(scale)
    !! Initialise a leaky ReLU activation function
    implicit none

    ! Arguments
    type(leaky_relu_type) :: initialise
    !! Leaky ReLU activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "leaky_relu"

    if(present(scale))then
       initialise%scale = scale
       initialise%apply_scaling = .true.
    else
       initialise%scale = 1._real32
    end if
  end function initialise
!###############################################################################


!###############################################################################
  function leaky_relu_activate(this, val) result(output)
    !! Apply leaky ReLU activation to 1D array
    !!
    !! Computes: f = max(0.01x, x)
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    ! allocate(output)
    if(this%apply_scaling)then
       output => max(val * 0.01_real32, val) * this%scale
    else
       output => max(val * 0.01_real32, val)
    end if
  end function leaky_relu_activate
!###############################################################################

end module athena__activation_leaky_relu
