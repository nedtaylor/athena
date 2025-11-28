module athena__activation_relu
  !! Module containing implementation of the ReLU activation function
  !!
  !! This module implements the Rectified Linear Unit activation function
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(*), max
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: relu_setup


  type, extends(activation_type) :: relu_type
     !! Type for ReLU activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate => relu_activate
  end type relu_type

  interface relu_setup
     procedure initialise
  end interface relu_setup



contains

!###############################################################################
  pure function initialise(scale)
    !! Initialise a ReLU activation function
    implicit none

    ! Arguments
    type(relu_type) :: initialise
    !! ReLU activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "relu"

    if(present(scale))then
       initialise%scale = scale
       initialise%apply_scaling = .true.
    else
       initialise%scale = 1._real32
    end if
    initialise%threshold = 0._real32

  end function initialise
!###############################################################################


!###############################################################################
  function relu_activate(this, val) result(output)
    !! Apply ReLU activation to 1D array
    !!
    !! Computes: f = max(0,x)
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    if(this%apply_scaling)then
       output => max(val, this%threshold) * this%scale
    else
       output => max(val, this%threshold)
    end if
  end function relu_activate
!###############################################################################

end module athena__activation_relu
