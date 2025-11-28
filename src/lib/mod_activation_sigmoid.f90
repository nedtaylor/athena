module athena__activation_sigmoid
  !! Module containing implementation of the sigmoid activation function
  !!
  !! This module implements the logistic sigmoid function for normalizing
  !! outputs between 0 and 1
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(-), &
       operator(*), operator(/), exp, merge, operator(.gt.), sigmoid
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: sigmoid_setup


  type, extends(activation_type) :: sigmoid_type
     !! Type for sigmoid activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate => sigmoid_activate
  end type sigmoid_type

  interface sigmoid_setup
     procedure initialise
  end interface sigmoid_setup



contains

!###############################################################################
  pure function initialise(threshold, scale)
    !! Initialise a sigmoid activation function
    implicit none

    ! Arguments
    type(sigmoid_type) :: initialise
    !! Sigmoid activation type
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "sigmoid"

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
    !initialise%scale = 1._real32
  end function initialise
!###############################################################################


!###############################################################################
  function sigmoid_activate(this, val) result(output)
    !! Apply sigmoid activation to 1D array
    !!
    !! Computes: f = 1/(1+exp(-x))
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values in range [0,1]

    if(this%apply_scaling)then
       output => sigmoid(val) * this%scale
    else
       output => sigmoid(val)
    end if
  end function sigmoid_activate
!###############################################################################

end module athena__activation_sigmoid
