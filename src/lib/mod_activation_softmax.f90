module athena__activation_softmax
  !! Module containing implementation of the softmax activation function
  !!
  !! This module implements the softmax activation function for normalising
  !! outputs into probability distributions
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(*)
  use athena__diffstruc_extd, only: softmax
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: softmax_setup


  type, extends(activation_type) :: softmax_type
     !! Type for softmax activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate => softmax_activate
  end type softmax_type

  interface softmax_setup
     procedure initialise
  end interface softmax_setup



contains

!###############################################################################
  pure function initialise(threshold, scale)
    !! Initialise a softmax activation function
    implicit none

    ! Arguments
    type(softmax_type) :: initialise
    !! Softmax activation type
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "softmax"

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
  end function initialise
!###############################################################################


!###############################################################################
  function softmax_activate(this, val) result(output)
    !! Apply softmax activation to 1D array
    !!
    !! Computes: f = exp(x-max)/sum(exp(x-max))
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Normalised probability distribution output

    !! compute softmax values
    if(this%apply_scaling)then
       output => softmax(val, dim=2) * this%scale
    else
       output => softmax(val, dim=2)
    end if
  end function softmax_activate
!###############################################################################

end module athena__activation_softmax
