module athena__activation_tanh
  !! Module containing implementation of the tanh activation function
  !!
  !! This module implements the hyperbolic tangent activation function
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(*), tanh
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: tanh_setup


  type, extends(activation_type) :: tanh_type
     !! Type for tanh activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate => tanh_activate
  end type tanh_type

  interface tanh_setup
     procedure initialise
  end interface tanh_setup



contains

!###############################################################################
  pure function initialise(threshold, scale)
    !! Initialise a tanh activation function
    implicit none

    ! Arguments
    type(tanh_type) :: initialise
    !! tanh activation type
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "tanh"

    if(present(scale))then
       initialise%scale = scale
       initialise%apply_scaling = .true.
    else
       initialise%scale = 1._real32
    end if

    !initialise%name = "tanh"
    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = min(huge(1._real32),32._real32)
    end if

  end function initialise
!###############################################################################


!###############################################################################
  function tanh_activate(this, val) result(output)
    !! Apply tanh activation to 1D array
    !!
    !! Applies the hyperbolic tangent function element-wise to input array:
    !! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    implicit none

    ! Arguments
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    if(this%apply_scaling)then
       output => tanh(val) * this%scale
    else
       output => tanh(val)
    end if
  end function tanh_activate
!###############################################################################

end module athena__activation_tanh
