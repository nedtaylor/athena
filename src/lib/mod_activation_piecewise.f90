!###############################################################################
module athena__activation_piecewise
  !! Module containing implementation of the piecewise activation function
  !! https://doi.org/10.48550/arXiv.1809.09534
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(*)
  use athena__diffstruc_extd, only: piecewise
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: piecewise_setup


  type, extends(activation_type) :: piecewise_type
     !! Type for piecewise activation function with overloaded procedures
     real(real32) :: gradient, limit
   contains
     procedure, pass(this) :: activate => piecewise_activate
  end type piecewise_type

  interface piecewise_setup
     procedure initialise
  end interface piecewise_setup



contains

!###############################################################################
  pure function initialise(scale, gradient, limit)
    !! Initialise a piecewise activation function
    implicit none

    ! Arguments
    type(piecewise_type) :: initialise
    !! Piecewise activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    real(real32), optional, intent(in) :: gradient
    !! Optional gradient parameter for piecewise function
    real(real32), optional, intent(in) :: limit
    !! Optional limit parameter for piecewise function
    !! -limit < x < limit

    initialise%name = "piecewise"

    if(present(scale))then
       initialise%scale = scale
       initialise%apply_scaling = .true.
    else
       initialise%scale = 1._real32
    end if
    if(present(gradient))then
       initialise%gradient = gradient
    else
       initialise%gradient = 0.1_real32
    end if
    if(present(limit))then
       initialise%limit = limit
    else
       initialise%limit = 1_real32
    end if

  end function initialise
!###############################################################################


!###############################################################################
  function piecewise_activate(this, val) result(output)
    !! Apply piecewise activation to 1D array
    !!
    !! Computes piecewise function:
    !! f = 0 if x ≤ min
    !! f = scale if x ≥ max
    !! f = scale * x + intercept otherwise
    implicit none

    ! Arguments
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    if(this%apply_scaling)then
       output => piecewise(val, this%gradient, this%limit) * this%scale
    else
       output => piecewise(val, this%gradient, this%limit)
    end if
  end function piecewise_activate
!###############################################################################

end module athena__activation_piecewise
