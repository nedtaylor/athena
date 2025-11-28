module athena__activation_linear
  !! Module containing implementation of the linear activation function
  !!
  !! This module implements a scaled linear function f(x) = scale * x
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(*)
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: linear_setup


  type, extends(activation_type) :: linear_type
   contains
     procedure, pass(this) :: activate => linear_activate
  end type linear_type

  interface linear_setup
     procedure initialise
  end interface linear_setup



contains

!###############################################################################
  pure function initialise(scale)
    !! Initialise a linear activation function
    implicit none

    ! Arguments
    type(linear_type) :: initialise
    !! Linear activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "linear"

    if(present(scale))then
       initialise%scale = scale
       initialise%apply_scaling = .true.
    else
       initialise%scale = 1._real32
    end if

  end function initialise
!###############################################################################


!###############################################################################
  function linear_activate(this, val) result(output)
    !! Apply linear activation to 1D array
    !!
    !! Computes: f = scale * x
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Scaled output values

    if(this%apply_scaling)then
       output => val * this%scale
    else
       output => val * 1._real32 ! multiplication by 1 to ensure new allocation
    end if
  end function linear_activate
!###############################################################################

end module athena__activation_linear
