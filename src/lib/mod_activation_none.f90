module athena__activation_none
  !! Module containing implementation of no activation function (i.e. linear)
  !!
  !! This module implements the identity function f(x) = x
  use coreutils, only: real32
  use diffstruc, only: array_type
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: none_setup


  type, extends(activation_type) :: none_type
   contains
     procedure, pass(this) :: activate => none_activate
  end type none_type

  interface none_setup
     procedure initialise
  end interface none_setup



contains

!###############################################################################
  pure function initialise()
    !! Initialise a none (no-op) activation function
    implicit none

    ! Arguments
    type(none_type) :: initialise
    !! None activation type

    initialise%name = "none"
    initialise%scale = 1.0_real32

  end function initialise
!###############################################################################


!###############################################################################
  function none_activate(this, val) result(output)
    !! Apply identity activation to 1D array
    !!
    !! Simply returns scaled input: f = scale * x
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Scaled output values

    output => val * 1._real32 ! multiplication by 1 to ensure new allocation
  end function none_activate
!###############################################################################

end module athena__activation_none
