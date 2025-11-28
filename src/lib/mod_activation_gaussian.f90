module athena__activation_gaussian
  !! Module containing implementation of the Gaussian activation function
  !!
  !! This module implements the Gaussian (bell curve) activation function
  use coreutils, only: real32
  use diffstruc, only: array_type, gaussian, operator(*)
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: gaussian_setup


  type, extends(activation_type) :: gaussian_type
     !! Type for Gaussian activation function with overloaded procedures
     real(real32) :: sigma
     !! Standard deviation parameter for Gaussian function
     real(real32) :: mu
     !! Mean parameter for Gaussian function
   contains
     procedure, pass(this) :: activate => gaussian_activate
  end type gaussian_type

  interface gaussian_setup
     procedure initialise
  end interface gaussian_setup



contains

!###############################################################################
  pure function initialise(threshold, scale, sigma, mu)
    !! Initialise a Gaussian activation function
    implicit none

    ! Arguments
    type(gaussian_type) :: initialise
    !! Gaussian activation type
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    real(real32), optional, intent(in) :: sigma
    !! Optional standard deviation parameter
    real(real32), optional, intent(in) :: mu
    !! Optional mean parameter

    initialise%name = "gaussian"

    if(present(scale))then
       initialise%scale = scale
       initialise%apply_scaling = .true.
    else
       initialise%scale = 1._real32
    end if

    if(present(sigma))then
       initialise%sigma = sigma
    else
       initialise%sigma = 1.5_real32
    end if

    if(present(mu))then
       initialise%mu = mu
    else
       initialise%mu = 0._real32
    end if

    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = min(huge(1._real32),16._real32) * &
            initialise%sigma
    end if

  end function initialise
!###############################################################################


!###############################################################################
  function gaussian_activate(this, val) result(output)
    !! Apply Gaussian activation to array
    !!
    !! Applies the Gaussian function element-wise to input array:
    !! f = exp(-x^2/(2σ^2))/(σ√(2π))
    implicit none

    ! Arguments
    class(gaussian_type), intent(in) :: this
    !! Gaussian activation type containing sigma parameter
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Gaussian activated output values

    if(this%apply_scaling)then
       output => gaussian(val, this%mu, this%sigma) * this%scale
    else
       output => gaussian(val, this%mu, this%sigma)
    end if

  end function gaussian_activate
!###############################################################################

end module athena__activation_gaussian
