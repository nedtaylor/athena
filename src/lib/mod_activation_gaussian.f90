!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_gaussian
  use constants, only: real12, pi
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: gaussian_type
     real(real12) :: sigma
   contains
     procedure :: activate => gaussian_activate
     procedure :: differentiate => gaussian_differentiate
  end type gaussian_type
  
  interface gaussian_setup
     procedure initialise
  end interface gaussian_setup
  
  
  private
  
  public :: gaussian_setup
  
  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  function initialise(threshold, scale, sigma)
    implicit none
    type(gaussian_type) :: initialise
    real(real12), optional, intent(in) :: threshold
    real(real12), optional, intent(in) :: scale
    real(real12), optional, intent(in) :: sigma

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12
    end if

    if(present(sigma))then
       initialise%sigma = sigma
    else
       initialise%sigma = 1.5_real12
    end if

    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = min(huge(1._real12),16._real12) * &
            initialise%sigma
    end if

  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! gaussian transfer function
!!! f = 1/(1+exp(-x))
!!!#############################################################################
  function gaussian_activate(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    if(abs(val).gt.this%threshold)then
       output = 0._real12
    else
       output = this%scale * 1._real12/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real12 * (val/this%sigma)**2._real12)
    end if
  end function gaussian_activate
!!!#############################################################################


!!!#############################################################################
!!! derivative of gaussian function
!!! df/dx = f * (1 - f)
!!!#############################################################################
  function gaussian_differentiate(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = -val/this%sigma**2._real12 * this%activate(val)
  end function gaussian_differentiate
!!!#############################################################################

end module activation_gaussian
