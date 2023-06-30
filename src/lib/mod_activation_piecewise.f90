!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_piecewise
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none

  type, extends(activation_type) :: piecewise_type
     real(real12) :: intercept, min, max
   contains
     procedure :: activate => piecewise_activate
     procedure :: differentiate => piecewise_differentiate
  end type piecewise_type

  interface piecewise_setup
     procedure initialise
  end interface piecewise_setup

  
  private
  
  public :: piecewise_setup

  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  function initialise(scale, intercept)
    implicit none
    type(piecewise_type) :: initialise
    real(real12), optional, intent(in) :: scale, intercept
    
    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12 !0.05_real12
    end if
    if(present(intercept))then
       initialise%intercept = intercept
    else
       initialise%intercept = 1._real12 !0.05_real12
    end if

    initialise%max = initialise%intercept/initialise%scale
    initialise%min = -initialise%max

  end function initialise
!!!#############################################################################

       
!!!#############################################################################
!!! Piecewise transfer function
!!! f = gradient * x
!!!#############################################################################
  function piecewise_activate(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    if(val.le.this%min)then
       output = 0._real12
    elseif(val.ge.this%max)then
       output = this%scale
    else
       output = this%scale * val + this%intercept
    end if
  end function piecewise_activate
!!!#############################################################################


!!!#############################################################################
!!! derivative of piecewise transfer function
!!! e.g. df/dx (gradient * x) = gradient
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  function piecewise_differentiate(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    if(val.le.this%min.or.val.ge.this%max)then
       output = 0._real12
    else
       output = this%scale
    end if
  end function piecewise_differentiate
!!!#############################################################################

end module activation_piecewise
