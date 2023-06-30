!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_linear
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none

  type, extends(activation_type) :: linear_type
   contains
     procedure :: activate => linear_activate
     procedure :: differentiate => linear_differentiate
  end type linear_type

  interface linear_setup
     procedure initialise
  end interface linear_setup

  
  private
  
  public :: linear_setup

  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  function initialise(scale)
    implicit none
    type(linear_type) :: initialise
    real(real12), optional, intent(in) :: scale
    
    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12 !0.05_real12
    end if

  end function initialise
!!!#############################################################################

       
!!!#############################################################################
!!! Linear transfer function
!!! f = gradient * x
!!!#############################################################################
  function linear_activate(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = this%scale * val
    !else
    !   output = 0.05_real12 * val
    !end if
  end function linear_activate
!!!#############################################################################


!!!#############################################################################
!!! derivative of linear transfer function
!!! e.g. df/dx (gradient * x) = gradient
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  function linear_differentiate(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = this%scale * val
    !else
    !   output = 0.05_real12 * val
    !end if
  end function linear_differentiate
!!!#############################################################################

end module activation_linear
