!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_none
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: none_type
   contains
     procedure :: activate => none_activate
     procedure :: differentiate => none_differentiate
  end type none_type
  
  interface none_setup
     procedure initialise
  end interface none_setup
  
  
  private
  
  public :: none_setup
  
  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  function initialise(scale)
    implicit none
    type(none_type) :: initialise
    real(real12), optional, intent(in) :: scale
    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12
    end if
  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! NONE transfer function
!!! x
!!!#############################################################################
  function none_activate(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = val * this%scale
  end function none_activate
!!!#############################################################################


!!!#############################################################################
!!! derivative of NONE transfer function
!!! e.g. df/dx = x
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  function none_differentiate(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = val * this%scale
  end function none_differentiate
!!!#############################################################################

end module activation_none
