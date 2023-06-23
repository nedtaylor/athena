!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_relu
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: relu_type
   contains
     procedure :: activate => relu_activate
     procedure :: differentiate => relu_differentiate
  end type relu_type
  
  interface relu_setup
     procedure initialise
  end interface relu_setup
  
  
  private
  
  public :: relu_setup
  
  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  function initialise()
    implicit none
    type(relu_type) :: initialise    
    !initialise%scale = 1._real12
  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! RELU transfer function
!!! f = max(0, x)
!!!#############################################################################
  function relu_activate(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = max(0._real12, val)
  end function relu_activate
!!!#############################################################################


!!!#############################################################################
!!! derivative of RELU transfer function
!!! e.g. df/dx (1*x) = 1
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  function relu_differentiate(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    if(val.ge.0._real12)then
       output = 1._real12
    else
       output = 0._real12
    end if
  end function relu_differentiate
!!!#############################################################################

end module activation_relu
