!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_leaky_relu
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: leaky_relu_type
   contains
     procedure :: activate => leaky_relu_activate
     procedure :: differentiate => leaky_relu_differentiate
  end type leaky_relu_type
  
  interface leaky_relu_setup
     procedure initialise
  end interface leaky_relu_setup
  
  
  private
  
  public :: leaky_relu_setup
  
  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  function initialise(scale)
    implicit none
    type(leaky_relu_type) :: initialise    
    real(real12), optional, intent(in) :: scale

    initialise%name = "leaky_relu"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12
    end if
  end function initialise
!!!#############################################################################
  

!!!#############################################################################
!!! leaky ReLU transfer function
!!! f = max(0.01*x, x)
!!!#############################################################################
  function leaky_relu_activate(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = max(0.01_real12*val, val) * this%scale
  end function leaky_relu_activate
!!!#############################################################################


!!!#############################################################################
!!! derivative of leaky ReLU transfer function
!!! e.g. df/dx (1.0*x) = 1.0
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  function leaky_relu_differentiate(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    if(val.ge.0._real12)then
       output = this%scale
    else
       output = 0.01_real12
    end if
  end function leaky_relu_differentiate
!!!#############################################################################

end module activation_leaky_relu
