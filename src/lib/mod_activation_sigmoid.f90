!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_sigmoid
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: sigmoid_type
   contains
     procedure :: activate => sigmoid_activate
     procedure :: differentiate => sigmoid_differentiate
  end type sigmoid_type
  
  interface sigmoid_setup
     procedure initialise
  end interface sigmoid_setup
  
  
  private
  
  public :: sigmoid_setup
  
  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  function initialise()
    implicit none
    type(sigmoid_type) :: initialise    
    !initialise%scale = 1._real12
  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! sigmoid transfer function
!!! f = 1/(1+exp(-x))
!!!#############################################################################
  function sigmoid_activate(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = 1._real12 /(1._real12 + exp(-val))
  end function sigmoid_activate
!!!#############################################################################


!!!#############################################################################
!!! derivative of sigmoid function
!!! df/dx = f * (1 - f)
!!!#############################################################################
  function sigmoid_differentiate(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = this%activate(val) * (1._real12 - this%activate(val))
  end function sigmoid_differentiate
!!!#############################################################################

end module activation_sigmoid
