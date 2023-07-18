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
  function initialise(threshold, scale)
    implicit none
    type(sigmoid_type) :: initialise
    real(real12), optional, intent(in) :: threshold
    real(real12), optional, intent(in) :: scale

    initialise%name = "sigmoid"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12
    end if

    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = -min(huge(1._real12),32._real12)
    end if
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

    if(val.lt.this%threshold)then
       output = 0._real12
    else
       output = this%scale /(1._real12 + exp(-val))
    end if
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

    output = this%scale * this%activate(val) * (this%scale - this%activate(val))
  end function sigmoid_differentiate
!!!#############################################################################

end module activation_sigmoid
