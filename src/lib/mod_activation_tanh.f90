!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_tanh
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: tanh_type
   contains
     procedure :: activate => tanh_activate
     procedure :: differentiate => tanh_differentiate
  end type tanh_type
  
  interface tanh_setup
     procedure initialise
  end interface tanh_setup
  
  
  private
  
  public :: tanh_setup
  
  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  function initialise(threshold, scale)
    implicit none
    type(tanh_type) :: initialise
    real(real12), optional, intent(in) :: threshold
    real(real12), optional, intent(in) :: scale

    initialise%name = "tanh"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12
    end if

    !initialise%name = "tanh"
    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = min(huge(1._real12),32._real12)
    end if

  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! tanh transfer function
!!! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
!!!#############################################################################
  function tanh_activate(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output


    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    if(abs(val).gt.this%threshold)then
       output = sign(1._real12, val) * this%scale
    else
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end if

  end function tanh_activate
!!!#############################################################################


!!!#############################################################################
!!! derivative of tanh function
!!! df/dx = 1 - f^2
!!!#############################################################################
  function tanh_differentiate(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real12), intent(in) :: val
    real(real12) :: output

    output = this%scale * &
         (1._real12 - (this%activate(val)/this%scale) ** 2._real12)

  end function tanh_differentiate
!!!#############################################################################

end module activation_tanh
