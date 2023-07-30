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
     procedure, pass(this) :: activate_1d => relu_activate_1d
     procedure, pass(this) :: activate_3d => relu_activate_3d
     procedure, pass(this) :: differentiate_1d => relu_differentiate_1d
     procedure, pass(this) :: differentiate_3d => relu_differentiate_3d
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
  function initialise(scale)
    implicit none
    type(relu_type) :: initialise
    real(real12), optional, intent(in) :: scale

    initialise%name = "relu"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12
    end if
    initialise%threshold = 0._real12

  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! RELU transfer function
!!! f = max(0, x)
!!!#############################################################################
  pure function relu_activate_1d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = max(this%threshold, val) * this%scale
  end function relu_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_activate_3d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = max(this%threshold, val) * this%scale
  end function relu_activate_3d
!!!#############################################################################


!!!#############################################################################
!!! derivative of RELU transfer function
!!! e.g. df/dx (1*x) = 1
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  pure function relu_differentiate_1d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_differentiate_3d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_3d
!!!#############################################################################

end module activation_relu
