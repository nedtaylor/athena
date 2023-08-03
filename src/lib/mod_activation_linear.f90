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
     procedure, pass(this) :: activate_1d => linear_activate_1d
     procedure, pass(this) :: activate_3d => linear_activate_3d
     procedure, pass(this) :: differentiate_1d => linear_differentiate_1d
     procedure, pass(this) :: differentiate_3d => linear_differentiate_3d
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
  pure function initialise(scale)
    implicit none
    type(linear_type) :: initialise
    real(real12), optional, intent(in) :: scale
    
    initialise%name = "linear"

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
  pure function linear_activate_1d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = this%scale * val
  end function linear_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_activate_3d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = this%scale * val
  end function linear_activate_3d
!!!#############################################################################


!!!#############################################################################
!!! derivative of linear transfer function
!!! e.g. df/dx (gradient * x) = gradient
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  pure function linear_differentiate_1d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = this%scale * val
  end function linear_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_differentiate_3d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = this%scale * val
  end function linear_differentiate_3d
!!!#############################################################################

end module activation_linear
