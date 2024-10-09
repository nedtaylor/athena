!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of no activation function (i.e. linear)
!!!#############################################################################
module activation_none
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: none_type
   contains
     procedure, pass(this) :: activate_1d => none_activate_1d
     procedure, pass(this) :: activate_2d => none_activate_2d
     procedure, pass(this) :: activate_3d => none_activate_3d
     procedure, pass(this) :: activate_4d => none_activate_4d
     procedure, pass(this) :: activate_5d => none_activate_5d
     procedure, pass(this) :: differentiate_1d => none_differentiate_1d
     procedure, pass(this) :: differentiate_2d => none_differentiate_2d
     procedure, pass(this) :: differentiate_3d => none_differentiate_3d
     procedure, pass(this) :: differentiate_4d => none_differentiate_4d
     procedure, pass(this) :: differentiate_5d => none_differentiate_5d
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
  pure function initialise(scale)
    implicit none
    type(none_type) :: initialise
    real(real12), optional, intent(in) :: scale

    initialise%name = "none"

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
  pure function none_activate_1d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = val * this%scale
  end function none_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function none_activate_2d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output

    output = val * this%scale
  end function none_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function none_activate_3d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = val * this%scale
  end function none_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function none_activate_4d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = val * this%scale
  end function none_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function none_activate_5d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = val * this%scale
  end function none_activate_5d
!!!#############################################################################


!!!#############################################################################
!!! derivative of NONE transfer function
!!! e.g. df/dx = x
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  pure function none_differentiate_1d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = this%scale
  end function none_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function none_differentiate_2d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output

    output = this%scale
  end function none_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function none_differentiate_3d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = this%scale
  end function none_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function none_differentiate_4d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = this%scale
  end function none_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function none_differentiate_5d(this, val) result(output)
    implicit none
    class(none_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = this%scale
  end function none_differentiate_5d
!!!#############################################################################

end module activation_none
