!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the linear activation function
!!!#############################################################################
module activation_linear
  use constants, only: real32
  use custom_types, only: activation_type
  implicit none

  type, extends(activation_type) :: linear_type
   contains
     procedure, pass(this) :: activate_1d => linear_activate_1d
     procedure, pass(this) :: activate_2d => linear_activate_2d
     procedure, pass(this) :: activate_3d => linear_activate_3d
     procedure, pass(this) :: activate_4d => linear_activate_4d
     procedure, pass(this) :: activate_5d => linear_activate_5d
     procedure, pass(this) :: differentiate_1d => linear_differentiate_1d
     procedure, pass(this) :: differentiate_2d => linear_differentiate_2d
     procedure, pass(this) :: differentiate_3d => linear_differentiate_3d
     procedure, pass(this) :: differentiate_4d => linear_differentiate_4d
     procedure, pass(this) :: differentiate_5d => linear_differentiate_5d
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
    real(real32), optional, intent(in) :: scale
    
    initialise%name = "linear"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32 !0.05_real32
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
    real(real32), dimension(:), intent(in) :: val
    real(real32), dimension(size(val,dim=1)) :: output

    output = this%scale * val
  end function linear_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_activate_2d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2)) :: output

    output = this%scale * val
  end function linear_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_activate_3d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = this%scale * val
  end function linear_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_activate_4d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = this%scale * val
  end function linear_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_activate_5d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = this%scale * val
  end function linear_activate_5d
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
    real(real32), dimension(:), intent(in) :: val
    real(real32), dimension(size(val,dim=1)) :: output

    output = this%scale * val
  end function linear_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_differentiate_2d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2)) :: output

    output = this%scale * val
  end function linear_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_differentiate_3d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = this%scale * val
  end function linear_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_differentiate_4d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = this%scale * val
  end function linear_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function linear_differentiate_5d(this, val) result(output)
    implicit none
    class(linear_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = this%scale * val
  end function linear_differentiate_5d
!!!#############################################################################

end module activation_linear
