!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the ReLU activation function
!!!#############################################################################
module activation_relu
  use constants, only: real32
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: relu_type
   contains
     procedure, pass(this) :: activate_1d => relu_activate_1d
     procedure, pass(this) :: activate_2d => relu_activate_2d
     procedure, pass(this) :: activate_3d => relu_activate_3d
     procedure, pass(this) :: activate_4d => relu_activate_4d
     procedure, pass(this) :: activate_5d => relu_activate_5d
     procedure, pass(this) :: differentiate_1d => relu_differentiate_1d
     procedure, pass(this) :: differentiate_2d => relu_differentiate_2d
     procedure, pass(this) :: differentiate_3d => relu_differentiate_3d
     procedure, pass(this) :: differentiate_4d => relu_differentiate_4d
     procedure, pass(this) :: differentiate_5d => relu_differentiate_5d
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
  pure function initialise(scale)
    implicit none
    type(relu_type) :: initialise
    real(real32), optional, intent(in) :: scale

    initialise%name = "relu"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32
    end if
    initialise%threshold = 0._real32

  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! RELU transfer function
!!! f = max(0, x)
!!!#############################################################################
  pure function relu_activate_1d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:), intent(in) :: val
    real(real32), dimension(size(val,dim=1)) :: output

    output = max(this%threshold, val) * this%scale
  end function relu_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_activate_2d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2)) :: output

    output = max(this%threshold, val) * this%scale
  end function relu_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_activate_3d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = max(this%threshold, val) * this%scale
  end function relu_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_activate_4d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = max(this%threshold, val) * this%scale
  end function relu_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_activate_5d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = max(this%threshold, val) * this%scale
  end function relu_activate_5d
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
    real(real32), dimension(:), intent(in) :: val
    real(real32), dimension(size(val,dim=1)) :: output

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_differentiate_2d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2)) :: output

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_differentiate_3d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_differentiate_4d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function relu_differentiate_5d(this, val) result(output)
    implicit none
    class(relu_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_5d
!!!#############################################################################

end module activation_relu
