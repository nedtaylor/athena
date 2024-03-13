!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the leaky ReLU activation function
!!!#############################################################################
module activation_leaky_relu
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: leaky_relu_type
   contains
     procedure, pass(this) :: activate_1d => leaky_relu_activate_1d
     procedure, pass(this) :: activate_2d => leaky_relu_activate_2d
     procedure, pass(this) :: activate_3d => leaky_relu_activate_3d
     procedure, pass(this) :: activate_4d => leaky_relu_activate_4d
     procedure, pass(this) :: activate_5d => leaky_relu_activate_5d
     procedure, pass(this) :: differentiate_1d => leaky_relu_differentiate_1d
     procedure, pass(this) :: differentiate_2d => leaky_relu_differentiate_2d
     procedure, pass(this) :: differentiate_3d => leaky_relu_differentiate_3d
     procedure, pass(this) :: differentiate_4d => leaky_relu_differentiate_4d
     procedure, pass(this) :: differentiate_5d => leaky_relu_differentiate_5d
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
  pure function initialise(scale)
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
  pure function leaky_relu_activate_1d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = max(0.01_real12*val, val) * this%scale
  end function leaky_relu_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function leaky_relu_activate_2d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output

    output = max(0.01_real12*val, val) * this%scale
  end function leaky_relu_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function leaky_relu_activate_3d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = max(0.01_real12*val, val) * this%scale
  end function leaky_relu_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function leaky_relu_activate_4d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = max(0.01_real12*val, val) * this%scale
  end function leaky_relu_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function leaky_relu_activate_5d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = max(0.01_real12*val, val) * this%scale
  end function leaky_relu_activate_5d
!!!#############################################################################


!!!#############################################################################
!!! derivative of leaky ReLU transfer function
!!! e.g. df/dx (1.0*x) = 1.0
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  pure function leaky_relu_differentiate_1d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    where(val.ge.0._real12)
       output = this%scale
    elsewhere
       output = 0.01_real12
    end where
  end function leaky_relu_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function leaky_relu_differentiate_2d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output

    where(val.ge.0._real12)
       output = this%scale
    elsewhere
       output = 0.01_real12
    end where
  end function leaky_relu_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function leaky_relu_differentiate_3d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    where(val.ge.0._real12)
       output = this%scale
    elsewhere
       output = 0.01_real12
    end where
  end function leaky_relu_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function leaky_relu_differentiate_4d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    where(val.ge.0._real12)
       output = this%scale
    elsewhere
       output = 0.01_real12
    end where
  end function leaky_relu_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function leaky_relu_differentiate_5d(this, val) result(output)
    implicit none
    class(leaky_relu_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    where(val.ge.0._real12)
       output = this%scale
    elsewhere
       output = 0.01_real12
    end where
  end function leaky_relu_differentiate_5d
!!!#############################################################################

end module activation_leaky_relu
