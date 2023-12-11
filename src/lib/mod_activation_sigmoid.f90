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
     procedure, pass(this) :: activate_1d => sigmoid_activate_1d
     procedure, pass(this) :: activate_2d => sigmoid_activate_2d
     procedure, pass(this) :: activate_3d => sigmoid_activate_3d
     procedure, pass(this) :: activate_4d => sigmoid_activate_4d
     procedure, pass(this) :: activate_5d => sigmoid_activate_5d
     procedure, pass(this) :: differentiate_1d => sigmoid_differentiate_1d
     procedure, pass(this) :: differentiate_2d => sigmoid_differentiate_2d
     procedure, pass(this) :: differentiate_3d => sigmoid_differentiate_3d
     procedure, pass(this) :: differentiate_4d => sigmoid_differentiate_4d
     procedure, pass(this) :: differentiate_5d => sigmoid_differentiate_5d
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
  pure function initialise(threshold, scale)
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
  pure function sigmoid_activate_1d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    where(val.lt.this%threshold)
       output = 0._real12
    elsewhere
       output = this%scale /(1._real12 + exp(-val))
    end where
  end function sigmoid_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function sigmoid_activate_2d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output

    where(val.lt.this%threshold)
       output = 0._real12
    elsewhere
       output = this%scale /(1._real12 + exp(-val))
    end where
  end function sigmoid_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function sigmoid_activate_3d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    where(val.lt.this%threshold)
       output = 0._real12
    elsewhere
       output = this%scale /(1._real12 + exp(-val))
    end where
  end function sigmoid_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function sigmoid_activate_4d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    where(val.lt.this%threshold)
       output = 0._real12
    elsewhere
       output = this%scale /(1._real12 + exp(-val))
    end where
  end function sigmoid_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function sigmoid_activate_5d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    where(val.lt.this%threshold)
       output = 0._real12
    elsewhere
       output = this%scale /(1._real12 + exp(-val))
    end where
  end function sigmoid_activate_5d
!!!#############################################################################


!!!#############################################################################
!!! derivative of sigmoid function
!!! df/dx = f * (1 - f)
!!!#############################################################################
  pure function sigmoid_differentiate_1d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = this%activate_1d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function sigmoid_differentiate_2d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output
    
    output = this%activate_2d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function sigmoid_differentiate_3d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output
    
    output = this%activate_3d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function sigmoid_differentiate_4d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    
    output = this%activate_4d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function sigmoid_differentiate_5d(this, val) result(output)
    implicit none
    class(sigmoid_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    
    output = this%activate_5d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_5d
!!!#############################################################################

end module activation_sigmoid
