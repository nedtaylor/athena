!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the tanh activation function
!!!#############################################################################
module activation_tanh
  use constants, only: real32
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: tanh_type
   contains
     procedure, pass(this) :: activate_1d => tanh_activate_1d
     procedure, pass(this) :: activate_2d => tanh_activate_2d
     procedure, pass(this) :: activate_3d => tanh_activate_3d
     procedure, pass(this) :: activate_4d => tanh_activate_4d
     procedure, pass(this) :: activate_5d => tanh_activate_5d
     procedure, pass(this) :: differentiate_1d => tanh_differentiate_1d
     procedure, pass(this) :: differentiate_2d => tanh_differentiate_2d
     procedure, pass(this) :: differentiate_3d => tanh_differentiate_3d
     procedure, pass(this) :: differentiate_4d => tanh_differentiate_4d
     procedure, pass(this) :: differentiate_5d => tanh_differentiate_5d
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
  pure function initialise(threshold, scale)
    implicit none
    type(tanh_type) :: initialise
    real(real32), optional, intent(in) :: threshold
    real(real32), optional, intent(in) :: scale

    initialise%name = "tanh"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32
    end if

    !initialise%name = "tanh"
    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = min(huge(1._real32),32._real32)
    end if

  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! tanh transfer function
!!! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
!!!#############################################################################
  pure function tanh_activate_1d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:), intent(in) :: val
    real(real32), dimension(size(val,dim=1)) :: output

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function tanh_activate_2d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2)) :: output

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function tanh_activate_3d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function tanh_activate_4d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function tanh_activate_5d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_5d
!!!#############################################################################


!!!#############################################################################
!!! derivative of tanh function
!!! df/dx = 1 - f^2
!!!#############################################################################
  pure function tanh_differentiate_1d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:), intent(in) :: val
    real(real32), dimension(size(val,dim=1)) :: output

    output = this%scale * &
         (1._real32 - (this%activate_1d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function tanh_differentiate_2d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2)) :: output

    output = this%scale * &
         (1._real32 - (this%activate_2d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function tanh_differentiate_3d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = this%scale * &
         (1._real32 - (this%activate_3d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function tanh_differentiate_4d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = this%scale * &
         (1._real32 - (this%activate_4d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function tanh_differentiate_5d(this, val) result(output)
    implicit none
    class(tanh_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = this%scale * &
         (1._real32 - (this%activate_5d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_5d
!!!#############################################################################

end module activation_tanh
