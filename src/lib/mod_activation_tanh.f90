module athena__activation_tanh
  !! Module containing implementation of the tanh activation function
  !! 
  !! This module implements the hyperbolic tangent activation function
  use athena__constants, only: real32
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: tanh_setup


  type, extends(activation_type) :: tanh_type
     !! Type for tanh activation function with overloaded procedures
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



contains
  
!###############################################################################
  pure function initialise(threshold, scale)
    !! Initialize a tanh activation function
    implicit none

    ! Arguments
    type(tanh_type) :: initialise
    !! tanh activation type
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

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
!###############################################################################
  
  
!###############################################################################
  pure function tanh_activate_1d(this, val) result(output)
    !! Apply tanh activation to 1D array
    !!
    !! Applies the hyperbolic tangent function element-wise to input array:
    !! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    implicit none

    ! Arguments
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Activated output values

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function tanh_activate_2d(this, val) result(output)
    !! Apply tanh activation to 2D array
    !!
    !! Applies the hyperbolic tangent function element-wise to input array:
    !! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    implicit none

    ! Arguments
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Activated output values

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function tanh_activate_3d(this, val) result(output)
    !! Apply tanh activation to 3D array
    !!
    !! Applies the hyperbolic tangent function element-wise to input array:
    !! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    implicit none
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Activated output values

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function tanh_activate_4d(this, val) result(output)
    !! Apply tanh activation to 4D array
    !!
    !! Applies the hyperbolic tangent function element-wise to input array:
    !! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    implicit none
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Activated output values

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function tanh_activate_5d(this, val) result(output)
    !! Apply tanh activation to 5D array
    !!
    !! Applies the hyperbolic tangent function element-wise to input array:
    !! f = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    implicit none
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Activated output values

    !! fix rounding errors of division of small numbers
    !! alt. could add an epsilon
    where(abs(val).gt.this%threshold)
       output = sign(1._real32, val) * this%scale
    elsewhere
       output = this%scale * (exp(val) - exp(-val))/(exp(val) + exp(-val))
    end where
  end function tanh_activate_5d
!###############################################################################


!###############################################################################
  pure function tanh_differentiate_1d(this, val) result(output)
    !! Differentiate tanh activation for 1D array
    !!
    !! Computes the derivative: df/dx = 1 - f^2
    implicit none
    
    ! Arguments
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values

    output = this%scale * &
         (1._real32 - (this%activate_1d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function tanh_differentiate_2d(this, val) result(output)
    !! Differentiate tanh activation for 2D array
    !!
    !! Computes the derivative: df/dx = 1 - f^2
    implicit none
    
    ! Arguments
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values

    output = this%scale * &
         (1._real32 - (this%activate_2d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function tanh_differentiate_3d(this, val) result(output)
    !! Differentiate tanh activation for 3D array
    !!
    !! Computes the derivative: df/dx = 1 - f^2
    implicit none
    
    ! Arguments
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output values

    output = this%scale * &
         (1._real32 - (this%activate_3d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function tanh_differentiate_4d(this, val) result(output)
    !! Differentiate tanh activation for 4D array
    !!
    !! Computes the derivative: df/dx = 1 - f^2
    implicit none
    
    ! Arguments
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output values

    output = this%scale * &
         (1._real32 - (this%activate_4d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function tanh_differentiate_5d(this, val) result(output)
    !! Differentiate tanh activation for 5D array
    !!
    !! Computes the derivative: df/dx = 1 - f^2
    implicit none
    
    ! Arguments
    class(tanh_type), intent(in) :: this
    !! Tanh activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values

    output = this%scale * &
         (1._real32 - (this%activate_5d(val)/this%scale) ** 2._real32)

  end function tanh_differentiate_5d
!###############################################################################

end module athena__activation_tanh
