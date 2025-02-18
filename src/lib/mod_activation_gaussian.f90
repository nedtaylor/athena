module athena__activation_gaussian
  !! Module containing implementation of the Gaussian activation function
  !!
  !! This module implements the Gaussian (bell curve) activation function
  use athena__constants, only: real32, pi
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: gaussian_setup


  type, extends(activation_type) :: gaussian_type
     !! Type for Gaussian activation function with overloaded procedures
     real(real32) :: sigma
     !! Standard deviation parameter for Gaussian function
   contains
     procedure, pass(this) :: activate_1d => gaussian_activate_1d
     procedure, pass(this) :: activate_2d => gaussian_activate_2d
     procedure, pass(this) :: activate_3d => gaussian_activate_3d
     procedure, pass(this) :: activate_4d => gaussian_activate_4d
     procedure, pass(this) :: activate_5d => gaussian_activate_5d
     procedure, pass(this) :: differentiate_1d => gaussian_differentiate_1d
     procedure, pass(this) :: differentiate_2d => gaussian_differentiate_2d
     procedure, pass(this) :: differentiate_3d => gaussian_differentiate_3d
     procedure, pass(this) :: differentiate_4d => gaussian_differentiate_4d
     procedure, pass(this) :: differentiate_5d => gaussian_differentiate_5d
  end type gaussian_type
  
  interface gaussian_setup
     procedure initialise
  end interface gaussian_setup

  
  
contains
  
!###############################################################################
  pure function initialise(threshold, scale, sigma)
    !! Initialise a Gaussian activation function
    implicit none

    ! Arguments
    type(gaussian_type) :: initialise
    !! Gaussian activation type
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    real(real32), optional, intent(in) :: sigma
    !! Optional standard deviation parameter

    initialise%name = "gaussian"
    
    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32
    end if

    if(present(sigma))then
       initialise%sigma = sigma
    else
       initialise%sigma = 1.5_real32
    end if

    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = min(huge(1._real32),16._real32) * &
            initialise%sigma
    end if

  end function initialise
!###############################################################################


!###############################################################################
  pure function gaussian_activate_1d(this, val) result(output)
    !! Apply Gaussian activation to 1D array
    !!
    !! Applies the Gaussian function element-wise to input array:
    !! f = exp(-x^2/(2σ^2))/(σ√(2π))
    implicit none

    ! Arguments
    class(gaussian_type), intent(in) :: this
    !! Gaussian activation type containing sigma parameter
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Gaussian activated output values

    where(abs(val).le.this%threshold)
       output = this%scale * 1._real32/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real32 * (val/this%sigma)**2._real32)
    elsewhere
       output = 0._real32
    end where
  end function gaussian_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_activate_2d(this, val) result(output)
    !! Apply Gaussian activation to 2D array
    !!
    !! Applies the Gaussian function element-wise to input array:
    !! f = exp(-x^2/(2σ^2))/(σ√(2π))
    implicit none

    ! Arguments
    class(gaussian_type), intent(in) :: this
    !! Gaussian activation type containing sigma parameter
    real(real32), dimension(:,:), intent(in) :: val 
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Gaussian activated output values

    where(abs(val).le.this%threshold)
       output = this%scale * 1._real32/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real32 * (val/this%sigma)**2._real32)
    elsewhere
       output = 0._real32
    end where
  end function gaussian_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_activate_3d(this, val) result(output)
    !! Apply Gaussian activation to 3D array
    !!
    !! Applies the Gaussian function element-wise to input array:
    !! f = exp(-x^2/(2σ^2))/(σ√(2π))
    implicit none

    ! Arguments
    class(gaussian_type), intent(in) :: this
    !! Gaussian activation type containing sigma parameter
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Gaussian activated output values

    where(abs(val).le.this%threshold)
       output = this%scale * 1._real32/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real32 * (val/this%sigma)**2._real32)
    elsewhere
       output = 0._real32
    end where
  end function gaussian_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_activate_4d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    where(abs(val).le.this%threshold)
       output = this%scale * 1._real32/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real32 * (val/this%sigma)**2._real32)
    elsewhere
       output = 0._real32
    end where
  end function gaussian_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_activate_5d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    where(abs(val).le.this%threshold)
       output = this%scale * 1._real32/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real32 * (val/this%sigma)**2._real32)
    elsewhere
       output = 0._real32
    end where
  end function gaussian_activate_5d
!###############################################################################


!###############################################################################
  pure function gaussian_differentiate_1d(this, val) result(output)
    !! Differentiate Gaussian activation for 1D array
    !!
    !! Computes the derivative: df/dx = -x/σ^2 * f(x)
    !! where f(x) is the Gaussian activation
    implicit none

    ! Arguments
    class(gaussian_type), intent(in) :: this
    !! Gaussian activation type containing sigma parameter
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values

    output = -val/this%sigma**2._real32 * this%activate_1d(val)
  end function gaussian_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_differentiate_2d(this, val) result(output)
    !! Differentiate Gaussian activation for 2D array
    !!
    !! Computes the derivative: df/dx = -x/σ^2 * f(x)
    !! where f(x) is the Gaussian activation
    implicit none

    ! Arguments
    class(gaussian_type), intent(in) :: this
    !! Gaussian activation type containing sigma parameter
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values

    output = -val/this%sigma**2._real32 * this%activate_2d(val)
  end function gaussian_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_differentiate_3d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = -val/this%sigma**2._real32 * this%activate_3d(val)
  end function gaussian_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_differentiate_4d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = -val/this%sigma**2._real32 * this%activate_4d(val)
  end function gaussian_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_differentiate_5d(this, val) result(output)
    !! Differentiate Gaussian activation for 5D array
    !!
    !! Computes the derivative: df/dx = -x/σ^2 * f(x)
    !! where f(x) is the Gaussian activation
    implicit none

    ! Arguments
    class(gaussian_type), intent(in) :: this
    !! Gaussian activation type containing sigma parameter
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
          size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values

    output = -val/this%sigma**2._real32 * this%activate_5d(val)
  end function gaussian_differentiate_5d
!###############################################################################

end module athena__activation_gaussian
