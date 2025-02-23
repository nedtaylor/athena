module athena__activation_relu
  !! Module containing implementation of the ReLU activation function
  !!
  !! This module implements the Rectified Linear Unit activation function
  use athena__constants, only: real32
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: relu_setup


  type, extends(activation_type) :: relu_type
     !! Type for ReLU activation function with overloaded procedures
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



contains

!###############################################################################
  pure function initialise(scale)
    !! Initialise a ReLU activation function
    implicit none

    ! Arguments
    type(relu_type) :: initialise
    !! ReLU activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "relu"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32
    end if
    initialise%threshold = 0._real32

  end function initialise
!###############################################################################


!###############################################################################
  pure function relu_activate_1d(this, val) result(output)
    !! Apply ReLU activation to 1D array
    !!
    !! Computes: f = max(0,x)
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Activated output values

    output = max(this%threshold, val) * this%scale
  end function relu_activate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function relu_activate_2d(this, val) result(output)
    !! Apply ReLU activation to 2D array
    !!
    !! Computes: f = max(0,x)
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Activated output values

    output = max(this%threshold, val) * this%scale
  end function relu_activate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function relu_activate_3d(this, val) result(output)
    !! Apply ReLU activation to 3D array
    !!
    !! Computes: f = max(0,x)
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Activated output values

    output = max(this%threshold, val) * this%scale
  end function relu_activate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function relu_activate_4d(this, val) result(output)
    !! Apply ReLU activation to 4D array
    !!
    !! Computes: f = max(0,x)
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Activated output values

    output = max(this%threshold, val) * this%scale
  end function relu_activate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function relu_activate_5d(this, val) result(output)
    !! Apply ReLU activation to 5D array
    !!
    !! Computes: f = max(0,x)
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Activated output values

    output = max(this%threshold, val) * this%scale
  end function relu_activate_5d
!###############################################################################


!###############################################################################
!!! derivative of RELU transfer function
!!! e.g. df/dx (1*x) = 1
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!###############################################################################
  pure function relu_differentiate_1d(this, val) result(output)
    !! Differentiate ReLU activation for 1D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0 otherwise
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function relu_differentiate_2d(this, val) result(output)
    !! Differentiate ReLU activation for 2D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0 otherwise
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function relu_differentiate_3d(this, val) result(output)
    !! Differentiate ReLU activation for 3D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0 otherwise
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output values

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function relu_differentiate_4d(this, val) result(output)
    !! Differentiate ReLU activation for 4D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0 otherwise
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output values

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function relu_differentiate_5d(this, val) result(output)
    !! Differentiate ReLU activation for 5D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0 otherwise
    implicit none

    ! Arguments
    class(relu_type), intent(in) :: this
    !! ReLU activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values

    where(val.ge.this%threshold)
       output = this%scale
    elsewhere
       output = this%threshold
    end where
  end function relu_differentiate_5d
!###############################################################################

end module athena__activation_relu
