module athena__activation_sigmoid
  !! Module containing implementation of the sigmoid activation function
  !!
  !! This module implements the logistic sigmoid function for normalizing
  !! outputs between 0 and 1
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(-), &
       operator(*), operator(/), exp, merge, operator(.gt.), sigmoid
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: sigmoid_setup


  type, extends(activation_type) :: sigmoid_type
     !! Type for sigmoid activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate_array => sigmoid_activate_array
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



contains

!###############################################################################
  pure function initialise(threshold, scale)
    !! Initialise a sigmoid activation function
    implicit none

    ! Arguments
    type(sigmoid_type) :: initialise
    !! Sigmoid activation type
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "sigmoid"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32
    end if

    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = -min(huge(1._real32),32._real32)
    end if
    !initialise%scale = 1._real32
  end function initialise
!###############################################################################


!###############################################################################
  function sigmoid_activate_array(this, val) result(output)
    !! Apply sigmoid activation to 1D array
    !!
    !! Computes: f = 1/(1+exp(-x))
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values in range [0,1]

    output => this%scale * sigmoid(val)
  end function sigmoid_activate_array
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_activate_1d(this, val) result(output)
    !! Apply sigmoid activation to 1D array
    !!
    !! Computes: f = 1/(1+exp(-x))
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Activated output values in range [0,1]

    where(val.lt.this%threshold)
       output = 0._real32
    elsewhere
       output = this%scale /(1._real32 + exp(-val))
    end where
  end function sigmoid_activate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_activate_2d(this, val) result(output)
    !! Apply sigmoid activation to 2D array
    !!
    !! Computes: f = 1/(1+exp(-x))
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Activated output values in range [0,1]

    where(val.lt.this%threshold)
       output = 0._real32
    elsewhere
       output = this%scale /(1._real32 + exp(-val))
    end where
  end function sigmoid_activate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_activate_3d(this, val) result(output)
    !! Apply sigmoid activation to 3D array
    !!
    !! Computes: f = 1/(1+exp(-x))
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Activated output values in range [0,1]

    where(val.lt.this%threshold)
       output = 0._real32
    elsewhere
       output = this%scale /(1._real32 + exp(-val))
    end where
  end function sigmoid_activate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_activate_4d(this, val) result(output)
    !! Apply sigmoid activation to 4D array
    !!
    !! Computes: f = 1/(1+exp(-x))
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Activated output values in range [0,1]

    where(val.lt.this%threshold)
       output = 0._real32
    elsewhere
       output = this%scale /(1._real32 + exp(-val))
    end where
  end function sigmoid_activate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_activate_5d(this, val) result(output)
    !! Apply sigmoid activation to 5D array
    !!
    !! Computes: f = 1/(1+exp(-x))
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Activated output values in range [0,1]

    where(val.lt.this%threshold)
       output = 0._real32
    elsewhere
       output = this%scale /(1._real32 + exp(-val))
    end where
  end function sigmoid_activate_5d
!###############################################################################


!###############################################################################
  function sigmoid_differentiate_1d(this, val) result(output)
    !! Differentiate sigmoid activation for 1D array
    !!
    !! Computes derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values

    output = this%activate_1d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_differentiate_2d(this, val) result(output)
    !! Differentiate sigmoid activation for 2D array
    !!
    !! Computes derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values

    output = this%activate_2d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_differentiate_3d(this, val) result(output)
    !! Differentiate sigmoid activation for 3D array
    !!
    !! Computes derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output values

    output = this%activate_3d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_differentiate_4d(this, val) result(output)
    !! Differentiate sigmoid activation for 4D array
    !!
    !! Computes derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output values

    output = this%activate_4d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function sigmoid_differentiate_5d(this, val) result(output)
    !! Differentiate sigmoid activation for 5D array
    !!
    !! Computes derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(sigmoid_type), intent(in) :: this
    !! Sigmoid activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values

    output = this%activate_5d(val)
    output = this%scale * output * (this%scale - output)
  end function sigmoid_differentiate_5d
!###############################################################################

end module athena__activation_sigmoid
