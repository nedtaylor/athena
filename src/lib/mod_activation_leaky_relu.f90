module athena__activation_leaky_relu
  !! Module containing implementation of the leaky ReLU activation function
  !!
  !! This module implements the Leaky Rectified Linear Unit function:
  !! f(x) = x if x > 0, 0.01x otherwise
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(-), &
       operator(*), operator(/), exp, max
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: leaky_relu_setup


  type, extends(activation_type) :: leaky_relu_type
   contains
     procedure, pass(this) :: activate_array => leaky_relu_activate_array
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



contains

!###############################################################################
  pure function initialise(scale)
    !! Initialise a leaky ReLU activation function
    implicit none

    ! Arguments
    type(leaky_relu_type) :: initialise
    !! Leaky ReLU activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "leaky_relu"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32
    end if
  end function initialise
!###############################################################################


!###############################################################################
  function leaky_relu_activate_array(this, val) result(output)
    !! Apply leaky ReLU activation to 1D array
    !!
    !! Computes: f = max(0.01x, x)
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Activated output values

    ! allocate(output)
    output => max(val * 0.01_real32, val) * this%scale
  end function leaky_relu_activate_array
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_activate_1d(this, val) result(output)
    !! Apply leaky ReLU activation to 1D array
    !!
    !! Computes: f = max(0.01x, x)
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Activated output values

    output = max(0.01_real32*val, val) * this%scale
  end function leaky_relu_activate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_activate_2d(this, val) result(output)
    !! Apply leaky ReLU activation to 2D array
    !!
    !! Computes: f = max(0.01x, x)
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Activated output values

    output = max(0.01_real32*val, val) * this%scale
  end function leaky_relu_activate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_activate_3d(this, val) result(output)
    !! Apply leaky ReLU activation to 3D array
    !!
    !! Computes: f = max(0.01x, x)
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Activated output values

    output = max(0.01_real32*val, val) * this%scale
  end function leaky_relu_activate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_activate_4d(this, val) result(output)
    !! Apply leaky ReLU activation to 4D array
    !!
    !! Computes: f = max(0.01x, x)
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Activated output values

    output = max(0.01_real32*val, val) * this%scale
  end function leaky_relu_activate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_activate_5d(this, val) result(output)
    !! Apply leaky ReLU activation to 5D array
    !!
    !! Computes: f = max(0.01x, x)
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Activated output values

    output = max(0.01_real32*val, val) * this%scale
  end function leaky_relu_activate_5d
!###############################################################################


!###############################################################################
  function leaky_relu_differentiate_1d(this, val) result(output)
    !! Differentiate leaky ReLU activation for 1D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0.01 otherwise
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values

    where(val.ge.0._real32)
       output = this%scale
    elsewhere
       output = 0.01_real32
    end where
  end function leaky_relu_differentiate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_differentiate_2d(this, val) result(output)
    !! Differentiate leaky ReLU activation for 2D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0.01 otherwise
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values

    where(val.ge.0._real32)
       output = this%scale
    elsewhere
       output = 0.01_real32
    end where
  end function leaky_relu_differentiate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_differentiate_3d(this, val) result(output)
    !! Differentiate leaky ReLU activation for 3D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0.01 otherwise
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output values

    where(val.ge.0._real32)
       output = this%scale
    elsewhere
       output = 0.01_real32
    end where
  end function leaky_relu_differentiate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_differentiate_4d(this, val) result(output)
    !! Differentiate leaky ReLU activation for 4D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0.01 otherwise
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output values

    where(val.ge.0._real32)
       output = this%scale
    elsewhere
       output = 0.01_real32
    end where
  end function leaky_relu_differentiate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function leaky_relu_differentiate_5d(this, val) result(output)
    !! Differentiate leaky ReLU activation for 5D array
    !!
    !! Computes derivative: df/dx = 1 if x > 0, 0.01 otherwise
    implicit none

    ! Arguments
    class(leaky_relu_type), intent(in) :: this
    !! Leaky ReLU activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values

    where(val.ge.0._real32)
       output = this%scale
    elsewhere
       output = 0.01_real32
    end where
  end function leaky_relu_differentiate_5d
!###############################################################################

end module athena__activation_leaky_relu
