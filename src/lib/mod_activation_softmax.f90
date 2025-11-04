module athena__activation_softmax
  !! Module containing implementation of the softmax activation function
  !!
  !! This module implements the softmax activation function for normalising
  !! outputs into probability distributions
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(-), &
       operator(*), operator(/), exp, sum, maxval
  use athena__diffstruc_extd, only: softmax
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: softmax_setup


  type, extends(activation_type) :: softmax_type
     !! Type for softmax activation function with overloaded procedures
   contains
     procedure, pass(this) :: activate_array => softmax_activate_array
     procedure, pass(this) :: activate_1d => softmax_activate_1d
     procedure, pass(this) :: activate_2d => softmax_activate_2d
     procedure, pass(this) :: activate_3d => softmax_activate_3d
     procedure, pass(this) :: activate_4d => softmax_activate_4d
     procedure, pass(this) :: activate_5d => softmax_activate_5d
     procedure, pass(this) :: differentiate_1d => softmax_differentiate_1d
     procedure, pass(this) :: differentiate_2d => softmax_differentiate_2d
     procedure, pass(this) :: differentiate_3d => softmax_differentiate_3d
     procedure, pass(this) :: differentiate_4d => softmax_differentiate_4d
     procedure, pass(this) :: differentiate_5d => softmax_differentiate_5d
  end type softmax_type

  interface softmax_setup
     procedure initialise
  end interface softmax_setup



contains

!###############################################################################
  pure function initialise(threshold, scale)
    !! Initialise a softmax activation function
    implicit none

    ! Arguments
    type(softmax_type) :: initialise
    !! Softmax activation type
    real(real32), optional, intent(in) :: threshold
    !! Optional threshold value for activation cutoff
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "softmax"

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
  end function initialise
!###############################################################################


!###############################################################################
  function softmax_activate_array(this, val) result(output)
    !! Apply softmax activation to 1D array
    !!
    !! Computes: f = exp(x-max)/sum(exp(x-max))
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    type(array_type), intent(in) :: val
    !! Input values
    type(array_type), pointer :: output
    !! Normalised probability distribution output

    !! compute softmax values
    output => softmax(val, dim=2)

  end function softmax_activate_array
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_activate_1d(this, val) result(output)
    !! Apply softmax activation to 1D array
    !!
    !! Computes: f = exp(x-max)/sum(exp(x-max))
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Normalised probability distribution output

    !! compute softmax values
    output = exp(val - maxval(val, dim=1))

    !! normalise softmax values
    output = output / sum(output, dim=1)

  end function softmax_activate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_activate_2d(this, val) result(output)
    !! Apply softmax activation to 2D array
    !!
    !! Computes: f = exp(x-max)/sum(exp(x-max))
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Normalised probability distribution output

    ! Local variables
    integer :: s
    !! Loop index

    do s = 1, size(val,2)
       !! compute softmax values
       output(:,s) = exp(val(:,s) - maxval(val(:,s)))

       !! normalise softmax values
       output(:,s) = output(:,s) / sum(output(:,s))
    end do

  end function softmax_activate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_activate_3d(this, val) result(output)
    !! Apply softmax activation to 3D array
    !!
    !! Computes: f = exp(x-max)/sum(exp(x-max))
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Normalised probability distribution output

    ! Local variables
    integer :: s
    !! Loop index

    do s=1,size(val,3)
       ! compute softmax values
       output(:,:,s) = exp(val(:,:,s) - maxval(val(:,:,s)))

       ! normalise softmax values
       output(:,:,s) = output(:,:,s) / sum(output(:,:,s))
    end do

  end function softmax_activate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_activate_4d(this, val) result(output)
    !! Apply softmax activation to 4D array
    !!
    !! Computes: f = exp(x-max)/sum(exp(x-max))
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Normalised probability distribution output

    ! Local variables
    integer :: s
    !! Loop index

    do s=1,size(val,4)
       ! compute softmax values
       output(:,:,:,s) = exp(val(:,:,:,s) - maxval(val(:,:,:,s)))

       ! normalise softmax values
       output(:,:,:,s) = output(:,:,:,s) / sum(output(:,:,:,s))
    end do

  end function softmax_activate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_activate_5d(this, val) result(output)
    !! Apply softmax activation to 5D array
    !!
    !! Computes: f = exp(x-max)/sum(exp(x-max))
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Normalised probability distribution output

    ! Local variables
    integer :: s
    !! Loop index

    do s=1,size(val,5)
       ! compute softmax values
       output(:,:,:,:,s) = exp(val(:,:,:,:,s) - maxval(val(:,:,:,:,s)))

       ! normalise softmax values
       output(:,:,:,:,s) = output(:,:,:,:,s) / sum(output(:,:,:,:,s))
    end do

  end function softmax_activate_5d
!###############################################################################


!###############################################################################
  function softmax_differentiate_1d(this, val) result(output)
    !! Differentiate softmax activation for 1D array
    !!
    !! Computes the derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values

    ! compute gradients for softmax layer
    output = this%activate_1d(val)
    output = output * (1._real32 - output)

  end function softmax_differentiate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_differentiate_2d(this, val) result(output)
    !! Differentiate softmax activation for 1D array
    !!
    !! Computes the derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values

    ! compute gradients for softmax layer
    output = this%activate_2d(val)
    output = output * (1._real32 - output)

  end function softmax_differentiate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_differentiate_3d(this, val) result(output)
    !! Differentiate softmax activation for 1D array
    !!
    !! Computes the derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output values

    ! compute gradients for softmax layer
    output = this%activate_3d(val)
    output = output * (1._real32 - output)

  end function softmax_differentiate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_differentiate_4d(this, val) result(output)
    !! Differentiate softmax activation for 1D array
    !!
    !! Computes the derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output values

    ! compute gradients for softmax layer
    output = this%activate_4d(val)
    output = output * (1._real32 - output)

  end function softmax_differentiate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  function softmax_differentiate_5d(this, val) result(output)
    !! Differentiate softmax activation for 1D array
    !!
    !! Computes the derivative: df/dx = f * (1 - f)
    implicit none

    ! Arguments
    class(softmax_type), intent(in) :: this
    !! Softmax activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values

    ! compute gradients for softmax layer
    output = this%activate_5d(val)
    output = output * (1._real32 - output)

  end function softmax_differentiate_5d
!###############################################################################

end module athena__activation_softmax
