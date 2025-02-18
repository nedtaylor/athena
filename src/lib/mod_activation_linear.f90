module athena__activation_linear
  !! Module containing implementation of the linear activation function
  !!
  !! This module implements a scaled linear function f(x) = scale * x
  use athena__constants, only: real32
  use athena__misc_types, only: activation_type
  implicit none

  type, extends(activation_type) :: linear_type
   contains
     procedure, pass(this) :: activate_1d => linear_activate_1d
     procedure, pass(this) :: activate_2d => linear_activate_2d
     procedure, pass(this) :: activate_3d => linear_activate_3d
     procedure, pass(this) :: activate_4d => linear_activate_4d
     procedure, pass(this) :: activate_5d => linear_activate_5d
     procedure, pass(this) :: differentiate_1d => linear_differentiate_1d
     procedure, pass(this) :: differentiate_2d => linear_differentiate_2d
     procedure, pass(this) :: differentiate_3d => linear_differentiate_3d
     procedure, pass(this) :: differentiate_4d => linear_differentiate_4d
     procedure, pass(this) :: differentiate_5d => linear_differentiate_5d
  end type linear_type

  interface linear_setup
     procedure initialise
  end interface linear_setup

  
  private
  
  public :: linear_setup

  
contains
  
!###############################################################################
  pure function initialise(scale)
    !! Initialise a linear activation function
    implicit none

    ! Arguments
    type(linear_type) :: initialise
    !! Linear activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    
    initialise%name = "linear"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32 !0.05_real32
    end if

  end function initialise
!###############################################################################

       
!###############################################################################
  pure function linear_activate_1d(this, val) result(output)
    !! Apply linear activation to 1D array
    !!
    !! Computes: f = scale * x
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Scaled output values

    output = this%scale * val
  end function linear_activate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function linear_activate_2d(this, val) result(output)
    !! Apply linear activation to 2D array
    !!
    !! Computes: f = scale * x
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Scaled output values

    output = this%scale * val
  end function linear_activate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function linear_activate_3d(this, val) result(output)
    !! Apply linear activation to 3D array
    !!
    !! Computes: f = scale * x
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Scaled output values

    output = this%scale * val
  end function linear_activate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function linear_activate_4d(this, val) result(output)
    !! Apply linear activation to 4D array
    !!
    !! Computes: f = scale * x
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Scaled output values

    output = this%scale * val
  end function linear_activate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function linear_activate_5d(this, val) result(output)
    !! Apply linear activation to 5D array
    !!
    !! Computes: f = scale * x
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Scaled output values

    output = this%scale * val
  end function linear_activate_5d
!###############################################################################


!###############################################################################
  pure function linear_differentiate_1d(this, val) result(output)
    !! Differentiate linear activation for 1D array
    !!
    !! Computes constant derivative: df/dx = scale
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output (constant)

    output = this%scale
  end function linear_differentiate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function linear_differentiate_2d(this, val) result(output)
    !! Differentiate linear activation for 2D array
    !!
    !! Computes constant derivative: df/dx = scale
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output (constant)

    output = this%scale
  end function linear_differentiate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function linear_differentiate_3d(this, val) result(output)
    !! Differentiate linear activation for 3D array
    !!
    !! Computes constant derivative: df/dx = scale
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output (constant)

    output = this%scale
  end function linear_differentiate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function linear_differentiate_4d(this, val) result(output)
    !! Differentiate linear activation for 4D array
    !!
    !! Computes constant derivative: df/dx = scale
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output (constant)

    output = this%scale
  end function linear_differentiate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function linear_differentiate_5d(this, val) result(output)
    !! Differentiate linear activation for 5D array
    !!
    !! Computes constant derivative: df/dx = scale
    implicit none

    ! Arguments
    class(linear_type), intent(in) :: this
    !! Linear activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output (constant)

    output = this%scale
  end function linear_differentiate_5d
!###############################################################################

end module athena__activation_linear
