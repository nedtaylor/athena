module athena__activation_none
  !! Module containing implementation of no activation function (i.e. linear)
  !!
  !! This module implements the identity function f(x) = x
  use athena__constants, only: real32
  use athena__misc_types, only: activation_type
  implicit none


  private

  public :: none_setup


  type, extends(activation_type) :: none_type
   contains
     procedure, pass(this) :: activate_1d => none_activate_1d
     procedure, pass(this) :: activate_2d => none_activate_2d
     procedure, pass(this) :: activate_3d => none_activate_3d
     procedure, pass(this) :: activate_4d => none_activate_4d
     procedure, pass(this) :: activate_5d => none_activate_5d
     procedure, pass(this) :: differentiate_1d => none_differentiate_1d
     procedure, pass(this) :: differentiate_2d => none_differentiate_2d
     procedure, pass(this) :: differentiate_3d => none_differentiate_3d
     procedure, pass(this) :: differentiate_4d => none_differentiate_4d
     procedure, pass(this) :: differentiate_5d => none_differentiate_5d
  end type none_type
  
  interface none_setup
     procedure initialise
  end interface none_setup

  
  
contains
  
!###############################################################################
  pure function initialise(scale)
    !! Initialise a none (no-op) activation function
    implicit none

    ! Arguments
    type(none_type) :: initialise
    !! None activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output

    initialise%name = "none"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32
    end if
  end function initialise
!###############################################################################
  
  
!###############################################################################
  pure function none_activate_1d(this, val) result(output)
    !! Apply identity activation to 1D array
    !!
    !! Simply returns scaled input: f = scale * x
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Scaled output values

    output = val * this%scale
  end function none_activate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function none_activate_2d(this, val) result(output)
    !! Apply identity activation to 2D array
    !!
    !! Simply returns scaled input: f = scale * x
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Scaled output values

    output = val * this%scale
  end function none_activate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function none_activate_3d(this, val) result(output)
    !! Apply identity activation to 3D array
    !!
    !! Simply returns scaled input: f = scale * x
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Scaled output values

    output = val * this%scale
  end function none_activate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function none_activate_4d(this, val) result(output)
    !! Apply identity activation to 4D array
    !!
    !! Simply returns scaled input: f = scale * x
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Scaled output values

    output = val * this%scale
  end function none_activate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function none_activate_5d(this, val) result(output)
    !! Apply identity activation to 5D array
    !!
    !! Simply returns scaled input: f = scale * x
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Scaled output values

    output = val * this%scale
  end function none_activate_5d
!###############################################################################


!###############################################################################
  pure function none_differentiate_1d(this, val) result(output)
    !! Differentiate identity activation for 1D array
    !!
    !! Returns constant scale factor as derivative: df/dx = scale
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values (constant)

    output = this%scale
  end function none_differentiate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function none_differentiate_2d(this, val) result(output)
    !! Differentiate identity activation for 2D array
    !!
    !! Returns constant scale factor as derivative: df/dx = scale
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values (constant)

    output = this%scale
  end function none_differentiate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function none_differentiate_3d(this, val) result(output)
    !! Differentiate identity activation for 3D array
    !!
    !! Returns constant scale factor as derivative: df/dx = scale
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output values (constant)

    output = this%scale
  end function none_differentiate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function none_differentiate_4d(this, val) result(output)
    !! Differentiate identity activation for 4D array
    !!
    !! Returns constant scale factor as derivative: df/dx = scale
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output values (constant)

    output = this%scale
  end function none_differentiate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function none_differentiate_5d(this, val) result(output)
    !! Differentiate identity activation for 5D array
    !!
    !! Returns constant scale factor as derivative: df/dx = scale
    implicit none

    ! Arguments
    class(none_type), intent(in) :: this
    !! None activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values (constant)

    output = this%scale
  end function none_differentiate_5d
!###############################################################################

end module athena__activation_none
