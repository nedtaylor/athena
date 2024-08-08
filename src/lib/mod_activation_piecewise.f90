!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the piecewise activation function
!!!#############################################################################
module activation_piecewise
  use constants, only: real32
  use custom_types, only: activation_type
  implicit none

  type, extends(activation_type) :: piecewise_type
     real(real32) :: intercept, min, max
   contains
     procedure, pass(this) :: activate_1d => piecewise_activate_1d
     procedure, pass(this) :: activate_2d => piecewise_activate_2d
     procedure, pass(this) :: activate_3d => piecewise_activate_3d
     procedure, pass(this) :: activate_4d => piecewise_activate_4d
     procedure, pass(this) :: activate_5d => piecewise_activate_5d
     procedure, pass(this) :: differentiate_1d => piecewise_differentiate_1d
     procedure, pass(this) :: differentiate_2d => piecewise_differentiate_2d
     procedure, pass(this) :: differentiate_3d => piecewise_differentiate_3d
     procedure, pass(this) :: differentiate_4d => piecewise_differentiate_4d
     procedure, pass(this) :: differentiate_5d => piecewise_differentiate_5d
  end type piecewise_type

  interface piecewise_setup
     procedure initialise
  end interface piecewise_setup

  
  private
  
  public :: piecewise_setup

  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  pure function initialise(scale, intercept)
    implicit none
    type(piecewise_type) :: initialise
    real(real32), optional, intent(in) :: scale, intercept
    
    initialise%name = "piecewise"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real32 !0.05_real32
    end if
    if(present(intercept))then
       initialise%intercept = intercept
    else
       initialise%intercept = 1._real32 !0.05_real32
    end if

    initialise%max = initialise%intercept/initialise%scale
    initialise%min = -initialise%max

  end function initialise
!!!#############################################################################

       
!!!#############################################################################
!!! Piecewise transfer function
!!! f = gradient * x
!!!#############################################################################
  pure function piecewise_activate_1d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:), intent(in) :: val
    real(real32), dimension(size(val,dim=1)) :: output

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_activate_2d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2)) :: output

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_activate_3d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_activate_4d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_activate_5d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_5d
!!!#############################################################################


!!!#############################################################################
!!! derivative of piecewise transfer function
!!! e.g. df/dx (gradient * x) = gradient
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!!!#############################################################################
  pure function piecewise_differentiate_1d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:), intent(in) :: val
    real(real32), dimension(size(val,dim=1)) :: output

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_differentiate_2d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2)) :: output

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_differentiate_3d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:,:,:), intent(in) :: val
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_differentiate_4d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_differentiate_5d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_5d
!!!#############################################################################

end module activation_piecewise
