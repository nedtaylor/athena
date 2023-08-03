!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_piecewise
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none

  type, extends(activation_type) :: piecewise_type
     real(real12) :: intercept, min, max
   contains
     procedure, pass(this) :: activate_1d => piecewise_activate_1d
     procedure, pass(this) :: activate_3d => piecewise_activate_3d
     procedure, pass(this) :: differentiate_1d => piecewise_differentiate_1d
     procedure, pass(this) :: differentiate_3d => piecewise_differentiate_3d
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
    real(real12), optional, intent(in) :: scale, intercept
    
    initialise%name = "piecewise"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12 !0.05_real12
    end if
    if(present(intercept))then
       initialise%intercept = intercept
    else
       initialise%intercept = 1._real12 !0.05_real12
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
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    where(val.le.this%min)
       output = 0._real12
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_activate_3d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    where(val.le.this%min)
       output = 0._real12
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_3d
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
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real12
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function piecewise_differentiate_3d(this, val) result(output)
    implicit none
    class(piecewise_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real12
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_3d
!!!#############################################################################

end module activation_piecewise
