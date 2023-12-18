!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor 
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module activation_gaussian
  use constants, only: real12, pi
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: gaussian_type
     real(real12) :: sigma
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
  
  
  private
  
  public :: gaussian_setup
  
  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  pure function initialise(threshold, scale, sigma)
    implicit none
    type(gaussian_type) :: initialise
    real(real12), optional, intent(in) :: threshold
    real(real12), optional, intent(in) :: scale
    real(real12), optional, intent(in) :: sigma

    initialise%name = "gaussian"
    
    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12
    end if

    if(present(sigma))then
       initialise%sigma = sigma
    else
       initialise%sigma = 1.5_real12
    end if

    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = min(huge(1._real12),16._real12) * &
            initialise%sigma
    end if

  end function initialise
!!!#############################################################################


!!!#############################################################################
!!! gaussian transfer function
!!! f = 1/(1+exp(-x))
!!!#############################################################################
  pure function gaussian_activate_1d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = 0._real12
    where(abs(val).le.this%threshold)
       output = this%scale * 1._real12/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real12 * (val/this%sigma)**2._real12)
    end where
  end function gaussian_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_activate_2d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output

    output = 0._real12
    where(abs(val).le.this%threshold)
       output = this%scale * 1._real12/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real12 * (val/this%sigma)**2._real12)
    end where
  end function gaussian_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_activate_3d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = 0._real12
    where(abs(val).le.this%threshold)
       output = this%scale * 1._real12/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real12 * (val/this%sigma)**2._real12)
    end where
  end function gaussian_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_activate_4d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = 0._real12
    where(abs(val).le.this%threshold)
       output = this%scale * 1._real12/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real12 * (val/this%sigma)**2._real12)
    end where
  end function gaussian_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_activate_5d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = 0._real12
    where(abs(val).le.this%threshold)
       output = this%scale * 1._real12/(sqrt(2*pi)*this%sigma) * &
            exp(-0.5_real12 * (val/this%sigma)**2._real12)
    end where
  end function gaussian_activate_5d
!!!#############################################################################


!!!#############################################################################
!!! derivative of gaussian function
!!! df/dx = f * (1 - f)
!!!#############################################################################
  pure function gaussian_differentiate_1d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    output = -val/this%sigma**2._real12 * this%activate_1d(val)
  end function gaussian_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_differentiate_2d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output

    output = -val/this%sigma**2._real12 * this%activate_2d(val)
  end function gaussian_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_differentiate_3d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    output = -val/this%sigma**2._real12 * this%activate_3d(val)
  end function gaussian_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_differentiate_4d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    output = -val/this%sigma**2._real12 * this%activate_4d(val)
  end function gaussian_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function gaussian_differentiate_5d(this, val) result(output)
    implicit none
    class(gaussian_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
          size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    output = -val/this%sigma**2._real12 * this%activate_5d(val)
  end function gaussian_differentiate_5d
!!!#############################################################################

end module activation_gaussian
