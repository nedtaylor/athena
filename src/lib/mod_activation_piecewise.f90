!###############################################################################
module athena__activation_piecewise
  !! Module containing implementation of the piecewise activation function
  use athena__constants, only: real32
  use athena__misc_types, only: activation_type
  implicit none

  type, extends(activation_type) :: piecewise_type
     !! Type for piecewise activation function with overloaded procedures
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
  
!###############################################################################
  pure function initialise(scale, intercept)
    !! Initialise a piecewise activation function 
    implicit none

    ! Arguments
    type(piecewise_type) :: initialise
    !! Piecewise activation type
    real(real32), optional, intent(in) :: scale
    !! Optional scale factor for activation output
    real(real32), optional, intent(in) :: intercept
    !! Optional intercept value
    
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
!###############################################################################

       
!###############################################################################
  pure function piecewise_activate_1d(this, val) result(output)
    !! Apply piecewise activation to 1D array
    !!
    !! Computes piecewise function:
    !! f = 0 if x ≤ min
    !! f = scale if x ≥ max
    !! f = scale * x + intercept otherwise
    implicit none

    ! Arguments
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Activated output values

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function piecewise_activate_2d(this, val) result(output)
    !! Apply piecewise activation to 2D array
    !!
    !! Computes piecewise function:
    !! f = 0 if x ≤ min
    !! f = scale if x ≥ max
    !! f = scale * x + intercept otherwise
    implicit none

    ! Arguments
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Activated output values

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function piecewise_activate_3d(this, val) result(output)
    !! Apply piecewise activation to 3D array
    !!
    !! Computes piecewise function:
    !! f = 0 if x ≤ min
    !! f = scale if x ≥ max
    !! f = scale * x + intercept otherwise
    implicit none

    ! Arguments
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Activated output values

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function piecewise_activate_4d(this, val) result(output)
    !! Apply piecewise activation to 4D array
    !!
    !! Computes piecewise function:
    !! f = 0 if x ≤ min
    !! f = scale if x ≥ max
    !! f = scale * x + intercept otherwise
    implicit none

    ! Arguments
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Activated output values

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function piecewise_activate_5d(this, val) result(output)
    !! Apply piecewise activation to 5D array
    !!
    !! Computes piecewise function:
    !! f = 0 if x ≤ min
    !! f = scale if x ≥ max
    !! f = scale * x + intercept otherwise
    implicit none

    ! Arguments
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Activated output values

    where(val.le.this%min)
       output = 0._real32
    elsewhere(val.ge.this%max)
       output = this%scale
    elsewhere
       output = this%scale * val + this%intercept
    end where
  end function piecewise_activate_5d
!###############################################################################


!###############################################################################
!!! derivative of piecewise transfer function
!!! e.g. df/dx (gradient * x) = gradient
!!! we are performing the derivative to identify what weight ...
!!! ... results in the minimum error
!###############################################################################
  pure function piecewise_differentiate_1d(this, val) result(output)
    !! Differentiate piecewise activation for 1D array
    !!
    !! Computes derivative: 
    !! df/dx = 0 if x ≤ min or x ≥ max
    !! df/dx = scale otherwise
    implicit none

    ! Arguments
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,dim=1)) :: output
    !! Differentiated output values

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_1d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function piecewise_differentiate_2d(this, val) result(output)
    !! Differentiate piecewise activation for 2D array
    !!
    !! Computes derivative: 
    !! df/dx = 0 if x ≤ min or x ≥ max
    !! df/dx = scale otherwise
    implicit none
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2)) :: output
    !! Differentiated output values

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_2d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function piecewise_differentiate_3d(this, val) result(output)
    !! Differentiate piecewise activation for 3D array
    !!
    !! Computes derivative: 
    !! df/dx = 0 if x ≤ min or x ≥ max
    !! df/dx = scale otherwise
    implicit none
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
    !! Differentiated output values

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_3d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function piecewise_differentiate_4d(this, val) result(output)
    !! Differentiate piecewise activation for 4D array
    !!
    !! Computes derivative: 
    !! df/dx = 0 if x ≤ min or x ≥ max
    !! df/dx = scale otherwise
    implicit none
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    !! Differentiated output values

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_4d
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
  pure function piecewise_differentiate_5d(this, val) result(output)
    !! Differentiate piecewise activation for 5D array
    !!
    !! Computes derivative: 
    !! df/dx = 0 if x ≤ min or x ≥ max
    !! df/dx = scale otherwise
    implicit none
    class(piecewise_type), intent(in) :: this
    !! Piecewise activation type
    real(real32), dimension(:,:,:,:,:), intent(in) :: val
    !! Input values
    real(real32), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output
    !! Differentiated output values

    where(val.le.this%min.or.val.ge.this%max)
       output = 0._real32
    elsewhere
       output = this%scale
    end where
  end function piecewise_differentiate_5d
!###############################################################################

end module athena__activation_piecewise
