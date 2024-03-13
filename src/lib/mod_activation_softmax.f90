!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of the softmax activation function
!!!#############################################################################
module activation_softmax
  use constants, only: real12
  use custom_types, only: activation_type
  implicit none
  
  type, extends(activation_type) :: softmax_type
   contains
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
  
  
  private
  
  public :: softmax_setup
  
  
contains
  
!!!#############################################################################
!!! initialisation
!!!#############################################################################
  pure function initialise(threshold, scale)
    implicit none
    type(softmax_type) :: initialise
    real(real12), optional, intent(in) :: threshold
    real(real12), optional, intent(in) :: scale

    initialise%name = "softmax"

    if(present(scale))then
       initialise%scale = scale
    else
       initialise%scale = 1._real12
    end if

    if(present(threshold))then
       initialise%threshold = threshold
    else
       initialise%threshold = -min(huge(1._real12),32._real12)
    end if
  end function initialise
!!!#############################################################################
  
  
!!!#############################################################################
!!! softmax transfer function
!!! f = exp(x-max)/sum(exp(x-max))
!!!#############################################################################
  pure function softmax_activate_1d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output
    
    !! compute softmax values
    output = exp(val - maxval(val))

    !! normalize softmax values
    output = output / sum(output)

  end function softmax_activate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function softmax_activate_2d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output
    
    integer :: s

    do s = 1, size(val,2)
      !! compute softmax values
      output(:,s) = exp(val(:,s) - maxval(val(:,s)))

      !! normalize softmax values
      output(:,s) = output(:,s) / sum(output(:,s))
    end do

  end function softmax_activate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function softmax_activate_3d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output
    
    integer :: s

    do s=1,size(val,3)
      !! compute softmax values
       output(:,:,s) = exp(val(:,:,s) - maxval(val(:,:,s)))

       !! normalize softmax values
       output(:,:,s) = output(:,:,s) / sum(output(:,:,s))
    end do

  end function softmax_activate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function softmax_activate_4d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output
    
    integer :: s

    do s=1,size(val,3)
      !! compute softmax values
      output(:,:,:,s) = exp(val(:,:,:,s) - maxval(val(:,:,:,s)))

      !! normalize softmax values
      output(:,:,:,s) = output(:,:,:,s) / sum(output(:,:,:,s))
    end do

  end function softmax_activate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function softmax_activate_5d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    integer :: s

    do s=1,size(val,3)
      !! compute softmax values
      output(:,:,:,:,s) = exp(val(:,:,:,:,s) - maxval(val(:,:,:,:,s)))

      !! normalize softmax values
      output(:,:,:,:,s) = output(:,:,:,:,s) / sum(output(:,:,:,:,s))
    end do

  end function softmax_activate_5d
!!!#############################################################################


!!!#############################################################################
!!! derivative of softmax function
!!! df/dx = f * (1 - f)
!!!#############################################################################
  pure function softmax_differentiate_1d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:), intent(in) :: val
    real(real12), dimension(size(val,dim=1)) :: output

    !! compute gradients for softmax layer
    output = this%activate_1d(val)
    output = output * (1._real12 - output)

  end function softmax_differentiate_1d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function softmax_differentiate_2d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2)) :: output

    !! compute gradients for softmax layer
    output = this%activate_2d(val)
    output = output * (1._real12 - output)

  end function softmax_differentiate_2d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function softmax_differentiate_3d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:,:,:), intent(in) :: val
    real(real12), dimension(size(val,1),size(val,2),size(val,3)) :: output

    !! compute gradients for softmax layer
    output = this%activate_3d(val)
    output = output * (1._real12 - output)

  end function softmax_differentiate_3d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function softmax_differentiate_4d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4)) :: output

    !! compute gradients for softmax layer
    output = this%activate_4d(val)
    output = output * (1._real12 - output)

  end function softmax_differentiate_4d
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function softmax_differentiate_5d(this, val) result(output)
    implicit none
    class(softmax_type), intent(in) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: val
    real(real12), dimension(&
         size(val,1),size(val,2),size(val,3),size(val,4),size(val,5)) :: output

    !! compute gradients for softmax layer
    output = this%activate_5d(val)
    output = output * (1._real12 - output)

  end function softmax_differentiate_5d
!!!#############################################################################

end module activation_softmax
