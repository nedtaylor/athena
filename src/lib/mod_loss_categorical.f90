!!! https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e#:~:text=Categorical%20cross%2Dentropy%20is%20used,%5D%20for%203%2Dclass%20problem.
!!! https://stackoverflow.com/questions/8612466/how-to-alias-a-function-name-in-fortran
module loss_categorical
  use constants, only: real12
  implicit none

  abstract interface
     pure function compute_loss_function(predicted, expected) result(output)
       import real12
       real(real12), dimension(:), intent(in) :: predicted, expected
       real(real12), dimension(size(predicted)) :: output
     end function compute_loss_function
  end interface
  
  abstract interface
     pure function total_loss_function(predicted, expected) result(output)
       import real12
       real(real12), dimension(:), intent(in) :: predicted, expected
       real(real12) :: output
     end function total_loss_function
  end interface

  private

  public :: compute_loss_derivative

  public :: compute_loss_function
  public :: compute_loss_bce
  public :: compute_loss_cce
  public :: compute_loss_mse
  public :: compute_loss_nll

  public :: total_loss_function
  public :: total_loss_bce
  public :: total_loss_cce
  public :: total_loss_mse
  public :: total_loss_nll


contains

!!!#############################################################################
!!! compute loss derivative
!!! for all cross entropy (and MSE and NLL) loss functions, the derivative ...
!!! ... of the loss function is
!!!#############################################################################
  pure function compute_loss_derivative(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted)) :: output

    output = predicted - expected
  end function compute_loss_derivative
!!!#############################################################################

!!!#############################################################################
!!! compute losses
!!! method: Binary cross entropy
!!!#############################################################################
  pure function compute_loss_bce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted)) :: output
    real(real12) :: epsilon

    epsilon = 1.E-10_real12
    output = -expected*log(predicted+epsilon)

  end function compute_loss_bce
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_bce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12) :: output

    output = sum(compute_loss_bce(predicted,expected))

  end function total_loss_bce
!!!#############################################################################


!!!#############################################################################
!!! compute 
!!! method: categorical cross entropy
!!!#############################################################################
  pure function compute_loss_cce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted)) :: output
    real(real12) :: epsilon

    epsilon = 1.E-10_real12
    output = -expected*log(predicted+epsilon)

  end function compute_loss_cce
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_cce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12) :: output

    output = sum(compute_loss_cce(predicted,expected))
    
  end function total_loss_cce
!!!#############################################################################


!!!!#############################################################################
!!!! compute loss derivative
!!!! method: categorical cross entropy
!!!! this is handled by the softmax backward subroutine
!!!!#############################################################################
!  subroutine compute_loss_derivative_cce(predicted, expected, gradient)
!    implicit none
!    integer, intent(in) :: expected
!    real(real12), dimension(:), intent(in) :: predicted
!    real(real12), dimension(:), intent(out) :: gradient
!    
!    gradient = predicted
!    gradient(expected) = predicted(expected) - 1._real12
!
!  end subroutine compute_loss_derivative_cce
!!!!#############################################################################


!!!#############################################################################
!!! compute losses
!!! method: mean squared error
!!!#############################################################################
  pure function compute_loss_mse(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted)) :: output

    output = ((predicted - expected)**2._real12) /(2*size(predicted))

  end function compute_loss_mse
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_mse(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12) :: output

    output = sum(compute_loss_mse(predicted,expected))
    
  end function total_loss_mse
!!!#############################################################################


!!!#############################################################################
!!! compute losses
!!! method: categorical cross entropy
!!!#############################################################################
  pure function compute_loss_nll(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted)) :: output
    real(real12) :: epsilon

    output = - log(expected - predicted + epsilon)

  end function compute_loss_nll
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_nll(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:), intent(in) :: predicted, expected
    real(real12) :: output

    output = sum(compute_loss_nll(predicted,expected))

  end function total_loss_nll
!!!#############################################################################

end module loss_categorical
