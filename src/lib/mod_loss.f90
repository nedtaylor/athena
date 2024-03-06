!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module loss
  use constants, only: real12
  implicit none

  abstract interface
     pure function compute_loss_function(predicted, expected) result(output)
       import real12
       real(real12), dimension(:,:), intent(in) :: predicted, expected
       real(real12), dimension(size(predicted,1),size(predicted,2)) :: output
     end function compute_loss_function
  end interface
  
  abstract interface
     pure function total_loss_function(predicted, expected) result(output)
       import real12
       real(real12), dimension(:,:), intent(in) :: predicted, expected
       real(real12), dimension(size(predicted,2)) :: output
     end function total_loss_function
  end interface

  private

  public :: compute_loss_derivative

  public :: compute_loss_function
  public :: compute_loss_bce
  public :: compute_loss_cce
  public :: compute_loss_mae
  public :: compute_loss_mse
  public :: compute_loss_nll

  public :: total_loss_function
  public :: total_loss_bce
  public :: total_loss_cce
  public :: total_loss_mae
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
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,1),size(predicted,2)) :: output

    output = predicted - expected
  end function compute_loss_derivative
!!!#############################################################################

!!!#############################################################################
!!! compute losses
!!! method: Binary cross entropy
!!!#############################################################################
  pure function compute_loss_bce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,1),size(predicted,2)) :: output
    real(real12) :: epsilon

    epsilon = 1.E-10_real12
    output = -expected*log(predicted+epsilon)

  end function compute_loss_bce
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_bce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,2)) :: output

    output = sum(compute_loss_bce(predicted,expected),dim=1)

  end function total_loss_bce
!!!#############################################################################


!!!#############################################################################
!!! compute 
!!! method: categorical cross entropy
!!!#############################################################################
  pure function compute_loss_cce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,1),size(predicted,2)) :: output
    real(real12) :: epsilon

    epsilon = 1.E-10_real12
    output = -expected * log(predicted + epsilon)

  end function compute_loss_cce
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_cce(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,2)) :: output

    output = sum(compute_loss_cce(predicted,expected),dim=1)
    
  end function total_loss_cce
!!!#############################################################################


!!!#############################################################################
!!! compute losses
!!! method: mean absolute error
!!!#############################################################################
  pure function compute_loss_mae(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,1),size(predicted,2)) :: output

    output = abs(predicted - expected) /(size(predicted,1))

  end function compute_loss_mae
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_mae(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,2)) :: output

    output = sum(compute_loss_mae(predicted,expected),dim=1)
    
  end function total_loss_mae
!!!#############################################################################


!!!#############################################################################
!!! compute losses
!!! method: mean squared error
!!!#############################################################################
  pure function compute_loss_mse(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,1),size(predicted,2)) :: output

    output = ((predicted - expected)**2._real12) /(2._real12*size(predicted,1))

  end function compute_loss_mse
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_mse(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,2)) :: output

    output = sum(compute_loss_mse(predicted,expected),dim=1)
    
  end function total_loss_mse
!!!#############################################################################


!!!#############################################################################
!!! compute losses
!!! method: negative log likelihood
!!!#############################################################################
  pure function compute_loss_nll(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,1),size(predicted,2)) :: output
    real(real12) :: epsilon

    output = - log(expected - predicted + epsilon)

  end function compute_loss_nll
!!!-----------------------------------------------------------------------------
!!!-----------------------------------------------------------------------------
  pure function total_loss_nll(predicted, expected) result(output)
    implicit none
    real(real12), dimension(:,:), intent(in) :: predicted, expected
    real(real12), dimension(size(predicted,2)) :: output

    output = sum(compute_loss_nll(predicted,expected),dim=1)

  end function total_loss_nll
!!!#############################################################################

end module loss
