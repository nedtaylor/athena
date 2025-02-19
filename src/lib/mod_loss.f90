module athena__loss
  !! Module containing functions to compute the loss of a model
  !!
  !! This module contains functions to compute the loss of a model
  !! The loss functions are used to determine how well a model is performing
  use athena__constants, only: real32
  implicit none


  private

  public :: compute_loss_derivative
  public :: compute_loss_hubber_derivative

  public :: compute_loss_function
  public :: compute_loss_bce
  public :: compute_loss_cce
  public :: compute_loss_mae
  public :: compute_loss_mse
  public :: compute_loss_nll
  public :: compute_loss_hubber

  public :: total_loss_function
  public :: total_loss_bce
  public :: total_loss_cce
  public :: total_loss_mae
  public :: total_loss_mse
  public :: total_loss_nll
  public :: total_loss_hubber


  abstract interface
     pure function compute_loss_function(predicted, expected) result(output)
       !! Compute the loss of a model
       import real32
       real(real32), dimension(:,:), intent(in) :: predicted, expected
       !! Predicted and expected values
       real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
       !! Loss of the model
     end function compute_loss_function
  end interface
  
  abstract interface
     !! compute the total loss function
     !! predicted = (R, in) predicted values
     !! expected  = (R, in) expected values
     !! output    = (R, in) loss function
     pure function total_loss_function(predicted, expected) result(output)
       !! Compute the total loss of a model
       import real32
       real(real32), dimension(:,:), intent(in) :: predicted, expected
       !! Predicted and expected values
       real(real32), dimension(size(predicted,2)) :: output
       !! Loss of the model
     end function total_loss_function
  end interface



contains

!###############################################################################
  pure function compute_loss_derivative(predicted, expected) result(output)
    !! Compute the derivative of the loss function
    !!
    !! This function computes the derivative of the loss function
    !! The derivative of the loss function is used to update the weights of
    !! the model
    !! For all cross entropy (and MSE and NLL) loss functions, the derivative
    !! of the loss function is simply the difference between the predicted and
    !! expected values
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Derivative of the loss function

    output = predicted - expected
  end function compute_loss_derivative
!###############################################################################


!###############################################################################
  pure function compute_loss_bce(predicted, expected) result(output)
    !! Compute the binary cross entropy loss of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Binary cross entropy loss

    ! Local variables
    real(real32) :: epsilon
    !! Small value to prevent log(0)

    epsilon = 1.E-10_real32
    output = -expected*log(predicted+epsilon)

  end function compute_loss_bce
!!!-----------------------------------------------------------------------------
  pure function total_loss_bce(predicted, expected) result(output)
    !! Compute the total binary cross entropy loss of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,2)) :: output
    !! Total binary cross entropy loss

    output = sum(compute_loss_bce(predicted,expected),dim=1)

  end function total_loss_bce
!###############################################################################


!###############################################################################
  pure function compute_loss_cce(predicted, expected) result(output)
    !! Compute the categorical cross entropy loss of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Categorical cross entropy loss

    ! Local variables
    real(real32) :: epsilon
    !! Small value to prevent log(0)

    epsilon = 1.E-10_real32
    output = -expected * log(predicted + epsilon)

  end function compute_loss_cce
!-------------------------------------------------------------------------------
  pure function total_loss_cce(predicted, expected) result(output)
    !! Compute the total categorical cross entropy loss of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,2)) :: output
    !! Total categorical cross entropy loss

    output = sum(compute_loss_cce(predicted,expected),dim=1)
    
  end function total_loss_cce
!###############################################################################


!###############################################################################
  pure function compute_loss_mae(predicted, expected) result(output)
    !! Compute the mean absolute error of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Mean absolute error

    output = abs(predicted - expected) !/(size(predicted,1))

  end function compute_loss_mae
!!!-----------------------------------------------------------------------------
  pure function total_loss_mae(predicted, expected) result(output)
    !! Compute the total mean absolute error of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,2)) :: output
    !! Total mean absolute error

    output = sum(compute_loss_mae(predicted,expected),dim=1) / size(predicted,1)
    
  end function total_loss_mae
!###############################################################################


!###############################################################################
  pure function compute_loss_mse(predicted, expected) result(output)
    !! Compute the mean squared error of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Mean squared error

    output = ((predicted - expected)**2._real32) /(2._real32)!*size(predicted,1))

  end function compute_loss_mse
!!!-----------------------------------------------------------------------------
  pure function total_loss_mse(predicted, expected) result(output)
    !! Compute the total mean squared error of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,2)) :: output
    !! Total mean squared error

    output = sum(compute_loss_mse(predicted,expected),dim=1) * &
         2._real32 / size(predicted,1)
    
  end function total_loss_mse
!###############################################################################


!###############################################################################
  pure function compute_loss_nll(predicted, expected) result(output)
    !! Compute the negative log likelihood of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Negative log likelihood

    ! Local variables
    real(real32) :: epsilon
    !! Small value to prevent log(0)

    epsilon = 1.E-10_real32
    output = - log(expected - predicted + epsilon)

  end function compute_loss_nll
!!!-----------------------------------------------------------------------------
  pure function total_loss_nll(predicted, expected) result(output)
    !! Compute the total negative log likelihood of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,2)) :: output
    !! Total negative log likelihood

    output = sum(compute_loss_nll(predicted,expected),dim=1) / size(predicted,1)

  end function total_loss_nll
!###############################################################################


!###############################################################################
  pure function compute_loss_hubber(predicted, expected) result(output)
    !! Compute the hubber loss of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Hubber loss

    ! Local variables
    real(real32) :: gamma
    !! Gamma value for the hubber loss function

    gamma = 1._real32

    where (abs(predicted - expected) .le. gamma)
       output = 0.5_real32 * (predicted - expected)**2._real32
    elsewhere
       output = gamma * (abs(predicted - expected) - 0.5_real32 * gamma)
    end where
  
  end function compute_loss_hubber
!!!-----------------------------------------------------------------------------
  pure function total_loss_hubber(predicted, expected) result(output)
    !! Compute the total hubber loss of a model
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,2)) :: output
    !! Total hubber loss
  
    output = sum(compute_loss_hubber(predicted,expected),dim=1) / size(predicted,1)
    
  end function total_loss_hubber
!!!-----------------------------------------------------------------------------
  pure function compute_loss_hubber_derivative(predicted, expected) &
       result(output)
    !! Compute the derivative of the hubber loss function
    implicit none

    ! Arguments
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Derivative of the hubber loss function

    ! Local variables
    real(real32) :: gamma
    !! Gamma value for the hubber loss function
    
    gamma = 1._real32

    where (abs(predicted - expected) .le. gamma)
       output = predicted - expected
    elsewhere
        output = gamma * sign(1._real32, predicted - expected)
    end where

  end function compute_loss_hubber_derivative
!###############################################################################

end module athena__loss
