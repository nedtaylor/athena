module athena__loss
  !! Module containing functions to compute the loss of a model
  !!
  !! This module contains functions to compute the loss of a model
  !! The loss functions are used to determine how well a model is performing
  use athena__constants, only: real32
  implicit none


  private

  public :: base_loss_type
  public :: bce_loss_type
  public :: cce_loss_type
  public :: mae_loss_type
  public :: mse_loss_type
  public :: nll_loss_type
  public :: huber_loss_type


  type, abstract :: base_loss_type
     !! Abstract type for loss functions
     character(len=:), allocatable :: name
     !! Name of the loss function
     real(real32) :: epsilon = 1.E-10_real32
     !! Small value to prevent log(0)
     logical :: requires_inputs = .false.
     !! Whether the loss function requires inputs to be passed
     integer :: batch_index = 1
     !! Index of the batch to compute the loss for
     integer :: sample_index = 1
     !! Index of the sample to compute the loss for
   contains
     procedure(compute_base), deferred, pass(this) :: compute
     !! Compute the loss of a model
     procedure, pass(this) :: compute_derivative => compute_derivative_base
     !! Compute the derivative of the loss function
  end type base_loss_type

  abstract interface
     pure module function compute_base(this, predicted, expected) result(output)
       !! Compute the loss of a model
       class(base_loss_type), intent(in) :: this
       !! Instance of the loss function type
       real(real32), dimension(:,:), intent(in) :: predicted, expected
       !! Predicted and expected values
       real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
       !! Loss of the model
     end function compute_base
  end interface

  interface
     pure module function compute_derivative_base(this, predicted, expected) &
          result(output)
       !! Compute the derivative of the loss function
       class(base_loss_type), intent(in) :: this
       !! Instance of the loss function type
       real(real32), dimension(:,:), intent(in) :: predicted, expected
       !! Predicted and expected values
       real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
       !! Derivative of the loss function
     end function compute_derivative_base
  end interface

!-------------------------------------------------------------------------------

  type, extends(base_loss_type) :: bce_loss_type
     !! Binary cross entropy loss function
   contains
     procedure :: compute => compute_bce
     !! Compute the loss of a model
  end type bce_loss_type

  interface bce_loss_type
     !! Interface for binary cross entropy loss function
     module function setup_loss_bce() result(loss)
       !! Set up binary cross entropy loss function
       type(bce_loss_type) :: loss
       !! Binary cross entropy loss function
     end function setup_loss_bce
  end interface bce_loss_type

!-------------------------------------------------------------------------------

  type, extends(base_loss_type) :: cce_loss_type
     !! Categorical cross entropy loss function
   contains
     procedure :: compute => compute_cce
     !! Compute the loss of a model
  end type cce_loss_type

  interface cce_loss_type
     !! Interface for categorical cross entropy loss function
     module function setup_loss_cce() result(loss)
       !! Set up categorical cross entropy loss function
       type(cce_loss_type) :: loss
       !! Categorical cross entropy loss function
     end function setup_loss_cce
  end interface cce_loss_type

!-------------------------------------------------------------------------------

  type, extends(base_loss_type) :: mae_loss_type
     !! Mean absolute error loss function
   contains
     procedure :: compute => compute_mae
     !! Compute the loss of a model
  end type mae_loss_type

  interface mae_loss_type
     !! Interface for mean absolute error loss function
     module function setup_loss_mae() result(loss)
       !! Set up mean absolute error loss function
       type(mae_loss_type) :: loss
       !! Mean absolute error loss function
     end function setup_loss_mae
  end interface mae_loss_type

!-------------------------------------------------------------------------------

  type, extends(base_loss_type) :: mse_loss_type
     !! Mean squared error loss function
   contains
     procedure :: compute => compute_mse
     !! Compute the loss of a model
  end type mse_loss_type

  interface mse_loss_type
     !! Interface for mean squared error loss function
     module function setup_loss_mse() result(loss)
       !! Set up mean squared error loss function
       type(mse_loss_type) :: loss
       !! Mean squared error loss function
     end function setup_loss_mse
  end interface mse_loss_type

!-------------------------------------------------------------------------------

  type, extends(base_loss_type) :: nll_loss_type
     !! Negative log likelihood loss function
   contains
     procedure :: compute => compute_nll
     !! Compute the loss of a model
  end type nll_loss_type

  interface nll_loss_type
     !! Interface for negative log likelihood loss function
     module function setup_loss_nll() result(loss)
       !! Set up negative log likelihood loss function
       type(nll_loss_type) :: loss
       !! Negative log likelihood loss function
     end function setup_loss_nll
  end interface nll_loss_type

!-------------------------------------------------------------------------------

  type, extends(base_loss_type) :: huber_loss_type
     !! Huber loss function
     real(real32) :: gamma = 1._real32
     !! Gamma value for the huber loss function
   contains
     procedure :: compute => compute_huber
     !! Compute the loss of a model
     procedure :: compute_derivative => compute_derivative_huber
     !! Compute the derivative of the loss function
  end type huber_loss_type

  interface huber_loss_type
     !! Interface for huber loss function
     module function setup_loss_huber() result(loss)
       !! Set up huber loss function
       type(huber_loss_type) :: loss
       !! Huber loss function
     end function setup_loss_huber
  end interface huber_loss_type

!-------------------------------------------------------------------------------



contains
!###############################################################################
  module function setup_loss_bce() result(loss)
    !! Set up binary cross entropy loss function
    implicit none

    ! Local variables
    type(bce_loss_type) :: loss
    !! Binary cross entropy loss function

    loss%name = 'bce'
  end function setup_loss_bce
!-------------------------------------------------------------------------------
  module function setup_loss_cce() result(loss)
    !! Set up categorical cross entropy loss function
    implicit none

    ! Local variables
    type(cce_loss_type) :: loss
    !! Categorical cross entropy loss function

    loss%name = 'cce'
  end function setup_loss_cce
!-------------------------------------------------------------------------------
  module function setup_loss_mae() result(loss)
    !! Set up mean absolute error loss function
    implicit none

    ! Local variables
    type(mae_loss_type) :: loss
    !! Mean absolute error loss function

    loss%name = 'mae'
  end function setup_loss_mae
!-------------------------------------------------------------------------------
  module function setup_loss_mse() result(loss)
    !! Set up mean squared error loss function
    implicit none

    ! Local variables
    type(mse_loss_type) :: loss
    !! Mean squared error loss function

    loss%name = 'mse'
  end function setup_loss_mse
!-------------------------------------------------------------------------------
  module function setup_loss_nll() result(loss)
    !! Set up negative log likelihood loss function
    implicit none

    ! Local variables
    type(nll_loss_type) :: loss
    !! Negative log likelihood loss function

    loss%name = 'nll'
  end function setup_loss_nll
!-------------------------------------------------------------------------------
  module function setup_loss_huber() result(loss)
    !! Set up huber loss function
    implicit none

    ! Local variables
    type(huber_loss_type) :: loss
    !! Huber loss function

    loss%name = 'hub'
  end function setup_loss_huber
!###############################################################################


!###############################################################################
  pure module function compute_derivative_base(this, predicted, expected) result(output)
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
    class(base_loss_type), intent(in) :: this
    !! Instance of the loss function type
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Derivative of the loss function

    output = predicted - expected
  end function compute_derivative_base
!###############################################################################


!###############################################################################
  pure function compute_bce(this, predicted, expected) result(output)
    !! Compute the binary cross entropy loss of a model
    implicit none

    ! Arguments
    class(bce_loss_type), intent(in) :: this
    !! Instance of the binary cross entropy loss function
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Binary cross entropy loss

    output = -expected*log(predicted+this%epsilon)

  end function compute_bce
!###############################################################################


!###############################################################################
  pure function compute_cce(this, predicted, expected) result(output)
    !! Compute the categorical cross entropy loss of a model
    implicit none

    ! Arguments
    class(cce_loss_type), intent(in) :: this
    !! Instance of the categorical cross entropy loss function
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Categorical cross entropy loss

    output = -expected * log(predicted + this%epsilon)

  end function compute_cce
!###############################################################################


!###############################################################################
  pure module function compute_mae(this, predicted, expected) result(output)
    !! Compute the mean absolute error of a model
    implicit none

    ! Arguments
    class(mae_loss_type), intent(in) :: this
    !! Instance of the mean absolute error loss function
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Mean absolute error

    output = abs(predicted - expected) !/(size(predicted,1))

  end function compute_mae
!###############################################################################


!###############################################################################
  pure module function compute_mse(this, predicted, expected) result(output)
    !! Compute the mean squared error of a model
    implicit none

    ! Arguments
    class(mse_loss_type), intent(in) :: this
    !! Instance of the mean squared error loss function
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Mean squared error

    output = ((predicted - expected)**2._real32) /(2._real32)!*size(predicted,1))

  end function compute_mse
!###############################################################################


!###############################################################################
  pure function compute_nll(this, predicted, expected) result(output)
    !! Compute the negative log likelihood of a model
    implicit none

    ! Arguments
    class(nll_loss_type), intent(in) :: this
    !! Instance of the negative log likelihood loss function
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Negative log likelihood

    output = - log(expected - predicted + this%epsilon)

  end function compute_nll
!###############################################################################


!###############################################################################
  pure function compute_huber(this, predicted, expected) result(output)
    !! Compute the huber loss of a model
    implicit none

    ! Arguments
    class(huber_loss_type), intent(in) :: this
    !! Instance of the huber loss function
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! huber loss

    where (abs(predicted - expected) .le. this%gamma)
       output = 0.5_real32 * (predicted - expected)**2._real32
    elsewhere
       output = this%gamma * (abs(predicted - expected) - 0.5_real32 * this%gamma)
    end where

  end function compute_huber
!-------------------------------------------------------------------------------
  pure function compute_derivative_huber(this, predicted, expected) &
       result(output)
    !! Compute the derivative of the huber loss function
    implicit none

    ! Arguments
    class(huber_loss_type), intent(in) :: this
    !! Instance of the huber loss function
    real(real32), dimension(:,:), intent(in) :: predicted, expected
    !! Predicted and expected values
    real(real32), dimension(size(predicted,1),size(predicted,2)) :: output
    !! Derivative of the huber loss function

    where (abs(predicted - expected) .le. this%gamma)
       output = predicted - expected
    elsewhere
       output = this%gamma * sign(1._real32, predicted - expected)
    end where

  end function compute_derivative_huber
!###############################################################################

end module athena__loss
