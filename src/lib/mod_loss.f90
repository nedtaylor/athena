module athena__loss
  !! Module containing loss function implementations
  !!
  !! This module implements loss functions that quantify the difference between
  !! model predictions and target values, guiding the optimisation process.
  !!
  !! Implemented loss functions:
  !!
  !! Mean Squared Error (MSE):
  !!   L = (1/N) Σ (y_pred - y_true)²
  !!   For regression, sensitive to outliers
  !!
  !! Mean Absolute Error (MAE):
  !!   L = (1/N) Σ |y_pred - y_true|
  !!   For regression, robust to outliers
  !!
  !! Binary Cross-Entropy:
  !!   L = -(1/N) Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
  !!   For binary classification (outputs in [0,1])
  !!
  !! Categorical Cross-Entropy:
  !!   L = -(1/N) Σ_i Σ_c y_{i,c} * log(ŷ_{i,c})
  !!   For multi-class classification with one-hot encoded targets
  !!
  !! Sparse Categorical Cross-Entropy:
  !!   L = -(1/N) Σ log(ŷ_{i,c_i})
  !!   For multi-class with integer class labels
  !!
  !! Huber Loss:
  !!   L = (1/N) Σ { 0.5*(y-ŷ)²           if |y-ŷ| ≤ δ
  !!               { δ*(|y-ŷ| - 0.5*δ)    otherwise
  !!   Combines MSE and MAE, robust to outliers while smooth near zero
  !!
  !! where N is number of samples, y is true value, ŷ is prediction
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(-), &
       operator(*), operator(/), operator(**), mean, sum, log, abs, merge
  use athena__diffstruc_extd, only: huber
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
     integer :: batch_index = 1
     !! Index of the batch to compute the loss for
     integer :: sample_index = 1
     !! Index of the sample to compute the loss for
   contains
     procedure(compute_base), deferred, pass(this) :: compute
     !! Compute the loss of a model
  end type base_loss_type

  interface
     module function compute_base(this, predicted, expected) result(output)
       !! Compute the loss of a model
       class(base_loss_type), intent(in), target :: this
       !! Instance of the physics-informed neural network loss function
       type(array_type), dimension(:,:), intent(inout), target :: predicted
       !! Predicted values
       type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
            expected
       !! Expected values
       type(array_type), pointer :: output
       !! Physics-informed neural network loss
     end function compute_base
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
  function compute_bce(this, predicted, expected) result(output)
    !! Compute the binary cross entropy loss of a model
    implicit none

    ! Arguments
    class(bce_loss_type), intent(in), target :: this
    !! Instance of the physics-informed neural network loss function
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    !! Predicted values
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    !! Expected values
    type(array_type), pointer :: output
    !! Binary cross entropy loss

    ! Local variables
    integer :: s, i
    !! Loop indices
    type(array_type), pointer :: ptr
    !! Temporary pointer for calculations

    output => mean(-expected(1,1) * log(predicted(1,1) + this%epsilon), dim=2)
    if(any(shape(predicted).gt.1))then
       do s = 1, size(predicted,2)
          do i = 1, size(predicted,1)
             if(.not.predicted(i,s)%allocated .or. &
                  .not.expected(i,s)%allocated) cycle
             ptr => mean(-expected(i,s) * log(predicted(i,s) + this%epsilon), dim=2)

             output => output + ptr
          end do
       end do
    end if

  end function compute_bce
!###############################################################################


!###############################################################################
  function compute_cce(this, predicted, expected) result(output)
    !! Compute the categorical cross entropy loss of a model
    implicit none

    ! Arguments
    class(cce_loss_type), intent(in), target :: this
    !! Instance of the physics-informed neural network loss function
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    !! Predicted values
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    !! Expected values
    type(array_type), pointer :: output
    !! Categorical cross entropy loss

    ! Local variables
    integer :: s, i
    !! Loop indices
    type(array_type), pointer :: ptr
    !! Temporary pointer for calculations

    output => -mean( sum( &
         expected(1,1) * log(predicted(1,1) + this%epsilon), &
         dim=1 ), dim=2)
    if(any(shape(predicted).gt.1))then
       do s = 1, size(predicted,2)
          do i = 1, size(predicted,1)
             if(.not.predicted(i,s)%allocated .or. &
                  .not.expected(i,s)%allocated) cycle
             ptr => mean( sum( &
                  expected(i,s) * log(predicted(i,s) + this%epsilon), &
                  dim=1 ), dim=2)

             output => output - ptr
          end do
       end do
    end if

  end function compute_cce
!###############################################################################


!###############################################################################
  function compute_mae(this, predicted, expected) result(output)
    !! Compute the mean absolute error of a model
    implicit none

    ! Arguments
    class(mae_loss_type), intent(in), target :: this
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    !! Predicted values
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    !! Expected values
    type(array_type), pointer :: output
    !! Mean absolute error

    ! Local variables
    integer :: s, i
    !! Loop indices
    type(array_type), pointer :: ptr
    !! Temporary pointer for calculations

    output => mean( abs( predicted(1,1) - expected(1,1) ) ) / &
         2._real32
    if(any(shape(predicted).gt.1))then
       do s = 1, size(predicted,2)
          do i = 1, size(predicted,1)
             if(.not.predicted(i,s)%allocated .or. &
                  .not.expected(i,s)%allocated) cycle
             ptr => mean( abs( predicted(i,s) - expected(i,s) ) ) / &
                  2._real32

             output => output + ptr
          end do
       end do
    end if

  end function compute_mae
!###############################################################################


!###############################################################################
  function compute_mse(this, predicted, expected) result(output)
    !! Compute the mean squared error of a model
    implicit none

    ! Arguments
    class(mse_loss_type), intent(in), target :: this
    !! Instance of the mean squared error loss function
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    !! Predicted values
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    !! Expected values
    type(array_type), pointer :: output
    !! Mean squared error loss

    ! Local variables
    integer :: s, i
    !! Loop indices
    type(array_type), pointer :: ptr
    !! Temporary pointer for calculations

    output => mean( ( predicted(1,1) - expected(1,1) )  ** 2._real32 ) / &
         2._real32
    if(any(shape(predicted).gt.1))then
       do s = 1, size(predicted,2)
          do i = 1, size(predicted,1)
             if(.not.predicted(i,s)%allocated .or. &
                  .not.expected(i,s)%allocated) cycle
             ptr => mean( ( predicted(i,s) - expected(i,s) )  ** 2._real32 ) / &
                  2._real32

             output => output + ptr
          end do
       end do
    end if

  end function compute_mse
!###############################################################################


!###############################################################################
  function compute_nll(this, predicted, expected) result(output)
    !! Compute the negative log likelihood of a model
    implicit none

    ! Arguments
    class(nll_loss_type), intent(in), target :: this
    !! Instance of the physics-informed neural network loss function
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    !! Predicted values
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    !! Expected values
    type(array_type), pointer :: output
    !! Negative log likelihood loss

    ! Local variables
    integer :: s, i
    !! Loop indices
    type(array_type), pointer :: ptr
    !! Temporary pointer for calculations

    output => mean(-log(expected(1,1) - predicted(1,1) + this%epsilon) )
    if(any(shape(predicted).gt.1))then
       do s = 1, size(predicted,2)
          do i = 1, size(predicted,1)
             if(.not.predicted(i,s)%allocated .or. &
                  .not.expected(i,s)%allocated) cycle
             ptr => mean(-log(expected(i,s) - predicted(i,s) + this%epsilon) )

             output => output + ptr
          end do
       end do
    end if

  end function compute_nll
!###############################################################################


!###############################################################################
  function compute_huber(this, predicted, expected) result(output)
    !! Compute the huber loss of a model
    implicit none

    ! Arguments
    class(huber_loss_type), intent(in), target :: this
    !! Instance of the huber loss function
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    !! Predicted values
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    !! Expected values
    type(array_type), pointer :: output
    !! Huber loss

    ! Local variables
    integer :: s, i
    !! Loop indices
    type(array_type), pointer :: ptr
    !! Temporary pointer for calculations

    ptr => predicted(1,1) - expected(1,1)
    output => mean( huber(predicted(1,1) - expected(1,1), this%gamma) )
    if(any(shape(predicted).gt.1))then
       do s = 1, size(predicted,2)
          do i = 1, size(predicted,1)
             if(.not.predicted(i,s)%allocated .or. &
                  .not.expected(i,s)%allocated) cycle
             ptr => predicted(i,s) - expected(i,s)

             output => output + mean( huber(ptr, this%gamma) )
          end do
       end do
    end if

    ! output => merge( &
    !      0.5_real32 * (ptr)**2._real32, &
    !      this%gamma * (abs(ptr) - 0.5_real32 * this%gamma), &
    !      abs(ptr) .le. this%gamma &
    ! )

  end function compute_huber
!###############################################################################


!###############################################################################
  module function compute_base(this, predicted, expected) result(output)
    !! Placeholder for compute function in base_loss_type
    implicit none

    ! Arguments
    class(base_loss_type), intent(in), target :: this
    !! Instance of the base loss function
    type(array_type), dimension(:,:), intent(inout), target :: predicted
    !! Predicted values
    type(array_type), dimension(size(predicted,1),size(predicted,2)), intent(in) :: &
         expected
    !! Expected values
    type(array_type), pointer :: output
    !! Loss value

  end function compute_base
!###############################################################################

end module athena__loss
