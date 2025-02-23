module athena__regulariser
  !! Module containing regularisation methods
  !!
  !! This module contains regularisation methods to prevent overfitting
  !! in neural networks
  use athena__constants, only: real32
  implicit none


  private

  public :: base_regulariser_type
  public :: l1_regulariser_type
  public :: l2_regulariser_type
  public :: l1l2_regulariser_type


  type, abstract :: base_regulariser_type
     !! Abstract type for regularisation
   contains
     procedure(regularise), deferred, pass(this) :: regularise
     !! Regularisation method
  end type base_regulariser_type

  abstract interface
    pure subroutine regularise(this, params, gradient, learning_rate)
      !! Regularise the parameters
      import :: base_regulariser_type, real32
      class(base_regulariser_type), intent(in) :: this
      !! Regulariser object
      real(real32), dimension(:),  intent(in) :: params
      !! Parameters to regularise
      real(real32), dimension(:), intent(inout) :: gradient
      !! Gradient of the parameters
      real(real32), intent(in) :: learning_rate
      !! Learning rate
    end subroutine regularise
  end interface

  type, extends(base_regulariser_type) :: l1_regulariser_type
     !! Type for L1 regularisation
     !!
     !! L1 regularisation is also known as Lasso regression
     !! It is used to prevent overfitting in neural networks
     real(real32) :: l1 = 0.01_real32
   contains
     procedure, pass(this) :: regularise => regularise_l1
     !! Regularisation method
  end type l1_regulariser_type

  type, extends(base_regulariser_type) :: l2_regulariser_type
     !! Type for L2 regularisation
     !!
     !! L2 regularisation is also known as Ridge regression
     !! It is used to prevent overfitting in neural networks
     !! L2 = L2 regularisation
     !! L2_decoupled = decoupled weight decay regularisation (AdamW)
     real(real32) :: l2 = 0.01_real32
     !! Regularisation parameter
     real(real32) :: l2_decoupled = 0.01_real32
     !! Decoupled weight decay regularisation parameter
     logical :: decoupled = .true.
     !! Use decoupled weight decay regularisation
   contains
     procedure, pass(this) :: regularise => regularise_l2
     !! Regularisation method
  end type l2_regulariser_type

  type, extends(base_regulariser_type) :: l1l2_regulariser_type
     !! Type for L1 and L2 regularisation
     real(real32) :: l1 = 0.01_real32
     !! L1 regularisation parameter
     real(real32) :: l2 = 0.01_real32
     !! L2 regularisation parameter
   contains
     procedure, pass(this) :: regularise => regularise_l1l2
     !! Regularisation method
  end type l1l2_regulariser_type



 contains

!###############################################################################
  pure subroutine regularise_l1(this, params, gradient, learning_rate)
    !! Regularise the parameters using L1 regularisation
    implicit none

    ! Arguments
    class(l1_regulariser_type), intent(in) :: this
    !! Instance of the L1 regulariser
    real(real32), dimension(:),  intent(in) :: params
    !! Parameters to regularise
    real(real32), dimension(:), intent(inout) :: gradient
    !! Gradient of the parameters
    real(real32), intent(in) :: learning_rate
    !! Learning rate

    gradient = gradient + learning_rate * this%l1 * sign(1._real32,params)

  end subroutine regularise_l1
!-------------------------------------------------------------------------------
  pure subroutine regularise_l2(this, params, gradient, learning_rate)
    !! Regularise the parameters using L2 regularisation
    implicit none

    ! Arguments
    class(l2_regulariser_type), intent(in) :: this
    !! Instance of the L2 regulariser
    real(real32), dimension(:),  intent(in) :: params
    !! Parameters to regularise
    real(real32), dimension(:), intent(inout) :: gradient
    !! Gradient of the parameters
    real(real32), intent(in) :: learning_rate
    !! Learning rate

    gradient = gradient + learning_rate * 2._real32 * this%l2 * params

  end subroutine regularise_l2
!-------------------------------------------------------------------------------
  pure subroutine regularise_l1l2(this, params, gradient, learning_rate)
    !! Regularise the parameters using L1 and L2 regularisation
    implicit none

    ! Arguments
    class(l1l2_regulariser_type), intent(in) :: this
    !! Instance of the L1 and L2 regulariser
    real(real32), dimension(:),  intent(in) :: params
    !! Parameters to regularise
    real(real32), dimension(:), intent(inout) :: gradient
    !! Gradient of the parameters
    real(real32), intent(in) :: learning_rate
    !! Learning rate

    gradient = gradient + learning_rate * &
         (this%l1 * sign(1._real32,params) + 2._real32 * this%l2 * params)

  end subroutine regularise_l1l2
!###############################################################################

end module athena__regulariser
