!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains regularisation methods and associated derived types
!!! module contains the following derived types:
!!! - base_regulariser_type - abstract base regulariser type
!!! - l1_regulariser_type   - L1 regulariser type
!!! - l2_regulariser_type   - L2 regulariser type
!!! - l1l2_regulariser_type - L1L2 regulariser type
!!!##################
!!! the base_regulariser_type contains the following deferred procedure:
!!! - regularise - regularise the gradient
!!!############################################################################# 
module regulariser
  use constants, only: real12
  implicit none


!!!-----------------------------------------------------------------------------
!!! regularise type
!!!-----------------------------------------------------------------------------
  type, abstract :: base_regulariser_type
   contains
     procedure(regularise), deferred, pass(this) :: regularise
  end type base_regulariser_type

  abstract interface
    pure subroutine regularise(this, params, gradient, learning_rate)
      import :: base_regulariser_type, real12
      class(base_regulariser_type), intent(in) :: this
      real(real12), dimension(:),  intent(in) :: params
      real(real12), dimension(:), intent(inout) :: gradient
      real(real12), intent(in) :: learning_rate
    end subroutine regularise
  end interface

  !! Lasso regression
  !! attempts to prevent overfitting
  type, extends(base_regulariser_type) :: l1_regulariser_type
     real(real12) :: l1 = 0.01_real12
   contains
     procedure, pass(this) :: regularise => regularise_l1
  end type l1_regulariser_type

  !! Ridge regression
  !! attempts to prevent overfitting
  type, extends(base_regulariser_type) :: l2_regulariser_type
     !! l2           = L2 regularisation
     !! l2_decoupled = decoupled weight decay regularisation (AdamW)
     real(real12) :: l2 = 0.01_real12
     real(real12) :: l2_decoupled = 0.01_real12
     logical :: decoupled = .true.
   contains
     procedure, pass(this) :: regularise => regularise_l2
  end type l2_regulariser_type

  type, extends(base_regulariser_type) :: l1l2_regulariser_type
     real(real12) :: l1 = 0.01_real12
     real(real12) :: l2 = 0.01_real12
   contains
     procedure, pass(this) :: regularise => regularise_l1l2
  end type l1l2_regulariser_type


  private

  public :: base_regulariser_type
  public :: l1_regulariser_type
  public :: l2_regulariser_type
  public :: l1l2_regulariser_type


 contains

!!!#############################################################################
!!! regularise
!!!#############################################################################
  pure subroutine regularise_l1(this, params, gradient, learning_rate)
    class(l1_regulariser_type), intent(in) :: this
    real(real12), dimension(:),  intent(in) :: params
    real(real12), dimension(:), intent(inout) :: gradient
    real(real12), intent(in) :: learning_rate

    gradient = gradient + learning_rate * this%l1 * sign(1._real12,params)

  end subroutine regularise_l1
!!!-----------------------------------------------------------------------------
  pure subroutine regularise_l2(this, params, gradient, learning_rate)
    class(l2_regulariser_type), intent(in) :: this
    real(real12), dimension(:),  intent(in) :: params
    real(real12), dimension(:), intent(inout) :: gradient
    real(real12), intent(in) :: learning_rate

    gradient = gradient + learning_rate * 2._real12 * this%l2 * params

  end subroutine regularise_l2
!!!-----------------------------------------------------------------------------
  pure subroutine regularise_l1l2(this, params, gradient, learning_rate)
    class(l1l2_regulariser_type), intent(in) :: this
    real(real12), dimension(:),  intent(in) :: params
    real(real12), dimension(:), intent(inout) :: gradient
    real(real12), intent(in) :: learning_rate

    gradient = gradient + learning_rate * &
         (this%l1 * sign(1._real12,params) + 2._real12 * this%l2 * params)

  end subroutine regularise_l1l2
!!!#############################################################################

end module regulariser