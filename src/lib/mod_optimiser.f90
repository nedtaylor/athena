!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module optimiser
  use constants, only: real12
  use clipper, only: clip_type
  implicit none

!!!------------------------------------------------------------------------
!!! regularise type
!!!------------------------------------------------------------------------
  !type regularisation_type
  !   class(*), allocatable :: l1, l2
  !end type regularisation_type

!!!-----------------------------------------------------------------------------
!!! learning parameter type
!!!-----------------------------------------------------------------------------
!!! MAKE THIS AN ABSTRACT TYPE WITH EXTENDED DERIVED TYPES FOR ADAM AND MOMENTUM
!!! THEN MAKE REGULARISATION A SUBTYPE
!!! NO, DON'T MAKE IT AN ABSTRACT TYPE, JUST MAKE IT A TYPE THAT HAS DERIVED TYPES FROM IT
  type optimiser_type
     character(:), allocatable :: method
     !! reduce learning rate on plateau parameters
     !integer :: wait = 0
     !integer :: patience = 0
     !real(real12) :: factor = 0._real12
     !real(real12) :: min_learning_rate = 0._real12
     real(real12) :: learning_rate
     integer :: iter
     !! momentum parameters
     real(real12) :: momentum = 0._real12  ! fraction of momentum based learning
     !! step decay parameters
     !real(real12) :: decay_rate = 0._real12
     !real(real12) :: decay_steps = 0._real12
     !! adam optimiser parameters
     real(real12) :: beta1 = 0._real12
     real(real12) :: beta2 = 0._real12
     real(real12) :: epsilon = 0._real12
     !real(real12) :: weight_decay  ! L2 regularisation on Adam (AdamW)
     logical :: regularise = .false.
     character(:), allocatable :: regularisation
     real(real12) :: l1 = 0._real12
     real(real12) :: l2 = 0._real12
     real(real12), allocatable, dimension(:) :: vector
     type(clip_type) :: clip_dict
   contains
     procedure, pass(this) :: optimise
     procedure, pass(this) :: minimise => optimise
     !procedure, private, pass(this) :: adam
  end type optimiser_type


  private

  public :: optimiser_type


contains

!!!#############################################################################
!!! 
!!!#############################################################################
  pure subroutine optimise(this, params, gradients)
    implicit none
    class(optimiser_type), intent(inout) :: this
    real(real12), dimension(:), intent(inout) :: params
    real(real12), dimension(:), intent(in) :: gradients

    real(real12) :: t_learning_rate
    real(real12), allocatable, dimension(:) :: lr_gradients

    lr_gradients = this%learning_rate * gradients

    !! adaptive learning method
    select case(this%method(1:1))
    case('m')!'momentum')
       !! momentum-based learning
       !! w = w - vel - lr * g
       this%vector = lr_gradients + &
            this%momentum * this%vector
    case('n')!('nesterov')
       !! nesterov momentum
       this%vector = - this%momentum * this%vector - &
            lr_gradients
    !case('a')!('adam')
    !   !! adam optimiser
    !   t_learning_rate = learning_rate
    !   call adam_optimiser(t_learning_rate, gradients, m, v)
    !   this%vector = t_learning_rate
    case default
       this%vector = lr_gradients
    end select

    !! regularisation
    if(this%regularise)then
       select case(this%regularisation)
       case('l1l2')
          !! L1L2 regularisation
          this%vector = this%vector + this%learning_rate * ( &
               this%l1 * sign(1._real12,params) + &
               2._real12 * this%l2 * params )
       case('l1')
          !! L1 regularisation
          this%vector = this%vector + this%learning_rate * &
               this%l1 * sign(1._real12,params)
       case('l2')
          !! L2 regularisation
          this%vector = this%vector + this%learning_rate * &
               2._real12 * this%l2 * params
       end select
    end if

    select case(this%method(1:1))
    case('n')!'nesterov')
       params = params + this%momentum * this%vector - &
            lr_gradients
    case default
       params = params - this%vector
    end select


  end subroutine optimise
!!!#############################################################################


!!!!############################################################################
!!!! adaptive learning rate
!!!! method: adam optimiser
!!!!         ... Adaptive Moment Estimation
!!!!############################################################################
!!!! learning_rate = initial learning rate hyperparameter
!!!! beta1 = exponential decay rate for first-moment estimates
!!!! beta2 = exponential decay rate for second-moment estimates
!!!! epsilon = small number for numerical stability
!!!! t = current iteration
!!!! m = first moment (m = mean of gradients)
!!!! v = second moment (v = variance of the gradients)
!  elemental subroutine adam(this, learning_rate, &
!       gradient, m, v)
!    implicit none
!    class(optimiser_type), intent(in) :: this
!    real(real12), intent(inout) :: learning_rate
!    real(real12), intent(in) :: gradient
!    real(real12), intent(inout) :: m, v
!
!    real :: m_norm, v_norm
!
!    !! update biased first moment estimate
!    m = this%beta1 * m + (1._real12 - this%beta1) * gradient
!
!    !! update biased second moment estimate
!    v = this%beta2 * v + (1._real12 - this%beta2) * gradient**2
!
!    !! normalised first moment estimate
!    m_norm = m / (1._real12 - this%beta1**this%iter)
!
!    !! normalised second moment estimate
!    v_norm = v / (1._real12 - this%beta2**this%iter)
!    
!    !! update learning rate
!    learning_rate = learning_rate * m_norm / (sqrt(v_norm) + this%epsilon)
!
!  end subroutine adam
!!!!############################################################################

  
  
  
end module optimiser
!!!#############################################################################

