!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module optimiser
  use constants, only: real12
  use clipper, only: clip_type
  use regulariser, only: base_regulariser_type
  implicit none

!!!-----------------------------------------------------------------------------
!!! learning parameter type
!!!-----------------------------------------------------------------------------
!!! MAKE THIS AN ABSTRACT TYPE WITH EXTENDED DERIVED TYPES FOR ADAM AND MOMENTUM
!!! THEN MAKE REGULARISATION A SUBTYPE
!!! NO, DON'T MAKE IT AN ABSTRACT TYPE, JUST MAKE IT A TYPE THAT HAS DERIVED TYPES FROM IT
  type, abstract :: base_optimiser_type !!base_optimiser_type
     character(:), allocatable :: method
     !! reduce learning rate on plateau parameters
     !integer :: wait = 0
     !integer :: patience = 0
     !real(real12) :: factor = 0._real12
     !real(real12) :: min_learning_rate = 0._real12
     !! step decay parameters
     !real(real12) :: decay_rate = 0._real12
     !real(real12) :: decay_steps = 0._real12
     !! adam optimiser parameters
     !real(real12) :: weight_decay  ! L2 regularisation on Adam (AdamW)

     integer :: iter = 0
     real(real12) :: learning_rate = 0.01_real12
     !! momentum parameters
     real(real12) :: momentum = 0._real12  ! fraction of momentum based learning
     logical :: regularisation = .false.
     class(base_regulariser_type), allocatable :: regulariser
     type(clip_type) :: clip_dict
   contains
     procedure, pass(this) :: init => init_base
     procedure(init_gradients), deferred, pass(this) :: init_gradients
     procedure(minimise), deferred, pass(this) :: minimise
  end type base_optimiser_type

  abstract interface
     pure subroutine init_gradients(this, num_params)
        import base_optimiser_type, base_regulariser_type, clip_type
        class(base_optimiser_type), intent(inout) :: this
        integer, intent(in) :: num_params
     end subroutine init_gradients

     pure subroutine minimise(this, param, gradient)
        import base_optimiser_type, real12
        class(base_optimiser_type), intent(inout) :: this
        real(real12), dimension(:), intent(inout) :: param
        real(real12), dimension(:), intent(in) :: gradient
     end subroutine minimise
  end interface

  type, extends(base_optimiser_type) :: sgd_optimiser_type
     logical :: nesterov = .false.
     real(real12), allocatable, dimension(:) :: velocity
   contains
     procedure, pass(this) :: init_gradients => init_gradients_sgd
     procedure, pass(this) :: minimise => minimise_sgd
  end type sgd_optimiser_type

  type, extends(base_optimiser_type) :: adam_optimiser_type
     real(real12) :: beta1 = 0.9_real12
     real(real12) :: beta2 = 0.999_real12
     real(real12) :: epsilon = 1.E-8_real12
     real(real12) :: weight_decay = 0._real12  ! L2 regularisation on Adam
     real(real12) :: weight_decay_decoupled = 0._real12  ! decoupled weight decay regularisation (AdamW)
     real(real12), allocatable, dimension(:) :: m
     real(real12), allocatable, dimension(:) :: v
   contains
     procedure, pass(this) :: init_gradients => init_gradients_adam
     procedure, pass(this) :: minimise => minimise_adam
  end type adam_optimiser_type


  private

  public :: base_optimiser_type
  public :: sgd_optimiser_type
  public :: adam_optimiser_type


contains

!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine init_base(this, num_params, regulariser, clip_dict)
    implicit none
    class(base_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params
    class(base_regulariser_type), optional, intent(in) :: regulariser
    type(clip_type), optional, intent(in) :: clip_dict
  
  
    !! apply regularisation
    if(present(regulariser)) this%regularisation = .true.
    if(this%regularisation) then
       allocate(this%regulariser, source=regulariser)
    end if

    !! apply clipping
    if(present(clip_dict)) then
       this%clip_dict = clip_dict
    end if

    !! initialise gradients
    call this%init_gradients(num_params)
    
  end subroutine init_base
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine init_gradients_sgd(this, num_params)
    implicit none
    class(sgd_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params
  

    !! initialise gradients
    allocate(this%velocity(num_params), source=0._real12)
    
  end subroutine init_gradients_sgd
!!!#############################################################################


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine minimise_sgd(this, param, gradient)
    implicit none
    class(sgd_optimiser_type), intent(inout) :: this
    real(real12), dimension(:), intent(inout) :: param
    real(real12), dimension(:), intent(in) :: gradient
  
  
    if(this%momentum.gt.1.E-8_real12)then !! adaptive learning method
       this%velocity = this%momentum * this%velocity - &
            this%learning_rate * gradient
    else !! standard learning method
        this%velocity = - this%learning_rate * gradient
    end if
  
    !! apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, this%velocity, this%learning_rate)
  
    !! update parameters
    if(this%nesterov)then
       param = param + this%momentum * this%velocity - &
            this%learning_rate * gradient
    else
       param = param + this%velocity
    end if
  
  end subroutine minimise_sgd
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine init_gradients_adam(this, num_params)
    implicit none
    class(adam_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params
  
  
    !! initialise gradients
    allocate(this%m(num_params), source=0._real12)
    allocate(this%v(num_params), source=0._real12)
    
  end subroutine init_gradients_adam
!!!#############################################################################


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine minimise_adam(this, param, gradient)
    implicit none
    class(adam_optimiser_type), intent(inout) :: this
    real(real12), dimension(:), intent(inout) :: param
    real(real12), dimension(:), intent(in) :: gradient
    
    real(real12), dimension(size(gradient,1)) :: velocity
    

    !! update iteration
    this%iter = this%iter + 1

    !! set up gradient temporary store
    velocity = gradient

    !! apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, velocity, this%learning_rate)
    
    !! adaptive learning method
    this%m = this%beta1 * this%m + &
         (1._real12 - this%beta1) * velocity
    this%v = this%beta2 * this%v + &
         (1._real12 - this%beta2) * velocity ** 2._real12
    
    !! update parameters
    associate( &
      m_hat => this%m / (1._real12 - this%beta1**this%iter), &
      v_hat => this%v / (1._real12 - this%beta2**this%iter) )
       param = param + &
            this%learning_rate * ( m_hat / (sqrt(v_hat) + this%epsilon) ) + &
            this%weight_decay_decoupled * param
    end associate
    
  end subroutine minimise_adam
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine optimise(this, param, gradient)
  implicit none
  class(base_optimiser_type), intent(inout) :: this
  real(real12), dimension(:), intent(inout) :: param
  real(real12), dimension(:), intent(in) :: gradient

  real(real12) :: t_learning_rate
  real(real12), allocatable, dimension(:) :: lr_gradient

 !  lr_gradient = this%learning_rate * gradient

 !  !! adaptive learning method
 !  select case(this%method(1:1))
 !  case('m')!'momentum')
 !     !! momentum-based learning
 !     !! w = w - vel - lr * g
 !     this%velocity = lr_gradient + &
 !          this%momentum * this%velocity
 !  case('n')!('nesterov')
 !     !! nesterov momentum
 !     this%velocity = - this%momentum * this%velocity - &
 !          lr_gradient
 !  !case('a')!('adam')
 !  !   !! adam optimiser
 !  !   t_learning_rate = learning_rate
 !  !   call adam_optimiser(t_learning_rate, gradient, m, v)
 !  !   this%velocity = t_learning_rate
 !  case default
 !     this%velocity = lr_gradient
 !  end select

 !  !! regularisation
 !  if(this%regularise)then
 !     select case(this%regularisation)
 !     case('l1l2')
 !        !! L1L2 regularisation
 !        this%velocity = this%velocity + this%learning_rate * ( &
 !             this%l1 * sign(1._real12,param) + &
 !             2._real12 * this%l2 * param )
 !     case('l1')
 !        !! L1 regularisation
 !        this%velocity = this%velocity + this%learning_rate * &
 !             this%l1 * sign(1._real12,param)
 !     case('l2')
 !        !! L2 regularisation
 !        this%velocity = this%velocity + this%learning_rate * &
 !             2._real12 * this%l2 * param
 !     end select
 !  end if

 !  select case(this%method(1:1))
 !  case('n')!'nesterov')
 !     param = param + this%momentum * this%velocity - &
 !          lr_gradient
 !  case default
 !     param = param - this%velocity
 !  end select


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
!!!! m = first moment (m = mean of gradient)
!!!! v = second moment (v = variance of the gradient)
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

