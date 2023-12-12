!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module optimiser
  use constants, only: real12
  use clipper, only: clip_type
  use regulariser, only: &
       base_regulariser_type, &
       l2_regulariser_type
  implicit none

!!!-----------------------------------------------------------------------------
!!! learning parameter type
!!!-----------------------------------------------------------------------------
  type :: base_optimiser_type !!base_optimiser_type
     !! iter = iteration number
     !! learning_rate = learning rate hyperparameter
     !! regularisation = apply regularisation
     !! regulariser = regularisation method
     !! clip_dict = clipping dictionary
     integer :: iter = 0
     real(real12) :: learning_rate = 0.01_real12
     logical :: regularisation = .false.
     class(base_regulariser_type), allocatable :: regulariser
     type(clip_type) :: clip_dict
   contains
     procedure, pass(this) :: init => init_base
     procedure, pass(this) :: init_gradients
     procedure, pass(this) :: minimise
  end type base_optimiser_type

  interface base_optimiser_type
     module function base_optimiser_setup( &
          learning_rate, &
          num_params, &
          regulariser, clip_dict) result(optimiser)
        real(real12), optional, intent(in) :: learning_rate
        integer, optional, intent(in) :: num_params
        class(base_regulariser_type), optional, intent(in) :: regulariser
        type(clip_type), optional, intent(in) :: clip_dict
        type(base_optimiser_type) :: optimiser
      end function base_optimiser_setup
  end interface base_optimiser_type


!!!-----------------------------------------------------------------------------

  type, extends(base_optimiser_type) :: sgd_optimiser_type
     logical :: nesterov = .false.
     real(real12) :: momentum = 0._real12  ! fraction of momentum based learning
     real(real12), allocatable, dimension(:) :: velocity
   contains
     procedure, pass(this) :: init_gradients => init_gradients_sgd
     procedure, pass(this) :: minimise => minimise_sgd
  end type sgd_optimiser_type

  interface sgd_optimiser_type
     module function sgd_optimiser_setup( &
          learning_rate, momentum, &
          nesterov, num_params, &
          regulariser, clip_dict) result(optimiser)
        real(real12), optional, intent(in) :: learning_rate, momentum
        logical, optional, intent(in) :: nesterov
        integer, optional, intent(in) :: num_params
        class(base_regulariser_type), optional, intent(in) :: regulariser
        type(clip_type), optional, intent(in) :: clip_dict  
        type(sgd_optimiser_type) :: optimiser    
     end function sgd_optimiser_setup
  end interface sgd_optimiser_type

  !!!-----------------------------------------------------------------------------

  type, extends(base_optimiser_type) :: adam_optimiser_type
     real(real12) :: beta1 = 0.9_real12
     real(real12) :: beta2 = 0.999_real12
     real(real12) :: epsilon = 1.E-8_real12
     real(real12), allocatable, dimension(:) :: m
     real(real12), allocatable, dimension(:) :: v
   contains
     procedure, pass(this) :: init_gradients => init_gradients_adam
     procedure, pass(this) :: minimise => minimise_adam
  end type adam_optimiser_type

  interface adam_optimiser_type
     module function adam_optimiser_setup( &
          learning_rate, &
          beta1, beta2, epsilon, &
          num_params, &
          regulariser, clip_dict) result(optimiser)
        real(real12), optional, intent(in) :: learning_rate
        real(real12), optional, intent(in) :: beta1, beta2, epsilon
        integer, optional, intent(in) :: num_params
        class(base_regulariser_type), optional, intent(in) :: regulariser
        type(clip_type), optional, intent(in) :: clip_dict  
        type(adam_optimiser_type) :: optimiser    
     end function adam_optimiser_setup
  end interface adam_optimiser_type

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


  private

  public :: base_optimiser_type
  public :: sgd_optimiser_type
  public :: adam_optimiser_type


contains

!!!#############################################################################
!!! set up optimiser
!!!#############################################################################
  module function base_optimiser_setup( &
      learning_rate, &
      num_params, &
      regulariser, clip_dict) result(optimiser)
    implicit none
    real(real12), optional, intent(in) :: learning_rate
    integer, optional, intent(in) :: num_params
    class(base_regulariser_type), optional, intent(in) :: regulariser
    type(clip_type), optional, intent(in) :: clip_dict

    type(base_optimiser_type) :: optimiser

    integer :: num_params_

  
    !! apply regularisation
    if(present(regulariser))then
       optimiser%regularisation = .true.
       if(allocated(optimiser%regulariser)) deallocate(optimiser%regulariser)
       allocate(optimiser%regulariser, source = regulariser)
    end if

    !! apply clipping
    if(present(clip_dict)) optimiser%clip_dict = clip_dict

    !! initialise general optimiser parameters
    if(present(learning_rate)) optimiser%learning_rate = learning_rate

    !! initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
    
   end function base_optimiser_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise optimiser
!!!#############################################################################
  subroutine init_base(this, num_params, regulariser, clip_dict)
    implicit none
    class(base_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params
    class(base_regulariser_type), optional, intent(in) :: regulariser
    type(clip_type), optional, intent(in) :: clip_dict
  
  
    !! apply regularisation
    if(present(regulariser))then
       this%regularisation = .true.
       if(allocated(this%regulariser)) deallocate(this%regulariser)
       allocate(this%regulariser, source = regulariser)
    end if

    !! apply clipping
    if(present(clip_dict)) then
       this%clip_dict = clip_dict
    end if

    !! initialise gradients
    call this%init_gradients(num_params)
    
  end subroutine init_base
!!!#############################################################################


!!!#############################################################################
!!! initialise gradients
!!!#############################################################################
  pure subroutine init_gradients(this, num_params)
    implicit none
    class(base_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params

    !allocate(this%velocity(num_params), source=0._real12)
    return
  end subroutine init_gradients
!!!#############################################################################


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine minimise(this, param, gradient)
    implicit none
    class(base_optimiser_type), intent(inout) :: this
    real(real12), dimension(:), intent(inout) :: param
    real(real12), dimension(:), intent(in) :: gradient

    !! update iteration
    this%iter = this%iter + 1

    !! update parameters
    param = param - this%learning_rate * gradient

  end subroutine minimise
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up optimiser
!!!#############################################################################
  module function sgd_optimiser_setup( &
       learning_rate, momentum, &
       nesterov, num_params, &
       regulariser, clip_dict) result(optimiser)
     implicit none
     real(real12), optional, intent(in) :: learning_rate, momentum
     logical, optional, intent(in) :: nesterov
     integer, optional, intent(in) :: num_params
     class(base_regulariser_type), optional, intent(in) :: regulariser
     type(clip_type), optional, intent(in) :: clip_dict
     
     type(sgd_optimiser_type) :: optimiser
     
     integer :: num_params_
     
     
     !! apply regularisation
     if(present(regulariser))then
        optimiser%regularisation = .true.
        if(allocated(optimiser%regulariser)) deallocate(optimiser%regulariser)
        allocate(optimiser%regulariser, source = regulariser)
     end if
      
     !! apply clipping
     if(present(clip_dict)) optimiser%clip_dict = clip_dict
     
     !! initialise general optimiser parameters
     if(present(learning_rate)) optimiser%learning_rate = learning_rate
     if(present(momentum)) optimiser%momentum = momentum

     !! initialise nesterov boolean
     if(present(nesterov)) optimiser%nesterov = nesterov
     
     !! initialise gradients
     if(present(num_params)) then
        num_params_ = num_params
     else
        num_params_ = 1
     end if
     call optimiser%init_gradients(num_params_)
  
  end function sgd_optimiser_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise gradients
!!!#############################################################################
  pure subroutine init_gradients_sgd(this, num_params)
    implicit none
    class(sgd_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params
  

    !! initialise gradients
    if(allocated(this%velocity)) deallocate(this%velocity)
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
!!! set up optimiser
!!!#############################################################################
  module function adam_optimiser_setup( &
       learning_rate, &
       beta1, beta2, epsilon, &
       num_params, &
       regulariser, clip_dict) result(optimiser)
     implicit none
     real(real12), optional, intent(in) :: learning_rate
     real(real12), optional, intent(in) :: beta1, beta2, epsilon
     integer, optional, intent(in) :: num_params
     class(base_regulariser_type), optional, intent(in) :: regulariser
     type(clip_type), optional, intent(in) :: clip_dict  
    
     type(adam_optimiser_type) :: optimiser
     
     integer :: num_params_
     
     
     !! apply regularisation
     if(present(regulariser))then
        optimiser%regularisation = .true.
        if(allocated(optimiser%regulariser)) deallocate(optimiser%regulariser)
        allocate(optimiser%regulariser, source = regulariser)
     end if
     
     !! apply clipping
     if(present(clip_dict)) optimiser%clip_dict = clip_dict

     !! initialise general optimiser parameters
     if(present(learning_rate)) optimiser%learning_rate = learning_rate

     !! initialise adam parameters
     if(present(beta1)) optimiser%beta1 = beta1
     if(present(beta2)) optimiser%beta2 = beta2
     if(present(epsilon)) optimiser%epsilon = epsilon
     
     !! initialise gradients
     if(present(num_params)) then
        num_params_ = num_params
     else
        num_params_ = 1
     end if
     call optimiser%init_gradients(num_params_)
  
  end function adam_optimiser_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise gradients
!!!#############################################################################
  pure subroutine init_gradients_adam(this, num_params)
    implicit none
    class(adam_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params
  
  
    !! initialise gradients
    if(allocated(this%m)) deallocate(this%m)
    if(allocated(this%v)) deallocate(this%v)
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
       select type(regulariser => this%regulariser)
       type is (l2_regulariser_type)
          select case(regulariser%decoupled)
          case(.true.)
             param = param - &
                  this%learning_rate * &
                  ( m_hat / (sqrt(v_hat) + this%epsilon) ) - &
                  regulariser%l2 * param
          case(.false.)
             param = param + &
                   this%learning_rate * &
                   ( ( m_hat + regulariser%l2 * param ) / &
                   (sqrt(v_hat) + this%epsilon) )
          end select
       class default
          param = param + &
               this%learning_rate * ( ( m_hat + param ) / &
               (sqrt(v_hat) + this%epsilon) )
       end select
    end associate
    
  end subroutine minimise_adam
!!!#############################################################################

end module optimiser
!!!#############################################################################

