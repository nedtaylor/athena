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
  use learning_rate_decay, only: base_lr_decay_type
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
     class(base_lr_decay_type), allocatable :: lr_decay
     type(clip_type) :: clip_dict
   contains
     procedure, pass(this) :: init => init_base
     procedure, pass(this) :: init_gradients => init_gradients_base
     procedure, pass(this) :: minimise => minimise_base
  end type base_optimiser_type

  interface base_optimiser_type
     module function optimiser_setup_base( &
          learning_rate, &
          num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
        real(real12), optional, intent(in) :: learning_rate
        integer, optional, intent(in) :: num_params
        class(base_regulariser_type), optional, intent(in) :: regulariser
        type(clip_type), optional, intent(in) :: clip_dict
        class(base_lr_decay_type), optional, intent(in) :: lr_decay
        type(base_optimiser_type) :: optimiser
      end function optimiser_setup_base
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
     module function optimiser_setup_sgd( &
          learning_rate, momentum, &
          nesterov, num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
        real(real12), optional, intent(in) :: learning_rate, momentum
        logical, optional, intent(in) :: nesterov
        integer, optional, intent(in) :: num_params
        class(base_regulariser_type), optional, intent(in) :: regulariser
        type(clip_type), optional, intent(in) :: clip_dict
        class(base_lr_decay_type), optional, intent(in) :: lr_decay 
        type(sgd_optimiser_type) :: optimiser
     end function optimiser_setup_sgd
  end interface sgd_optimiser_type

!!!-----------------------------------------------------------------------------

  type, extends(base_optimiser_type) :: rmsprop_optimiser_type
     real(real12) :: beta = 0._real12
     real(real12) :: epsilon = 1.E-8_real12
     real(real12), allocatable, dimension(:) :: moving_avg
   contains
     procedure, pass(this) :: init_gradients => init_gradients_rmsprop
     procedure, pass(this) :: minimise => minimise_rmsprop
  end type rmsprop_optimiser_type

  interface rmsprop_optimiser_type
     module function optimiser_setup_rmsprop( &
          learning_rate, beta, &
          epsilon, num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
        real(real12), optional, intent(in) :: learning_rate, beta, epsilon
        integer, optional, intent(in) :: num_params
        class(base_regulariser_type), optional, intent(in) :: regulariser
        type(clip_type), optional, intent(in) :: clip_dict
        class(base_lr_decay_type), optional, intent(in) :: lr_decay 
        type(rmsprop_optimiser_type) :: optimiser
     end function optimiser_setup_rmsprop
  end interface rmsprop_optimiser_type

!!!-----------------------------------------------------------------------------

  type, extends(base_optimiser_type) :: adagrad_optimiser_type
     real(real12) :: epsilon = 1.E-8_real12
     real(real12), allocatable, dimension(:) :: sum_squares
   contains
     procedure, pass(this) :: init_gradients => init_gradients_adagrad
     procedure, pass(this) :: minimise => minimise_adagrad
  end type adagrad_optimiser_type

  interface adagrad_optimiser_type
     module function optimiser_setup_adagrad( &
          learning_rate, &
          epsilon, num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
        real(real12), optional, intent(in) :: learning_rate, epsilon
        integer, optional, intent(in) :: num_params
        class(base_regulariser_type), optional, intent(in) :: regulariser
        type(clip_type), optional, intent(in) :: clip_dict
        class(base_lr_decay_type), optional, intent(in) :: lr_decay
        type(adagrad_optimiser_type) :: optimiser
     end function optimiser_setup_adagrad
  end interface adagrad_optimiser_type

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
     module function optimiser_setup_adam( &
          learning_rate, &
          beta1, beta2, epsilon, &
          num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
        real(real12), optional, intent(in) :: learning_rate
        real(real12), optional, intent(in) :: beta1, beta2, epsilon
        integer, optional, intent(in) :: num_params
        class(base_regulariser_type), optional, intent(in) :: regulariser
        type(clip_type), optional, intent(in) :: clip_dict
        class(base_lr_decay_type), optional, intent(in) :: lr_decay
        type(adam_optimiser_type) :: optimiser
     end function optimiser_setup_adam
  end interface adam_optimiser_type

     !! reduce learning rate on plateau parameters
     !integer :: wait = 0
     !integer :: patience = 0
     !real(real12) :: factor = 0._real12
     !real(real12) :: min_learning_rate = 0._real12


  private

  public :: base_optimiser_type
  public :: sgd_optimiser_type
  public :: rmsprop_optimiser_type
  public :: adagrad_optimiser_type
  public :: adam_optimiser_type


contains

!!!#############################################################################
!!! set up optimiser
!!!#############################################################################
  module function optimiser_setup_base( &
      learning_rate, &
      num_params, &
      regulariser, clip_dict, lr_decay) result(optimiser)
    implicit none
    real(real12), optional, intent(in) :: learning_rate
    integer, optional, intent(in) :: num_params
    class(base_regulariser_type), optional, intent(in) :: regulariser
    type(clip_type), optional, intent(in) :: clip_dict
    class(base_lr_decay_type), optional, intent(in) :: lr_decay

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

    !! initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    end if

    !! initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
    
   end function optimiser_setup_base
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
  pure subroutine init_gradients_base(this, num_params)
    implicit none
    class(base_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params

    !allocate(this%velocity(num_params), source=0._real12)
    return
  end subroutine init_gradients_base
!!!#############################################################################


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine minimise_base(this, param, gradient)
    implicit none
    class(base_optimiser_type), intent(inout) :: this
    real(real12), dimension(:), intent(inout) :: param
    real(real12), dimension(:), intent(inout) :: gradient

    real(real12) :: learning_rate


    !! decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1

    !! update parameters
    param = param - learning_rate * gradient

  end subroutine minimise_base
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up optimiser
!!!#############################################################################
  module function optimiser_setup_sgd( &
       learning_rate, momentum, &
       nesterov, num_params, &
       regulariser, clip_dict, lr_decay) result(optimiser)
     implicit none
     real(real12), optional, intent(in) :: learning_rate, momentum
     logical, optional, intent(in) :: nesterov
     integer, optional, intent(in) :: num_params
     class(base_regulariser_type), optional, intent(in) :: regulariser
     type(clip_type), optional, intent(in) :: clip_dict
     class(base_lr_decay_type), optional, intent(in) :: lr_decay
     
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

    !! initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    end if

     !! initialise nesterov boolean
     if(present(nesterov)) optimiser%nesterov = nesterov
     
     !! initialise gradients
     if(present(num_params)) then
        num_params_ = num_params
     else
        num_params_ = 1
     end if
     call optimiser%init_gradients(num_params_)
  
  end function optimiser_setup_sgd
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
    real(real12), dimension(:), intent(inout) :: gradient

    real(real12) :: learning_rate


    !! decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1

    !! apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, gradient, learning_rate)

    if(this%momentum.gt.1.E-8_real12)then !! adaptive learning method
       this%velocity = this%momentum * this%velocity - &
            learning_rate * gradient
    else !! standard learning method
       this%velocity = - learning_rate * gradient
    end if

    !! update parameters
    if(this%nesterov)then
       param = param + this%momentum * this%velocity - &
            learning_rate * gradient
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
  module function optimiser_setup_rmsprop( &
      learning_rate, &
      beta, epsilon, &
      num_params, &
      regulariser, clip_dict, lr_decay) result(optimiser)
    implicit none
    real(real12), optional, intent(in) :: learning_rate
    real(real12), optional, intent(in) :: beta, epsilon
    integer, optional, intent(in) :: num_params
    class(base_regulariser_type), optional, intent(in) :: regulariser
    type(clip_type), optional, intent(in) :: clip_dict
    class(base_lr_decay_type), optional, intent(in) :: lr_decay
  
    type(rmsprop_optimiser_type) :: optimiser
  
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
  
    !! initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    end if

    !! initialise adam parameters
    if(present(beta)) optimiser%beta = beta
    if(present(epsilon)) optimiser%epsilon = epsilon
  
    !! initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
  
  end function optimiser_setup_rmsprop
!!!#############################################################################


!!!#############################################################################
!!! initialise gradients
!!!#############################################################################
  pure subroutine init_gradients_rmsprop(this, num_params)
    implicit none
    class(rmsprop_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params
  
  
    !! initialise gradients
    if(allocated(this%moving_avg)) deallocate(this%moving_avg)
    allocate(this%moving_avg(num_params), source=0._real12) !1.E-8_real12)
    
  end subroutine init_gradients_rmsprop
!!!#############################################################################


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine minimise_rmsprop(this, param, gradient)
    implicit none
    class(rmsprop_optimiser_type), intent(inout) :: this
    real(real12), dimension(:), intent(inout) :: param
    real(real12), dimension(:), intent(inout) :: gradient

    real(real12) :: learning_rate


    !! decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1
  
    !! apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, gradient, learning_rate)

    this%moving_avg = this%beta * this%moving_avg + &
         (1._real12 - this%beta) * gradient ** 2._real12 

    !! update parameters
    param = param - learning_rate * gradient / &
         (sqrt(this%moving_avg + this%epsilon))
  
  end subroutine minimise_rmsprop
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up optimiser
!!!#############################################################################
  module function optimiser_setup_adagrad( &
      learning_rate, &
      epsilon, &
      num_params, &
      regulariser, clip_dict, lr_decay) result(optimiser)
    implicit none
    real(real12), optional, intent(in) :: learning_rate
    real(real12), optional, intent(in) :: epsilon
    integer, optional, intent(in) :: num_params
    class(base_regulariser_type), optional, intent(in) :: regulariser
    type(clip_type), optional, intent(in) :: clip_dict
    class(base_lr_decay_type), optional, intent(in) :: lr_decay
  
    type(adagrad_optimiser_type) :: optimiser
  
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
  
    !! initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    end if

    !! initialise adam parameters
    if(present(epsilon)) optimiser%epsilon = epsilon
  
    !! initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
  
  end function optimiser_setup_adagrad
!!!#############################################################################


!!!#############################################################################
!!! initialise gradients
!!!#############################################################################
  pure subroutine init_gradients_adagrad(this, num_params)
    implicit none
    class(adagrad_optimiser_type), intent(inout) :: this
    integer, intent(in) :: num_params
  
  
    !! initialise gradients
    if(allocated(this%sum_squares)) deallocate(this%sum_squares)
    allocate(this%sum_squares(num_params), source=0._real12) !1.E-8_real12)
    
  end subroutine init_gradients_adagrad
!!!#############################################################################


!!!#############################################################################
!!! minimise the loss function by applying gradients to the parameters
!!!#############################################################################
  pure subroutine minimise_adagrad(this, param, gradient)
    implicit none
    class(adagrad_optimiser_type), intent(inout) :: this
    real(real12), dimension(:), intent(inout) :: param
    real(real12), dimension(:), intent(inout) :: gradient

    real(real12) :: learning_rate


    !! decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1
  
    !! apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, this%sum_squares, learning_rate)

    this%sum_squares = this%sum_squares + gradient ** 2._real12 

    !! update parameters
    param = param - learning_rate * gradient / &
         (sqrt(this%sum_squares + this%epsilon))

  end subroutine minimise_adagrad
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up optimiser
!!!#############################################################################
  module function optimiser_setup_adam( &
      learning_rate, &
      beta1, beta2, epsilon, &
      num_params, &
      regulariser, clip_dict, lr_decay) result(optimiser)
    implicit none
    real(real12), optional, intent(in) :: learning_rate
    real(real12), optional, intent(in) :: beta1, beta2, epsilon
    integer, optional, intent(in) :: num_params
    class(base_regulariser_type), optional, intent(in) :: regulariser
    type(clip_type), optional, intent(in) :: clip_dict
    class(base_lr_decay_type), optional, intent(in) :: lr_decay

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

    !! initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    end if

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

  end function optimiser_setup_adam
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
    real(real12), dimension(:), intent(inout) :: gradient

    real(real12) :: learning_rate


    !! decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1

    !! apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, gradient, learning_rate)
    
    !! adaptive learning method
    this%m = this%beta1 * this%m + &
         (1._real12 - this%beta1) * gradient
    this%v = this%beta2 * this%v + &
         (1._real12 - this%beta2) * gradient ** 2._real12
    
    !! update parameters
    associate( &
         m_hat => this%m / (1._real12 - this%beta1**this%iter), &
         v_hat => this%v / (1._real12 - this%beta2**this%iter) )
       select type(regulariser => this%regulariser)
       type is (l2_regulariser_type)
          select case(regulariser%decoupled)
          case(.true.)
             param = param - &
                  learning_rate * &
                  ( m_hat / (sqrt(v_hat) + this%epsilon) + &
                  regulariser%l2 * param )
          case(.false.)
             param = param + &
                   learning_rate * &
                   ( ( m_hat + regulariser%l2 * param ) / &
                   (sqrt(v_hat) + this%epsilon) )
          end select
       class default
          param = param + &
               learning_rate * ( ( m_hat + param ) / &
               (sqrt(v_hat) + this%epsilon) )
       end select
    end associate
    
  end subroutine minimise_adam
!!!#############################################################################

end module optimiser
!!!#############################################################################

