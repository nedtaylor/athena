module athena__optimiser
  !! Module containing implementations of optimisation methods
  !!
  !! This module contains implementations of optimisation methods used to
  !! minimise the loss function of a neural network.
  !! Attribution statement:
  !! The following module is based on code from the neural-fortran library
  !! https://github.com/modern-fortran/neural-fortran/blob/main/src/nf/nf_optimizers.f90
  !! The implementation of optimiser_base_type is based on the ...
  !! ... optimizer_base_type from the neural-fortran library
  !! The same applies to the implementation of the sgd_optimiser_type, ...
  !! ... rmsprop_optimiser_type, adagrad_optimiser_type, and adam_optimiser_type
  use athena__constants, only: real32
  use athena__clipper, only: clip_type
  use athena__regulariser, only: base_regulariser_type, l2_regulariser_type
  use athena__learning_rate_decay, only: base_lr_decay_type
  implicit none


  private

  public :: base_optimiser_type
  public :: sgd_optimiser_type
  public :: rmsprop_optimiser_type
  public :: adagrad_optimiser_type
  public :: adam_optimiser_type


!-------------------------------------------------------------------------------

  type :: base_optimiser_type
     !! Base optimiser type
     integer :: iter = 0
     !! Iteration number
     real(real32) :: learning_rate = 0.01_real32
     !! Learning rate hyperparameter
     logical :: regularisation = .false.
     !! Apply regularisation
     class(base_regulariser_type), allocatable :: regulariser
     !! Regularisation method
     class(base_lr_decay_type), allocatable :: lr_decay
     !! Learning rate decay method
     type(clip_type) :: clip_dict
     !! Clipping dictionary
   contains
     procedure, pass(this) :: init => init_base
     !! Initialise base optimiser
     procedure, pass(this) :: init_gradients => init_gradients_base
     !! Initialise gradients
     procedure, pass(this) :: minimise => minimise_base
     !! Apply gradients to parameters to minimise loss using base optimiser
  end type base_optimiser_type

  interface base_optimiser_type
     !! Interface for setting up the base optimiser
     module function optimiser_setup_base( &
          learning_rate, num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
       !! Set up the base optimiser
       real(real32), optional, intent(in) :: learning_rate
       !! Learning rate
       integer, optional, intent(in) :: num_params
       !! Number of parameters
       class(base_regulariser_type), optional, intent(in) :: regulariser
       !! Regularisation method
       type(clip_type), optional, intent(in) :: clip_dict
       !! Clipping dictionary
       class(base_lr_decay_type), optional, intent(in) :: lr_decay
       !! Learning rate decay method
       type(base_optimiser_type) :: optimiser
       !! Instance of the base optimiser
     end function optimiser_setup_base
  end interface base_optimiser_type

!-------------------------------------------------------------------------------

  type, extends(base_optimiser_type) :: sgd_optimiser_type
     !! Stochastic gradient descent optimiser type
     logical :: nesterov = .false.
     !! Nesterov momentum
     real(real32) :: momentum = 0._real32
     !! Fraction of momentum-based learning
     real(real32), allocatable, dimension(:) :: velocity
     !! Velocity for momentum
   contains
     procedure, pass(this) :: init_gradients => init_gradients_sgd
     !! Initialise gradients for SGD
     procedure, pass(this) :: minimise => minimise_sgd
     !! Apply gradients to parameters to minimise loss using SGD optimiser
  end type sgd_optimiser_type

  interface sgd_optimiser_type
     !! Interface for setting up the SGD optimiser
     module function optimiser_setup_sgd( &
          learning_rate, momentum, &
          nesterov, num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
       !! Set up the SGD optimiser
       real(real32), optional, intent(in) :: learning_rate, momentum
       !! Learning rate and momentum
       logical, optional, intent(in) :: nesterov
       !! Nesterov momentum
       integer, optional, intent(in) :: num_params
       !! Number of parameters
       class(base_regulariser_type), optional, intent(in) :: regulariser
       !! Regularisation method
       type(clip_type), optional, intent(in) :: clip_dict
       !! Clipping dictionary
       class(base_lr_decay_type), optional, intent(in) :: lr_decay
       !! Learning rate decay method
       type(sgd_optimiser_type) :: optimiser
       !! Instance of the SGD optimiser
     end function optimiser_setup_sgd
  end interface sgd_optimiser_type

!-------------------------------------------------------------------------------

  type, extends(base_optimiser_type) :: rmsprop_optimiser_type
     !! RMSprop optimiser type
     real(real32) :: beta = 0._real32
     !! Beta parameter
     real(real32) :: epsilon = 1.E-8_real32
     !! Epsilon parameter
     real(real32), allocatable, dimension(:) :: moving_avg
     !! Moving average
   contains
     procedure, pass(this) :: init_gradients => init_gradients_rmsprop
     !! Initialise gradients for RMSprop
     procedure, pass(this) :: minimise => minimise_rmsprop
     !! Apply gradients to parameters to minimise loss using RMSprop optimiser
  end type rmsprop_optimiser_type

  interface rmsprop_optimiser_type
     !! Interface for setting up the RMSprop optimiser
     module function optimiser_setup_rmsprop( &
          learning_rate, beta, &
          epsilon, num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
       !! Set up the RMSprop optimiser
       real(real32), optional, intent(in) :: learning_rate, beta, epsilon
       !! Learning rate, beta, and epsilon
       integer, optional, intent(in) :: num_params
       !! Number of parameters
       class(base_regulariser_type), optional, intent(in) :: regulariser
       !! Regularisation method
       type(clip_type), optional, intent(in) :: clip_dict
       !! Clipping dictionary
       class(base_lr_decay_type), optional, intent(in) :: lr_decay
       !! Learning rate decay method
       type(rmsprop_optimiser_type) :: optimiser
       !! Instance of the RMSprop optimiser
     end function optimiser_setup_rmsprop
  end interface rmsprop_optimiser_type

!-------------------------------------------------------------------------------

  type, extends(base_optimiser_type) :: adagrad_optimiser_type
     !! Adagrad optimiser type
     real(real32) :: epsilon = 1.E-8_real32
     !! Epsilon parameter
     real(real32), allocatable, dimension(:) :: sum_squares
     !! Sum of squares of gradients
   contains
     procedure, pass(this) :: init_gradients => init_gradients_adagrad
     !! Initialise gradients for Adagrad
     procedure, pass(this) :: minimise => minimise_adagrad
     !! Apply gradients to parameters to minimise loss using Adagrad optimiser
  end type adagrad_optimiser_type

  interface adagrad_optimiser_type
     !! Interface for setting up the Adagrad optimiser
     module function optimiser_setup_adagrad( &
          learning_rate, &
          epsilon, num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
       !! Set up the Adagrad optimiser
       real(real32), optional, intent(in) :: learning_rate, epsilon
       !! Learning rate and epsilon
       integer, optional, intent(in) :: num_params
       !! Number of parameters
       class(base_regulariser_type), optional, intent(in) :: regulariser
       !! Regularisation method
       type(clip_type), optional, intent(in) :: clip_dict
       !! Clipping dictionary
       class(base_lr_decay_type), optional, intent(in) :: lr_decay
       !! Learning rate decay method
       type(adagrad_optimiser_type) :: optimiser
       !! Instance of the Adagrad optimiser
     end function optimiser_setup_adagrad
  end interface adagrad_optimiser_type

!-------------------------------------------------------------------------------

  type, extends(base_optimiser_type) :: adam_optimiser_type
     !! Adam optimiser type
     real(real32) :: beta1 = 0.9_real32
     !! Beta1 parameter
     real(real32) :: beta2 = 0.999_real32
     !! Beta2 parameter
     real(real32) :: epsilon = 1.E-8_real32
     !! Epsilon parameter
     real(real32), allocatable, dimension(:) :: m
     !! First moment estimate
     real(real32), allocatable, dimension(:) :: v
     !! Second moment estimate
   contains
     procedure, pass(this) :: init_gradients => init_gradients_adam
     !! Initialise gradients for Adam
     procedure, pass(this) :: minimise => minimise_adam
     !! Apply gradients to parameters to minimise loss using Adam optimiser
  end type adam_optimiser_type

  interface adam_optimiser_type
     !! Interface for setting up the Adam optimiser
     module function optimiser_setup_adam( &
          learning_rate, &
          beta1, beta2, epsilon, &
          num_params, &
          regulariser, clip_dict, lr_decay) result(optimiser)
       !! Set up the Adam optimiser
       real(real32), optional, intent(in) :: learning_rate
       !! Learning rate
       real(real32), optional, intent(in) :: beta1, beta2, epsilon
       !! Beta1, beta2, and epsilon
       integer, optional, intent(in) :: num_params
       !! Number of parameters
       class(base_regulariser_type), optional, intent(in) :: regulariser
       !! Regularisation method
       type(clip_type), optional, intent(in) :: clip_dict
       !! Clipping dictionary
       class(base_lr_decay_type), optional, intent(in) :: lr_decay
       !! Learning rate decay method
       type(adam_optimiser_type) :: optimiser
       !! Instance of the Adam optimiser
     end function optimiser_setup_adam
  end interface adam_optimiser_type



contains

!###############################################################################
  module function optimiser_setup_base( &
      learning_rate, num_params, &
      regulariser, clip_dict, lr_decay) result(optimiser)
    !! Set up the base optimiser
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: learning_rate
    !! Learning rate
    integer, optional, intent(in) :: num_params
    !! Number of parameters
    class(base_regulariser_type), optional, intent(in) :: regulariser
    !! Regularisation method
    type(clip_type), optional, intent(in) :: clip_dict
    !! Clipping dictionary
    class(base_lr_decay_type), optional, intent(in) :: lr_decay
    !! Learning rate decay method

    type(base_optimiser_type) :: optimiser
    !! Instance of the base optimiser

    ! Local variables
    integer :: num_params_
    !! Number of parameters


    ! Apply regularisation
    if(present(regulariser))then
       optimiser%regularisation = .true.
       if(allocated(optimiser%regulariser)) deallocate(optimiser%regulariser)
       allocate(optimiser%regulariser, source = regulariser)
    end if

    ! Apply clipping
    if(present(clip_dict)) optimiser%clip_dict = clip_dict

    ! Initialise general optimiser parameters
    if(present(learning_rate)) optimiser%learning_rate = learning_rate

    ! Initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    else
       allocate(optimiser%lr_decay, source = base_lr_decay_type())
    end if

    ! Initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
  end function optimiser_setup_base
!###############################################################################


!###############################################################################
  subroutine init_base(this, num_params, regulariser, clip_dict)
    !! Initialise base optimiser
    implicit none

    ! Arguments
    class(base_optimiser_type), intent(inout) :: this
    !! Instance of the base optimiser
    integer, intent(in) :: num_params
    !! Number of parameters
    class(base_regulariser_type), optional, intent(in) :: regulariser
    !! Regularisation method
    type(clip_type), optional, intent(in) :: clip_dict
    !! Clipping dictionary


    ! Apply regularisation
    if(present(regulariser))then
       this%regularisation = .true.
       if(allocated(this%regulariser)) deallocate(this%regulariser)
       allocate(this%regulariser, source = regulariser)
    end if

    ! Apply clipping
    if(present(clip_dict)) this%clip_dict = clip_dict

    ! Initialise gradients
    call this%init_gradients(num_params)
  end subroutine init_base
!###############################################################################


!###############################################################################
  pure subroutine init_gradients_base(this, num_params)
    !! Initialise gradients for base optimiser
    implicit none

    ! Arguments
    class(base_optimiser_type), intent(inout) :: this
    !! Instance of the base optimiser
    integer, intent(in) :: num_params
    !! Number of parameters

    !allocate(this%velocity(num_params), source=0._real32)
  end subroutine init_gradients_base
!###############################################################################


!###############################################################################
  pure subroutine minimise_base(this, param, gradient)
    !! Apply gradients to parameters to minimise loss using base optimiser
    implicit none

    ! Arguments
    class(base_optimiser_type), intent(inout) :: this
    !! Instance of the base optimiser
    real(real32), dimension(:), intent(inout) :: param
    !! Parameters
    real(real32), dimension(:), intent(inout) :: gradient
    !! Gradients

    ! Local variables
    real(real32) :: learning_rate
    !! Learning rate


    ! Decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1

    ! Update parameters
    param = param - learning_rate * gradient
  end subroutine minimise_base
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function optimiser_setup_sgd( &
       learning_rate, momentum, &
       nesterov, num_params, &
       regulariser, clip_dict, lr_decay) result(optimiser)
    !! Set up the SGD optimiser
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: learning_rate, momentum
    !! Learning rate and momentum
    logical, optional, intent(in) :: nesterov
    !! Nesterov momentum
    integer, optional, intent(in) :: num_params
    !! Number of parameters
    class(base_regulariser_type), optional, intent(in) :: regulariser
    !! Regularisation method
    type(clip_type), optional, intent(in) :: clip_dict
    !! Clipping dictionary
    class(base_lr_decay_type), optional, intent(in) :: lr_decay
    !! Learning rate decay method

    type(sgd_optimiser_type) :: optimiser
    !! Instance of the SGD optimiser

    ! Local variables
    integer :: num_params_
    !! Number of parameters


    ! Apply regularisation
    if(present(regulariser))then
       optimiser%regularisation = .true.
       if(allocated(optimiser%regulariser)) deallocate(optimiser%regulariser)
       allocate(optimiser%regulariser, source = regulariser)
    end if

    ! Apply clipping
    if(present(clip_dict)) optimiser%clip_dict = clip_dict

    ! Initialise general optimiser parameters
    if(present(learning_rate)) optimiser%learning_rate = learning_rate
    if(present(momentum)) optimiser%momentum = momentum

    ! Initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    else
       allocate(optimiser%lr_decay, source = base_lr_decay_type())
    end if

    ! Initialise nesterov boolean
    if(present(nesterov)) optimiser%nesterov = nesterov

    ! Initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
  end function optimiser_setup_sgd
!###############################################################################


!###############################################################################
  pure subroutine init_gradients_sgd(this, num_params)
    !! Initialise gradients for SGD optimiser
    implicit none

    ! Arguments
    class(sgd_optimiser_type), intent(inout) :: this
    !! Instance of the SGD optimiser
    integer, intent(in) :: num_params
    !! Number of parameters


    ! Initialise gradients
    if(allocated(this%velocity)) deallocate(this%velocity)
    allocate(this%velocity(num_params), source=0._real32)
  end subroutine init_gradients_sgd
!###############################################################################


!###############################################################################
  pure subroutine minimise_sgd(this, param, gradient)
    !! Apply gradients to parameters to minimise loss using SGD optimiser
    implicit none

    ! Arguments
    class(sgd_optimiser_type), intent(inout) :: this
    !! Instance of the SGD optimiser
    real(real32), dimension(:), intent(inout) :: param
    !! Parameters
    real(real32), dimension(:), intent(inout) :: gradient
    !! Gradients

    ! Local variables
    real(real32) :: learning_rate
    !! Learning rate


    ! Decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1

    ! Apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, gradient, learning_rate)

    gradient = - learning_rate * gradient
    ! Update parameters
    if(this%momentum.gt.1.E-8_real32)then
       !! Adaptive learning method
       this%velocity = this%momentum * this%velocity + gradient
       if(this%nesterov)then
          param = param + this%momentum * this%velocity + gradient
       else
          param = param + this%velocity
       end if
    else
       ! Standard learning method
       this%velocity = gradient
       param = param + this%velocity
    end if
  end subroutine minimise_sgd
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function optimiser_setup_rmsprop( &
      learning_rate, beta, epsilon, &
      num_params, regulariser, clip_dict, lr_decay) result(optimiser)
    !! Set up the RMSprop optimiser
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: learning_rate
    !! Learning rate
    real(real32), optional, intent(in) :: beta, epsilon
    !! Beta and epsilon
    integer, optional, intent(in) :: num_params
    !! Number of parameters
    class(base_regulariser_type), optional, intent(in) :: regulariser
    !! Regularisation method
    type(clip_type), optional, intent(in) :: clip_dict
    !! Clipping dictionary
    class(base_lr_decay_type), optional, intent(in) :: lr_decay
    !! Learning rate decay method

    type(rmsprop_optimiser_type) :: optimiser
    !! Instance of the RMSprop optimiser

    ! Local variables
    integer :: num_params_
    !! Number of parameters


    ! Apply regularisation
    if(present(regulariser))then
       optimiser%regularisation = .true.
       if(allocated(optimiser%regulariser)) deallocate(optimiser%regulariser)
       allocate(optimiser%regulariser, source = regulariser)
    end if

    ! Apply clipping
    if(present(clip_dict)) optimiser%clip_dict = clip_dict

    ! Initialise general optimiser parameters
    if(present(learning_rate)) optimiser%learning_rate = learning_rate

    ! Initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    else
       allocate(optimiser%lr_decay, source = base_lr_decay_type())
    end if

    ! Initialise RMSprop parameters
    if(present(beta)) optimiser%beta = beta
    if(present(epsilon)) optimiser%epsilon = epsilon

    ! Initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
  end function optimiser_setup_rmsprop
!###############################################################################


!###############################################################################
  pure subroutine init_gradients_rmsprop(this, num_params)
    !! Initialise gradients for RMSprop optimiser
    implicit none

    ! Arguments
    class(rmsprop_optimiser_type), intent(inout) :: this
    !! Instance of the RMSprop optimiser
    integer, intent(in) :: num_params
    !! Number of parameters


    ! Initialise gradients
    if(allocated(this%moving_avg)) deallocate(this%moving_avg)
    allocate(this%moving_avg(num_params), source=0._real32)
  end subroutine init_gradients_rmsprop
!###############################################################################


!###############################################################################
  pure subroutine minimise_rmsprop(this, param, gradient)
    !! Apply gradients to parameters to minimise loss using RMSprop optimiser
    implicit none

    ! Arguments
    class(rmsprop_optimiser_type), intent(inout) :: this
    !! Instance of the RMSprop optimiser
    real(real32), dimension(:), intent(inout) :: param
    !! Parameters
    real(real32), dimension(:), intent(inout) :: gradient
    !! Gradients

    ! Local variables
    real(real32) :: learning_rate
    !! Learning rate


    ! Decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1

    ! Apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, gradient, learning_rate)

    this%moving_avg = this%beta * this%moving_avg + &
         (1._real32 - this%beta) * gradient ** 2._real32

    ! Update parameters
    param = param - learning_rate * gradient / &
         (sqrt(this%moving_avg + this%epsilon))
  end subroutine minimise_rmsprop
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function optimiser_setup_adagrad( &
      learning_rate, epsilon, &
      num_params, regulariser, clip_dict, lr_decay) result(optimiser)
    !! Set up the Adagrad optimiser
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: learning_rate
    !! Learning rate
    real(real32), optional, intent(in) :: epsilon
    !! Epsilon
    integer, optional, intent(in) :: num_params
    !! Number of parameters
    class(base_regulariser_type), optional, intent(in) :: regulariser
    !! Regularisation method
    type(clip_type), optional, intent(in) :: clip_dict
    !! Clipping dictionary
    class(base_lr_decay_type), optional, intent(in) :: lr_decay
    !! Learning rate decay method

    type(adagrad_optimiser_type) :: optimiser
    !! Instance of the Adagrad optimiser

    ! Local variables
    integer :: num_params_
    !! Number of parameters


    ! Apply regularisation
    if(present(regulariser))then
       optimiser%regularisation = .true.
       if(allocated(optimiser%regulariser)) deallocate(optimiser%regulariser)
       allocate(optimiser%regulariser, source = regulariser)
    end if

    ! Apply clipping
    if(present(clip_dict)) optimiser%clip_dict = clip_dict

    ! Initialise general optimiser parameters
    if(present(learning_rate)) optimiser%learning_rate = learning_rate

    ! Initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    else
       allocate(optimiser%lr_decay, source = base_lr_decay_type())
    end if

    ! Initialise Adagrad parameters
    if(present(epsilon)) optimiser%epsilon = epsilon

    ! Initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
  end function optimiser_setup_adagrad
!###############################################################################


!###############################################################################
  pure subroutine init_gradients_adagrad(this, num_params)
    !! Initialise gradients for Adagrad optimiser
    implicit none

    ! Arguments
    class(adagrad_optimiser_type), intent(inout) :: this
    !! Instance of the Adagrad optimiser
    integer, intent(in) :: num_params
    !! Number of parameters


    ! Initialise gradients
    if(allocated(this%sum_squares)) deallocate(this%sum_squares)
    allocate(this%sum_squares(num_params), source=0._real32)
  end subroutine init_gradients_adagrad
!###############################################################################


!###############################################################################
  pure subroutine minimise_adagrad(this, param, gradient)
    !! Apply gradients to parameters to minimise loss using Adagrad optimiser
    implicit none

    ! Arguments
    class(adagrad_optimiser_type), intent(inout) :: this
    !! Instance of the Adagrad optimiser
    real(real32), dimension(:), intent(inout) :: param
    !! Parameters
    real(real32), dimension(:), intent(inout) :: gradient
    !! Gradients

    real(real32) :: learning_rate
    !! Learning rate


    ! Decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1

    ! Apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, gradient, learning_rate)

    this%sum_squares = this%sum_squares + gradient ** 2._real32

    ! Update parameters
    param = param - learning_rate * gradient / &
         (sqrt(this%sum_squares + this%epsilon))
  end subroutine minimise_adagrad
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function optimiser_setup_adam( &
      learning_rate, beta1, beta2, epsilon, &
      num_params, regulariser, clip_dict, lr_decay) result(optimiser)
    !! Set up the Adam optimiser
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: learning_rate
    !! Learning rate
    real(real32), optional, intent(in) :: beta1, beta2, epsilon
    !! Beta1, beta2, and epsilon
    integer, optional, intent(in) :: num_params
    !! Number of parameters
    class(base_regulariser_type), optional, intent(in) :: regulariser
    !! Regularisation method
    type(clip_type), optional, intent(in) :: clip_dict
    !! Clipping dictionary
    class(base_lr_decay_type), optional, intent(in) :: lr_decay
    !! Learning rate decay method

    type(adam_optimiser_type) :: optimiser
    !! Instance of the Adam optimiser

    ! Local variables
    integer :: num_params_
    !! Number of parameters


    ! Apply regularisation
    if(present(regulariser))then
       optimiser%regularisation = .true.
       if(allocated(optimiser%regulariser)) deallocate(optimiser%regulariser)
       allocate(optimiser%regulariser, source = regulariser)
    end if

    ! Apply clipping
    if(present(clip_dict)) optimiser%clip_dict = clip_dict

    ! Initialise general optimiser parameters
    if(present(learning_rate)) optimiser%learning_rate = learning_rate

    ! Initialise learning rate decay
    if(present(lr_decay)) then
       if(allocated(optimiser%lr_decay)) deallocate(optimiser%lr_decay)
       allocate(optimiser%lr_decay, source = lr_decay)
    else
       allocate(optimiser%lr_decay, source = base_lr_decay_type())
    end if

    ! Initialise Adam parameters
    if(present(beta1)) optimiser%beta1 = beta1
    if(present(beta2)) optimiser%beta2 = beta2
    if(present(epsilon)) optimiser%epsilon = epsilon

    ! Initialise gradients
    if(present(num_params)) then
       num_params_ = num_params
    else
       num_params_ = 1
    end if
    call optimiser%init_gradients(num_params_)
  end function optimiser_setup_adam
!###############################################################################


!###############################################################################
  pure subroutine init_gradients_adam(this, num_params)
    !! Initialise gradients for Adam optimiser
    implicit none

    ! Arguments
    class(adam_optimiser_type), intent(inout) :: this
    !! Instance of the Adam optimiser
    integer, intent(in) :: num_params
    !! Number of parameters


    ! Initialise gradients
    if(allocated(this%m)) deallocate(this%m)
    if(allocated(this%v)) deallocate(this%v)
    allocate(this%m(num_params), source=0._real32)
    allocate(this%v(num_params), source=0._real32)
  end subroutine init_gradients_adam
!###############################################################################


!###############################################################################
  pure subroutine minimise_adam(this, param, gradient)
    !! Apply gradients to parameters to minimise loss using Adam optimiser
    implicit none

    ! Arguments
    class(adam_optimiser_type), intent(inout) :: this
    !! Instance of the Adam optimiser
    real(real32), dimension(:), intent(inout) :: param
    !! Parameters
    real(real32), dimension(:), intent(inout) :: gradient
    !! Gradients

    ! Local variables
    real(real32) :: learning_rate
    !! Learning rate


    ! Decay learning rate and update iteration
    learning_rate = this%lr_decay%get_lr(this%learning_rate, this%iter)
    this%iter = this%iter + 1

    ! Apply regularisation
    if(this%regularisation) &
         call this%regulariser%regularise( &
         param, gradient, learning_rate)

    ! Adaptive learning method
    this%m = this%beta1 * this%m + &
         (1._real32 - this%beta1) * gradient
    this%v = this%beta2 * this%v + &
         (1._real32 - this%beta2) * gradient ** 2._real32

    ! Update parameters
    associate( &
         m_hat => this%m / (1._real32 - this%beta1**this%iter), &
         v_hat => this%v / (1._real32 - this%beta2**this%iter) )
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
!###############################################################################

end module athena__optimiser
