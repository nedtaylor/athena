module athena__learning_rate_decay
  !! Module containing learning decay rate types and procedures
  use coreutils, only: real32
  implicit none


  private

  public :: base_lr_decay_type
  public :: exp_lr_decay_type
  public :: step_lr_decay_type
  public :: inv_lr_decay_type


!-------------------------------------------------------------------------------

  type base_lr_decay_type
     !! Type for learning rate decay
     character(len=20) :: name
     !! Name of the learning rate decay type
     real(real32) :: initial_learning_rate
     !! Initial learning rate
     real(real32) :: decay_rate
     !! Decay rate for learning rate
     logical :: iterate_per_epoch = .false.
     !! Whether to iterate learning rate decay per epoch
   contains
     procedure :: get_lr => lr_decay_none
     !! Procedure to get the learning rate
  end type base_lr_decay_type

  interface base_lr_decay_type
     !! Interface for base learning rate decay type
     module function setup_lr_decay_base() result(lr_decay)
       !! Set up base learning rate decay type
       type(base_lr_decay_type) :: lr_decay
       !! Base learning rate decay type
     end function setup_lr_decay_base
  end interface base_lr_decay_type

!-------------------------------------------------------------------------------

  type, extends(base_lr_decay_type) :: exp_lr_decay_type
     !! Type for exponential learning rate decay
   contains
     procedure :: get_lr => lr_decay_exp
     !! Procedure to get the learning rate
  end type exp_lr_decay_type

  interface exp_lr_decay_type
     !! Interface for exponential learning rate decay type
     module function setup_lr_decay_exp(decay_rate) result(lr_decay)
       !! Set up exponential learning rate decay type
       real(real32), optional, intent(in) :: decay_rate
       !! Decay rate for learning rate
       type(exp_lr_decay_type) :: lr_decay
       !! Exponential learning rate decay type
     end function setup_lr_decay_exp
  end interface exp_lr_decay_type

!-------------------------------------------------------------------------------

  type, extends(base_lr_decay_type) :: step_lr_decay_type
     !! Type for step learning rate decay
     integer :: decay_steps
     !! Number of steps for learning rate decay
   contains
     procedure :: get_lr => lr_decay_step
     !! Procedure to get the learning rate
  end type step_lr_decay_type

  interface step_lr_decay_type
     !! Interface for step learning rate decay type
     module function setup_lr_decay_step(decay_rate, decay_steps) &
          result(lr_decay)
       !! Set up step learning rate decay type
       real(real32), optional, intent(in) :: decay_rate
       !! Decay rate for learning rate
       integer, optional, intent(in) :: decay_steps
       !! Number of steps for learning rate decay
       type(step_lr_decay_type) :: lr_decay
       !! Step learning rate decay type
     end function setup_lr_decay_step
  end interface step_lr_decay_type

!-------------------------------------------------------------------------------

  type, extends(base_lr_decay_type) :: inv_lr_decay_type
     !! Type for inverse learning rate decay
     real(real32) :: decay_power
     !! Power for learning rate decay
   contains
     procedure :: get_lr => lr_decay_inv
     !! Procedure to get the learning rate
  end type inv_lr_decay_type

  interface inv_lr_decay_type
     !! Interface for inverse learning rate decay type
     module function setup_lr_decay_inv(decay_rate, decay_power) &
          result(lr_decay)
       !! Set up inverse learning rate decay type
       real(real32), optional, intent(in) :: decay_rate, decay_power
       !! Decay rate for learning rate
       type(inv_lr_decay_type) :: lr_decay
       !! Inverse learning rate decay type
     end function setup_lr_decay_inv
  end interface inv_lr_decay_type



contains

!###############################################################################
  module function setup_lr_decay_base() result(lr_decay)
    !! Set up base learning rate decay type
    implicit none

    ! Output variable
    type(base_lr_decay_type) :: lr_decay
    !! Instance of the base learning rate decay type

    lr_decay%name = "base"
    lr_decay%decay_rate = 0._real32

  end function setup_lr_decay_base
!-------------------------------------------------------------------------------
  module function setup_lr_decay_exp(decay_rate) result(lr_decay)
    !! Set up exponential learning rate decay type
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: decay_rate
    !! Decay rate for learning rate
    type(exp_lr_decay_type) :: lr_decay
    !! Exponential learning rate decay type

    lr_decay%name = "exp"
    if(present(decay_rate))then
       lr_decay%decay_rate = decay_rate
    else
       lr_decay%decay_rate = 0.9_real32
    end if

  end function setup_lr_decay_exp
!-------------------------------------------------------------------------------
  module function setup_lr_decay_step(decay_rate, decay_steps) result(lr_decay)
    !! Set up step learning rate decay type
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: decay_rate
    !! Decay rate for learning rate
    integer, optional, intent(in) :: decay_steps
    !! Number of steps for learning rate decay
    type(step_lr_decay_type) :: lr_decay
    !! Step learning rate decay type

    lr_decay%name = "step"
    if(present(decay_rate))then
       lr_decay%decay_rate = decay_rate
    else
       lr_decay%decay_rate = 0.1_real32
    end if
    if(present(decay_steps))then
       lr_decay%decay_steps = decay_steps
    else
       lr_decay%decay_steps = 100
    end if
    lr_decay%iterate_per_epoch = .true.

  end function setup_lr_decay_step
!-------------------------------------------------------------------------------
  module function setup_lr_decay_inv(decay_rate, decay_power) result(lr_decay)
    !! Set up inverse learning rate decay type
    implicit none

    ! Arguments
    real(real32), optional, intent(in) :: decay_rate, decay_power
    !! Decay rate for learning rate
    type(inv_lr_decay_type) :: lr_decay
    !! Inverse learning rate decay type

    lr_decay%name = "inv"
    if(present(decay_rate))then
       lr_decay%decay_rate = decay_rate
    else
       lr_decay%decay_rate = 0.001_real32
    end if
    if(present(decay_power))then
       lr_decay%decay_power = decay_power
    else
       lr_decay%decay_power = 1._real32
    end if

  end function setup_lr_decay_inv
!###############################################################################


!###############################################################################
  pure function lr_decay_none(this, learning_rate, iteration) result(output)
    !! Get the learning rate for the base decay type
    implicit none

    ! Arguments
    class(base_lr_decay_type), intent(in) :: this
    !! Instance of the base learning rate decay type
    real(real32), intent(in) :: learning_rate
    !! Initial learning rate
    integer, intent(in) :: iteration
    !! Iteration number
    real(real32) :: output
    !! Learning rate

    output = learning_rate

  end function lr_decay_none
!-------------------------------------------------------------------------------
  pure function lr_decay_exp(this, learning_rate, iteration) result(output)
    !! Get the learning rate for the exponential decay type
    implicit none

    ! Arguments
    class(exp_lr_decay_type), intent(in) :: this
    !! Instance of the exponential learning rate decay type
    real(real32), intent(in) :: learning_rate
    !! Initial learning rate
    integer, intent(in) :: iteration
    !! Iteration number
    real(real32) :: output
    !! Learning rate

    output = learning_rate * exp(- iteration * this%decay_rate)

  end function lr_decay_exp
!-------------------------------------------------------------------------------
  pure function lr_decay_step(this, learning_rate, iteration) result(output)
    !! Get the learning rate for the step decay type
    implicit none

    ! Arguments
    class(step_lr_decay_type), intent(in) :: this
    !! Instance of the step learning rate decay type
    real(real32), intent(in) :: learning_rate
    !! Initial learning rate
    integer, intent(in) :: iteration
    !! Iteration number
    real(real32) :: output
    !! Learning rate

    output = learning_rate * this%decay_rate ** (iteration / this%decay_steps)

  end function lr_decay_step
!-------------------------------------------------------------------------------
  pure function lr_decay_inv(this, learning_rate, iteration) result(output)
    !! Get the learning rate for the inverse decay type
    implicit none

    ! Arguments
    class(inv_lr_decay_type), intent(in) :: this
    !! Instance of the inverse learning rate decay type
    real(real32), intent(in) :: learning_rate
    !! Initial learning rate
    integer, intent(in) :: iteration
    !! Iteration number
    real(real32) :: output
    !! Learning rate

    output = learning_rate * &
         (1._real32 + this%decay_rate * iteration) ** (- this%decay_power)

  end function lr_decay_inv
!###############################################################################

end module athena__learning_rate_decay
