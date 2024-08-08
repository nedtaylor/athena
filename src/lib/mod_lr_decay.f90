!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains learning rate decay types and procedures
!!! module contains the following derived types:
!!! - base_lr_decay_type - base learning rate decay type
!!! - exp_lr_decay_type  - exponential learning rate decay type
!!! - step_lr_decay_type - step learning rate decay type
!!! - inv_lr_decay_type  - inverse learning rate decay type
!!!##################
!!! <NAME>_lr_decay_type contains the following procedures:
!!! - get_lr - returns the updated learning rate
!!!##################
!!! module contains the following procedures:
!!! - setup_lr_decay_<NAME> - sets up the <NAME> learning rate decay type
!!!#############################################################################
module learning_rate_decay
  use constants, only: real32
  implicit none


  type base_lr_decay_type
     real(real32) :: initial_learning_rate
     real(real32) :: decay_rate
   contains
     procedure :: get_lr => lr_decay_none
  end type base_lr_decay_type

  interface base_lr_decay_type
     module function setup_lr_decay_base() result(lr_decay)
       type(base_lr_decay_type) :: lr_decay
     end function setup_lr_decay_base
  end interface base_lr_decay_type

!!!-----------------------------------------------------------------------------

  type, extends(base_lr_decay_type) :: exp_lr_decay_type
   contains
     procedure :: get_lr => lr_decay_exp
  end type exp_lr_decay_type

  interface exp_lr_decay_type
     module function setup_lr_decay_exp(decay_rate) result(lr_decay)
       real(real32), optional, intent(in) :: decay_rate
       type(exp_lr_decay_type) :: lr_decay
     end function setup_lr_decay_exp
  end interface exp_lr_decay_type

!!!-----------------------------------------------------------------------------

  type, extends(base_lr_decay_type) :: step_lr_decay_type
     integer :: decay_steps
   contains
     procedure :: get_lr => lr_decay_step
  end type step_lr_decay_type

  interface step_lr_decay_type
     module function setup_lr_decay_step(decay_rate, decay_steps) &
          result(lr_decay)
       real(real32), optional, intent(in) :: decay_rate
       integer, optional, intent(in) :: decay_steps
       type(step_lr_decay_type) :: lr_decay
     end function setup_lr_decay_step
  end interface step_lr_decay_type

!!!-----------------------------------------------------------------------------

  type, extends(base_lr_decay_type) :: inv_lr_decay_type
     real(real32) :: decay_power
   contains
     procedure :: get_lr => lr_decay_inv
  end type inv_lr_decay_type

  interface inv_lr_decay_type
     module function setup_lr_decay_inv(decay_rate, decay_power) &
          result(lr_decay)
       real(real32), optional, intent(in) :: decay_rate, decay_power
       type(inv_lr_decay_type) :: lr_decay
     end function setup_lr_decay_inv
  end interface inv_lr_decay_type

!!!-----------------------------------------------------------------------------


  private

  public :: base_lr_decay_type
  public :: exp_lr_decay_type
  public :: step_lr_decay_type
  public :: inv_lr_decay_type


contains

!!!#############################################################################
!!! set up learning rate types
!!!#############################################################################
  module function setup_lr_decay_base() result(lr_decay)
    type(base_lr_decay_type) :: lr_decay

    lr_decay%decay_rate = 0._real32

  end function setup_lr_decay_base
!!!-----------------------------------------------------------------------------
  module function setup_lr_decay_exp(decay_rate) result(lr_decay)
    real(real32), optional, intent(in) :: decay_rate
    type(exp_lr_decay_type) :: lr_decay

    if(present(decay_rate))then
       lr_decay%decay_rate = decay_rate
    else
       lr_decay%decay_rate = 0.9_real32
    end if

  end function setup_lr_decay_exp
!!!-----------------------------------------------------------------------------
  module function setup_lr_decay_step(decay_rate, decay_steps) result(lr_decay)
    real(real32), optional, intent(in) :: decay_rate
    integer, optional, intent(in) :: decay_steps
    type(step_lr_decay_type) :: lr_decay

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

  end function setup_lr_decay_step
!!!-----------------------------------------------------------------------------
  module function setup_lr_decay_inv(decay_rate, decay_power) result(lr_decay)
    real(real32), optional, intent(in) :: decay_rate, decay_power
    type(inv_lr_decay_type) :: lr_decay

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
!!!#############################################################################


!!!#############################################################################
!!! learning rate decay procedures
!!!#############################################################################
  pure function lr_decay_none(this, learning_rate, iteration) result(output)
    implicit none
    class(base_lr_decay_type), intent(in) :: this
    real(real32), intent(in) :: learning_rate
    integer, intent(in) :: iteration

    real(real32) :: output

    output = learning_rate

  end function lr_decay_none
!!!-----------------------------------------------------------------------------
  pure function lr_decay_exp(this, learning_rate, iteration) result(output)
    implicit none
    class(exp_lr_decay_type), intent(in) :: this
    real(real32), intent(in) :: learning_rate
    integer, intent(in) :: iteration

    real(real32) :: output

    output = learning_rate * exp(- iteration * this%decay_rate)

  end function lr_decay_exp
!!!-----------------------------------------------------------------------------
  pure function lr_decay_step(this, learning_rate, iteration) result(output)
    implicit none
    class(step_lr_decay_type), intent(in) :: this
    real(real32), intent(in) :: learning_rate
    integer, intent(in) :: iteration

    real(real32) :: output

    output = learning_rate * this%decay_rate ** (iteration / this%decay_steps)

  end function lr_decay_step
!!!-----------------------------------------------------------------------------
  pure function lr_decay_inv(this, learning_rate, iteration) result(output)
    implicit none
    class(inv_lr_decay_type), intent(in) :: this
    real(real32), intent(in) :: learning_rate
    integer, intent(in) :: iteration

    real(real32) :: output

    output = learning_rate * &
         (1._real32 + this%decay_rate * iteration) ** (- this%decay_power)

  end function lr_decay_inv
!!!#############################################################################

end module learning_rate_decay
