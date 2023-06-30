module misc_ml
  use constants, only: real12
  implicit none


  private

  public :: get_padding_half

  public :: step_decay
  public :: reduce_lr_on_plateau
  public :: adam_optimiser


contains

!!!########################################################################
!!! return width of padding from kernel/filter size
!!!########################################################################
  function get_padding_half(width) result(output)
    implicit none
    integer, intent(in) :: width
    integer :: output
    
    output = ( width - (1 - mod(width,2)) - 1 ) / 2
        
  end function get_padding_half
!!!########################################################################


!!!########################################################################
!!! adaptive learning rate
!!! method: step decay
!!!########################################################################
  subroutine step_decay(learning_rate, epoch, decay_rate, decay_steps)
    implicit none
    integer, intent(in) :: epoch
    integer, intent(in) :: decay_steps
    real(real12), intent(inout) :: learning_rate
    real(real12), intent(in) :: decay_rate

    !! calculate new learning rate
    learning_rate = learning_rate * &
         decay_rate**((epoch - 1._real12) / decay_steps)

  end subroutine step_decay
!!!########################################################################


!!!########################################################################
!!! adaptive learning rate
!!! method: reduce learning rate on plateau
!!!########################################################################
  subroutine reduce_lr_on_plateau(learning_rate, &
       metric_value, patience, factor, min_lr, & 
       best_metric_value, wait)
    implicit none
    integer, intent(in) :: patience
    integer, intent(inout) :: wait
    real(real12), intent(inout) :: learning_rate
    real(real12), intent(in) :: metric_value
    real(real12), intent(in) :: factor
    real(real12), intent(in) :: min_lr
    real(real12), intent(inout) :: best_metric_value

    !! check if the metric value has improved
    if (metric_value.lt.best_metric_value) then
       best_metric_value = metric_value
       wait = 0
    else
       wait = wait + 1
       if (wait.ge.patience) then
          learning_rate = learning_rate * factor
          if (learning_rate.lt.min_lr) then
             learning_rate = min_lr
          endif
          wait = 0
       endif
    endif

  end subroutine reduce_lr_on_plateau
!!!########################################################################


!!!########################################################################
!!! adaptive learning rate
!!! method: adam optimiser
!!!         ... Adaptive Moment Estimation
!!!########################################################################
!!! learning_rate = initial learning rate hyperparameter
!!! beta1 = exponential decay rate for first-moment estimates
!!! beta2 = exponential decay rate for second-moment estimates
!!! epsilon = small number for numerical stability
!!! t = current iteration
!!! m = first moment (m = mean of gradients)
!!! v = second moment (v = variance of the gradients)
  subroutine adam_optimiser(learning_rate, &
       gradient, m, v, t, &
       beta1, beta2, epsilon)
    implicit none
    integer, intent(inout) :: t
    real(real12), intent(in) :: gradient
    real(real12), intent(inout) :: m, v
    real(real12), intent(inout) :: learning_rate
    real(real12), intent(in) :: beta1, beta2, epsilon

    real :: m_norm, v_norm

    !! update biased first moment estimate
    m = beta1 * m + (1._real12 - beta1) * gradient

    !! update biased second moment estimate
    v = beta2 * v + (1._real12 - beta2) * gradient**2

    !! normalised first moment estimate
    m_norm = m / (1._real12 - beta1**t)

    !! normalised second moment estimate
    v_norm = v / (1._real12 - beta2**t)
    
    !! update learning rate
    learning_rate = learning_rate * m_norm / (sqrt(v_norm) + epsilon)

  end subroutine adam_optimiser
!!!########################################################################

end module misc_ml
