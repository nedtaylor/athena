module misc_ml
  use constants, only: real12
  use custom_types, only: learning_parameters_type
  implicit none


  

  private

  public :: get_padding_half

  public :: step_decay
  public :: reduce_lr_on_plateau
  public :: adam_optimiser
  public :: update_weight

  public :: drop_block, generate_bernoulli_mask


contains

!!!########################################################################
!!! DropBlock method for dropping random blocks of data from an image
!!!########################################################################
!!! https://proceedings.neurips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
!!! https://pub.towardsai.net/dropblock-a-new-regularization-technique-e926bbc74adb
!!! input = input data
!!!         ... channels are provided independently
!!!         ... this tries to prevent the network from relying too ...
!!!         ... heavily one one set of activations
!!! keep_prob   = probability of keeping a unit, as in traditional dropout
!!!               ... (default = 0.75-0.95)
!!! block_size  = width of block (default = 5)
!!! gamma       = how many activation units to drop
  subroutine drop_block(input, mask, block_size)
    implicit none
    real(real12), dimension(:,:), intent(inout) :: input
    logical, dimension(:,:), intent(in) :: mask
    integer, intent(in) :: block_size

    integer :: i, j, x, y, start_idx, end_idx, mask_size

    mask_size = size(mask, dim=1)
    start_idx = -(block_size - 1)/2 !centre should be zero
    end_idx = (block_size -1)/2 + (1 - mod(block_size,2)) !centre should be zero

    ! gamma = (1 - keep_prob)/block_size**2 * input_size**2/(input_size - block_size + 1)**2

    do j = 1, mask_size
       do i = 1, mask_size
          if (.not.mask(i, j))then
             do x=start_idx,end_idx,1
                do y=start_idx,end_idx,1
                   input(i - start_idx + x, j - start_idx + y) = 0._real12
                end do
             end do
          endif
       end do
    end do

    input = input * size(mask,dim=1) * size(mask,dim=2) / count(mask)

  end subroutine drop_block
!!!########################################################################


!!!########################################################################
!!! 
!!!########################################################################
  subroutine generate_bernoulli_mask(mask, gamma, seed)
    implicit none
    logical, dimension(:,:), intent(out) :: mask
    real, intent(in) :: gamma
    integer, optional, intent(in) :: seed
    real(real12), allocatable, dimension(:,:) :: mask_real
    integer :: i, j

    !! IF seed GIVEN, INITIALISE
    ! assume random number already seeded and don't need to again
    !call random_seed()  ! Initialize random number generator
    allocate(mask_real(size(mask,1), size(mask,2)))
    call random_number(mask_real)  ! Generate random values in [0,1)

    !! Apply threshold to create binary mask
    do j = 1, size(mask, dim=2)
       do i = 1, size(mask, dim=1)
          if(mask_real(i, j).gt.gamma)then
             mask(i, j) = .false. !0 = drop
          else
             mask(i, j) = .true.  !1 = keep
          end if
       end do
    end do
    
  end subroutine generate_bernoulli_mask
!!!########################################################################


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
       metric_value, patience, factor, min_learning_rate, & 
       best_metric_value, wait)
    implicit none
    integer, intent(in) :: patience
    integer, intent(inout) :: wait
    real(real12), intent(inout) :: learning_rate
    real(real12), intent(in) :: metric_value
    real(real12), intent(in) :: factor
    real(real12), intent(in) :: min_learning_rate
    real(real12), intent(inout) :: best_metric_value

    !! check if the metric value has improved
    if (metric_value.lt.best_metric_value) then
       best_metric_value = metric_value
       wait = 0
    else
       wait = wait + 1
       if (wait.ge.patience) then
          learning_rate = learning_rate * factor
          if (learning_rate.lt.min_learning_rate) then
             learning_rate = min_learning_rate
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
  elemental subroutine adam_optimiser(learning_rate, &
       gradient, m, v, t, &
       beta1, beta2, epsilon)
    implicit none
    integer, intent(in) :: t
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


!!!########################################################################
!!! 
!!!########################################################################
elemental subroutine update_weight(learning_rate, weight, weight_incr, &
     gradient, iteration, parameters, m, v)
  implicit none
  integer, intent(in) :: iteration
  real(real12), intent(in) :: learning_rate
  real(real12), intent(inout) :: weight
  real(real12), intent(inout) :: weight_incr
  real(real12), intent(inout) :: gradient
  real(real12), optional, intent(inout) :: m, v
  type(learning_parameters_type), intent(in) :: parameters

  real(real12) :: t_learning_rate
  

  !! momentum-based learning
  if(parameters%method.eq.'momentum')then
     !! reversed weight applier to match keras, improves convergence
     !! w = w + vel - lr * g
     weight_incr = learning_rate * gradient - &
          parameters%momentum * weight_incr
  !! nesterov momentum
  elseif(parameters%method.eq.'nesterov')then
     weight_incr = - parameters%momentum * weight_incr - &
          learning_rate * gradient
  !! adam optimiser
  elseif(parameters%method.eq.'adam')then
     t_learning_rate = learning_rate
     call adam_optimiser(t_learning_rate, gradient, &
          m, v, iteration, &
          parameters%beta1, parameters%beta2, &
          parameters%epsilon)
     weight_incr = t_learning_rate
  else
     weight_incr = learning_rate * gradient
  end if

  if(.not.parameters%regularisation.eq.'')then
     !! L1L2 regularisation
     if(parameters%regularisation.eq.'l1l2')then
        weight_incr = weight_incr + learning_rate * ( &
             parameters%l1 * sign(1._real12,weight) + &
             parameters%l2 * weight )
     !! L1 regularisation
     elseif(parameters%regularisation.eq.'l1')then
        weight_incr = weight_incr + learning_rate * &
             parameters%l1 * sign(1._real12,weight)
     !! L2 regularisation
     elseif(parameters%regularisation.eq.'l2')then
        weight_incr = weight_incr + learning_rate * &
             parameters%l2 * weight
     end if
  end if

  if(parameters%method.eq.'nesterov')then
     weight = weight + parameters%momentum * weight_incr - &
          learning_rate * gradient
  else
     weight = weight - weight_incr
  end if


end subroutine update_weight
!!!########################################################################

end module misc_ml
