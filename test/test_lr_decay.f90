program test_lr_decay
  use misc_ml, only: step_decay, reduce_lr_on_plateau
  use learning_rate_decay
  implicit none

  real :: learning_rate
  real, parameter :: init_learning_rate = 0.1E0
  real, parameter :: decay_rate = 0.1E0
  integer, parameter :: num_epoch = 20
  integer, parameter :: decay_steps = 10
  real, parameter :: min_learning_rate = 1.E-6
  real, parameter :: factor = 0.1E0

  integer :: i
  integer :: epoch, wait
  real :: best_metric_value = 0.0E0
  real :: expected_learning_rate
  logical :: success = .true.

  class(base_lr_decay_type), allocatable :: lr_decay


  !! test step_decay
  do epoch = 1, 20
    learning_rate = init_learning_rate
    call step_decay(learning_rate, epoch, decay_rate, decay_steps)
     if ( abs( &
          learning_rate - &
          init_learning_rate * decay_rate**((epoch - 1.E0) / &
          decay_steps) ) .gt. &
          1.E-6 ) then
        success = .false.
        write(0,*) "step_decay failed"
     end if
  end do

  !! test reduce_lr_on_plateau
  learning_rate = init_learning_rate
  do i = 1, 5
     wait = 6
     expected_learning_rate = max( learning_rate * factor , min_learning_rate)
     call reduce_lr_on_plateau(learning_rate, 0.5E0, 5, factor, &
          1.E-6, best_metric_value, wait)

     if( abs( expected_learning_rate - learning_rate ) .gt. 1.E-6 )then
        success = .false.
        write(0,*) "reduce_lr_on_plateau failed"
        write(*,*) "learning_rate = ", learning_rate, expected_learning_rate
     end if
  end do

  !! test base learning rate decay
  lr_decay = base_lr_decay_type()
  select type(lr_decay)
  type is (base_lr_decay_type)
     if(abs(lr_decay%decay_rate).gt.1.E-6)then
        write(0,*) "base decay rate failed to initialise"
        success = .false.
     end if
  class default
     write(0,*) "base_lr_decay_type failed"
     success = .false.
  end select
  if(abs(learning_rate - &
       lr_decay%get_lr(learning_rate, 1)).gt.1.E-6)then
     write(0,*) "base learning rate decay failed"
     success = .false.
  end if

  !! test exponential learning rate decay
  lr_decay = exp_lr_decay_type()
  if(abs(lr_decay%decay_rate-0.9E0).gt.1.E-6)then
     write(0,*) "step decay rate failed to initialise"
     success = .false.
  end if
  lr_decay = exp_lr_decay_type(decay_rate=0.2E0)
  select type(lr_decay)
  type is (exp_lr_decay_type)
     if(abs(lr_decay%decay_rate-0.2E0).gt.1.E-6)then
        write(0,*) "exponential decay rate failed to initialise"
        success = .false.
     end if
  class default
     write(0,*) "exp_lr_decay_type failed"
     success = .false.
  end select
  if(abs(learning_rate * exp( -1 * 0.2E0) - &
       lr_decay%get_lr(learning_rate, 1)) .gt. 1.E-6)then
     write(0,*) "expoential learning rate decay failed"
     success = .false.
  end if

  !! test step learning rate decay
  lr_decay = step_lr_decay_type()
  if(abs(lr_decay%decay_rate-0.1E0).gt.1.E-6)then
     write(0,*) "step decay rate failed to initialise"
     success = .false.
  end if
  lr_decay = step_lr_decay_type(decay_rate=0.2E0, decay_steps=10)
  select type(lr_decay)
  type is (step_lr_decay_type)
     if(abs(lr_decay%decay_rate-0.2E0).gt.1.E-6)then
        write(0,*) "step decay rate failed to initialise"
        success = .false.
     end if
     if(lr_decay%decay_steps.ne.10)then
        write(0,*) "number of decay steps failed to initialise"
        success = .false.
     end if
  class default
     write(0,*) "step_lr_decay_type failed"
     success = .false.
  end select
  if(abs(learning_rate * 0.2E0 ** (11/10) - &
       lr_decay%get_lr(learning_rate, 11)) .gt. 1.E-6)then
     write(0,*) "step learning rate decay failed"
     success = .false.
  end if

  !! test inverse learning rate decay
  lr_decay = inv_lr_decay_type()
  if(abs(lr_decay%decay_rate-0.001E0).gt.1.E-6)then
     write(0,*) "inv decay rate failed to initialise"
     success = .false.
  end if
  lr_decay = inv_lr_decay_type(decay_rate=0.2E0, decay_power=2.E0)
  select type(lr_decay)
  type is (inv_lr_decay_type)
     if(abs(lr_decay%decay_rate-0.2E0).gt.1.E-6)then
        write(0,*) "inverse decay rate failed to initialise"
        success = .false.
     end if
     if(abs(lr_decay%decay_power-2.E0).gt.1.E-6)then
        write(0,*) "decay power failed to initialise"
        success = .false.
     end if
  class default
     write(0,*) "inv_lr_decay_type failed"
     success = .false.
  end select
  if(abs(learning_rate * (1.E0 + 0.2E0 * 1)**(-2.E0) - &
       lr_decay%get_lr(learning_rate, 1)) .gt. 1.E-6)then
     write(0,*) "inverse learning rate decay failed"
     success = .false.
  end if

!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_lr_decay passed all tests'
  else
     write(0,*) 'test_lr_decay failed one or more tests'
     stop 1
  end if

end program test_lr_decay