program test_lr_decay
  use misc_ml, only: step_decay, reduce_lr_on_plateau
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