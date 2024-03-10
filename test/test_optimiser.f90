program test_optimiser
  use optimiser
  use regulariser
  use learning_rate_decay
  implicit none

  class(base_optimiser_type), allocatable :: optimiser


  integer, parameter :: num_params = 10
  real, dimension(num_params) :: param = 0.E0
  real, dimension(num_params) :: gradient = 0.E0

  logical :: success = .true.


!!!-----------------------------------------------------------------------------
!!! test empty base optimiser
!!!-----------------------------------------------------------------------------
  allocate(optimiser, source=base_optimiser_type())


!!!-----------------------------------------------------------------------------
!!! test base optimiser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser)
  allocate(optimiser, source=base_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser)
  type is (base_optimiser_type)
     write(*,*) "base_optimiser_type"
  class default
     write(0,*) "Failed to allocate base optimiser"
     success = .false.
  end select
  call optimiser%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test sgd optimiser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser)
  allocate(optimiser, source=sgd_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       momentum = 0.9, &
       nesterov = .true., &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser)
  type is (sgd_optimiser_type)
     write(*,*) "sgd_optimiser_type"
  class default
     write(0,*) "Failed to allocate sgd optimiser"
     success = .false.
  end select
  call optimiser%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test rmsprop optimiser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser)
  allocate(optimiser, source=rmsprop_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta = 0.9, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser)
  type is (rmsprop_optimiser_type)
     write(*,*) "rmsprop_optimiser_type"
  class default
     write(0,*) "Failed to allocate rmsprop optimiser"
     success = .false.
  end select
  call optimiser%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test adagrad optimiser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser)
  allocate(optimiser, source=adagrad_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser)
  type is (adagrad_optimiser_type)
     write(*,*) "adagrad_optimiser_type"
  class default
     write(0,*) "Failed to allocate adagrad optimiser"
     success = .false.
  end select
  call optimiser%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test adam optimiser with l1 regulariser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser)
  allocate(optimiser, source=adam_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta1 = 0.9, &
       beta2 = 0.999, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser)
  type is (adam_optimiser_type)
     write(*,*) "adam_optimiser_type"
  class default
     write(0,*) "Failed to allocate adam optimiser"
     success = .false.
  end select
  call optimiser%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test adam optimiser with l2 regulariser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser)
  allocate(optimiser, source=adam_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta1 = 0.9, &
       beta2 = 0.999, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l2_regulariser_type() &
       ))
  select type(optimiser)
  type is (adam_optimiser_type)
     write(*,*) "adam_optimiser_type"
  class default
     write(0,*) "Failed to allocate adam optimiser"
     success = .false.
  end select
  call optimiser%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test adam optimiser with l2 decoupled regulariser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser)
  allocate(optimiser, source=adam_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta1 = 0.9, &
       beta2 = 0.999, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l2_regulariser_type(decoupled=.false.) &
       ))
  select type(optimiser)
  type is (adam_optimiser_type)
     write(*,*) "adam_optimiser_type"
  class default
     write(0,*) "Failed to allocate adam optimiser"
     success = .false.
  end select
  call optimiser%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! check for any failed tests
!!!-----------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_optimiser passed all tests'
  else
     write(0,*) 'test_optimiser failed one or more tests'
     stop 1
  end if


end program test_optimiser