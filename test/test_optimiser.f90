program test_optimiser
  use optimiser
  use regulariser
  use learning_rate_decay
  implicit none

  class(base_optimiser_type), allocatable :: optimiser_var


  integer, parameter :: num_params = 10
  real, dimension(num_params) :: param = 0.E0
  real, dimension(num_params) :: gradient = 0.E0

  logical :: success = .true.


!!!-----------------------------------------------------------------------------
!!! test empty base optimiser_var
!!!-----------------------------------------------------------------------------
  allocate(optimiser_var, source=base_optimiser_type())


!!!-----------------------------------------------------------------------------
!!! test base optimiser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=base_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser_var)
  type is (base_optimiser_type)
     write(*,*) "base_optimiser_type"
  class default
     write(0,*) "Failed to allocate base optimiser"
     success = .false.
  end select
  call optimiser_var%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test sgd optimiser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=sgd_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       momentum = 0.9, &
       nesterov = .true., &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser_var)
  type is (sgd_optimiser_type)
     write(*,*) "sgd_optimiser_type"
  class default
     write(0,*) "Failed to allocate sgd optimiser"
     success = .false.
  end select
  call optimiser_var%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test rmsprop optimiser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=rmsprop_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta = 0.9, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser_var)
  type is (rmsprop_optimiser_type)
     write(*,*) "rmsprop_optimiser_type"
  class default
     write(0,*) "Failed to allocate rmsprop optimiser"
     success = .false.
  end select
  call optimiser_var%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test adagrad optimiser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=adagrad_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser_var)
  type is (adagrad_optimiser_type)
     write(*,*) "adagrad_optimiser_type"
  class default
     write(0,*) "Failed to allocate adagrad optimiser"
     success = .false.
  end select
  call optimiser_var%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test adam optimiser with l1 regulariser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=adam_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta1 = 0.9, &
       beta2 = 0.999, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
       ))
  select type(optimiser_var)
  type is (adam_optimiser_type)
     write(*,*) "adam_optimiser_type"
  class default
     write(0,*) "Failed to allocate adam optimiser"
     success = .false.
  end select
  call optimiser_var%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test adam optimiser with l2 regulariser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=adam_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta1 = 0.9, &
       beta2 = 0.999, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l2_regulariser_type() &
       ))
  select type(optimiser_var)
  type is (adam_optimiser_type)
     write(*,*) "adam_optimiser_type"
  class default
     write(0,*) "Failed to allocate adam optimiser"
     success = .false.
  end select
  call optimiser_var%minimise(param, gradient)


!!!-----------------------------------------------------------------------------
!!! test adam optimiser with l2 decoupled regulariser
!!!-----------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=adam_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta1 = 0.9, &
       beta2 = 0.999, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l2_regulariser_type(decoupled=.false.) &
       ))
  select type(optimiser_var)
  type is (adam_optimiser_type)
     write(*,*) "adam_optimiser_type"
  class default
     write(0,*) "Failed to allocate adam optimiser"
     success = .false.
  end select
  call optimiser_var%minimise(param, gradient)


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