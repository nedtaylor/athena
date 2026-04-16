program test_optimiser
  use coreutils, only: real32
  use athena__optimiser
  use athena__regulariser
  use athena__learning_rate_decay
  implicit none

  class(base_optimiser_type), allocatable :: optimiser_var
  type(plp_optimiser_type) :: plp_behaviour


  integer, parameter :: num_params = 10
  integer :: i
  real(real32), parameter :: tol = 1.E-6_real32
  real(real32), dimension(num_params) :: param = 0._real32
  real(real32), dimension(num_params) :: gradient = 0._real32
  real(real32), dimension(1) :: plp_param = 0._real32
  real(real32), dimension(1) :: plp_gradient = 0._real32
  real(real32), dimension(3) :: plp_gradient_schedule = [ &
       1._real32, 2._real32, 4._real32]

  logical :: success = .true.


!-------------------------------------------------------------------------------
! test empty base optimiser_var
!-------------------------------------------------------------------------------
  allocate(optimiser_var, source=base_optimiser_type())


!-------------------------------------------------------------------------------
! test base optimiser
!-------------------------------------------------------------------------------
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
  param = 0._real32
  gradient = 0._real32
  optimiser_var%iter = optimiser_var%iter + 1
  call optimiser_var%minimise(param, gradient)


!-------------------------------------------------------------------------------
! test sgd optimiser
!-------------------------------------------------------------------------------
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
  param = 0._real32
  gradient = 0._real32
  optimiser_var%iter = optimiser_var%iter + 1
  call optimiser_var%minimise(param, gradient)


!-------------------------------------------------------------------------------
! test rmsprop optimiser
!-------------------------------------------------------------------------------
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
  param = 0._real32
  gradient = 0._real32
  optimiser_var%iter = optimiser_var%iter + 1
  call optimiser_var%minimise(param, gradient)


!-------------------------------------------------------------------------------
! test adagrad optimiser
!-------------------------------------------------------------------------------
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
  param = 0._real32
  gradient = 0._real32
  optimiser_var%iter = optimiser_var%iter + 1
  call optimiser_var%minimise(param, gradient)


!-------------------------------------------------------------------------------
! test adam optimiser with l1 regulariser
!-------------------------------------------------------------------------------
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
  param = 0._real32
  gradient = 0._real32
  optimiser_var%iter = optimiser_var%iter + 1
  call optimiser_var%minimise(param, gradient)


!-------------------------------------------------------------------------------
! test adam optimiser with l2 regulariser
!-------------------------------------------------------------------------------
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
  param = 0._real32
  gradient = 0._real32
  optimiser_var%iter = optimiser_var%iter + 1
  call optimiser_var%minimise(param, gradient)


!-------------------------------------------------------------------------------
! test adam optimiser with l2 decoupled regulariser
!-------------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=adam_optimiser_type( &
       learning_rate = 0.1, &
       num_params = num_params, &
       beta1 = 0.9, &
       beta2 = 0.999, &
       epsilon = 1e-8, &
       lr_decay = base_lr_decay_type(), &
       regulariser=l2_regulariser_type(decoupled=.true.) &
  ))
  select type(optimiser_var)
  type is (adam_optimiser_type)
     write(*,*) "adam_optimiser_type"
  class default
     write(0,*) "Failed to allocate adam optimiser"
     success = .false.
  end select
  param = 0._real32
  gradient = 0._real32
  optimiser_var%iter = optimiser_var%iter + 1
  call optimiser_var%minimise(param, gradient)


!-------------------------------------------------------------------------------
! test plp optimiser
!-------------------------------------------------------------------------------
  deallocate(optimiser_var)
  allocate(optimiser_var, source=plp_optimiser_type( &
       learning_rate = 0.1_real32, &
       num_params = num_params, &
       momentum = 0.9_real32, &
       nesterov = .true., &
       lr_decay = base_lr_decay_type(), &
       regulariser=l1_regulariser_type() &
  ))
  select type(optimiser_var)
  type is (plp_optimiser_type)
     write(*,*) "plp_optimiser_type"
  class default
     write(0,*) "Failed to allocate plp optimiser"
     success = .false.
  end select
  param = 0._real32
  gradient = 0._real32
  optimiser_var%iter = optimiser_var%iter + 1
  call optimiser_var%minimise(param, gradient)


!-------------------------------------------------------------------------------
! test plp prediction behaviour
!-------------------------------------------------------------------------------
  plp_behaviour = plp_optimiser_type( &
       learning_rate = 1._real32, &
       num_params = 1)
  plp_param = 0._real32
  do i = 1, size(plp_gradient_schedule)
     plp_gradient(1) = plp_gradient_schedule(i)
     plp_behaviour%iter = plp_behaviour%iter + 1
     call plp_behaviour%minimise(plp_param, plp_gradient)
  end do
  if(abs(plp_param(1) + 8._real32).gt.tol)then
     write(0,*) "PLP prediction produced", plp_param(1), "expected -8.0"
     success = .false.
  end if

  plp_gradient(1) = 1._real32
  plp_behaviour%iter = plp_behaviour%iter + 1
  call plp_behaviour%minimise(plp_param, plp_gradient)
  if(abs(plp_param(1) + 9._real32).gt.tol)then
     write(0,*) "PLP cycle restart produced", plp_param(1), "expected -9.0"
     success = .false.
  end if


!-------------------------------------------------------------------------------
! check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_optimiser passed all tests'
  else
     write(0,*) 'test_optimiser failed one or more tests'
     stop 1
  end if


end program test_optimiser
