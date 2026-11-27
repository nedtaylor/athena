program benchmark_simple_athena
  !! Matched simple-function training benchmark using Athena.
  !!
  !! Matches benchmark_simple_neural_fortran: input 3 -> dense 5 tanh ->
  !! dense 2 sigmoid, x = [0.2, 0.4, 0.6], y = [0.123456, 0.246802],
  !! MSE loss, SGD learning rate 2, batch 1. Athena's MSE is mean/2, so the
  !! doubled rate matches neural-fortran's mean loss with rate 1.
  !! One 501-step pass, timed with a nanosecond wall clock.
  use iso_fortran_env, only: real32, real64
  use athena, only: network_type, full_layer_type, base_optimiser_type, array_type
  use bench_timer, only: wall_time
  implicit none

  type(network_type) :: network
  real(real32), allocatable :: x(:,:), y(:,:), prediction(:,:)
  type(array_type) :: y_array(1,1)
  type(array_type), pointer :: loss
  integer, parameter :: num_iterations = 500
  integer :: seed_size, n
  integer, allocatable :: seed(:)
  real(real64) :: t0, t1
  real :: cpu0, cpu1

  call random_seed(size=seed_size)
  allocate(seed(seed_size), source=1)
  call random_seed(put=seed)

  x = reshape([0.2_real32, 0.4_real32, 0.6_real32], [3,1])
  y = reshape([0.123456_real32, 0.246802_real32], [2,1])
  call y_array(1,1)%allocate(source=y, temporary=.false.)

  call network%add(full_layer_type(num_inputs=3, num_outputs=5, &
       activation="tanh"))
  call network%add(full_layer_type(num_outputs=2, activation="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=2._real32), &
       loss_method="mse", verbose=-1)
  call network%set_batch_size(1)

  call cpu_time(cpu0)
  t0 = wall_time()
  do n = 0, num_iterations
     call network%forward(x)
     network%expected_array = y_array
     loss => network%loss_eval(1, 1)
     call loss%grad_reverse()
     call network%update()
     call loss%release_expression()
     nullify(loss)
  end do
  t1 = wall_time()
  call cpu_time(cpu1)

  prediction = network%predict(input=x)
  write(*,'("RESULT example=simple library=athena train_s=",F10.6, &
       &" cpu_s=",F10.6," final_mse=",ES12.4)') &
  t1-t0, cpu1-cpu0, &
  sum((prediction - y)**2) / real(size(prediction), real32)
end program benchmark_simple_athena
