program benchmark_sine_athena
  !! Matched sine-function training benchmark using Athena.
  !!
  !! Matches benchmark_sine_neural_fortran: input 1 -> dense 5 tanh ->
  !! dense 1 sigmoid, 100,001 random samples from y = (sin(x)+1)/2 with
  !! x in [0, 2*pi], MSE loss, SGD learning rate 2, batch 1. Athena's MSE is
  !! mean/2, so the doubled rate matches neural-fortran's mean loss with rate 1.
  use iso_fortran_env, only: real32, real64
  use athena, only: network_type, full_layer_type, base_optimiser_type, array_type
  use bench_timer, only: wall_time
  implicit none

  type(network_type) :: network
  real(real32), dimension(1,1) :: x, y
  real(real32), dimension(1,30) :: x_test, y_test, y_pred
  type(array_type) :: y_array(1,1)
  type(array_type), pointer :: loss
  real(real32), parameter :: pi = 3.14159265358979323846_real32
  integer, parameter :: num_iterations = 100000
  integer, parameter :: test_size = 30
  integer :: seed_size, i, n
  integer, allocatable :: seed(:)
  real(real64) :: t0, t1
  real :: cpu0, cpu1

  call random_seed(size=seed_size)
  allocate(seed(seed_size), source=1)
  call random_seed(put=seed)

  do i = 1, test_size
     x_test(1,i) = (i - 1) * 2._real32 * pi / test_size
     y_test(1,i) = (sin(x_test(1,i)) + 1._real32) / 2._real32
  end do
  call y_array(1,1)%allocate(array_shape=[1,1], temporary=.false.)

  call network%add(full_layer_type(num_inputs=1, num_outputs=5, &
       activation="tanh"))
  call network%add(full_layer_type(num_outputs=1, activation="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=2._real32), &
       loss_method="mse", verbose=-1)
  call network%set_batch_size(1)

  ! Re-seed after construction so both apps draw the same x sequence.
  call random_seed(put=seed)

  call cpu_time(cpu0)
  t0 = wall_time()
  do n = 0, num_iterations
     call random_number(x)
     x = x * 2._real32 * pi
     y = (sin(x) + 1._real32) / 2._real32
     y_array(1,1)%val = y

     call network%forward(x)
     network%expected_array = y_array
     loss => network%loss_eval(1, 1)
     call loss%grad_reverse()
     call network%update()
     ! Release the consumed graph, as network%train does, so that 100k
     ! iterations do not accumulate expression history.
     call loss%release_expression()
     nullify(loss)
  end do
  t1 = wall_time()
  call cpu_time(cpu1)

  y_pred = network%predict(input=x_test)
  write(*,'("RESULT example=sine library=athena train_s=",F10.6, &
       &" cpu_s=",F10.6," test_mse=",ES12.4)') &
  t1-t0, cpu1-cpu0, &
  sum((y_pred - y_test)**2) / real(size(y_pred), real32)
end program benchmark_sine_athena
