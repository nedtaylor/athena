program benchmark_simple_neural_fortran
  !! Matched simple-function training benchmark using neural-fortran.
  !!
  !! Same architecture, data, MSE loss and iteration count as
  !! benchmark_simple_athena. Athena uses its mean/2 MSE with learning rate 2;
  !! this app uses the mean MSE with learning rate 1 for matching gradients.
  !! One 501-step pass, timed with a nanosecond wall clock.
  use iso_fortran_env, only: real64
  use nf, only: network, input, dense, tanhf, sigmoid, sgd, mse
  use bench_timer, only: wall_time
  implicit none

  type(network) :: net
  type(sgd) :: optimizer
  real, allocatable :: x(:), y(:), prediction(:)
  integer, parameter :: num_iterations = 500
  integer :: seed_size, n
  integer, allocatable :: seed(:)
  real(real64) :: t0, t1
  real :: cpu0, cpu1

  call random_seed(size=seed_size)
  allocate(seed(seed_size), source=1)
  call random_seed(put=seed)

  net = network([ &
       input(3), &
       dense(5, activation=tanhf()), &
       dense(2, activation=sigmoid()) &
  ])
  net%loss = mse()
  optimizer = sgd(learning_rate=1.)

  x = [0.2, 0.4, 0.6]
  y = [0.123456, 0.246802]

  call cpu_time(cpu0)
  t0 = wall_time()
  do n = 0, num_iterations
     call net%forward(x)
     call net%backward(y)
     call net%update(optimizer=optimizer)
  end do
  t1 = wall_time()
  call cpu_time(cpu1)

  prediction = net%predict(x)
  write(*,'("RESULT example=simple library=neural-fortran train_s=",F10.6, &
       &" cpu_s=",F10.6," final_mse=",ES12.4)') &
  t1-t0, cpu1-cpu0, &
  sum((prediction - y)**2) / size(prediction)
end program benchmark_simple_neural_fortran
