program benchmark_sine_neural_fortran
  !! Matched sine-function training benchmark using neural-fortran.
  !!
  !! Same architecture, sampling, MSE loss and iteration count as
  !! benchmark_sine_athena. Athena uses its mean/2 MSE with learning rate 2;
  !! this app uses the mean MSE with learning rate 1 for matching gradients.
  use iso_fortran_env, only: real64
  use nf, only: network, input, dense, tanhf, sigmoid, sgd, mse
  use bench_timer, only: wall_time
  implicit none

  type(network) :: net
  type(sgd) :: optimizer
  real :: x(1), y(1)
  real, parameter :: pi = 4 * atan(1.)
  integer, parameter :: num_iterations = 100000
  integer, parameter :: test_size = 30
  real :: xtest(test_size), ytest(test_size), ypred(test_size)
  integer :: seed_size, i, n
  integer, allocatable :: seed(:)
  real(real64) :: t0, t1
  real :: cpu0, cpu1

  call random_seed(size=seed_size)
  allocate(seed(seed_size), source=1)
  call random_seed(put=seed)

  net = network([ &
       input(1), &
       dense(5, activation=tanhf()), &
       dense(1, activation=sigmoid()) &
  ])
  net%loss = mse()
  optimizer = sgd(learning_rate=1.)

  xtest = [((i - 1) * 2 * pi / test_size, i = 1, test_size)]
  ytest = (sin(xtest) + 1) / 2

  ! Re-seed after construction so both apps draw the same x sequence.
  call random_seed(put=seed)

  call cpu_time(cpu0)
  t0 = wall_time()
  do n = 0, num_iterations
     call random_number(x)
     x = x * 2 * pi
     y = (sin(x) + 1) / 2

     call net%forward(x)
     call net%backward(y)
     call net%update(optimizer=optimizer)
  end do
  t1 = wall_time()
  call cpu_time(cpu1)

  ypred = [(net%predict([xtest(i)]), i = 1, test_size)]
  write(*,'("RESULT example=sine library=neural-fortran train_s=",F10.6, &
       &" cpu_s=",F10.6," test_mse=",ES12.4)') &
  t1-t0, cpu1-cpu0, &
  sum((ypred - ytest)**2) / size(ypred)
end program benchmark_sine_neural_fortran
