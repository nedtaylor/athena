program benchmark_neural_fortran
  !! One-epoch MNIST training benchmark using neural-fortran.
  !!
  !! Same data split, architecture and hyperparameters as benchmark_athena.
  use iso_fortran_env, only: real32
  use nf, only: network, input, reshape, conv, maxpool, dense, relu, &
       softmax, sgd, mse, label_digits
  use mnist_data, only: load_mnist
  implicit none

  type(network) :: net
  real(real32), allocatable :: x_train(:,:), x_test(:,:)
  real(real32), allocatable :: y_train_digits(:,:), y_test_digits(:,:)
  integer, allocatable :: labels_train(:), labels_test(:)
  real(real32), allocatable :: labels_train_real(:), labels_test_real(:)
  character(len=1024) :: dir
  integer :: i, n_train, seed(8), good
  integer :: count0, count1, rate
  real :: t0, t1

  dir = '../../../MNIST-txt'
  if(command_argument_count() >= 1) call get_command_argument(1, dir)
  n_train = 50000

  seed = 42
  call random_seed(put=seed)
  call load_mnist(trim(dir), n_train, x_train, labels_train, x_test, labels_test)
  allocate(labels_train_real(n_train), labels_test_real(size(x_test,2)))
  labels_train_real = real(labels_train, real32)
  labels_test_real = real(labels_test, real32)
  y_train_digits = label_digits(labels_train_real)
  y_test_digits = label_digits(labels_test_real)

  net = network([ &
       input(784), &
       reshape(1, 28, 28), &
       conv(filters=32, kernel_width=3, kernel_height=3, activation=relu()), &
       maxpool(pool_width=2, pool_height=2, stride=2), &
       dense(100, activation=relu()), &
       dense(10, activation=softmax()) &
  ])

  call cpu_time(t0)
  call system_clock(count0, rate)
  call net%train(x_train, y_train_digits, 32, 1, &
       optimizer=sgd(learning_rate=0.01), loss=mse())
  call system_clock(count1)
  call cpu_time(t1)

  good = 0
  do i = 1, size(x_test,2)
     if(all(maxloc(net%predict(x_test(:,i))) == maxloc(y_test_digits(:,i)))) &
          good = good + 1
  end do
  write(*,'("RESULT library=neural-fortran train_s=",F8.3," cpu_s=",F8.3, &
       &" test_acc=",F6.4)') real(count1-count0)/real(rate), t1-t0, &
  real(good)/real(size(x_test,2))
end program benchmark_neural_fortran
