program benchmark_athena
  !! One-epoch MNIST training benchmark using Athena.
  !!
  !! Architecture and setting match benchmark_neural_fortran:
  !!   conv2d(32, 3x3, relu) -> maxpool2d(2,2) -> dense(100, relu)
  !!   -> dense(10, softmax), MSE loss, SGD (no momentum), batch 32.
  use iso_fortran_env, only: real32
  use athena, only: network_type, conv2d_layer_type, maxpool2d_layer_type, &
       full_layer_type, sgd_optimiser_type
  use mnist_data, only: load_mnist
  implicit none

  type(network_type) :: net
  type(sgd_optimiser_type) :: optimiser
  real(real32), allocatable :: x_train(:,:), x_test(:,:)
  real(real32), allocatable :: x4_train(:,:,:,:), x4_test(:,:,:,:)
  real(real32), allocatable :: y_train(:,:), y_test(:,:)
  integer, allocatable :: labels_train(:), labels_test(:)
  character(len=1024) :: dir
  integer :: i, n_train, seed(8)
  integer :: count0, count1, rate
  real :: t0, t1

  dir = '../../../MNIST-txt'
  if(command_argument_count() >= 1) call get_command_argument(1, dir)
  n_train = 50000

  seed = 42
  call random_seed(put=seed)
  call load_mnist(trim(dir), n_train, x_train, labels_train, x_test, labels_test)

  allocate(x4_train(28,28,1,n_train), x4_test(28,28,1,size(x_test,2)))
  x4_train = reshape(x_train, [28,28,1,n_train])
  x4_test = reshape(x_test, [28,28,1,size(x_test,2)])
  allocate(y_train(10,n_train), y_test(10,size(x_test,2)))
  y_train = 0._real32
  y_test = 0._real32
  do i = 1, n_train
     y_train(labels_train(i)+1, i) = 1._real32
  end do
  do i = 1, size(x_test,2)
     y_test(labels_test(i)+1, i) = 1._real32
  end do

  call net%add(conv2d_layer_type( &
       input_shape = [28,28,1], num_filters = 32, kernel_size = 3, &
       stride = 1, padding = "none", activation = "relu" ))
  call net%add(maxpool2d_layer_type(pool_size = 2, stride = 2))
  call net%add(full_layer_type(num_outputs = 100, activation = "relu"))
  call net%add(full_layer_type(num_outputs = 10, activation = "softmax"))

  optimiser = sgd_optimiser_type(learning_rate = 0.01_real32, momentum = 0.0_real32)
  ! Accuracy is evaluated only after the timed epoch, as in the
  ! neural-fortran benchmark; per-batch scoring would be extra work.
  call net%compile(optimiser = optimiser, loss_method = "mse", &
       batch_size = 32, verbose = -1)

  call cpu_time(t0)
  call system_clock(count0, rate)
  call net%train(x4_train, y_train, 1, 32, &
       shuffle_batches = .true., verbose = -1)
  call system_clock(count1)
  call cpu_time(t1)

  call net%set_accuracy("cat")
  call net%test(x4_test, y_test, verbose = -1)
  write(*,'("RESULT library=athena train_s=",F8.3," cpu_s=",F8.3, &
       &" test_acc=",F6.4," test_loss=",F8.5)') &
  real(count1-count0)/real(rate), t1-t0, net%accuracy_val, net%loss_val
end program benchmark_athena
