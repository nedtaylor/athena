!! This file contains a modified version of the "sine" example found in ...
!! ... neural fortran:
!! https://github.com/modern-fortran/neural-fortran/blob/main/example/sine.f90
program sine
  use athena
  use constants_mnist, only: real12, pi

  implicit none

  type(network_type) :: network
  real(real12), dimension(1,1) :: x, y
  
  integer, parameter :: num_iterations = 10000
  integer, parameter :: test_size = 30

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  real(real12), dimension(1,test_size) :: x_test, y_test, y_pred

  integer :: i, n
  

  !! set random seed
  seed_size = 8
  call random_seed(size=seed_size)
  seed = [1,1,1,1,1,1,1,1]
  call random_seed(put=seed)

  write(*,*) "Sine function approximation using a neural network"
  write(*,*) "--------------------------------------------------"
  write(*,*) "Based on the example from the book 'Neural Networks and Deep Learning'???"
  write(*,*) "Based on example found in the neural-fortran code"

  !! create network
  ! call network%add(input1d_layer_type(input_shape=[1]))
  call network%add(full_layer_type(num_inputs=1,num_outputs=5, activation_function="tanh"))
  call network%add(full_layer_type(num_outputs=1, activation_function="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=1._real12), &
       loss_method="mse", metrics=["loss"], verbose=1)
  call network%set_batch_size(1)

  !! create test data
  do i = 1, test_size
     x_test(1,i) = ( ( i - 1 ) * 2._real12 * pi ) / test_size
     y_test(1,i) = ( sin(x_test(1,i)) + 1._real12 ) / 2._real12
  end do

  !! train network
  write(*,*) "Training network"
  write(*,*) "----------------"
  write(*,*) "Iteration, Loss"
  do n = 0, num_iterations
    call random_number(x)
    x = x * 2._real12 * pi
    y = (sin(x) + 1._real12) / 2._real12

    call network%set_batch_size(1)
    call network%forward(x)
    call network%backward(y)
    call network%update()

    if (mod(n, 1000) == 0) then
      y_pred(:,:) = network%predict(input=x_test(:,:))
      write(*,'(I7,1X,F9.6)') n, sum((y_pred - y_test)**2) / size(y_pred)
    end if

  end do


end program sine
