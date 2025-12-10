program sine
  !! Sine function approximation using a fully-connected neural network
  !!
  !! This example demonstrates universal approximation capabilities of neural
  !! networks by learning to approximate the sine function \( y = \sin(x) \)
  !! over the domain \( x \in [0, 2\pi] \).
  !!
  !! ## Objective
  !!
  !! Learn the mapping:
  !! $$f_{\theta}: x \mapsto \sin(x)$$
  !!
  !! where \( \theta \) represents the network parameters (weights and biases).
  !!
  !! ## Network Architecture
  !!
  !! - Input layer: 1 feature (\( x \))
  !! - Hidden layer: 5 neurons with tanh activation
  !! - Output layer: 1 neuron with sigmoid activation
  !!
  !! The network approximates the sine function through:
  !! $$\hat{y} = \sigma\left(\mathbf{w}_2^T \tanh(\mathbf{W}_1 x + \mathbf{b}_1) + b_2\right)$$
  !!
  !! ## Training
  !!
  !! Minimizes mean squared error over 10,000 iterations with random sampling:
  !! $$\mathcal{L} = (y - \hat{y})^2 = (\sin(x) - f_{\theta}(x))^2$$
  !!
  !! ## Reference
  !!
  !! Modified version of the "sine" example from neural-fortran:
  !! https://github.com/modern-fortran/neural-fortran/blob/main/example/sine.f90
  use athena
  use coreutils, only: real32
  use constants_mnist, only: pi

  implicit none

  type(network_type) :: network
  real(real32), dimension(1,1) :: x, y
  type(array_type) :: x_array(1), y_array(1,1)
  type(array_type), pointer :: loss

  integer, parameter :: num_iterations = 10000
  integer, parameter :: test_size = 30

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  real(real32), dimension(1,test_size) :: x_test, y_test, y_pred

  integer :: i, n


  !-----------------------------------------------------------------------------
  ! set random seed
  !-----------------------------------------------------------------------------
  seed_size = 8
  call random_seed(size=seed_size)
  seed = [1,1,1,1,1,1,1,1]
  call random_seed(put=seed)

  write(*,*) "Sine function approximation using a neural network"
  write(*,*) "--------------------------------------------------"
  write(*,*) "Based on the example from the book 'Neural Networks and Deep Learning'???"
  write(*,*) "Based on example found in the neural-fortran code"


  !-----------------------------------------------------------------------------
  ! create network
  !-----------------------------------------------------------------------------
  ! call network%add(input1d_layer_type(input_shape=[1]))
  call network%add(full_layer_type(num_inputs=1,num_outputs=5, &
       activation="tanh"))
  call network%add(full_layer_type(num_outputs=1, activation="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=1._real32), &
       loss_method="mse", metrics=["loss"], verbose=1)
  call network%set_batch_size(1)


  !-----------------------------------------------------------------------------
  ! create train data
  !-----------------------------------------------------------------------------
  do i = 1, test_size
     x_test(1,i) = ( ( i - 1 ) * 2._real32 * pi ) / test_size
     y_test(1,i) = ( sin(x_test(1,i)) + 1._real32 ) / 2._real32
  end do
  call x_array(1)%allocate(array_shape=[1,1])
  call y_array(1,1)%allocate(array_shape=[1,1])


  !-----------------------------------------------------------------------------
  ! train network
  !-----------------------------------------------------------------------------
  write(*,*) "Training network"
  write(*,*) "----------------"
  write(*,*) "Iteration, Loss"
  do n = 0, num_iterations
     call random_number(x)
     x = x * 2._real32 * pi
     y = (sin(x) + 1._real32) / 2._real32
     x_array(1)%val = x
     y_array(1,1)%val = y

     call network%set_batch_size(1)
     call network%forward(x)
     network%expected_array = y_array
     loss => network%loss_eval(1, 1)
     call loss%grad_reverse()
     call network%update()

     if (mod(n, 1000) == 0) then
        y_pred(:,:) = network%predict(input=x_test(:,:))
        write(*,'(I7,1X,F9.6)') n, sum((y_pred - y_test)**2) / size(y_pred)
     end if

  end do

end program sine
