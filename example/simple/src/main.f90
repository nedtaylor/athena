program simple
  use athena
  use constants_minst, only: real12, pi

  implicit none

  type(network_type) :: network
  real(real12), allocatable, dimension(:,:) :: x, y
  
  integer, parameter :: num_iterations = 500

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  integer :: i, n
  

  !! set random seed
  seed_size = 8
  call random_seed(size=seed_size)
  seed = [1,1,1,1,1,1,1,1]
  call random_seed(put=seed)

  write(*,*) "Simple function approximation using a fully-connected neural network"
  write(*,*) "--------------------------------------------------"
  write(*,*) "Based on example provided in the neural-fortran code"

  !! create network
  ! call network%add(input1d_layer_type(input_shape=[1]))
  call network%add(full_layer_type(num_inputs=3,num_outputs=5, activation_function="tanh"))
  call network%add(full_layer_type(num_outputs=2, activation_function="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=1._real12), &
       loss_method="mse", metrics=["loss"], verbose=1)
  call network%set_batch_size(1)

  !! create train data
  x = reshape([0.2, 0.4, 0.6], [3,1])
  y = reshape([0.123456, 0.246802], [2,1])

  !! train network
  write(*,*) "Training network"
  write(*,*) "----------------"
  write(*,*) "Iteration, Loss"
  do n = 0, num_iterations

    call network%set_batch_size(1)
    call network%forward(x)
    call network%backward(y)
    call network%update()

    if (mod(n, 50) == 0) &
        write(*,'(I7,2(1X,F9.6))') n, network%predict(input=x)

  end do

end program simple
