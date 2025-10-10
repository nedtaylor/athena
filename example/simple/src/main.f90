
program simple
  !! Simple function approximation using a fully-connected neural network
  !!
  !! This file contains a modified version of the "simple" example found in neural fortran:
  !! https://github.com/modern-fortran/neural-fortran/blob/main/example/simple.f90
  use athena
  use coreutils, only: real32
  use constants_mnist, only: pi

  implicit none

  type(network_type) :: network
  real(real32), allocatable, dimension(:,:) :: x, y, prediction
  type(array_type) :: x_array(1), y_array(1,1)
  type(array_type), allocatable :: loss(:,:)

  integer, parameter :: num_iterations = 500

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  integer :: i, n


  !-----------------------------------------------------------------------------
  ! set random seed
  !-----------------------------------------------------------------------------
  call random_seed(size=seed_size)
  allocate(seed(seed_size), source = 1)
  call random_seed(put=seed)

  write(*,*) "Simple function approximation using a fully-connected neural network"
  write(*,*) "--------------------------------------------------"
  write(*,*) "Based on example provided in the neural-fortran code"


  !-----------------------------------------------------------------------------
  ! create network
  !-----------------------------------------------------------------------------
  ! call network%add(input1d_layer_type(input_shape=[1]))
  call network%add(full_layer_type(num_inputs=3,num_outputs=5, &
       activation_function="tanh"))
  call network%add(full_layer_type(num_outputs=2, activation_function="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=1._real32), &
       loss_method="mse", metrics=["loss"], verbose=1)
  call network%set_batch_size(1)


  !-----------------------------------------------------------------------------
  ! create train data
  !-----------------------------------------------------------------------------
  x = reshape([0.2, 0.4, 0.6], [3,1])
  y = reshape([0.123456, 0.246802], [2,1])
  call x_array(1)%allocate(source=x)
  call y_array(1,1)%allocate(source=y)


  !-----------------------------------------------------------------------------
  ! train network
  !-----------------------------------------------------------------------------
  write(*,*) "Training network"
  write(*,*) "----------------"
  write(*,*) "Iteration, Loss"
  do n = 0, num_iterations

     call network%set_batch_size(1)
     call network%forward(x)
     loss = network%loss%compute_generic( &
          network%model(network%leaf_vertices(1))%layer%output,  y_array, x_array &
     )
     call loss(1,1)%grad_reverse()
     ! call network%backward(y)
     call network%update()

     prediction = network%predict(input=x)
     if (mod(n, 50) == 0) write(*,'(I7,2(1X,F9.6))') n, prediction

  end do

end program simple
