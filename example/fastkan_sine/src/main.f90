program fastkan_sine
  !! Sine function approximation using a FastKAN (RBF) layer
  !!
  !! This example demonstrates that a single FastKAN layer with Gaussian RBF
  !! activations can learn to approximate y = (sin(pi*x) + 1) / 2 over [-1, 1].
  !!
  !! The FastKAN layer uses radial basis function (RBF) activations:
  !!   phi_{i,k}(x) = exp(-0.5 * ((x - c_{i,k}) / sigma_{i,k})^2)
  !! followed by a linear combination across all basis functions.
  use athena
  use coreutils, only: real32

  implicit none

  type(network_type) :: network
  real(real32), dimension(1,1) :: x, y
  type(array_type), pointer :: loss

  integer, parameter :: num_iterations = 10000
  integer, parameter :: test_size = 30

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  real(real32), dimension(1,test_size) :: x_test, y_test, y_pred
  real(real32), parameter :: pi = 4.0_real32 * atan(1.0_real32)
  real(real32) :: test_mse

  integer :: i, n


  !-----------------------------------------------------------------------------
  ! set random seed for reproducibility
  !-----------------------------------------------------------------------------
  seed_size = 8
  allocate(seed(seed_size))
  seed = 42
  call random_seed(put=seed)

  write(*,*) "Sine function approximation using a FastKAN layer (RBF)"
  write(*,*) "----------------------------------------------------------"
  write(*,*) ""


  !-----------------------------------------------------------------------------
  ! create network with a single KAN layer
  !-----------------------------------------------------------------------------
  call network%add(fastkan_layer_type( &
       num_inputs=1, num_outputs=1, n_basis=10))
  call network%compile( &
       optimiser = sgd_optimiser_type(learning_rate=0.01_real32), &
       loss_method="mse", metrics=["loss"], verbose=1)
  call network%set_batch_size(1)


  !-----------------------------------------------------------------------------
  ! create test data: x in [-1,1], y = (sin(pi*x) + 1) / 2
  !-----------------------------------------------------------------------------
  do i = 1, test_size
     x_test(1,i) = -1.0_real32 + 2.0_real32 * real(i - 1, real32) / &
          real(test_size - 1, real32)
     y_test(1,i) = ( sin(pi * x_test(1,i)) + 1.0_real32 ) / 2.0_real32
  end do

  allocate(network%expected_array(1,1))
  call network%expected_array(1,1)%allocate(array_shape=[1,1])


  !-----------------------------------------------------------------------------
  ! train network
  !-----------------------------------------------------------------------------
  write(*,*) "Training network"
  write(*,*) "----------------"
  write(*,'(A10,A12)') "Iteration", "Test MSE"
  do n = 0, num_iterations
     call random_number(x)
     x = x * 2.0_real32 - 1.0_real32  ! [-1, 1]
     y = (sin(pi * x) + 1.0_real32) / 2.0_real32

     network%expected_array(1,1)%val = y

     call network%set_batch_size(1)
     call network%forward(x)
     loss => network%loss_eval(1, 1)
     call loss%grad_reverse()
     call network%update()

     if (mod(n, 1000) == 0) then
        y_pred(:,:) = network%predict(input=x_test(:,:))
        test_mse = sum((y_pred - y_test)**2) / size(y_pred)
        write(*,'(I10,F12.6)') n, test_mse
     end if

     call loss%nullify_graph()
     loss => null()
  end do


  !-----------------------------------------------------------------------------
  ! final evaluation
  !-----------------------------------------------------------------------------
  write(*,*) ""
  write(*,*) "Final predictions vs targets:"
  write(*,'(A10,A12,A12)') "x", "target", "predicted"
  y_pred(:,:) = network%predict(input=x_test(:,:))
  do i = 1, test_size
     write(*,'(F10.4,F12.4,F12.4)') x_test(1,i), y_test(1,i), y_pred(1,i)
  end do

  test_mse = sum((y_pred - y_test)**2) / size(y_pred)
  write(*,*) ""
  write(*,'(A,F10.6)') " Final test MSE: ", test_mse

  deallocate(seed)

end program fastkan_sine
