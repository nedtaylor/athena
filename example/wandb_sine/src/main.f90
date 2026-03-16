program wandb_sine
  !! Sine function approximation with wandb experiment tracking.
  !!
  !! This example demonstrates how to use the `athena_wandb` module to log
  !! training metrics and hyper-parameters to Weights & Biases.
  !!
  !! This file has been generated using LLM AI code agents and has not yet been
  !! thoroughly tested.
  !!
  !! It trains a small fully-connected network to approximate sin(x)
  !! over [0, 2*pi] (the same task as the `sine` example), while
  !! sending loss curves and config to wandb.
  !!
  !! ## Usage
  !!
  !! 1.  Make sure `wandb` is installed: `pip install wandb`
  !! 2.  Log in once:  `wandb login`
  !! 3.  Build and run:
  !!
  !!     ```bash
  !!     cmake -S example/wandb_sine -B build_wandb_sine
  !!     cmake --build build_wandb_sine
  !!     ./build_wandb_sine/wandb_sine
  !!     ```
  !!
  !! 4.  Open the printed wandb URL to see the dashboard.
  !!
  use athena
  use athena_wandb
  use coreutils, only: real32

  implicit none

  type(network_type) :: network
  real(real32), dimension(1,1) :: x, y
  type(array_type) :: x_array(1), y_array(1,1)
  type(array_type), pointer :: loss

  integer, parameter :: num_iterations = 10000
  integer, parameter :: num_hidden     = 5
  integer, parameter :: test_size      = 30
  real(real32), parameter :: learning_rate = 1.0_real32
  real(real32), parameter :: pi = 4.0_real32 * atan(1.0_real32)

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  real(real32), dimension(1,test_size) :: x_test, y_test, y_pred
  real(real32) :: mse

  integer :: i, n


  !-----------------------------------------------------------------------------
  ! set random seed (for reproducibility)
  !-----------------------------------------------------------------------------
  seed_size = 8
  call random_seed(size=seed_size)
  seed = [1, 1, 1, 1, 1, 1, 1, 1]
  call random_seed(put=seed)


  !-----------------------------------------------------------------------------
  ! initialise wandb
  !-----------------------------------------------------------------------------
  write(*,*) "Initializing wandb run..."
  call wandb_init(project="athena-examples", name="sine-approximation")
  write(*,*) "wandb run initialized. Logging hyper-parameters and training metrics..."

  ! log hyper-parameters
  call wandb_config_set("num_iterations", num_iterations)
  call wandb_config_set("num_hidden",     num_hidden)
  call wandb_config_set("learning_rate",  learning_rate)
  call wandb_config_set("test_size",      test_size)
  call wandb_config_set("activation",     "tanh")


  !-----------------------------------------------------------------------------
  ! create network
  !-----------------------------------------------------------------------------
  call network%add(full_layer_type( &
       num_inputs=1, num_outputs=num_hidden, activation="tanh"))
  call network%add(full_layer_type( &
       num_outputs=1, activation="sigmoid"))
  call network%compile( &
       optimiser = base_optimiser_type(learning_rate=learning_rate), &
       loss_method="mse", metrics=["loss"], verbose=1)
  call network%set_batch_size(1)


  !-----------------------------------------------------------------------------
  ! prepare test data
  !-----------------------------------------------------------------------------
  do i = 1, test_size
     x_test(1,i) = (real(i - 1, real32) * 2.0_real32 * pi) / test_size
     y_test(1,i) = (sin(x_test(1,i)) + 1.0_real32) / 2.0_real32
  end do

  call x_array(1)%allocate(array_shape=[1,1])
  call y_array(1,1)%allocate(array_shape=[1,1])


  !-----------------------------------------------------------------------------
  ! train network
  !-----------------------------------------------------------------------------
  write(*,*) "Training network with wandb logging"
  write(*,*) "------------------------------------"
  write(*,*) "Iteration,      Loss"

  do n = 0, num_iterations
     call random_number(x)
     x = x * 2.0_real32 * pi
     y = (sin(x) + 1.0_real32) / 2.0_real32

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
        mse = sum((y_pred - y_test)**2) / size(y_pred)
        write(*,'(I7,1X,F12.6)') n, mse

        ! log to wandb
        call wandb_log("mse", mse, step=n)
     end if
  end do


  !-----------------------------------------------------------------------------
  ! final evaluation
  !-----------------------------------------------------------------------------
  y_pred(:,:) = network%predict(input=x_test(:,:))
  mse = sum((y_pred - y_test)**2) / size(y_pred)
  write(*,*) ""
  write(*,'("Final MSE: ",F12.6)') mse
  call wandb_log("final_mse", mse, step=num_iterations)


  !-----------------------------------------------------------------------------
  ! finish wandb run
  !-----------------------------------------------------------------------------
  call wandb_finish()

  write(*,*) "Done. Check your wandb dashboard for results."

end program wandb_sine
