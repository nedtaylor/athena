program wandb_network_sine
  !! Sine-function approximation using wandb_network_type for automatic logging.
  !!
  !! This example demonstrates how the `wandb_network_type` derived type
  !! automatically logs training metrics (loss and accuracy) to Weights &
  !! Biases at the end of every epoch — no manual `wandb_log` calls needed
  !! in the training loop.
  !!
  !! Compared to the `wandb_sine` example (which manually calls wandb_log),
  !! this example uses the higher-level `network_type%train` interface and
  !! shows that simply switching from `network_type` to `wandb_network_type`
  !! is sufficient to gain W&B logging.
  !!
  !! ## Prerequisites
  !!
  !! 1.  Install wandb and log in once:
  !!     ```bash
  !!     pip install wandb && wandb login
  !!     ```
  !!
  !! 2.  Build with the `_WANDB` macro (already set in fpm.toml):
  !!     ```bash
  !!     source /path/to/wandb-fortran/tools/setup_env.sh
  !!     fpm run wandb_network_sine --example
  !!     ```
  !!
  !! 3.  Open the printed wandb URL to see the loss / accuracy curves.
  !!
#ifdef _WANDB
  use athena
  use athena_wandb
  use coreutils, only: real32

  implicit none

  !-----------------------------------------------------------------------------
  ! Hyper-parameters
  !-----------------------------------------------------------------------------
  integer,      parameter :: num_epochs    = 100
  integer,      parameter :: batch_size    = 32
  integer,      parameter :: num_train     = 512
  integer,      parameter :: num_val       = 128
  integer,      parameter :: num_hidden    = 32
  real(real32), parameter :: learning_rate = 1.0e-3_real32
  real(real32), parameter :: pi = 4.0_real32 * atan(1.0_real32)


  !-----------------------------------------------------------------------------
  ! Network and data declarations
  !-----------------------------------------------------------------------------
  type(wandb_network_type) :: net

  real(real32), dimension(1, num_train) :: x_train, y_train
  real(real32), dimension(1, num_val)   :: x_val,   y_val, y_pred

  real(real32) :: mse
  integer      :: i


  !-----------------------------------------------------------------------------
  ! Reproducible random seed
  !-----------------------------------------------------------------------------
  call random_seed()


  !-----------------------------------------------------------------------------
  ! Build training and validation datasets
  ! Target:  y = (sin(x) + 1) / 2   normalised to [0, 1]
  !-----------------------------------------------------------------------------
  do i = 1, num_train
     x_train(1, i) = real(i - 1, real32) * 2.0_real32 * pi / real(num_train, real32)
     y_train(1, i) = (sin(x_train(1, i)) + 1.0_real32) / 2.0_real32
  end do
  do i = 1, num_val
     x_val(1, i) = real(i - 1, real32) * 2.0_real32 * pi / real(num_val, real32)
     y_val(1, i) = (sin(x_val(1, i)) + 1.0_real32) / 2.0_real32
  end do


  !-----------------------------------------------------------------------------
  ! Build network
  !-----------------------------------------------------------------------------
  call net%add(full_layer_type(num_inputs=1,   num_outputs=num_hidden, &
       activation="tanh"))
  call net%add(full_layer_type(num_outputs=num_hidden, &
       activation="tanh"))
  call net%add(full_layer_type(num_outputs=1, &
       activation="sigmoid"))
  call net%compile( &
       optimiser       = adam_optimiser_type(learning_rate=learning_rate), &
       loss_method     = "mse", &
       accuracy_method = "mse", &
       metrics         = ["loss", "mae "], &
       batch_size      = batch_size, &
       verbose         = 0 &
  )


  !-----------------------------------------------------------------------------
  ! Initialise W&B run and log hyper-parameters
  !-----------------------------------------------------------------------------
  write(*, '(A)') "Initialising wandb run..."
  call net%wandb_setup(project="athena-examples", name="wandb-network-sine")

  call wandb_config_set("num_epochs",    num_epochs)
  call wandb_config_set("batch_size",    batch_size)
  call wandb_config_set("num_hidden",    num_hidden)
  call wandb_config_set("learning_rate", real(learning_rate, kind=8))
  call wandb_config_set("num_train",     num_train)
  call wandb_config_set("activation",    "tanh")
  call wandb_config_set("optimiser",     "adam")

  write(*, '(A)') "wandb run initialised."


  !-----------------------------------------------------------------------------
  ! Train — wandb_network_type logs loss & accuracy automatically each epoch
  !
  ! Disable the absolute-loss convergence threshold so all num_epochs run
  ! and every epoch is logged to W&B.
  !-----------------------------------------------------------------------------
  net%metrics(1)%threshold = 0.0_real32
  net%metrics(2)%threshold = 0.0_real32
  write(*, '(A)') ""
  write(*, '(A)') "Training network (loss/accuracy logged to W&B each epoch)..."
  call net%train( &
       input        = x_train, &
       output       = y_train, &
       num_epochs   = num_epochs, &
       verbose      = 0 &
  )


  !-----------------------------------------------------------------------------
  ! Evaluate on validation set
  !-----------------------------------------------------------------------------
  y_pred = net%predict(input=x_val)
  mse    = sum((y_pred - y_val)**2) / real(size(y_pred), real32)
  write(*, '(A)') ""
  write(*, '("Validation MSE: ", F10.6)') mse

  ! Log the final validation MSE explicitly
  call wandb_log("val_mse", real(mse, kind=8), step=num_epochs)


  !-----------------------------------------------------------------------------
  ! Finish W&B run
  !-----------------------------------------------------------------------------
  call wandb_finish()
  call wandb_shutdown()
  write(*, '(A)') "Done. Check your W&B dashboard for results."

#else
  implicit none
  write(*, '(A)') "wandb_network_sine: built without _WANDB — W&B support disabled."
  write(*, '(A)') "Recompile with the _WANDB macro to enable this example."
#endif

end program wandb_network_sine
