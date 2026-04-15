program wandb_sweep_example
  !! Hyperparameter sweep example using `athena_wandb`.
  !!
  !! Demonstrates how to:
  !!   1. Register a Bayesian sweep with `wandb_sweep`.
  !!   2. Start a sweep agent in a background thread with
  !!      `wandb_sweep_start_agent`.
  !!   3. For each run: call `wandb_sweep_next_params` (blocks until
  !!      wandb.agent calls wandb.init and samples hyperparameters),
  !!      read the values with `wandb_config_get`, train, log, then
  !!      call `wandb_sweep_run_done` to signal the agent.
  !!
  !! The sweep searches over:
  !!   - learning_rate  : log-uniform in [1e-3, 1.0]
  !!   - num_hidden     : one of {4, 8, 16, 32}
  !!   - activation     : one of {"tanh", "relu", "sigmoid"}
  !!
  !! Each run trains a small network to approximate sin(x) and reports
  !! the final MSE so the sweep can identify the best combination.
  !!
  !! ## Usage
  !!
  !!   source tools/setup_wf_env.sh             # or set FPM_CFLAGS/LDFLAGS
  !!   fpm run wandb_sweep --example
  !!
  use athena
  use athena_wandb
  use coreutils, only: real32

  implicit none

  integer, parameter :: NUM_SWEEP_RUNS = 5   ! how many runs to execute
  integer, parameter :: NUM_ITERATIONS = 5000
  integer, parameter :: TEST_SIZE      = 30
  real(real32), parameter :: PI = 4.0_real32 * atan(1.0_real32)

  type(wandb_sweep_config_type) :: sweep_config
  character(len=256) :: sweep_id
  character(len=64)  :: project
  integer :: run_idx

  project  = "athena-sweep"
  sweep_id = ''

  ! --------------------------------------------------------------------------
  ! 1.  Build the sweep configuration with the typed builder API.
  ! --------------------------------------------------------------------------
  call sweep_config%set_method("bayes")
  call sweep_config%set_metric("final_mse", "minimize")

  ! Continuous hyperparameter: log-uniform range
  call sweep_config%add_param_range("learning_rate", &
       min_val=1.0e-3_real32, max_val=1.0_real32, &
       distribution="log_uniform_values")

  ! Discrete integer values
  call sweep_config%add_param_values("num_hidden", [4, 8, 16, 32])

  ! Discrete string values (pad to equal declared length)
  call sweep_config%add_param_values("activation", &
       ["tanh   ", "relu   ", "sigmoid"])

  ! --------------------------------------------------------------------------
  ! 2.  Register the sweep and obtain its ID.
  ! --------------------------------------------------------------------------
  write(*,'(a)') "Registering wandb sweep..."
  write(*,'(2a)') "  config JSON: ", sweep_config%to_json()
  call wandb_sweep( &
       config   = sweep_config, &
       project  = trim(project), &
       sweep_id = sweep_id        &
  )

  if(len_trim(sweep_id) .eq. 0)then
     write(*,'(a)') "ERROR: failed to create sweep (is wandb logged in?)."
     stop 1
  end if
  write(*,'(2a)') "Sweep ID: ", trim(sweep_id)
  write(*,'(a)') ""

  ! --------------------------------------------------------------------------
  ! 3.  Start the sweep agent in a background Python thread.
  !
  !     The agent runs NUM_SWEEP_RUNS iterations.  For each run it calls
  !     wandb.init() which contacts the sweep controller and receives a fresh
  !     set of hyperparameters in wandb.config.  It then signals
  !     wandb_sweep_next_params and waits until wandb_sweep_run_done() is
  !     called before finishing the run and moving on.
  ! --------------------------------------------------------------------------
  call wandb_sweep_start_agent( &
       sweep_id = trim(sweep_id), &
       project  = trim(project),  &
       count    = NUM_SWEEP_RUNS  &
  )

  ! --------------------------------------------------------------------------
  ! 4.  Training loop: one iteration per sweep run.
  ! --------------------------------------------------------------------------
  do run_idx = 1, NUM_SWEEP_RUNS
     call run_training(run_idx, trim(project))
  end do

  ! Shut down the embedded Python interpreter after all runs are done.
  call wandb_shutdown()

  write(*,'(a)') ""
  write(*,'(a)') "Sweep complete.  Open your wandb dashboard to compare runs."

contains

  !----------------------------------------------------------------------------
  ! run_training
  !
  ! Executes a single sweep run:
  !   - calls wandb_sweep_next_params to block until the agent has called
  !     wandb.init() and sampled hyperparameters are available
  !   - reads hyperparameters from wandb.config
  !   - builds and trains the network
  !   - logs metrics
  !   - calls wandb_sweep_run_done() so the agent finishes this run
  !----------------------------------------------------------------------------
  subroutine run_training(run_idx, project)
    integer,          intent(in) :: run_idx
    character(len=*), intent(in) :: project

    type(network_type) :: network
    real(real32), dimension(1,1) :: x, y
    type(array_type)         :: y_arr(1,1)
    type(array_type), pointer :: loss

    ! Hyperparameters read from sweep config
    real(real32)      :: lr
    integer           :: num_hidden
    character(len=64) :: activation

    character(len=512) :: params_json
    real(real32), dimension(1,TEST_SIZE) :: x_test, y_test, y_pred
    real(real32) :: mse
    integer :: i, n

    ! ------------------------------------------------------------------
    ! a) Block until wandb.agent has called wandb.init() and the
    !    sweep controller has provided sampled hyperparameters.
    ! ------------------------------------------------------------------
    call wandb_sweep_next_params(params_json)

    ! ------------------------------------------------------------------
    ! b) Read the sampled hyperparameters from wandb.config.
    ! ------------------------------------------------------------------
    call wandb_config_get("learning_rate", lr,        default_value=0.01_real32)
    call wandb_config_get("num_hidden",    num_hidden, default_value=8)
    call wandb_config_get("activation",    activation, default_value="tanh")

    write(*,'(a,i0,a)')     "--- Run ", run_idx, " ---"
    write(*,'(2a)')         "  params JSON  : ", trim(params_json)
    write(*,'(a,f8.5)')     "  learning_rate: ", lr
    write(*,'(a,i0)')       "  num_hidden   : ", num_hidden
    write(*,'(2a)')         "  activation   : ", trim(activation)

    ! ------------------------------------------------------------------
    ! c) Build network using sampled hyperparameters.
    ! ------------------------------------------------------------------
    call network%add(full_layer_type( &
         num_inputs=1, num_outputs=num_hidden, &
         activation=trim(activation)))
    call network%add(full_layer_type( &
         num_outputs=1, activation="sigmoid"))
    call network%compile( &
         optimiser   = base_optimiser_type(learning_rate=lr), &
         loss_method = "mse", &
         metrics     = ["loss"], &
         verbose     = 0)
    call network%set_batch_size(1)

    ! Prepare test data: sin(x) normalised to [0,1].
    do i = 1, TEST_SIZE
       x_test(1,i) = (real(i - 1, real32) * 2.0_real32 * PI) / TEST_SIZE
       y_test(1,i) = (sin(x_test(1,i)) + 1.0_real32) / 2.0_real32
    end do

    call y_arr(1,1)%allocate(array_shape=[1,1])

    ! ------------------------------------------------------------------
    ! d) Training loop.
    ! ------------------------------------------------------------------
    do n = 1, NUM_ITERATIONS
       call random_number(x(1,1))
       x(1,1) = x(1,1) * 2.0_real32 * PI
       y(1,1) = (sin(x(1,1)) + 1.0_real32) / 2.0_real32

       y_arr(1,1)%val = y

       call network%set_batch_size(1)
       call network%forward(x)
       network%expected_array = y_arr
       loss => network%loss_eval(1, 1)
       call loss%grad_reverse()
       call network%update()

       if(mod(n, 500) .eq. 0)then
          y_pred = network%predict(input=x_test)
          mse    = sum((y_pred - y_test)**2) / real(TEST_SIZE, real32)
          call wandb_log("mse", mse, step=n)
       end if
    end do

    ! Final evaluation.
    y_pred = network%predict(input=x_test)
    mse    = sum((y_pred - y_test)**2) / real(TEST_SIZE, real32)

    call wandb_log("final_mse", mse)
    write(*,'(a,f10.6)') "  final_mse    : ", mse

    ! ------------------------------------------------------------------
    ! e) Signal the agent that training is done for this run.
    !    The callback calls wandb.finish() and the agent requests the
    !    next run's hyperparameters from the sweep controller.
    ! ------------------------------------------------------------------
    call wandb_sweep_run_done()

  end subroutine run_training


  !----------------------------------------------------------------------------
  ! Minimal integer-to-string helper (avoids write-to-string portability issues)
  !----------------------------------------------------------------------------
  function int_to_str(n) result(s)
    integer, intent(in) :: n
    character(len=12) :: s
    write(s,'(i0)') n
  end function int_to_str

end program wandb_sweep_example
