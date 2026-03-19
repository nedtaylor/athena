module athena_wandb
  !! Athena interface to Weights & Biases (wandb) experiment tracking.
  !!
  !! This module re-exports the full public API of the `wf` (wandb-fortran)
  !! package so that user code only needs a single `use athena_wandb`.
  !!
  !! When the library is compiled with the preprocessor macro `_WANDB`
  !! (i.e. `-D_WANDB` is passed to the compiler), this module also provides
  !! `wandb_network_type` — a thin `network_type` extension whose `train`
  !! method automatically logs epoch loss and accuracy to W&B after each
  !! epoch.
  !!
  !! ## Usage (basic wandb API)
  !!
  !! ```fortran
  !! use athena_wandb
  !! call wandb_init(project="my-project", name="run-01")
  !! call wandb_config_set("learning_rate", 0.001_real32)
  !! do epoch = 1, 100
  !!    ! ... training ...
  !!    call wandb_log("loss", loss_val, step=epoch)
  !! end do
  !! call wandb_finish()
  !! ```
  !!
  !! ## Usage (automatic logging via wandb_network_type)
  !!
  !! Requires `-D_WANDB` at compile time (set `macros = ["_WANDB"]` in
  !! `[preprocess.cpp]` of `fpm.toml`).
  !!
  !! ```fortran
  !! use athena
  !! use athena_wandb          ! provides wandb_network_type
  !! type(wandb_network_type) :: net
  !! ! ... add layers, compile ...
  !! call net%wandb_setup(project="my-project", name="run-01")
  !! call net%train(x, y, num_epochs=50, verbose=0)   ! logs per epoch
  !! call wandb_finish()
  !! ```
  !!
#ifdef _WANDB
  use wf, only: &
       wandb_init,              &
       wandb_log,               &
       wandb_finish,            &
       wandb_shutdown,          &
       wandb_config_set,        &
       wandb_config_get,        &
       wandb_sweep_config_type, &
       wandb_sweep,             &
       wandb_agent,             &
       wandb_sweep_start_agent, &
       wandb_sweep_next_params, &
       wandb_sweep_run_done
  use athena__network, only: network_type
  use coreutils, only: real32
#endif
  implicit none

  private

#ifdef _WANDB
  !---------------------------------------------------------------------------
  ! Re-export wandb-fortran public API
  !---------------------------------------------------------------------------
  public :: wandb_init
  public :: wandb_log
  public :: wandb_finish
  public :: wandb_shutdown
  public :: wandb_config_set
  public :: wandb_config_get
  public :: wandb_sweep_config_type
  public :: wandb_sweep
  public :: wandb_agent
  public :: wandb_sweep_start_agent
  public :: wandb_sweep_next_params
  public :: wandb_sweep_run_done


  !---------------------------------------------------------------------------
  ! wandb_network_type
  !---------------------------------------------------------------------------
  public :: wandb_network_type

  type, extends(network_type) :: wandb_network_type
     !! Extension of `network_type` with automatic W&B metric logging.
     !!
     !! Call `wandb_setup` once before training to initialise the W&B run,
     !! then call `train` as normal.  After every epoch the hook logs:
     !!   - "loss"     — mean batch loss over the epoch
     !!   - "accuracy" — mean batch accuracy over the epoch
     !!
     character(len=:), allocatable :: wandb_project
     !! W&B project name (set by `wandb_setup`)
     character(len=:), allocatable :: wandb_run_name
     !! W&B run display name (set by `wandb_setup`)
     logical :: log_batch_metrics = .false.
     !! Reserved for future use: log per-batch metrics in addition to epoch metrics
   contains
     procedure :: post_epoch_hook => wandb_post_epoch_hook
     !! Override the base no-op hook to log epoch metrics to W&B
     procedure :: wandb_setup
     !! Initialise the W&B run and store project / run-name metadata
  end type wandb_network_type

contains

!###############################################################################
  subroutine wandb_setup(this, project, name)
    !! Initialise a Weights & Biases run for this network.
    !!
    !! Call this once before `train`.  It is a thin wrapper around
    !! `wandb_init` that also stores the project and run name on the type so
    !! they are available for diagnostics.
    implicit none

    ! Arguments
    class(wandb_network_type), intent(inout) :: this
    !! Network instance
    character(*), intent(in) :: project
    !! W&B project name
    character(*), intent(in), optional :: name
    !! W&B run display name (optional)

    this%wandb_project = trim(project)
    if (present(name)) then
       this%wandb_run_name = trim(name)
       call wandb_init(project=project, name=name)
    else
       this%wandb_run_name = ""
       call wandb_init(project=project)
    end if

  end subroutine wandb_setup
!###############################################################################


!###############################################################################
  subroutine wandb_post_epoch_hook(this, epoch, loss, accuracy)
    !! Called automatically at the end of each training epoch.
    !! Logs "loss" and "accuracy" to the current W&B run.
    implicit none

    ! Arguments
    class(wandb_network_type), intent(inout) :: this
    !! Network instance
    integer, intent(in) :: epoch
    !! Current epoch number (1-based)
    real(real32), intent(in) :: loss
    !! Mean loss over the epoch
    real(real32), intent(in) :: accuracy
    !! Mean accuracy over the epoch

    call wandb_log("loss",     loss,     step=epoch)
    call wandb_log("accuracy", accuracy, step=epoch)

  end subroutine wandb_post_epoch_hook
!###############################################################################

#endif

end module athena_wandb
