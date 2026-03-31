module athena_rollout_baseline
  !! Athena baseline training and rollout utilities.
  !!
  !! This module trains a practical Athena surrogate on the Python-generated
  !! one-step dataset and rolls it out autoregressively on the standard thermal
  !! relaxation initial condition used by the Python benchmark.
  use coreutils, only: real32
  use athena, only: network_type, full_layer_type, adam_optimiser_type, &
     fixed_lno_layer_type, clip_type
  implicit none

  private

  public :: athena_rollout_config_type
  public :: load_rollout_config
  public :: load_dataset_matrix
  public :: build_rollout_network
  public :: make_rollout_initial_condition
  public :: write_history_file
   public :: normalise_temperature
   public :: denormalise_temperature

  real(real32), parameter :: pi_real32 = 3.14159265358979323846_real32

  type :: athena_rollout_config_type
     integer :: grid_size = 112
     integer :: input_dim = 224
     integer :: output_dim = 112
     integer :: n_train = 0
     integer :: n_val = 0
     integer :: n_train_trajectories = 0
     integer :: n_val_trajectories = 0
     integer :: trajectory_length = 0
     integer :: rollout_steps = 200
     real(real32) :: dt = 2.0e-11_real32
     real(real32) :: dx = 1.0e-8_real32
     real(real32) :: bc_left = 100.0_real32
     real(real32) :: bc_right = 200.0_real32
     real(real32) :: tau = 1.0e-9_real32
     real(real32) :: alpha = 1.0e-4_real32
     real(real32) :: rho_cp = 1.0e6_real32
     real(real32) :: temp_ref = 200.0_real32
     real(real32) :: delta_t = 100.0_real32
  end type athena_rollout_config_type

contains

  subroutine load_rollout_config(file_name, config)
    !! Read rollout metadata exported from Python.
    character(len=*), intent(in) :: file_name
    type(athena_rollout_config_type), intent(out) :: config

    character(len=512) :: line
    character(len=128) :: key, value
    integer :: unit, ios, pos

    open(newunit=unit, file=file_name, status='old', action='read', iostat=ios)
    if (ios /= 0) stop 'Failed to open Athena metadata file'

    do
       read(unit,'(A)',iostat=ios) line
       if (ios /= 0) exit
       pos = index(line, ':')
       if (pos <= 0) cycle
       key = adjustl(trim(line(1:pos-1)))
       value = adjustl(trim(line(pos+1:)))
       if (len_trim(value) > 0) then
          if (value(len_trim(value):len_trim(value)) == ',') value = trim(value(:len_trim(value)-1))
       end if
       value = trim(adjustl(value))
       if (len_trim(value) > 1) then
          if (value(1:1) == '"') value = value(2:len_trim(value)-1)
       end if
       select case (trim(key))
       case ('"grid_size"')
          read(value,*) config%grid_size
       case ('"input_dim"')
          read(value,*) config%input_dim
       case ('"output_dim"')
          read(value,*) config%output_dim
       case ('"n_train"')
          read(value,*) config%n_train
       case ('"n_val"')
          read(value,*) config%n_val
       case ('"n_train_trajectories"')
          read(value,*) config%n_train_trajectories
       case ('"n_val_trajectories"')
          read(value,*) config%n_val_trajectories
       case ('"trajectory_length"')
          read(value,*) config%trajectory_length
       case ('"dt"')
          read(value,*) config%dt
       case ('"dx"')
          read(value,*) config%dx
       case ('"bc_left"')
          read(value,*) config%bc_left
       case ('"bc_right"')
          read(value,*) config%bc_right
       case ('"tau"')
          read(value,*) config%tau
       case ('"alpha"')
          read(value,*) config%alpha
       case ('"rho_cp"')
          read(value,*) config%rho_cp
       case ('"rollout_steps"')
          read(value,*) config%rollout_steps
       case ('"temp_ref"')
          read(value,*) config%temp_ref
       case ('"delta_t"')
          read(value,*) config%delta_t
       end select
    end do
    close(unit)
  end subroutine load_rollout_config

  subroutine normalise_temperature(field, config)
    !! Convert dimensional temperature to the training scale.
    real(real32), intent(inout) :: field(:)
    type(athena_rollout_config_type), intent(in) :: config

    field = (field - config%temp_ref) / config%delta_t
  end subroutine normalise_temperature

  subroutine denormalise_temperature(field, config)
    !! Convert scaled temperature back to Kelvin.
    real(real32), intent(inout) :: field(:)
    type(athena_rollout_config_type), intent(in) :: config

    field = config%temp_ref + config%delta_t * field
  end subroutine denormalise_temperature

  subroutine load_dataset_matrix(file_name, num_rows, num_cols, matrix)
    !! Load a whitespace-delimited matrix written by numpy.savetxt.
    character(len=*), intent(in) :: file_name
    integer, intent(in) :: num_rows
    integer, intent(in) :: num_cols
    real(real32), allocatable, intent(out) :: matrix(:,:)

    integer :: unit, ios, i
    real(real32), allocatable :: row_data(:)

    allocate(matrix(num_cols, num_rows))
    allocate(row_data(num_cols))

    open(newunit=unit, file=file_name, status='old', action='read', iostat=ios)
    if (ios /= 0) stop 'Failed to open dataset matrix file'

    do i = 1, num_rows
       read(unit, *, iostat=ios) row_data
       if (ios /= 0) stop 'Failed while reading dataset matrix file'
       matrix(:, i) = row_data
    end do

    close(unit)
    deallocate(row_data)
  end subroutine load_dataset_matrix

  subroutine build_rollout_network(network, config)
    !! Build an Athena Laplace Neural Operator surrogate reproducing the
    !! Python Cattaneo-LNO architecture.
    !!
    !! Python architecture (cattaneo_lno.py):
    !!   Conv1d(2→width) → 4×[InstanceNorm + LaplaceConv1d(width,width,modes) + SiLU]
    !!   → WaveDiffusionHead → IterativeCorrector → RelaxationGate
    !!   width=64, modes=16, activation='swish' (SiLU), num_no_layers=4
    !!
   !! Athena fallback baseline (structurally similar, but not exact):
    !!   full(224→64, swish)                        ← Conv1d(2→64) equivalent
    !!   → laplace_nop(64→64, 16 modes, swish)      ← lno_blocks[0]
    !!   → laplace_nop(64→64, 16 modes, swish)      ← lno_blocks[1]
    !!   → laplace_nop(64→64, 16 modes, swish)      ← lno_blocks[2]
    !!   → laplace_nop(64→64, 16 modes, swish)      ← lno_blocks[3]
    !!   → full(64→112, linear)                      ← output ΔT prediction
    type(network_type), intent(inout) :: network
    type(athena_rollout_config_type), intent(in) :: config

    integer, parameter :: width = 64   !! Python: width=64
    integer, parameter :: modes = 16   !! Python: modes=16

    !! -- Input projection: Conv1d(2→64) equivalent (224→64, swish) -----------
    call network%add(full_layer_type( &
         num_inputs=config%input_dim, &
         num_outputs=width, &
         activation='swish'))

    !! -- LNO layer 1 (64→64, 16 modes, swish) — Python lno_blocks[0] --------
   call network%add(fixed_lno_layer_type( &
         num_outputs=width, &
         num_modes=modes, &
         kernel_initialiser='lecun_normal', &
         activation='swish'))

    !! -- LNO layer 2 (64→64, 16 modes, swish) — Python lno_blocks[1] --------
   call network%add(fixed_lno_layer_type( &
         num_outputs=width, &
         num_modes=modes, &
         kernel_initialiser='lecun_normal', &
         activation='swish'))

    !! -- LNO layer 3 (64→64, 16 modes, swish) — Python lno_blocks[2] --------
   call network%add(fixed_lno_layer_type( &
         num_outputs=width, &
         num_modes=modes, &
         kernel_initialiser='lecun_normal', &
         activation='swish'))

    !! -- LNO layer 4 (64→64, 16 modes, swish) — Python lno_blocks[3] --------
   call network%add(fixed_lno_layer_type( &
         num_outputs=width, &
         num_modes=modes, &
         kernel_initialiser='lecun_normal', &
         activation='swish'))

    !! -- Output projection (64→112, linear) — predicts ΔT --------------------
    call network%add(full_layer_type( &
         num_outputs=config%output_dim, &
         activation='linear'))

       !! -- Compile: Adam lr=1e-4 + gradient clipping ---------------------------
       !! Athena's laplace_nop becomes unstable with swish at 1e-3 because its
       !! spectral weights are unbounded.  1e-4 is the highest stable rate here.
    call network%compile( &
         optimiser=adam_optimiser_type( &
             learning_rate=1.0e-4_real32, &
              clip_dict=clip_type(clip_norm=1.0_real32)), &
         loss_method='mse', metrics=['loss'], batch_size=32, verbose=1)
  end subroutine build_rollout_network

  subroutine make_rollout_initial_condition(config, state)
    !! Build the same non-equilibrium initial condition used by Python.
    type(athena_rollout_config_type), intent(in) :: config
    real(real32), allocatable, intent(out) :: state(:)

    integer :: i
    real(real32) :: x_norm, delta_t

    allocate(state(config%grid_size))
    delta_t = config%bc_right - config%bc_left
    do i = 1, config%grid_size
       x_norm = real(i - 1, real32) / real(max(1, config%grid_size - 1), real32)
       state(i) = config%bc_left + delta_t * x_norm + 1.2_real32 * delta_t * sin(pi_real32 * x_norm)
    end do
  end subroutine make_rollout_initial_condition

  subroutine write_history_file(file_name, history)
    !! Write rollout history as whitespace-delimited rows.
    character(len=*), intent(in) :: file_name
    real(real32), intent(in) :: history(:,:)

    integer :: unit, ios, i

    open(newunit=unit, file=file_name, status='replace', action='write', iostat=ios)
    if (ios /= 0) stop 'Failed to open output history file'
    do i = 1, size(history, 1)
       write(unit,'(*(ES16.8,1X))') history(i, :)
    end do
    close(unit)
  end subroutine write_history_file

end module athena_rollout_baseline