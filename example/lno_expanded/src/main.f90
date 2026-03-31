program train_rollout_fortran
   !! Train a custom Fortran LNO surrogate and export an autoregressive rollout.
  use coreutils, only: real32
   use custom_lno_trainable, only: custom_lno_model_type, custom_lno_init, custom_lno_train, custom_lno_rollout
  use athena_rollout_baseline, only: &
       athena_rollout_config_type, &
       load_rollout_config, &
       load_dataset_matrix, &
       make_rollout_initial_condition, &
     write_history_file
  implicit none

  type(athena_rollout_config_type) :: config
   type(custom_lno_model_type) :: model

  real(real32), allocatable :: train_inputs(:,:), train_targets(:,:)
  real(real32), allocatable :: val_inputs(:,:), val_targets(:,:)
   real(real32), allocatable :: train_tau(:), train_bc_left(:), train_bc_right(:)
   real(real32), allocatable :: val_tau(:), val_bc_left(:), val_bc_right(:)
   real(real32), allocatable :: train_trajectories(:,:,:), val_trajectories(:,:,:)
   real(real32), allocatable :: train_trajectory_tau(:), train_trajectory_bc_left(:), train_trajectory_bc_right(:)
   real(real32), allocatable :: val_trajectory_tau(:), val_trajectory_bc_left(:), val_trajectory_bc_right(:)
   real(real32), allocatable :: init_state(:), history(:,:)
      real(real32) :: learning_rate, rollout_weight, filter_strength, alpha_scalar, lambda_steady_state
   logical :: use_causal_mask, use_cosine_schedule
   character(len=32) :: spectral_filter

   integer :: rollout_steps, num_epochs, batch_size, train_rollout_steps
      integer :: train_rollout_steps_min, rollout_warmup_epochs, num_corrections, rng_seed, steady_state_every_n_batches

  num_epochs = get_env_int('FORTRAN_NUM_EPOCHS', 10)
  batch_size = get_env_int('FORTRAN_BATCH_SIZE', 32)
   learning_rate = get_env_real('FORTRAN_LR', 1.0e-3_real32)
   rollout_weight = get_env_real('FORTRAN_ROLLOUT_WEIGHT', 0.0_real32)
   train_rollout_steps = get_env_int('FORTRAN_ROLLOUT_STEPS', 4)
   train_rollout_steps_min = get_env_int('FORTRAN_ROLLOUT_STEPS_MIN', 3)
   rollout_warmup_epochs = get_env_int('FORTRAN_ROLLOUT_WARMUP_EPOCHS', 15)
   num_corrections = get_env_int('FORTRAN_NUM_CORRECTIONS', 5)
   rng_seed = get_env_int('FORTRAN_SEED', 42)
   lambda_steady_state = get_env_real('FORTRAN_LAMBDA_STEADY_STATE', 0.1_real32)
   steady_state_every_n_batches = get_env_int('FORTRAN_STEADY_STATE_EVERY_N_BATCHES', 4)
   filter_strength = get_env_real('FORTRAN_FILTER_STRENGTH', 1.0_real32)
   use_causal_mask = get_env_logical('FORTRAN_USE_CAUSAL_MASK', .false.)
   use_cosine_schedule = get_env_logical('FORTRAN_USE_COSINE_SCHEDULE', .true.)
   spectral_filter = get_env_string('FORTRAN_SPECTRAL_FILTER', 'exponential')

  call load_rollout_config('data/metadata.json', config)
  call load_dataset_matrix('data/train_inputs.txt', config%n_train, config%input_dim, train_inputs)
  call load_dataset_matrix('data/train_targets.txt', config%n_train, config%output_dim, train_targets)
  call load_dataset_matrix('data/val_inputs.txt', config%n_val, config%input_dim, val_inputs)
  call load_dataset_matrix('data/val_targets.txt', config%n_val, config%output_dim, val_targets)
   call load_dataset_vector('data/train_tau.txt', config%n_train, train_tau)
   call load_dataset_vector('data/train_bc_left.txt', config%n_train, train_bc_left)
   call load_dataset_vector('data/train_bc_right.txt', config%n_train, train_bc_right)
   call load_dataset_vector('data/val_tau.txt', config%n_val, val_tau)
   call load_dataset_vector('data/val_bc_left.txt', config%n_val, val_bc_left)
   call load_dataset_vector('data/val_bc_right.txt', config%n_val, val_bc_right)
   call load_trajectory_tensor('data/train_trajectories.txt', config%n_train_trajectories, config%trajectory_length, config%grid_size, train_trajectories)
   call load_trajectory_tensor('data/val_trajectories.txt', config%n_val_trajectories, config%trajectory_length, config%grid_size, val_trajectories)
   call load_dataset_vector('data/train_trajectory_tau.txt', config%n_train_trajectories, train_trajectory_tau)
   call load_dataset_vector('data/train_trajectory_bc_left.txt', config%n_train_trajectories, train_trajectory_bc_left)
   call load_dataset_vector('data/train_trajectory_bc_right.txt', config%n_train_trajectories, train_trajectory_bc_right)
   call load_dataset_vector('data/val_trajectory_tau.txt', config%n_val_trajectories, val_trajectory_tau)
   call load_dataset_vector('data/val_trajectory_bc_left.txt', config%n_val_trajectories, val_trajectory_bc_left)
   call load_dataset_vector('data/val_trajectory_bc_right.txt', config%n_val_trajectories, val_trajectory_bc_right)

  train_inputs(1:config%grid_size, :) = (train_inputs(1:config%grid_size, :) - config%temp_ref) / config%delta_t
  train_inputs(config%grid_size + 1:config%input_dim, :) = &
     (train_inputs(config%grid_size + 1:config%input_dim, :) - config%temp_ref) / config%delta_t
  train_targets = (train_targets - config%temp_ref) / config%delta_t
  val_inputs(1:config%grid_size, :) = (val_inputs(1:config%grid_size, :) - config%temp_ref) / config%delta_t
  val_inputs(config%grid_size + 1:config%input_dim, :) = &
     (val_inputs(config%grid_size + 1:config%input_dim, :) - config%temp_ref) / config%delta_t
  val_targets = (val_targets - config%temp_ref) / config%delta_t
  train_trajectories = (train_trajectories - config%temp_ref) / config%delta_t
  val_trajectories = (val_trajectories - config%temp_ref) / config%delta_t
  train_bc_left = (train_bc_left - config%temp_ref) / config%delta_t
  train_bc_right = (train_bc_right - config%temp_ref) / config%delta_t
  val_bc_left = (val_bc_left - config%temp_ref) / config%delta_t
  val_bc_right = (val_bc_right - config%temp_ref) / config%delta_t
  train_trajectory_bc_left = (train_trajectory_bc_left - config%temp_ref) / config%delta_t
  train_trajectory_bc_right = (train_trajectory_bc_right - config%temp_ref) / config%delta_t
  val_trajectory_bc_left = (val_trajectory_bc_left - config%temp_ref) / config%delta_t
  val_trajectory_bc_right = (val_trajectory_bc_right - config%temp_ref) / config%delta_t

  !! Residual targets: ΔT = T_{target} - T_n (in normalised space)
  !! Python internally predicts increments, not absolute temps.
  train_targets = train_targets - train_inputs(1:config%grid_size, :)
  val_targets   = val_targets   - val_inputs(1:config%grid_size, :)

   alpha_scalar = config%alpha
   call set_random_seed(rng_seed)

      call custom_lno_init(model, config%grid_size, width=64, modes=16, num_blocks=4, lr=learning_rate, &
         bc_left_norm=(config%bc_left - config%temp_ref) / config%delta_t, &
         bc_right_norm=(config%bc_right - config%temp_ref) / config%delta_t, &
         alpha=config%alpha, tau=config%tau, dt=config%dt, dx=config%dx, &
         num_corrections=num_corrections, &
         use_causal_mask=use_causal_mask, spectral_filter=trim(spectral_filter), filter_strength=filter_strength)

   write(*,'(A)') 'Training custom Fortran LNO rollout model'
  write(*,'(A,I0)') '  epochs: ', num_epochs
   write(*,'(A,I0)') '  batch size: ', batch_size
     write(*,'(A,ES12.4)') '  learning rate: ', learning_rate
   write(*,'(A,ES12.4)') '  rollout weight: ', rollout_weight
   write(*,'(A,I0)') '  rollout steps: ', train_rollout_steps
   write(*,'(A,I0)') '  rollout min steps: ', train_rollout_steps_min
   write(*,'(A,I0)') '  rollout warmup epochs: ', rollout_warmup_epochs
   write(*,'(A,I0)') '  num corrections: ', num_corrections
   write(*,'(A,I0)') '  RNG seed: ', rng_seed
   write(*,'(A,ES12.4)') '  steady-state weight: ', lambda_steady_state
   write(*,'(A,I0)') '  steady-state every n batches: ', steady_state_every_n_batches
   write(*,'(A,L1)') '  cosine schedule: ', use_cosine_schedule
      write(*,'(A,L1)') '  use causal mask: ', use_causal_mask
      write(*,'(A,A)') '  spectral filter: ', trim(spectral_filter)
      write(*,'(A,ES12.4)') '  filter strength: ', filter_strength
  write(*,'(A,I0)') '  train samples: ', config%n_train
  write(*,'(A,I0)') '  validation samples: ', config%n_val
  write(*,'(A,I0)') '  train trajectories: ', config%n_train_trajectories
  write(*,'(A,I0)') '  validation trajectories: ', config%n_val_trajectories
   call custom_lno_train(model, train_inputs, train_targets, val_inputs, val_targets, num_epochs, batch_size, &
        alpha_scalar, config%dt, config%dx, train_tau, train_bc_left, train_bc_right, val_tau, val_bc_left, val_bc_right, &
        train_trajectories=train_trajectories, train_trajectory_tau=train_trajectory_tau, &
        train_trajectory_bc_left=train_trajectory_bc_left, train_trajectory_bc_right=train_trajectory_bc_right, &
        val_trajectories=val_trajectories, val_trajectory_tau=val_trajectory_tau, &
        val_trajectory_bc_left=val_trajectory_bc_left, val_trajectory_bc_right=val_trajectory_bc_right, &
        rollout_weight=rollout_weight, rollout_steps=train_rollout_steps, rollout_steps_min=train_rollout_steps_min, &
      rollout_warmup_epochs=rollout_warmup_epochs, use_cosine_schedule=use_cosine_schedule, &
      lambda_steady_state=lambda_steady_state, steady_state_every_n_batches=steady_state_every_n_batches)

  call make_rollout_initial_condition(config, init_state)

  rollout_steps = config%rollout_steps
  allocate(history(rollout_steps + 1, config%grid_size), source=0.0_real32)
   call custom_lno_rollout(model, init_state, rollout_steps, history, config%temp_ref, config%delta_t, alpha_scalar, &
        val_trajectory_tau(1), val_trajectory_bc_left(1), val_trajectory_bc_right(1), config%dt, config%dx)
   history(:, 1) = config%bc_left
   history(:, config%grid_size) = config%bc_right

  call write_history_file('results/fortran_rollout_history.txt', history)

   write(*,'(A)') 'Saved custom Fortran rollout history to results/fortran_rollout_history.txt'

contains

   subroutine load_dataset_vector(file_name, num_values, vector)
      character(len=*), intent(in) :: file_name
      integer, intent(in) :: num_values
      real(real32), allocatable, intent(out) :: vector(:)
      real(real32), allocatable :: matrix(:,:)

      call load_dataset_matrix(file_name, num_values, 1, matrix)
      allocate(vector(num_values))
      vector = matrix(1, :)
      deallocate(matrix)
   end subroutine load_dataset_vector

   subroutine load_trajectory_tensor(file_name, num_trajectories, trajectory_length, grid_size, trajectories)
      character(len=*), intent(in) :: file_name
      integer, intent(in) :: num_trajectories, trajectory_length, grid_size
      real(real32), allocatable, intent(out) :: trajectories(:,:,:)
      real(real32), allocatable :: flat_matrix(:,:)
      integer :: traj_idx, start_col, end_col

      call load_dataset_matrix(file_name, num_trajectories * trajectory_length, grid_size, flat_matrix)
      allocate(trajectories(grid_size, trajectory_length, num_trajectories))
      do traj_idx = 1, num_trajectories
         start_col = (traj_idx - 1) * trajectory_length + 1
         end_col = start_col + trajectory_length - 1
         trajectories(:, :, traj_idx) = flat_matrix(:, start_col:end_col)
      end do
      deallocate(flat_matrix)
   end subroutine load_trajectory_tensor

   subroutine set_random_seed(seed)
      integer, intent(in) :: seed
      integer :: i, n
      integer, allocatable :: seed_values(:)

      call random_seed(size=n)
      allocate(seed_values(n))
      do i = 1, n
         seed_values(i) = seed + 37 * (i - 1)
      end do
      call random_seed(put=seed_values)
      deallocate(seed_values)
   end subroutine set_random_seed

   integer function get_env_int(name, default_value)
      character(len=*), intent(in) :: name
      integer, intent(in) :: default_value
      character(len=32) :: value
      integer :: status

      get_env_int = default_value
      call get_environment_variable(name, value, status=status)
      if (status == 0 .and. len_trim(value) > 0) read(value, *, iostat=status) get_env_int
      if (status /= 0) get_env_int = default_value
   end function get_env_int

   real(real32) function get_env_real(name, default_value)
      character(len=*), intent(in) :: name
      real(real32), intent(in) :: default_value
      character(len=32) :: value
      integer :: status

      get_env_real = default_value
      call get_environment_variable(name, value, status=status)
      if (status == 0 .and. len_trim(value) > 0) read(value, *, iostat=status) get_env_real
      if (status /= 0) get_env_real = default_value
   end function get_env_real

   logical function get_env_logical(name, default_value)
      character(len=*), intent(in) :: name
      logical, intent(in) :: default_value
      character(len=32) :: value
      integer :: status

      get_env_logical = default_value
      call get_environment_variable(name, value, status=status)
      if (status /= 0 .or. len_trim(value) == 0) return
      select case (trim(adjustl(value)))
      case ('1', 'true', 'TRUE', 'yes', 'YES', 'on', 'ON')
         get_env_logical = .true.
      case ('0', 'false', 'FALSE', 'no', 'NO', 'off', 'OFF')
         get_env_logical = .false.
      end select
   end function get_env_logical

   character(len=32) function get_env_string(name, default_value)
      character(len=*), intent(in) :: name
      character(len=*), intent(in) :: default_value
      character(len=32) :: value
      integer :: status

      get_env_string = default_value
      call get_environment_variable(name, value, status=status)
      if (status == 0 .and. len_trim(value) > 0) get_env_string = trim(adjustl(value))
   end function get_env_string
end program train_rollout_fortran