program train_rollout_fortran
   !! Train a custom Fortran LNO surrogate and export an autoregressive rollout.
  use coreutils, only: real32
      use custom_lno_trainable, only: custom_lno_model_type, custom_lno_init, custom_lno_train, custom_lno_rollout, &
         numerical_gradient_check, set_runtime_conditions_field, load_weights_from_file
  use athena_rollout_baseline, only: &
       athena_rollout_config_type, &
       load_rollout_config, &
       load_dataset_matrix, &
     write_history_file
  implicit none

  type(athena_rollout_config_type) :: config
   type(custom_lno_model_type) :: model

  real(real32), allocatable :: train_inputs(:,:), train_targets(:,:)
  real(real32), allocatable :: val_inputs(:,:), val_targets(:,:)
   real(real32), allocatable :: train_tau(:), train_bc_left(:), train_bc_right(:)
   real(real32), allocatable :: val_tau(:), val_bc_left(:), val_bc_right(:)
      real(real32), allocatable :: train_tau_field(:,:), val_tau_field(:,:)
   real(real32), allocatable :: train_trajectories(:,:,:), val_trajectories(:,:,:)
   real(real32), allocatable :: train_trajectory_tau(:), train_trajectory_bc_left(:), train_trajectory_bc_right(:)
   real(real32), allocatable :: val_trajectory_tau(:), val_trajectory_bc_left(:), val_trajectory_bc_right(:)
      real(real32), allocatable :: train_trajectory_tau_field(:,:), val_trajectory_tau_field(:,:)
   real(real32), allocatable :: init_state(:), history(:,:)
      real(real32) :: learning_rate, rollout_weight, filter_strength, alpha_scalar, lambda_steady_state
      real(real32) :: lambda_gain, lambda_coeff_reg, lambda_cattaneo, lambda_energy, lambda_characteristic, lambda_contraction
      real(real32) :: data_loss_floor_k, input_noise_std
      real(real32) :: eval_alpha, eval_tau, eval_bc_left, eval_bc_right
   logical :: use_causal_mask, use_cosine_schedule
   character(len=32) :: spectral_filter
   character(len=256) :: weights_file

   integer :: rollout_steps, num_epochs, batch_size, train_rollout_steps
      integer :: train_rollout_steps_min, rollout_warmup_epochs, num_corrections, rng_seed, steady_state_every_n_batches
      integer :: physics_warmup_epochs, timestep_jump
      real(real32) :: dt_eff

  num_epochs = get_env_int('FORTRAN_NUM_EPOCHS', 10)
  batch_size = get_env_int('FORTRAN_BATCH_SIZE', 32)
   learning_rate = get_env_real('FORTRAN_LR', 1.0e-3_real32)
   rollout_weight = get_env_real('FORTRAN_ROLLOUT_WEIGHT', 0.1_real32)
   train_rollout_steps = get_env_int('FORTRAN_ROLLOUT_STEPS', 20)
   train_rollout_steps_min = get_env_int('FORTRAN_ROLLOUT_STEPS_MIN', 3)
   rollout_warmup_epochs = get_env_int('FORTRAN_ROLLOUT_WARMUP_EPOCHS', 20)
   num_corrections = get_env_int('FORTRAN_NUM_CORRECTIONS', 5)
   rng_seed = get_env_int('FORTRAN_SEED', 42)
   lambda_steady_state = get_env_real('FORTRAN_LAMBDA_STEADY_STATE', 0.1_real32)
   steady_state_every_n_batches = get_env_int('FORTRAN_STEADY_STATE_EVERY_N_BATCHES', 4)
   physics_warmup_epochs = get_env_int('FORTRAN_PHYSICS_WARMUP_EPOCHS', 10)
   lambda_gain = get_env_real('FORTRAN_LAMBDA_GAIN', 1.0_real32)
   lambda_coeff_reg = get_env_real('FORTRAN_LAMBDA_COEFF_REG', 0.05_real32)
   lambda_cattaneo = get_env_real('FORTRAN_LAMBDA_CATTANEO', 1.0_real32)
   lambda_energy = get_env_real('FORTRAN_LAMBDA_ENERGY', 0.1_real32)
   lambda_characteristic = get_env_real('FORTRAN_LAMBDA_CHARACTERISTIC', 0.0_real32)
   lambda_contraction = get_env_real('FORTRAN_LAMBDA_CONTRACTION', 0.2_real32)
   data_loss_floor_k = get_env_real('FORTRAN_DATA_LOSS_FLOOR_K', 1.0e-3_real32)
   input_noise_std = get_env_real('FORTRAN_INPUT_NOISE_STD', 3.0e-3_real32)
   filter_strength = get_env_real('FORTRAN_FILTER_STRENGTH', 1.0_real32)
   use_causal_mask = get_env_logical('FORTRAN_USE_CAUSAL_MASK', .false.)
   use_cosine_schedule = get_env_logical('FORTRAN_USE_COSINE_SCHEDULE', .true.)
   spectral_filter = get_env_string('FORTRAN_SPECTRAL_FILTER', 'exponential')
   timestep_jump = get_env_int('FORTRAN_TIMESTEP_JUMP', 200)

  call load_rollout_config('data/metadata.json', config)
   ! metadata dt is already effective dt (dt_base * timestep_jump),
   ! so do NOT multiply by timestep_jump again
   dt_eff = config%dt
  call load_dataset_matrix('data/train_inputs.txt', config%n_train, config%input_dim, train_inputs)
  call load_dataset_matrix('data/train_targets.txt', config%n_train, config%output_dim, train_targets)
  call load_dataset_matrix('data/val_inputs.txt', config%n_val, config%input_dim, val_inputs)
  call load_dataset_matrix('data/val_targets.txt', config%n_val, config%output_dim, val_targets)
   call load_dataset_vector('data/train_tau.txt', config%n_train, train_tau)
   call load_dataset_matrix('data/train_tau_field.txt', config%n_train, config%grid_size, train_tau_field)
   call load_dataset_vector('data/train_bc_left.txt', config%n_train, train_bc_left)
   call load_dataset_vector('data/train_bc_right.txt', config%n_train, train_bc_right)
   call load_dataset_vector('data/val_tau.txt', config%n_val, val_tau)
   call load_dataset_matrix('data/val_tau_field.txt', config%n_val, config%grid_size, val_tau_field)
   call load_dataset_vector('data/val_bc_left.txt', config%n_val, val_bc_left)
   call load_dataset_vector('data/val_bc_right.txt', config%n_val, val_bc_right)
   call load_trajectory_tensor('data/train_trajectories.txt', config%n_train_trajectories, config%trajectory_length, config%grid_size, train_trajectories)
   call load_trajectory_tensor('data/val_trajectories.txt', config%n_val_trajectories, config%trajectory_length, config%grid_size, val_trajectories)
   call load_dataset_vector('data/train_trajectory_tau.txt', config%n_train_trajectories, train_trajectory_tau)
   call load_dataset_matrix('data/train_trajectory_tau_field.txt', config%n_train_trajectories, config%grid_size, train_trajectory_tau_field)
   call load_dataset_vector('data/train_trajectory_bc_left.txt', config%n_train_trajectories, train_trajectory_bc_left)
   call load_dataset_vector('data/train_trajectory_bc_right.txt', config%n_train_trajectories, train_trajectory_bc_right)
   call load_dataset_vector('data/val_trajectory_tau.txt', config%n_val_trajectories, val_trajectory_tau)
   call load_dataset_vector('data/val_trajectory_bc_left.txt', config%n_val_trajectories, val_trajectory_bc_left)
   call load_dataset_vector('data/val_trajectory_bc_right.txt', config%n_val_trajectories, val_trajectory_bc_right)
   call load_dataset_matrix('data/val_trajectory_tau_field.txt', config%n_val_trajectories, config%grid_size, val_trajectory_tau_field)

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
         alpha=config%alpha, tau=config%tau, dt=dt_eff, dx=config%dx, &
         num_corrections=num_corrections, &
         use_causal_mask=use_causal_mask, spectral_filter=trim(spectral_filter), filter_strength=filter_strength)

      model%lambda_gain = lambda_gain
      model%lambda_coeff_reg = lambda_coeff_reg
      model%lambda_cattaneo = lambda_cattaneo
      model%lambda_energy = lambda_energy
      model%lambda_characteristic = lambda_characteristic
      model%lambda_contraction = lambda_contraction
      model%input_noise_std = input_noise_std
      model%data_loss_floor_star_sq = (data_loss_floor_k / max(config%delta_t, 1.0e-20_real32)) ** 2

   ! Run numerical gradient check on first training sample
   call set_runtime_conditions_field(model, train_bc_left(1), train_bc_right(1), alpha_scalar, train_tau_field(:, 1), dt_eff, config%dx)
   call numerical_gradient_check(model, train_inputs(:, 1), train_targets(:, 1))

   write(*,'(A)') 'Training custom Fortran LNO rollout model'
  write(*,'(A,I0)') '  epochs: ', num_epochs
   write(*,'(A,I0)') '  batch size: ', batch_size
     write(*,'(A,ES12.4)') '  learning rate: ', learning_rate
   write(*,'(A,ES12.4)') '  rollout weight: ', rollout_weight
   write(*,'(A,I0)') '  rollout steps: ', train_rollout_steps
   write(*,'(A,I0)') '  rollout min steps: ', train_rollout_steps_min
   write(*,'(A,I0)') '  rollout warmup epochs: ', rollout_warmup_epochs
   write(*,'(A,I0)') '  num corrections: ', num_corrections
   write(*,'(A,I0)') '  timestep_jump: ', timestep_jump
   write(*,'(A,ES12.4)') '  dt_eff: ', dt_eff
   write(*,'(A,I0)') '  RNG seed: ', rng_seed
   write(*,'(A,ES12.4)') '  steady-state weight: ', lambda_steady_state
   write(*,'(A,I0)') '  steady-state every n batches: ', steady_state_every_n_batches
   write(*,'(A,I0)') '  physics warmup epochs: ', physics_warmup_epochs
   write(*,'(A,ES12.4)') '  lambda gain: ', lambda_gain
   write(*,'(A,ES12.4)') '  lambda coeff reg: ', lambda_coeff_reg
   write(*,'(A,ES12.4)') '  lambda cattaneo: ', lambda_cattaneo
   write(*,'(A,ES12.4)') '  lambda energy: ', lambda_energy
   write(*,'(A,ES12.4)') '  lambda characteristic: ', lambda_characteristic
   write(*,'(A,ES12.4)') '  lambda contraction: ', lambda_contraction
   write(*,'(A,ES12.4)') '  data loss floor [K]: ', data_loss_floor_k
   write(*,'(A,ES12.4)') '  input noise std [T*]: ', input_noise_std
   write(*,'(A,L1)') '  cosine schedule: ', use_cosine_schedule
      write(*,'(A,L1)') '  use causal mask: ', use_causal_mask
      write(*,'(A,A)') '  spectral filter: ', trim(spectral_filter)
      write(*,'(A,ES12.4)') '  filter strength: ', filter_strength
  write(*,'(A,I0)') '  train samples: ', config%n_train
  write(*,'(A,I0)') '  validation samples: ', config%n_val
  write(*,'(A,I0)') '  train trajectories: ', config%n_train_trajectories
  write(*,'(A,I0)') '  validation trajectories: ', config%n_val_trajectories

   ! Check if we should load pre-trained weights instead of training
   call get_environment_variable('FORTRAN_LOAD_WEIGHTS', weights_file)
   if (len_trim(weights_file) > 0) then
      write(*,'(A,A)') 'Loading pre-trained weights from: ', trim(weights_file)
      call load_weights_from_file(model, trim(weights_file))
   else
   call custom_lno_train(model, train_inputs, train_targets, val_inputs, val_targets, num_epochs, batch_size, &
        alpha_scalar, dt_eff, config%dx, train_tau, train_bc_left, train_bc_right, val_tau, val_bc_left, val_bc_right, &
        train_trajectories=train_trajectories, train_trajectory_tau=train_trajectory_tau, &
        train_trajectory_bc_left=train_trajectory_bc_left, train_trajectory_bc_right=train_trajectory_bc_right, &
        val_trajectories=val_trajectories, val_trajectory_tau=val_trajectory_tau, &
        val_trajectory_bc_left=val_trajectory_bc_left, val_trajectory_bc_right=val_trajectory_bc_right, &
        rollout_weight=rollout_weight, rollout_steps=train_rollout_steps, rollout_steps_min=train_rollout_steps_min, &
      rollout_warmup_epochs=rollout_warmup_epochs, use_cosine_schedule=use_cosine_schedule, &
         lambda_steady_state=lambda_steady_state, steady_state_every_n_batches=steady_state_every_n_batches, &
             physics_warmup_epochs=physics_warmup_epochs, train_tau_field=train_tau_field, val_tau_field=val_tau_field, &
             train_trajectory_tau_field=train_trajectory_tau_field, val_trajectory_tau_field=val_trajectory_tau_field)
   end if

   eval_alpha = get_env_real('FORTRAN_EVAL_ALPHA', config%alpha)
   eval_tau = get_env_real('FORTRAN_EVAL_TAU', config%tau)
   eval_bc_left = get_env_real('FORTRAN_EVAL_BC_LEFT', config%bc_left)
   eval_bc_right = get_env_real('FORTRAN_EVAL_BC_RIGHT', config%bc_right)

  call make_rollout_initial_condition_custom(config%grid_size, eval_bc_left, eval_bc_right, init_state)

  rollout_steps = get_env_int('FORTRAN_EVAL_ROLLOUT_STEPS', config%rollout_steps)
  write(*,'(A,I0)') '  eval rollout steps: ', rollout_steps
   write(*,'(A,ES12.5)') '  eval alpha: ', eval_alpha
   write(*,'(A,ES12.5)') '  eval tau: ', eval_tau
   write(*,'(A,F10.4)') '  eval BC left: ', eval_bc_left
   write(*,'(A,F10.4)') '  eval BC right: ', eval_bc_right
  allocate(history(rollout_steps + 1, config%grid_size), source=0.0_real32)
   if (get_env_logical('FORTRAN_USE_TAU_FIELD', .false.)) then
      write(*,'(A)') '  Using per-grid-point tau field for rollout'
    call custom_lno_rollout(model, init_state, rollout_steps, history, config%temp_ref, config%delta_t, eval_alpha, &
       eval_tau, (eval_bc_left - config%temp_ref) / config%delta_t, (eval_bc_right - config%temp_ref) / config%delta_t, dt_eff, config%dx, &
           tau_field=val_trajectory_tau_field(:, 1))
   else
    call custom_lno_rollout(model, init_state, rollout_steps, history, config%temp_ref, config%delta_t, eval_alpha, &
       eval_tau, (eval_bc_left - config%temp_ref) / config%delta_t, (eval_bc_right - config%temp_ref) / config%delta_t, dt_eff, config%dx)
   end if
   history(:, 1) = eval_bc_left
   history(:, config%grid_size) = eval_bc_right

  call write_history_file('results/fortran_rollout_history.txt', history)

   write(*,'(A)') 'Saved custom Fortran rollout history to results/fortran_rollout_history.txt'

contains

   subroutine make_rollout_initial_condition_custom(grid_size, bc_left, bc_right, state)
      integer, intent(in) :: grid_size
      real(real32), intent(in) :: bc_left, bc_right
      real(real32), allocatable, intent(out) :: state(:)

      integer :: i
      real(real32) :: x_norm, delta_temp, pi_value

      allocate(state(grid_size))
      delta_temp = bc_right - bc_left
      pi_value = acos(-1.0_real32)
      do i = 1, grid_size
         x_norm = real(i - 1, real32) / real(max(1, grid_size - 1), real32)
         state(i) = bc_left + delta_temp * x_norm + 1.2_real32 * delta_temp * sin(pi_value * x_norm)
      end do
   end subroutine make_rollout_initial_condition_custom

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