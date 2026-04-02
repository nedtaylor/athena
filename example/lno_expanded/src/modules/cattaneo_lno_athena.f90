module cattaneo_lno_athena
  !! Top-level Fortran Cattaneo-LNO translation.
  !!
  !! The forward path mirrors the active Python CattaneoLNO execution while
  !! still using Athena layers for all pointwise convolutions and parameter
  !! storage.
  use coreutils, only: real32
  use athena, only: conv1d_layer_type
  use cattaneo_lno_athena_runtime_utils, only: &
     build_unit_interval, &
     extend_dirichlet_field, &
     expand_field_channel, &
     replicate_pad_field, &
     squeeze_single_channel
  use cattaneo_lno_athena_config, only: cattaneo_lno_config_type
  use cattaneo_lno_athena_predictor, only: second_order_predictor_athena_type
  implicit none

  private

  public :: cattaneo_lno_config_type
  public :: cattaneo_lno_output_type
  public :: cattaneo_lno_athena_type

  type :: cattaneo_lno_output_type
    real(real32), allocatable :: T_pred(:,:)
    real(real32), allocatable :: wave_speed(:)
    logical, allocatable :: stable(:)
    real(real32), allocatable :: T_increment(:,:)
    real(real32), allocatable :: T_pred_star(:,:)
    real(real32), allocatable :: T_n_star(:,:)
    real(real32), allocatable :: Fo(:,:)
    real(real32), allocatable :: Fo_substep(:,:)
    real(real32), allocatable :: Ve(:,:)
    real(real32), allocatable :: tau_dt(:,:)
    real(real32), allocatable :: a_theta(:,:)
    real(real32), allocatable :: b_theta(:,:)
    real(real32), allocatable :: hidden_state(:,:,:)
  end type cattaneo_lno_output_type

  type :: cattaneo_lno_athena_type
    !! Top-level model matching the Python module boundary.
     type(cattaneo_lno_config_type) :: config
     !! Shared configuration.
     integer :: input_channels = 5
     !! Input channel count from the Python Fo/Ve formulation.
     integer :: extended_grid = 0
     !! Grid size seen by Athena-backed layers.
    real(real32) :: characteristic_time = 1.0_real32
    !! Characteristic diffusion time used for dimensionless dt.
     type(conv1d_layer_type) :: input_proj
    !! Input projection retained for Python parameter parity.
     type(second_order_predictor_athena_type) :: predictor
    !! Python-matched predictor implementation.
  contains
     procedure :: init => init_cattaneo_lno_athena
    procedure :: forward => forward_cattaneo_lno_athena
     procedure :: num_params => num_params_cattaneo_lno_athena
     procedure :: print_summary => print_cattaneo_lno_summary
  end type cattaneo_lno_athena_type

contains

  subroutine init_cattaneo_lno_athena(this, config)
    !! Initialise the top-level Athena-backed model scaffold.
    class(cattaneo_lno_athena_type), intent(inout) :: this
    type(cattaneo_lno_config_type), intent(in), optional :: config

    if (present(config)) this%config = config
    this%extended_grid = this%config%extended_grid()
    this%characteristic_time = this%config%domain_length ** 2 / this%config%alpha_ref

    this%input_proj = conv1d_layer_type( &
         input_shape=[this%extended_grid, this%input_channels], &
         num_filters=this%config%width, &
         kernel_size=1, &
         activation="none")

    call this%predictor%init(this%config, this%extended_grid)
  end subroutine init_cattaneo_lno_athena

  integer function num_params_cattaneo_lno_athena(this)
      !! Return the total learnable parameter count for the scaffold.
    class(cattaneo_lno_athena_type), intent(in) :: this

    num_params_cattaneo_lno_athena = this%input_proj%get_num_params() + this%predictor%num_params()
  end function num_params_cattaneo_lno_athena

  subroutine forward_cattaneo_lno_athena(this, T_n, T_nm1, q, tau, alpha, rho_cp, bc_left, bc_right, dt, dx, output, T_history, hidden_state)
   class(cattaneo_lno_athena_type), intent(inout) :: this
   real(real32), intent(in) :: T_n(:,:)
   real(real32), intent(in) :: T_nm1(:,:)
   real(real32), intent(in) :: q(:,:)
   real(real32), intent(in) :: tau(:,:)
   real(real32), intent(in) :: alpha(:,:)
   real(real32), intent(in) :: rho_cp(:,:)
   real(real32), intent(in) :: bc_left(:)
   real(real32), intent(in) :: bc_right(:)
   real(real32), intent(in) :: dt
   real(real32), intent(in) :: dx
   type(cattaneo_lno_output_type), intent(out) :: output
   real(real32), intent(in), optional :: T_history(:,:,:)
   real(real32), intent(in), optional :: hidden_state(:,:,:)

   integer :: step_idx
   real(real32) :: dt_sub
   real(real32), allocatable :: T_curr(:,:), T_prev(:,:), T_curr_star(:,:), T_n_star(:,:), Fo_macro(:,:), current_hidden(:,:,:)
   type(cattaneo_lno_output_type) :: step_output

   if (size(q, 1) < 0 .or. size(rho_cp, 1) < 0) stop 'Unreachable input shape guard'
   if (present(T_history)) then
   end if

   if (this%config%num_internal_steps <= 1) then
     if (present(hidden_state)) then
       call single_step_forward_cattaneo_lno_athena(this, T_n, T_nm1, tau, alpha, bc_left, bc_right, dt, dx, output, hidden_state)
     else
       call single_step_forward_cattaneo_lno_athena(this, T_n, T_nm1, tau, alpha, bc_left, bc_right, dt, dx, output)
     end if
     return
   end if

   dt_sub = dt / real(this%config%num_internal_steps, real32)
   T_curr = T_n
   T_prev = T_nm1
   if (present(hidden_state)) current_hidden = hidden_state

   do step_idx = 1, this%config%num_internal_steps
     if (allocated(current_hidden)) then
       call single_step_forward_cattaneo_lno_athena(this, T_curr, T_prev, tau, alpha, bc_left, bc_right, dt_sub, dx, step_output, current_hidden)
     else
       call single_step_forward_cattaneo_lno_athena(this, T_curr, T_prev, tau, alpha, bc_left, bc_right, dt_sub, dx, step_output)
     end if

     T_prev = T_curr
     T_curr = step_output%T_pred
     if (allocated(step_output%hidden_state)) then
       current_hidden = step_output%hidden_state
     else if (allocated(current_hidden)) then
       deallocate(current_hidden)
     end if
   end do

   output = step_output
   T_n_star = (T_n - this%config%temp_ref) / this%config%delta_temp
   T_curr_star = (T_curr - this%config%temp_ref) / this%config%delta_temp
   call compute_fo_field(this, alpha, dt, dx, Fo_macro)

   output%T_pred = T_curr
   output%T_pred_star = T_curr_star - T_n_star
   output%T_n_star = T_n_star
   output%T_increment = output%T_pred_star * this%config%delta_temp
   output%Fo = Fo_macro
   output%Fo_substep = Fo_macro / real(this%config%num_internal_steps, real32)
  end subroutine forward_cattaneo_lno_athena

  subroutine compute_fo_field(this, alpha, dt, dx, Fo)
   class(cattaneo_lno_athena_type), intent(in) :: this
   real(real32), intent(in) :: alpha(:,:)
   real(real32), intent(in) :: dt
   real(real32), intent(in) :: dx
   real(real32), allocatable, intent(out) :: Fo(:,:)

   allocate(Fo(size(alpha, 1), size(alpha, 2)))
   Fo = alpha * dt / max(dx * dx, 1.0e-30_real32)
   Fo = max(this%config%fo_min, min(this%config%fo_max, Fo))
  end subroutine compute_fo_field

  subroutine compute_ve_field(this, alpha, tau, Ve)
   class(cattaneo_lno_athena_type), intent(in) :: this
   real(real32), intent(in) :: alpha(:,:)
   real(real32), intent(in) :: tau(:,:)
   real(real32), allocatable, intent(out) :: Ve(:,:)

   allocate(Ve(size(alpha, 1), size(alpha, 2)))
   Ve = sqrt(max(alpha * tau, 0.0_real32)) / max(this%config%domain_length, 1.0e-30_real32)
  end subroutine compute_ve_field

  subroutine single_step_forward_cattaneo_lno_athena(this, T_n, T_nm1, tau, alpha, bc_left, bc_right, dt, dx, output, hidden_state)
   class(cattaneo_lno_athena_type), intent(inout) :: this
   real(real32), intent(in) :: T_n(:,:)
   real(real32), intent(in) :: T_nm1(:,:)
   real(real32), intent(in) :: tau(:,:)
   real(real32), intent(in) :: alpha(:,:)
   real(real32), intent(in) :: bc_left(:)
   real(real32), intent(in) :: bc_right(:)
   real(real32), intent(in) :: dt
   real(real32), intent(in) :: dx
   type(cattaneo_lno_output_type), intent(out) :: output
   real(real32), intent(in), optional :: hidden_state(:,:,:)

   integer :: actual_grid, batch_size
   real(real32) :: dt_star, dx_star
   real(real32), allocatable :: T_n_star(:,:), T_nm1_star(:,:), bc_left_star(:), bc_right_star(:)
   real(real32), allocatable :: Fo_field(:,:), Ve_field(:,:), tau_dt_field(:,:), T_n_ext(:,:), T_nm1_ext(:,:)
   real(real32), allocatable :: Fo_ext(:,:), Ve_ext(:,:), tau_dt_ext(:,:), xi(:), boundary_mask(:)
   real(real32), allocatable :: c_star_field(:,:), c_star(:), T_inc_ext(:,:,:), a_theta(:,:), b_theta(:,:), sec_diff_T(:,:)
   real(real32), allocatable :: backbone_features(:,:,:), new_hidden(:,:,:), T_inc_interior_ch(:,:,:), T_inc_interior_gated_ch(:,:,:)
   real(real32), allocatable :: backbone_interior(:,:,:), T_inc_interior(:,:), G_inc(:,:), shaped_increment(:,:), T_full_star(:,:)
   real(real32), allocatable :: T_pred(:,:), wave_speed(:), bc_inc_left(:), bc_inc_right(:)
   logical, allocatable :: stable(:)

   actual_grid = size(T_n, 1)
   batch_size = size(T_n, 2)

   T_n_star = (T_n - this%config%temp_ref) / this%config%delta_temp
   T_nm1_star = (T_nm1 - this%config%temp_ref) / this%config%delta_temp
   bc_left_star = (bc_left - this%config%temp_ref) / this%config%delta_temp
   bc_right_star = (bc_right - this%config%temp_ref) / this%config%delta_temp

   call compute_fo_field(this, alpha, dt, dx, Fo_field)
   call compute_ve_field(this, alpha, tau, Ve_field)
   tau_dt_field = tau / max(dt, 1.0e-30_real32)

   dx_star = dx / max(this%config%domain_length, 1.0e-30_real32)
   dt_star = dt / max(this%characteristic_time, 1.0e-30_real32)

   if (this%config%use_ghost_cells) then
    call extend_dirichlet_field(T_n_star, bc_left_star, bc_right_star, T_n_ext)
    call extend_dirichlet_field(T_nm1_star, bc_left_star, bc_right_star, T_nm1_ext)
    call replicate_pad_field(Fo_field, Fo_ext)
    call replicate_pad_field(Ve_field, Ve_ext)
    call replicate_pad_field(tau_dt_field, tau_dt_ext)
   else
    T_n_ext = T_n_star
    T_nm1_ext = T_nm1_star
    Fo_ext = Fo_field
    Ve_ext = Ve_field
    tau_dt_ext = tau_dt_field
   end if

   c_star_field = sqrt(alpha / max(tau, 1.0e-30_real32)) * this%config%domain_length / this%config%alpha_ref
   c_star = maxval(c_star_field, dim=1)
   call build_unit_interval(actual_grid, xi, boundary_mask)

   if (present(hidden_state)) then
     call this%predictor%forward(Fo_ext, Ve_ext, tau_dt_ext, dt_star, dx_star, c_star, T_nm1_ext, T_n_ext, &
        T_inc_ext, a_theta, b_theta, sec_diff_T, backbone_features, hidden_state, new_hidden)
   else
     call this%predictor%forward(Fo_ext, Ve_ext, tau_dt_ext, dt_star, dx_star, c_star, T_nm1_ext, T_n_ext, &
        T_inc_ext, a_theta, b_theta, sec_diff_T, backbone_features, new_hidden=new_hidden)
   end if

   if (this%config%use_ghost_cells) then
     T_inc_interior_ch = T_inc_ext(2:actual_grid + 1, :, :)
     backbone_interior = backbone_features(2:actual_grid + 1, :, :)
   else
     T_inc_interior_ch = T_inc_ext
     backbone_interior = backbone_features
   end if

   call this%predictor%relaxation_gate%forward(backbone_interior, T_inc_interior_ch, T_n_star, bc_left_star, bc_right_star, xi, dt_star, T_inc_interior_gated_ch)
   call squeeze_single_channel(T_inc_interior_gated_ch, T_inc_interior)

   bc_inc_left = bc_left_star - T_n_star(1, :)
   bc_inc_right = bc_right_star - T_n_star(actual_grid, :)
   G_inc = spread(bc_inc_left, dim=1, ncopies=actual_grid) + &
      spread(bc_inc_right - bc_inc_left, dim=1, ncopies=actual_grid) * spread(xi, dim=2, ncopies=batch_size)
   shaped_increment = G_inc + spread(boundary_mask, dim=2, ncopies=batch_size) * T_inc_interior
   T_full_star = T_n_star + shaped_increment
   T_pred = this%config%temp_ref + T_full_star * this%config%delta_temp

   allocate(stable(batch_size), wave_speed(batch_size))
   stable = c_star * dt_star / max(dx_star, 1.0e-30_real32) < this%config%cfl_threshold
   wave_speed = c_star * this%config%domain_length / this%characteristic_time

   output%T_pred = T_pred
   output%wave_speed = wave_speed
   output%stable = stable
   output%T_increment = shaped_increment * this%config%delta_temp
   output%T_pred_star = shaped_increment
   output%T_n_star = T_n_star
   output%Fo = Fo_field
   output%Fo_substep = Fo_field
   output%Ve = Ve_field
   output%tau_dt = tau_dt_field
   output%a_theta = a_theta
   output%b_theta = b_theta
   if (allocated(new_hidden)) output%hidden_state = new_hidden
  end subroutine single_step_forward_cattaneo_lno_athena

  subroutine print_cattaneo_lno_summary(this, unit)
    !! Print a summary of the translated architecture.
    class(cattaneo_lno_athena_type), intent(in) :: this
    integer, intent(in), optional :: unit
    integer :: out_unit
    integer :: i

    out_unit = 6
    if (present(unit)) out_unit = unit

    write(out_unit,'(A)') 'Fortran Cattaneo-LNO Architecture'
    write(out_unit,'(A)') '==============================='
    write(out_unit,'(A,I0)') 'grid_size: ', this%config%grid_size
    write(out_unit,'(A,L1)') 'use_ghost_cells: ', this%config%use_ghost_cells
    write(out_unit,'(A,I0)') 'extended_grid: ', this%extended_grid
    write(out_unit,'(A,I0)') 'modes: ', this%config%modes
    write(out_unit,'(A,I0)') 'width: ', this%config%width
    write(out_unit,'(A,I0)') 'num_no_layers: ', this%config%num_no_layers
    write(out_unit,'(A,I0)') 'history_len: ', this%config%history_len
    write(out_unit,'(A,L1)') 'use_recurrent_memory: ', this%config%use_recurrent_memory
    write(out_unit,'(A,I0)') 'memory_channels: ', this%config%memory_channels
    write(out_unit,'(A,I0)') 'num_corrections: ', this%config%num_corrections
    write(out_unit,'(A,I0)') 'timestep_jump: ', this%config%timestep_jump
    write(out_unit,'(A,ES10.3)') 'learning_rate: ', this%config%learning_rate
    write(out_unit,'(A,ES10.3)') 'domain_length: ', this%config%domain_length
    write(out_unit,'(A,ES10.3)') 'alpha_ref: ', this%config%alpha_ref
    write(out_unit,'(A,ES10.3)') 'temp_ref: ', this%config%temp_ref
    write(out_unit,'(A,ES10.3)') 'delta_temp: ', this%config%delta_temp
    write(out_unit,'(A,ES10.3)') 'cfl_threshold: ', this%config%cfl_threshold
    write(out_unit,'(A)') ''
    write(out_unit,'(A)') 'Fortran module mappings'
    write(out_unit,'(A,I0)') '  input_proj params: ', this%input_proj%get_num_params()
    write(out_unit,'(A,I0)') '  predictor.lno_proj params: ', this%predictor%lno_proj%get_num_params()
    do i = 1, size(this%predictor%lno_blocks)
       write(out_unit,'(A,I0,A,I0)') '  predictor.lno_blocks(', i, ') params: ', &
            this%predictor%lno_blocks(i)%num_params()
    end do
    write(out_unit,'(A,I0)') '  predictor.rnn_proj params: ', this%predictor%rnn_proj%get_num_params()
    if (this%predictor%use_recurrent_memory) then
      write(out_unit,'(A,I0)') '  predictor.memory_cell params: ', this%predictor%memory_cell%num_params()
      write(out_unit,'(A,I0)') '  predictor.memory_fusion params: ', this%predictor%memory_fusion%num_params()
    end if
    write(out_unit,'(A,I0)') '  predictor.coeff_head params: ', this%predictor%coeff_head%num_params()
    write(out_unit,'(A,I0)') '  predictor.corrector params: ', this%predictor%corrector%num_params()
    write(out_unit,'(A,I0)') '  predictor.relaxation_gate params: ', this%predictor%relaxation_gate%num_params()
    write(out_unit,'(A)') ''
    write(out_unit,'(A)') 'Python-matched forward path'
    write(out_unit,'(A)') '  - Dimensionless Fo/Ve/tau_dt fields with ghost-cell extension'
    write(out_unit,'(A)') '  - InstanceNorm1d plus Athena dynamic LNO blocks'
    write(out_unit,'(A)') '  - Structure-preserving coefficient, corrector, and relaxation heads'
    write(out_unit,'(A)') '  - Hard Dirichlet shaping and optional recurrent memory'
    write(out_unit,'(A)') ''
    write(out_unit,'(A,I0)') 'Total learnable parameters: ', this%num_params()
  end subroutine print_cattaneo_lno_summary

end module cattaneo_lno_athena