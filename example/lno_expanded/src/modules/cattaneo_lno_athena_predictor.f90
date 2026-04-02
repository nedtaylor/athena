module cattaneo_lno_athena_predictor
  !! Athena-backed predictor scaffold for the Python Cattaneo-LNO layout.
  !!
  !! This module groups the predictor-local blocks into a stable interface so
  !! that the top-level model wiring stays stable while Athena-facing block
  !! implementations continue to evolve.
  use coreutils, only: real32
  use athena, only: conv1d_layer_type
  use cattaneo_lno_athena_runtime_utils, only: conv1d_forward_real, expand_field_channel, instance_norm1d_real, second_difference_replicate
  use athena_dynamic_lno_layer_adapter, only: dynamic_lno_block_athena_type
  use cattaneo_lno_athena_config, only: cattaneo_lno_config_type
  use cattaneo_lno_athena_blocks, only: &
       conv1d_gru_cell_athena_type, &
       memory_fusion_athena_type, &
       wave_diffusion_head_athena_type, &
       iterative_corrector_athena_type, &
       relaxation_gate_athena_type
  implicit none

  private

  public :: second_order_predictor_athena_type

  type :: second_order_predictor_athena_type
     !! Structural predictor scaffold matching the Python module boundary.
     type(conv1d_layer_type) :: lno_proj
     !! Initial projection of temperature state into backbone width.
    type(dynamic_lno_block_athena_type), allocatable :: lno_blocks(:)
     !! Repeated LNO blocks.
     type(conv1d_layer_type) :: rnn_proj
     !! Projection of recurrent inputs into backbone width.
     type(conv1d_gru_cell_athena_type) :: memory_cell
     !! GRU memory stand-in.
     type(memory_fusion_athena_type) :: memory_fusion
     !! Memory fusion block.
     type(wave_diffusion_head_athena_type) :: coeff_head
     !! Coefficient head block.
     type(iterative_corrector_athena_type) :: corrector
     !! Iterative corrector block.
     type(relaxation_gate_athena_type) :: relaxation_gate
     !! Relaxation gate block.
     logical :: use_recurrent_memory = .false.
     !! Whether memory-cell blocks are active.
     integer :: channels = 0
     !! Backbone width.
     integer :: grid_size = 0
     !! Working grid size including ghost cells when present.
     integer :: modes = 0
     !! Structural mode count used by LNO stand-ins.
   contains
     procedure :: init => init_second_order_predictor
     procedure :: forward => forward_second_order_predictor
     !! Initialise the predictor scaffold.
     procedure :: num_params => num_params_second_order_predictor
     !! Return the total learnable parameter count.
  end type second_order_predictor_athena_type

contains

  subroutine init_second_order_predictor(this, config, grid_size)
    !! Initialise the predictor scaffold from the shared configuration.
    class(second_order_predictor_athena_type), intent(inout) :: this
    type(cattaneo_lno_config_type), intent(in) :: config
    integer, intent(in) :: grid_size
    integer :: i

    this%channels = config%width
    this%grid_size = grid_size
    this%modes = config%modes
    this%use_recurrent_memory = config%use_recurrent_memory

    this%lno_proj = conv1d_layer_type( &
         input_shape=[grid_size, 1], &
         num_filters=config%width, &
         kernel_size=1, &
         activation="none")

    allocate(this%lno_blocks(config%num_no_layers))
    do i = 1, size(this%lno_blocks)
       call this%lno_blocks(i)%init(config%width, config%modes, grid_size, config%activation, &
          config%max_amp, config%amp_sharpness, config%pole_offset_scale, &
          config%pole_min, config%pole_max, config%spectral_filter, &
          config%filter_strength, config%use_causal_mask, config%causal_safety)
    end do

    this%rnn_proj = conv1d_layer_type( &
         input_shape=[grid_size, 2], &
         num_filters=config%width, &
         kernel_size=1, &
         activation="none")

    if (config%use_recurrent_memory) then
       call this%memory_cell%init(config%width, config%memory_channels, grid_size)
       call this%memory_fusion%init(config%width, config%memory_channels, grid_size)
    end if

    call this%coeff_head%init(config%width, grid_size)
    call this%corrector%init(config%width, grid_size, config%num_corrections)
    call this%relaxation_gate%init(config%width, grid_size)
  end subroutine init_second_order_predictor

  integer function num_params_second_order_predictor(this)
    !! Return the total learnable parameter count in the predictor scaffold.
    class(second_order_predictor_athena_type), intent(in) :: this
    integer :: i

    num_params_second_order_predictor = this%lno_proj%get_num_params() + this%rnn_proj%get_num_params() + &
         this%coeff_head%num_params() + this%corrector%num_params() + this%relaxation_gate%num_params()
    do i = 1, size(this%lno_blocks)
       num_params_second_order_predictor = num_params_second_order_predictor + this%lno_blocks(i)%num_params()
    end do
    if (this%use_recurrent_memory) then
       num_params_second_order_predictor = num_params_second_order_predictor + &
            this%memory_cell%num_params() + this%memory_fusion%num_params()
    end if
  end function num_params_second_order_predictor

    subroutine forward_second_order_predictor(this, Fo_ext, Ve_ext, tau_dt_ext, dt_star, dx_star, c_star, &
       T_nm1_star_ext, T_n_star_ext, T_inc, a_theta, b_theta, sec_diff_T, backbone_features, hidden_state, new_hidden)
     class(second_order_predictor_athena_type), intent(inout) :: this
     real(real32), intent(in) :: Fo_ext(:,:)
     real(real32), intent(in) :: Ve_ext(:,:)
     real(real32), intent(in) :: tau_dt_ext(:,:)
     real(real32), intent(in) :: dt_star
     real(real32), intent(in) :: dx_star
     real(real32), intent(in) :: c_star(:)
     real(real32), intent(in) :: T_nm1_star_ext(:,:)
     real(real32), intent(in) :: T_n_star_ext(:,:)
     real(real32), allocatable, intent(out) :: T_inc(:,:,:)
     real(real32), allocatable, intent(out) :: a_theta(:,:)
     real(real32), allocatable, intent(out) :: b_theta(:,:)
     real(real32), allocatable, intent(out) :: sec_diff_T(:,:)
     real(real32), allocatable, intent(out) :: backbone_features(:,:,:)
     real(real32), intent(in), optional :: hidden_state(:,:,:)
     real(real32), allocatable, intent(out) :: new_hidden(:,:,:)

     integer :: block_idx
     real(real32), allocatable :: x(:,:,:), x_norm(:,:,:), x_next(:,:,:), rnn_input(:,:,:), rnn_features(:,:,:)
     real(real32), allocatable :: h(:,:,:), h_fused(:,:,:), coeffs(:,:,:), diff_T_ext(:,:), T_inc_init(:,:), T_inc_init_ch(:,:,:)

     call expand_field_channel(T_n_star_ext, x)
     call conv1d_forward_real(this%lno_proj, x, x_next)
     x = x_next

     do block_idx = 1, size(this%lno_blocks)
       call instance_norm1d_real(x, x_norm)
       call this%lno_blocks(block_idx)%forward(x_norm, dt_star, dx_star, c_star, x_next)
       x = x_next
     end do

     h = x
     diff_T_ext = T_n_star_ext - T_nm1_star_ext
     allocate(rnn_input(size(T_n_star_ext, 1), 2, size(T_n_star_ext, 2)))
     rnn_input(:, 1, :) = T_n_star_ext
     rnn_input(:, 2, :) = diff_T_ext
     call conv1d_forward_real(this%rnn_proj, rnn_input, rnn_features)

     if (this%use_recurrent_memory) then
       if (present(hidden_state)) then
         call this%memory_cell%forward(rnn_features, hidden_state, new_hidden)
       else
         call this%memory_cell%forward(rnn_features, h_new=new_hidden)
       end if
       call this%memory_fusion%forward(h, new_hidden, h_fused)
       h = h_fused
     end if

     backbone_features = h
     call this%coeff_head%forward(h, Fo_ext, Ve_ext, tau_dt_ext, coeffs)

     allocate(a_theta(size(Fo_ext, 1), size(Fo_ext, 2)), b_theta(size(Fo_ext, 1), size(Fo_ext, 2)))
     a_theta = coeffs(:, 1, :)
     b_theta = coeffs(:, 2, :)

     call second_difference_replicate(T_n_star_ext, sec_diff_T)
     T_inc_init = a_theta * sec_diff_T + b_theta * diff_T_ext
     call expand_field_channel(T_inc_init, T_inc_init_ch)
     call this%corrector%forward(h, T_inc_init_ch, dt_star, T_inc)
    end subroutine forward_second_order_predictor

end module cattaneo_lno_athena_predictor