module cattaneo_lno_athena_blocks
  !! Custom building blocks for a Python-matched Cattaneo-LNO scaffold.
  !!
  !! Laplace-specific scaffolding now lives in custom_laplace_layer so
  !! the remaining blocks here only cover the non-spectral parts of the model.
  use coreutils, only: real32
  use athena, only: conv1d_layer_type
  use cattaneo_lno_athena_runtime_utils, only: conv1d_forward_real, concat_channels, softplus_real
  use custom_laplace_layer, only: custom_laplace_block_type
  implicit none

  private

  public :: custom_laplace_block_type
  public :: conv1d_gru_cell_athena_type
  public :: memory_fusion_athena_type
  public :: wave_diffusion_head_athena_type
  public :: iterative_corrector_athena_type
  public :: relaxation_gate_athena_type

    real(real32), parameter :: max_correction_frac = 0.01_real32

  type :: conv1d_gru_cell_athena_type
     !! Pointwise spatial GRU cell assembled from Athena 1x1 convolutions.
     type(conv1d_layer_type) :: w_rz
     !! Joint reset and update gate projection.
     type(conv1d_layer_type) :: w_h
     !! Candidate hidden-state projection.
      integer :: hidden_channels = 0
   contains
     procedure :: init => init_conv1d_gru_cell
      procedure :: forward => forward_conv1d_gru_cell
     procedure :: num_params => num_params_conv1d_gru_cell
  end type conv1d_gru_cell_athena_type

  type :: memory_fusion_athena_type
     !! Gated memory fusion block matching the Python module boundary.
     type(conv1d_layer_type) :: proj
     !! Memory projection into backbone width.
     type(conv1d_layer_type) :: gate
     !! Gate producing per-channel fusion weights.
   contains
     procedure :: init => init_memory_fusion
     procedure :: forward => forward_memory_fusion
     procedure :: num_params => num_params_memory_fusion
  end type memory_fusion_athena_type

  type :: wave_diffusion_head_athena_type
     !! Coefficient head mirroring the Python residual MLP layout.
     type(conv1d_layer_type) :: conv1
     type(conv1d_layer_type) :: conv2
     type(conv1d_layer_type) :: conv3
   contains
     procedure :: init => init_wave_diffusion_head
     procedure :: forward => forward_wave_diffusion_head
     procedure :: num_params => num_params_wave_diffusion_head
  end type wave_diffusion_head_athena_type

  type :: iterative_corrector_athena_type
     !! Iterative correction block with learnable per-step scales.
     type(conv1d_layer_type) :: conv1
     type(conv1d_layer_type) :: conv2
     type(conv1d_layer_type) :: conv3
     real(real32), allocatable :: step_sizes(:)
     !! Learnable step sizes represented explicitly in the scaffold.
   contains
     procedure :: init => init_iterative_corrector
     procedure :: forward => forward_iterative_corrector
     procedure :: num_params => num_params_iterative_corrector
  end type iterative_corrector_athena_type

  type :: relaxation_gate_athena_type
     !! Relaxation gate block matching the Python gate MLP layout.
     type(conv1d_layer_type) :: conv1
     type(conv1d_layer_type) :: conv2
     real(real32) :: relax_strength = -2.94_real32
     !! Softplus-parameterised relaxation strength from the Python model.
   contains
     procedure :: init => init_relaxation_gate
     procedure :: forward => forward_relaxation_gate
     procedure :: num_params => num_params_relaxation_gate
  end type relaxation_gate_athena_type

contains

  subroutine init_conv1d_gru_cell(this, input_channels, hidden_channels, grid_size)
    !! Initialise the pointwise GRU cell stand-in.
    class(conv1d_gru_cell_athena_type), intent(inout) :: this
    integer, intent(in) :: input_channels
    integer, intent(in) :: hidden_channels
    integer, intent(in) :: grid_size

    this%hidden_channels = hidden_channels
    this%w_rz = conv1d_layer_type( &
         input_shape=[grid_size, input_channels + hidden_channels], &
         num_filters=2 * hidden_channels, &
         kernel_size=1, &
         activation="none")
    this%w_h = conv1d_layer_type( &
         input_shape=[grid_size, input_channels + hidden_channels], &
         num_filters=hidden_channels, &
         kernel_size=1, &
         activation="none")
  end subroutine init_conv1d_gru_cell

  integer function num_params_conv1d_gru_cell(this)
    !! Return the learnable parameter count for the GRU cell stand-in.
    class(conv1d_gru_cell_athena_type), intent(in) :: this

    num_params_conv1d_gru_cell = this%w_rz%get_num_params() + this%w_h%get_num_params()
  end function num_params_conv1d_gru_cell

  subroutine forward_conv1d_gru_cell(this, x, hidden_state, h_new)
    class(conv1d_gru_cell_athena_type), intent(inout) :: this
    real(real32), intent(in) :: x(:,:,:)
    real(real32), intent(in), optional :: hidden_state(:,:,:)
    real(real32), allocatable, intent(out) :: h_new(:,:,:)

    real(real32), allocatable :: h_prev(:,:,:), gate_input(:,:,:), rz(:,:,:), candidate_input(:,:,:), h_tilde(:,:,:)

    allocate(h_prev(size(x, 1), this%hidden_channels, size(x, 3)), source=0.0_real32)
    if (present(hidden_state)) h_prev = hidden_state

    call concat_channels(x, h_prev, gate_input)
    call conv1d_forward_real(this%w_rz, gate_input, rz)
    rz = 1.0_real32 / (1.0_real32 + exp(-rz))

    allocate(candidate_input(size(x, 1), size(x, 2) + this%hidden_channels, size(x, 3)))
    candidate_input(:, 1:size(x, 2), :) = x
    candidate_input(:, size(x, 2) + 1:, :) = rz(:, 1:this%hidden_channels, :) * h_prev
    call conv1d_forward_real(this%w_h, candidate_input, h_tilde)
    h_tilde = tanh(h_tilde)

    allocate(h_new(size(h_prev, 1), size(h_prev, 2), size(h_prev, 3)))
    h_new = (1.0_real32 - rz(:, this%hidden_channels + 1:, :)) * h_prev + rz(:, this%hidden_channels + 1:, :) * h_tilde
  end subroutine forward_conv1d_gru_cell

  subroutine init_memory_fusion(this, backbone_channels, memory_channels, grid_size)
    !! Initialise the gated memory fusion block.
    class(memory_fusion_athena_type), intent(inout) :: this
    integer, intent(in) :: backbone_channels
    integer, intent(in) :: memory_channels
    integer, intent(in) :: grid_size

    this%proj = conv1d_layer_type( &
         input_shape=[grid_size, memory_channels], &
         num_filters=backbone_channels, &
         kernel_size=1, &
         activation="none")
    this%gate = conv1d_layer_type( &
         input_shape=[grid_size, backbone_channels + memory_channels], &
         num_filters=backbone_channels, &
         kernel_size=1, &
         activation="sigmoid")
  end subroutine init_memory_fusion

  integer function num_params_memory_fusion(this)
    !! Return the learnable parameter count for the memory fusion block.
    class(memory_fusion_athena_type), intent(in) :: this

    num_params_memory_fusion = this%proj%get_num_params() + this%gate%get_num_params()
  end function num_params_memory_fusion

  subroutine forward_memory_fusion(this, h_backbone, memory, output)
    class(memory_fusion_athena_type), intent(inout) :: this
    real(real32), intent(in) :: h_backbone(:,:,:)
    real(real32), intent(in) :: memory(:,:,:)
    real(real32), allocatable, intent(out) :: output(:,:,:)

    real(real32), allocatable :: projected(:,:,:), gate_input(:,:,:), gate_values(:,:,:)

    call conv1d_forward_real(this%proj, memory, projected)
    call concat_channels(h_backbone, memory, gate_input)
    call conv1d_forward_real(this%gate, gate_input, gate_values)

    allocate(output(size(h_backbone, 1), size(h_backbone, 2), size(h_backbone, 3)))
    output = h_backbone + gate_values * projected
  end subroutine forward_memory_fusion

  subroutine init_wave_diffusion_head(this, channels, grid_size)
    !! Initialise the coefficient head scaffold.
    class(wave_diffusion_head_athena_type), intent(inout) :: this
    integer, intent(in) :: channels
    integer, intent(in) :: grid_size
    integer :: hidden_channels

    hidden_channels = max(1, channels / 2)
    this%conv1 = conv1d_layer_type( &
         input_shape=[grid_size, channels + 2], &
         num_filters=channels, &
         kernel_size=1, &
         activation="swish")
    this%conv2 = conv1d_layer_type( &
         input_shape=[grid_size, channels], &
         num_filters=hidden_channels, &
         kernel_size=1, &
         activation="swish")
    this%conv3 = conv1d_layer_type( &
         input_shape=[grid_size, hidden_channels], &
         num_filters=1, &
         kernel_size=1, &
         activation="none")
  end subroutine init_wave_diffusion_head

  integer function num_params_wave_diffusion_head(this)
    !! Return the learnable parameter count for the coefficient head.
    class(wave_diffusion_head_athena_type), intent(in) :: this

    num_params_wave_diffusion_head = this%conv1%get_num_params() + &
         this%conv2%get_num_params() + this%conv3%get_num_params()
  end function num_params_wave_diffusion_head

  subroutine forward_wave_diffusion_head(this, h, Fo, Ve, tau_dt, coeffs)
    class(wave_diffusion_head_athena_type), intent(inout) :: this
    real(real32), intent(in) :: h(:,:,:)
    real(real32), intent(in) :: Fo(:,:)
    real(real32), intent(in) :: Ve(:,:)
    real(real32), intent(in) :: tau_dt(:,:)
    real(real32), allocatable, intent(out) :: coeffs(:,:,:)

    real(real32), allocatable :: coeff_input(:,:,:), hidden1(:,:,:), hidden2(:,:,:), raw_a(:,:,:)
    real(real32), allocatable :: a_target(:,:), b_target(:,:), bound_a(:,:)

    allocate(a_target(size(Fo, 1), size(Fo, 2)), b_target(size(Fo, 1), size(Fo, 2)), bound_a(size(Fo, 1), size(Fo, 2)))
    a_target = Fo / (1.0_real32 + tau_dt)
    b_target = tau_dt / (1.0_real32 + tau_dt)
    bound_a = max_correction_frac * max(abs(a_target), 1.0e-6_real32)

    allocate(coeff_input(size(h, 1), size(h, 2) + 2, size(h, 3)))
    coeff_input(:, 1:size(h, 2), :) = h
    coeff_input(:, size(h, 2) + 1, :) = log(1.0_real32 + Fo)
    coeff_input(:, size(h, 2) + 2, :) = Ve

    call conv1d_forward_real(this%conv1, coeff_input, hidden1)
    call conv1d_forward_real(this%conv2, hidden1, hidden2)
    call conv1d_forward_real(this%conv3, hidden2, raw_a)

    allocate(coeffs(size(Fo, 1), 2, size(Fo, 2)))
    coeffs(:, 1, :) = a_target + bound_a * tanh(raw_a(:, 1, :))
    coeffs(:, 2, :) = b_target
  end subroutine forward_wave_diffusion_head

  subroutine init_iterative_corrector(this, channels, grid_size, num_iterations)
    !! Initialise the iterative corrector scaffold.
    class(iterative_corrector_athena_type), intent(inout) :: this
    integer, intent(in) :: channels
    integer, intent(in) :: grid_size
    integer, intent(in) :: num_iterations
    integer :: hidden_channels

    hidden_channels = max(1, channels / 2)
    this%conv1 = conv1d_layer_type( &
         input_shape=[grid_size, channels + 1], &
         num_filters=channels, &
         kernel_size=1, &
         activation="swish")
    this%conv2 = conv1d_layer_type( &
         input_shape=[grid_size, channels], &
         num_filters=hidden_channels, &
         kernel_size=1, &
         activation="swish")
    this%conv3 = conv1d_layer_type( &
         input_shape=[grid_size, hidden_channels], &
         num_filters=1, &
         kernel_size=1, &
         activation="none")
    allocate(this%step_sizes(num_iterations))
    this%step_sizes = 0.1_real32
  end subroutine init_iterative_corrector

  integer function num_params_iterative_corrector(this)
    !! Return the learnable parameter count for the iterative corrector.
    class(iterative_corrector_athena_type), intent(in) :: this

    num_params_iterative_corrector = this%conv1%get_num_params() + &
         this%conv2%get_num_params() + this%conv3%get_num_params()
    if (allocated(this%step_sizes)) then
       num_params_iterative_corrector = num_params_iterative_corrector + size(this%step_sizes)
    end if
  end function num_params_iterative_corrector

    subroutine forward_iterative_corrector(this, h, T_inc_init, dt_star, T_inc)
     class(iterative_corrector_athena_type), intent(inout) :: this
     real(real32), intent(in) :: h(:,:,:)
     real(real32), intent(in) :: T_inc_init(:,:,:)
     real(real32), intent(in) :: dt_star
     real(real32), allocatable, intent(out) :: T_inc(:,:,:)

     integer :: batch_idx, step_idx
     real(real32), allocatable :: signal_gate(:,:), corr_input(:,:,:), hidden1(:,:,:), hidden2(:,:,:), delta(:,:,:)

     if (dt_star < 0.0_real32) then
     end if

     allocate(T_inc(size(T_inc_init, 1), size(T_inc_init, 2), size(T_inc_init, 3)), source=T_inc_init)
     allocate(signal_gate(size(T_inc_init, 1), size(T_inc_init, 3)))
     do batch_idx = 1, size(T_inc_init, 3)
       signal_gate(:, batch_idx) = abs(T_inc_init(:, 1, batch_idx)) / max(maxval(abs(T_inc_init(:, 1, batch_idx))), 1.0e-12_real32)
     end do

     do step_idx = 1, size(this%step_sizes)
       call concat_channels(h, T_inc, corr_input)
       call conv1d_forward_real(this%conv1, corr_input, hidden1)
       call conv1d_forward_real(this%conv2, hidden1, hidden2)
       call conv1d_forward_real(this%conv3, hidden2, delta)
       do batch_idx = 1, size(T_inc, 3)
         T_inc(:, 1, batch_idx) = T_inc(:, 1, batch_idx) + this%step_sizes(step_idx) * delta(:, 1, batch_idx) * signal_gate(:, batch_idx)
       end do
     end do
    end subroutine forward_iterative_corrector

  subroutine init_relaxation_gate(this, channels, grid_size)
    !! Initialise the relaxation gate scaffold.
    class(relaxation_gate_athena_type), intent(inout) :: this
    integer, intent(in) :: channels
    integer, intent(in) :: grid_size
    integer :: hidden_channels

    hidden_channels = max(1, channels / 2)
    this%conv1 = conv1d_layer_type( &
         input_shape=[grid_size, channels + 1], &
         num_filters=hidden_channels, &
         kernel_size=1, &
         activation="swish")
    this%conv2 = conv1d_layer_type( &
         input_shape=[grid_size, hidden_channels], &
         num_filters=1, &
         kernel_size=1, &
         activation="sigmoid")
  end subroutine init_relaxation_gate

  integer function num_params_relaxation_gate(this)
    !! Return the learnable parameter count for the relaxation gate.
    class(relaxation_gate_athena_type), intent(in) :: this

    num_params_relaxation_gate = this%conv1%get_num_params() + &
         this%conv2%get_num_params() + 1
  end function num_params_relaxation_gate

  subroutine forward_relaxation_gate(this, h, T_inc, T_n_star, bc_left_star, bc_right_star, xi, dt_star, T_inc_gated)
    class(relaxation_gate_athena_type), intent(inout) :: this
    real(real32), intent(in) :: h(:,:,:)
    real(real32), intent(in) :: T_inc(:,:,:)
    real(real32), intent(in) :: T_n_star(:,:)
    real(real32), intent(in) :: bc_left_star(:)
    real(real32), intent(in) :: bc_right_star(:)
    real(real32), intent(in) :: xi(:)
    real(real32), intent(in) :: dt_star
    real(real32), allocatable, intent(out) :: T_inc_gated(:,:,:)

    real(real32), allocatable :: gate_input(:,:,:), hidden1(:,:,:), gate(:,:,:), T_ref(:,:), relax_dir(:,:)
    real(real32) :: relax_strength_value

    if (dt_star < 0.0_real32) then
    end if

    call concat_channels(h, T_inc, gate_input)
    call conv1d_forward_real(this%conv1, gate_input, hidden1)
    call conv1d_forward_real(this%conv2, hidden1, gate)

    relax_strength_value = softplus_real(this%relax_strength)
    allocate(T_ref(size(T_n_star, 1), size(T_n_star, 2)), relax_dir(size(T_n_star, 1), size(T_n_star, 2)))
    T_ref = spread(bc_left_star, dim=1, ncopies=size(xi)) + &
         spread(bc_right_star - bc_left_star, dim=1, ncopies=size(xi)) * spread(xi, dim=2, ncopies=size(T_n_star, 2))
    relax_dir = T_ref - T_n_star

    allocate(T_inc_gated(size(T_inc, 1), size(T_inc, 2), size(T_inc, 3)))
    T_inc_gated(:, 1, :) = T_inc(:, 1, :) + gate(:, 1, :) * relax_strength_value * relax_dir
  end subroutine forward_relaxation_gate

end module cattaneo_lno_athena_blocks