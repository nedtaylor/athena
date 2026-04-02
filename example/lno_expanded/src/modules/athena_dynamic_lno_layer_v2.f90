module athena_dynamic_lno_layer_v2
  !> Python-matched replacement for Athena's dynamic LNO layer.
  !!
  !! This project-local v2 layer keeps Athena's block-oriented API style while
  !! switching the spectral operator to the bounded-polar Laplace kernel used
  !! by the current near-parity Fortran path. It is intended as the closest
  !! Athena-facing replacement for the built-in dynamic LNO layer without
  !! requiring changes inside the vendored Athena dependency.
  use coreutils, only: real32
  use athena, only: conv1d_layer_type
  use cattaneo_lno_athena_runtime_utils, only: conv1d_forward_real, sigmoid_real, silu_real, softplus_real
  implicit none

  private

  public :: dynamic_lno_layer_v2_type
  public :: dynamic_lno_block_v2_type

  real(real32), parameter :: pi_real32 = 3.14159265358979323846_real32

  type :: dynamic_lno_layer_v2_type
     integer :: in_channels = 0
     integer :: out_channels = 0
     integer :: modes = 0
     integer :: grid_size = 0
     real(real32) :: max_amp = 1.0_real32
     real(real32) :: amp_sharpness = 1.0_real32
     real(real32) :: pole_offset_scale = 0.1_real32
     real(real32) :: pole_min = 0.1_real32
     real(real32) :: pole_max = 100.0_real32
     real(real32) :: filter_strength = 4.0_real32
     real(real32) :: causal_safety = 1.0_real32
     logical :: use_causal_mask = .true.
     character(len=32) :: spectral_filter = 'exponential'
     real(real32), allocatable :: weight_log_amp(:,:,:)
     real(real32), allocatable :: weight_phase(:,:,:)
     real(real32), allocatable :: log_poles(:)
     real(real32), allocatable :: pole_mlp_w(:,:)
     real(real32), allocatable :: pole_mlp_b(:)
     real(real32), allocatable :: filter_values(:)
     real(real32), allocatable :: dist(:,:)
   contains
     procedure :: init => init_dynamic_lno_layer_v2
     procedure :: forward => forward_dynamic_lno_layer_v2
     procedure :: num_params => num_params_dynamic_lno_layer_v2
  end type dynamic_lno_layer_v2_type

  type :: dynamic_lno_block_v2_type
     type(conv1d_layer_type) :: pointwise
     type(dynamic_lno_layer_v2_type) :: spectral
     logical :: requires_custom_instance_norm = .true.
     logical :: spectral_mapping_is_approximate = .false.
   contains
     procedure :: init => init_dynamic_lno_block_v2
     procedure :: forward => forward_dynamic_lno_block_v2
     procedure :: num_params => num_params_dynamic_lno_block_v2
  end type dynamic_lno_block_v2_type

contains

  subroutine init_dynamic_lno_layer_v2(this, in_channels, out_channels, modes, grid_size, &
       max_amp, amp_sharpness, pole_offset_scale, pole_min, pole_max, &
       spectral_filter, filter_strength, use_causal_mask, causal_safety)
    class(dynamic_lno_layer_v2_type), intent(inout) :: this
    integer, intent(in) :: in_channels, out_channels, modes, grid_size
    real(real32), intent(in) :: max_amp, amp_sharpness, pole_offset_scale
    real(real32), intent(in) :: pole_min, pole_max, filter_strength, causal_safety
    character(len=*), intent(in) :: spectral_filter
    logical, intent(in) :: use_causal_mask

    integer :: k
    real(real32) :: target_amp, target_sigmoid, init_bias, init_scale, mlp_scale

    this%in_channels = in_channels
    this%out_channels = out_channels
    this%modes = modes
    this%grid_size = grid_size
    this%max_amp = max_amp
    this%amp_sharpness = amp_sharpness
    this%pole_offset_scale = pole_offset_scale
    this%pole_min = pole_min
    this%pole_max = pole_max
    this%spectral_filter = spectral_filter
    this%filter_strength = filter_strength
    this%use_causal_mask = use_causal_mask
    this%causal_safety = causal_safety

    allocate(this%weight_log_amp(modes, in_channels, out_channels))
    allocate(this%weight_phase(modes, in_channels, out_channels))
    allocate(this%log_poles(modes))
    allocate(this%pole_mlp_w(modes, in_channels))
    allocate(this%pole_mlp_b(modes))
    allocate(this%filter_values(modes))
    allocate(this%dist(grid_size, grid_size))

    target_amp = sqrt(2.0_real32 / real(max(1, modes * in_channels), real32))
    target_sigmoid = min(0.99_real32, max(0.01_real32, target_amp / max(max_amp, 1.0e-6_real32)))
    init_bias = log(target_sigmoid / (1.0_real32 - target_sigmoid)) / max(amp_sharpness, 1.0e-6_real32)
    init_scale = 1.0_real32 / sqrt(real(max(1, in_channels * out_channels * modes), real32))
    mlp_scale = 1.0_real32 / sqrt(real(max(1, in_channels), real32))

    call random_normal_3d(this%weight_log_amp, init_bias, init_scale)
    call random_normal_3d(this%weight_phase, 0.0_real32, 1.0_real32)
    call random_normal_2d(this%pole_mlp_w, 0.0_real32, mlp_scale)
    this%pole_mlp_b = 0.0_real32

    do k = 1, modes
       this%log_poles(k) = log(1.0_real32) + &
            (log(50.0_real32) - log(1.0_real32)) * real(k - 1, real32) / real(max(1, modes - 1), real32)
    end do

    call build_spectral_filter(this%filter_values, spectral_filter, filter_strength)
    call build_distance_matrix(this%dist)
  end subroutine init_dynamic_lno_layer_v2

  integer function num_params_dynamic_lno_layer_v2(this)
    class(dynamic_lno_layer_v2_type), intent(in) :: this

    num_params_dynamic_lno_layer_v2 = 0
    if (allocated(this%weight_log_amp)) num_params_dynamic_lno_layer_v2 = num_params_dynamic_lno_layer_v2 + size(this%weight_log_amp)
    if (allocated(this%weight_phase)) num_params_dynamic_lno_layer_v2 = num_params_dynamic_lno_layer_v2 + size(this%weight_phase)
    if (allocated(this%log_poles)) num_params_dynamic_lno_layer_v2 = num_params_dynamic_lno_layer_v2 + size(this%log_poles)
    if (allocated(this%pole_mlp_w)) num_params_dynamic_lno_layer_v2 = num_params_dynamic_lno_layer_v2 + size(this%pole_mlp_w)
    if (allocated(this%pole_mlp_b)) num_params_dynamic_lno_layer_v2 = num_params_dynamic_lno_layer_v2 + size(this%pole_mlp_b)
  end function num_params_dynamic_lno_layer_v2

  subroutine forward_dynamic_lno_layer_v2(this, x, dt_star, dx_star, c_star, output)
    class(dynamic_lno_layer_v2_type), intent(in) :: this
    real(real32), intent(in) :: x(:,:,:)
    real(real32), intent(in), optional :: dt_star, dx_star
    real(real32), intent(in), optional :: c_star(:)
    real(real32), allocatable, intent(out) :: output(:,:,:)

    integer :: batch_idx, grid_idx, input_idx, output_idx, mode_idx
    integer :: grid_size, batch_size, num_poles
    real(real32) :: c_eff, max_dist_norm, row_norm, pole_value, weight_amp
    real(real32), allocatable :: x_mean(:), pole_offsets(:), poles(:)
    real(real32), allocatable :: dist(:,:), causal(:,:), kernels(:,:,:), kernel_x(:,:,:), weights(:,:,:)

    if (present(dx_star)) then
    end if

    grid_size = size(x, 1)
    batch_size = size(x, 3)
    num_poles = min(this%modes, max(1, grid_size))

    allocate(x_mean(this%in_channels), pole_offsets(this%modes), poles(this%modes))
    allocate(kernels(num_poles, grid_size, grid_size), kernel_x(grid_size, this%in_channels, batch_size))
    allocate(weights(num_poles, this%in_channels, this%out_channels), output(grid_size, this%out_channels, batch_size))

    if (grid_size == this%grid_size .and. allocated(this%dist)) then
       allocate(dist(grid_size, grid_size), source=this%dist)
    else
       allocate(dist(grid_size, grid_size))
       call build_distance_matrix(dist)
    end if
    allocate(causal(grid_size, grid_size), source=1.0_real32)

    do input_idx = 1, this%in_channels
       x_mean(input_idx) = sum(x(:, input_idx, :)) / real(grid_size * batch_size, real32)
    end do

    do mode_idx = 1, this%modes
       pole_offsets(mode_idx) = tanh(dot_product(this%pole_mlp_w(mode_idx, :), x_mean) + this%pole_mlp_b(mode_idx))
       poles(mode_idx) = min(this%pole_max, max(this%pole_min, &
            softplus_real(this%log_poles(mode_idx) + this%pole_offset_scale * pole_offsets(mode_idx))))
    end do

    if (this%use_causal_mask .and. present(c_star) .and. present(dt_star) .and. present(dx_star)) then
       c_eff = this%causal_safety * maxval(c_star)
       max_dist_norm = c_eff * dt_star
       do batch_idx = 1, grid_size
          do grid_idx = 1, grid_size
             if (dist(grid_idx, batch_idx) > max_dist_norm) causal(grid_idx, batch_idx) = 0.0_real32
          end do
       end do
    end if

    do mode_idx = 1, num_poles
       do output_idx = 1, this%out_channels
          do input_idx = 1, this%in_channels
             weight_amp = this%max_amp * sigmoid_real(this%amp_sharpness * this%weight_log_amp(mode_idx, input_idx, output_idx))
             weights(mode_idx, input_idx, output_idx) = weight_amp * cos(this%weight_phase(mode_idx, input_idx, output_idx))
          end do
       end do
    end do

    do mode_idx = 1, num_poles
       do grid_idx = 1, grid_size
          row_norm = 0.0_real32
          pole_value = poles(mode_idx)
          do batch_idx = 1, grid_size
             kernels(mode_idx, grid_idx, batch_idx) = exp(-pole_value * dist(grid_idx, batch_idx)) * causal(grid_idx, batch_idx)
             row_norm = row_norm + kernels(mode_idx, grid_idx, batch_idx)
          end do
          row_norm = max(row_norm, 1.0e-8_real32)
          kernels(mode_idx, grid_idx, :) = kernels(mode_idx, grid_idx, :) * this%filter_values(mode_idx) / row_norm
       end do
    end do

    output = 0.0_real32
    do mode_idx = 1, num_poles
       do batch_idx = 1, batch_size
          do input_idx = 1, this%in_channels
             do grid_idx = 1, grid_size
                kernel_x(grid_idx, input_idx, batch_idx) = sum(kernels(mode_idx, grid_idx, :) * x(:, input_idx, batch_idx))
             end do
          end do
       end do

       do batch_idx = 1, batch_size
          do output_idx = 1, this%out_channels
             do input_idx = 1, this%in_channels
                output(:, output_idx, batch_idx) = output(:, output_idx, batch_idx) + &
                     weights(mode_idx, input_idx, output_idx) * kernel_x(:, input_idx, batch_idx)
             end do
          end do
       end do
    end do
  end subroutine forward_dynamic_lno_layer_v2

  subroutine init_dynamic_lno_block_v2(this, channels, modes, grid_size, activation, max_amp, amp_sharpness, &
       pole_offset_scale, pole_min, pole_max, spectral_filter, filter_strength, &
       use_causal_mask, causal_safety)
    class(dynamic_lno_block_v2_type), intent(inout) :: this
    integer, intent(in) :: channels, modes, grid_size
    character(len=*), intent(in) :: activation
    real(real32), intent(in) :: max_amp, amp_sharpness, pole_offset_scale
    real(real32), intent(in) :: pole_min, pole_max, filter_strength, causal_safety
    character(len=*), intent(in) :: spectral_filter
    logical, intent(in) :: use_causal_mask

    if (len_trim(activation) < 0) then
    end if

    this%pointwise = conv1d_layer_type( &
         input_shape=[grid_size, channels], &
         num_filters=channels, &
         kernel_size=1, &
         activation='none')
    call this%spectral%init(channels, channels, modes, grid_size, &
         max_amp=max_amp, amp_sharpness=amp_sharpness, &
         pole_offset_scale=pole_offset_scale, pole_min=pole_min, pole_max=pole_max, &
         spectral_filter=spectral_filter, filter_strength=filter_strength, &
         use_causal_mask=use_causal_mask, causal_safety=causal_safety)
  end subroutine init_dynamic_lno_block_v2

  integer function num_params_dynamic_lno_block_v2(this)
    class(dynamic_lno_block_v2_type), intent(in) :: this

    num_params_dynamic_lno_block_v2 = this%pointwise%get_num_params() + this%spectral%num_params()
  end function num_params_dynamic_lno_block_v2

  subroutine forward_dynamic_lno_block_v2(this, x, dt_star, dx_star, c_star, output)
    class(dynamic_lno_block_v2_type), intent(inout) :: this
    real(real32), intent(in) :: x(:,:,:)
    real(real32), intent(in), optional :: dt_star, dx_star
    real(real32), intent(in), optional :: c_star(:)
    real(real32), allocatable, intent(out) :: output(:,:,:)

    real(real32), allocatable :: pointwise(:,:,:)

    call conv1d_forward_real(this%pointwise, x, pointwise)
    pointwise = silu_real(pointwise)
    call this%spectral%forward(pointwise, dt_star, dx_star, c_star, output)
  end subroutine forward_dynamic_lno_block_v2

  subroutine build_spectral_filter(filter_values, spectral_filter, filter_strength)
    real(real32), intent(out) :: filter_values(:)
    character(len=*), intent(in) :: spectral_filter
    real(real32), intent(in) :: filter_strength

    integer :: k, n_modes, cutoff
    real(real32) :: k_norm

    n_modes = size(filter_values)
    if (n_modes <= 1) then
       filter_values = 1.0_real32
       return
    end if

    select case (trim(spectral_filter))
    case ('none', 'None')
       filter_values = 1.0_real32
    case ('exponential')
       do k = 1, n_modes
          k_norm = real(k - 1, real32) / real(n_modes - 1, real32)
          filter_values(k) = exp(-filter_strength * k_norm * k_norm)
       end do
    case ('raised_cosine')
       do k = 1, n_modes
          k_norm = real(k - 1, real32) / real(n_modes - 1, real32)
          filter_values(k) = 0.5_real32 * (1.0_real32 + cos(pi_real32 * k_norm))
       end do
    case ('sharp_cutoff')
       cutoff = int(real(n_modes, real32) * 2.0_real32 / 3.0_real32)
       filter_values = 1.0_real32
       if (cutoff < n_modes) filter_values(cutoff + 1:n_modes) = 0.0_real32
    case ('dealias')
       cutoff = int(real(n_modes, real32) * filter_strength)
       cutoff = max(1, min(n_modes, cutoff))
       filter_values = 1.0_real32
       if (cutoff < n_modes) filter_values(cutoff + 1:n_modes) = 0.0_real32
    case ('transient_optimized')
       cutoff = int(real(n_modes, real32) * 0.8_real32)
       cutoff = max(1, min(n_modes, cutoff))
       filter_values = 1.0_real32
       if (cutoff < n_modes) then
          do k = cutoff + 1, n_modes
             filter_values(k) = 1.0_real32 - real(k - cutoff, real32) / real(n_modes - cutoff, real32)
          end do
       end if
    case default
       filter_values = 1.0_real32
    end select
  end subroutine build_spectral_filter

  subroutine build_distance_matrix(dist)
    real(real32), intent(out) :: dist(:,:)

    integer :: i, j, grid_size
    real(real32) :: denom

    grid_size = size(dist, 1)
    denom = real(max(1, grid_size - 1), real32)
    do j = 1, grid_size
       do i = 1, grid_size
          dist(i, j) = abs(real(i - 1, real32) - real(j - 1, real32)) / denom
       end do
    end do
  end subroutine build_distance_matrix

  subroutine random_normal_2d(arr, mean, std)
    real(real32), intent(out) :: arr(:,:)
    real(real32), intent(in) :: mean, std

    integer :: i, j
    real(real32) :: u1, u2, z0, z1

    do j = 1, size(arr, 2)
       i = 1
       do while (i <= size(arr, 1))
          call random_number(u1)
          call random_number(u2)
          u1 = max(u1, 1.0e-10_real32)
          z0 = sqrt(-2.0_real32 * log(u1)) * cos(2.0_real32 * pi_real32 * u2)
          z1 = sqrt(-2.0_real32 * log(u1)) * sin(2.0_real32 * pi_real32 * u2)
          arr(i, j) = mean + std * z0
          if (i + 1 <= size(arr, 1)) arr(i + 1, j) = mean + std * z1
          i = i + 2
       end do
    end do
  end subroutine random_normal_2d

  subroutine random_normal_3d(arr, mean, std)
    real(real32), intent(out) :: arr(:,:,:)
    real(real32), intent(in) :: mean, std

    integer :: i, j, k
    real(real32) :: u1, u2, z0, z1

    do k = 1, size(arr, 3)
       do j = 1, size(arr, 2)
          i = 1
          do while (i <= size(arr, 1))
             call random_number(u1)
             call random_number(u2)
             u1 = max(u1, 1.0e-10_real32)
             z0 = sqrt(-2.0_real32 * log(u1)) * cos(2.0_real32 * pi_real32 * u2)
             z1 = sqrt(-2.0_real32 * log(u1)) * sin(2.0_real32 * pi_real32 * u2)
             arr(i, j, k) = mean + std * z0
             if (i + 1 <= size(arr, 1)) arr(i + 1, j, k) = mean + std * z1
             i = i + 2
          end do
       end do
    end do
  end subroutine random_normal_3d

end module athena_dynamic_lno_layer_v2