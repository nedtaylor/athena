module athena_dynamic_lno_layer_adapter
  use coreutils, only: real32
  use athena, only: conv1d_layer_type, dynamic_lno_layer_type
  use cattaneo_lno_athena_runtime_utils, only: conv1d_forward_real, dynamic_lno_forward_real, silu_real
  implicit none

  private

  public :: dynamic_lno_block_athena_type

  type :: dynamic_lno_block_athena_type
     type(conv1d_layer_type) :: pointwise
     type(dynamic_lno_layer_type) :: spectral
     logical :: requires_custom_instance_norm = .true.
     logical :: spectral_mapping_is_approximate = .true.
   contains
     procedure :: init => init_dynamic_lno_block_athena
     procedure :: forward => forward_dynamic_lno_block_athena
     procedure :: num_params => num_params_dynamic_lno_block_athena
  end type dynamic_lno_block_athena_type

contains

  subroutine init_dynamic_lno_block_athena(this, channels, modes, grid_size, activation, max_amp, amp_sharpness, &
       pole_offset_scale, pole_min, pole_max, spectral_filter, filter_strength, &
       use_causal_mask, causal_safety)
    class(dynamic_lno_block_athena_type), intent(inout) :: this
    integer, intent(in) :: channels, modes, grid_size
    character(len=*), intent(in) :: activation
    real(real32), intent(in) :: max_amp, amp_sharpness, pole_offset_scale
    real(real32), intent(in) :: pole_min, pole_max, filter_strength, causal_safety
    character(len=*), intent(in) :: spectral_filter
    logical, intent(in) :: use_causal_mask

    if (len_trim(activation) < 0) then
    end if
    if (max_amp < 0.0_real32) then
    end if
    if (amp_sharpness < 0.0_real32) then
    end if
    if (pole_offset_scale < 0.0_real32) then
    end if
    if (pole_min < 0.0_real32) then
    end if
    if (pole_max < 0.0_real32) then
    end if
    if (filter_strength < 0.0_real32) then
    end if
    if (causal_safety < 0.0_real32) then
    end if
    if (len_trim(spectral_filter) < 0) then
    end if
    if (use_causal_mask) then
    end if

    this%pointwise = conv1d_layer_type( &
         input_shape=[grid_size, channels], &
         num_filters=channels, &
         kernel_size=1, &
         activation='none')
    this%spectral = dynamic_lno_layer_type( &
         num_outputs=grid_size, &
         num_modes=modes, &
         num_inputs=grid_size, &
         use_bias=.false., &
         activation='none')
  end subroutine init_dynamic_lno_block_athena

  integer function num_params_dynamic_lno_block_athena(this)
    class(dynamic_lno_block_athena_type), intent(in) :: this

    num_params_dynamic_lno_block_athena = this%pointwise%get_num_params() + this%spectral%get_num_params()
  end function num_params_dynamic_lno_block_athena

  subroutine forward_dynamic_lno_block_athena(this, x, dt_star, dx_star, c_star, output)
    class(dynamic_lno_block_athena_type), intent(inout) :: this
    real(real32), intent(in) :: x(:,:,:)
    real(real32), intent(in), optional :: dt_star, dx_star
    real(real32), intent(in), optional :: c_star(:)
    real(real32), allocatable, intent(out) :: output(:,:,:)

    real(real32), allocatable :: pointwise(:,:,:)

    if (present(dt_star)) then
    end if
    if (present(dx_star)) then
    end if
    if (present(c_star)) then
    end if

    call conv1d_forward_real(this%pointwise, x, pointwise)
    pointwise = silu_real(pointwise)
    call dynamic_lno_forward_real(this%spectral, pointwise, output)
  end subroutine forward_dynamic_lno_block_athena

end module athena_dynamic_lno_layer_adapter