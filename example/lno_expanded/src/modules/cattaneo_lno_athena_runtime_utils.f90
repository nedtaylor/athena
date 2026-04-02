module cattaneo_lno_athena_runtime_utils
  use coreutils, only: real32
  use athena, only: conv1d_layer_type, dynamic_lno_layer_type
  use diffstruc, only: array_type
  implicit none

  private

  public :: sigmoid_real
  public :: silu_real
  public :: softplus_real
  public :: conv1d_forward_real
  public :: dynamic_lno_forward_real
  public :: concat_channels
  public :: expand_field_channel
  public :: squeeze_single_channel
  public :: instance_norm1d_real
  public :: extend_dirichlet_field
  public :: replicate_pad_field
  public :: second_difference_replicate
  public :: build_unit_interval

contains

  elemental real(real32) function sigmoid_real(x)
    real(real32), intent(in) :: x

    if (x >= 0.0_real32) then
       sigmoid_real = 1.0_real32 / (1.0_real32 + exp(-x))
    else
       sigmoid_real = exp(x) / (1.0_real32 + exp(x))
    end if
  end function sigmoid_real

  elemental real(real32) function silu_real(x)
    real(real32), intent(in) :: x

    silu_real = x * sigmoid_real(x)
  end function silu_real

  elemental real(real32) function softplus_real(x)
    real(real32), intent(in) :: x

    if (x > 20.0_real32) then
       softplus_real = x
    else if (x < -20.0_real32) then
       softplus_real = exp(x)
    else
       softplus_real = log(1.0_real32 + exp(x))
    end if
  end function softplus_real

  subroutine conv1d_forward_real(layer, input, output)
    type(conv1d_layer_type), intent(inout) :: layer
    real(real32), intent(in) :: input(:,:,:)
    real(real32), allocatable, intent(out) :: output(:,:,:)

    type(array_type) :: input_array(1,1)
    type(array_type), pointer :: output_array(:,:)
    integer :: batch_idx
    integer :: output_shape(2)

    call input_array(1,1)%allocate( &
         array_shape=[size(input, 1), size(input, 2), size(input, 3)], &
         source=0.0_real32)
    call input_array(1,1)%set(input)

    output_array => layer%forward_eval(input_array)
   output_shape = output_array(1,1)%shape
   allocate(output(output_shape(1), output_shape(2), size(output_array(1,1)%val, 2)))
    do batch_idx = 1, size(output, 3)
     output(:, :, batch_idx) = reshape(output_array(1,1)%val(:, batch_idx), output_shape)
    end do

    call input_array(1,1)%deallocate()
  end subroutine conv1d_forward_real

  subroutine dynamic_lno_forward_real(layer, input, output)
   type(dynamic_lno_layer_type), intent(inout) :: layer
   real(real32), intent(in) :: input(:,:,:)
   real(real32), allocatable, intent(out) :: output(:,:,:)

   type(array_type) :: input_array(1,1)
   type(array_type), pointer :: output_array(:,:)
   real(real32), allocatable :: packed_input(:,:)
   integer :: batch_idx, channel_idx, sample_idx
   integer :: grid_size, num_channels, batch_size, packed_batch, output_grid_size

   grid_size = size(input, 1)
   num_channels = size(input, 2)
   batch_size = size(input, 3)
   packed_batch = num_channels * batch_size

   allocate(packed_input(grid_size, packed_batch))
   do batch_idx = 1, batch_size
     do channel_idx = 1, num_channels
       sample_idx = channel_idx + (batch_idx - 1) * num_channels
       packed_input(:, sample_idx) = input(:, channel_idx, batch_idx)
     end do
   end do

   call input_array(1,1)%allocate(array_shape=[grid_size, packed_batch], source=0.0_real32)
   call input_array(1,1)%set(packed_input)

   output_array => layer%forward_eval(input_array)
   output_grid_size = output_array(1,1)%shape(1)
   allocate(output(output_grid_size, num_channels, batch_size))

   do batch_idx = 1, batch_size
     do channel_idx = 1, num_channels
       sample_idx = channel_idx + (batch_idx - 1) * num_channels
       output(:, channel_idx, batch_idx) = output_array(1,1)%val(:, sample_idx)
     end do
   end do

   deallocate(packed_input)
   call input_array(1,1)%deallocate()
  end subroutine dynamic_lno_forward_real

  subroutine concat_channels(left, right, output)
    real(real32), intent(in) :: left(:,:,:)
    real(real32), intent(in) :: right(:,:,:)
    real(real32), allocatable, intent(out) :: output(:,:,:)

    allocate(output(size(left, 1), size(left, 2) + size(right, 2), size(left, 3)))
    output(:, 1:size(left, 2), :) = left
    output(:, size(left, 2) + 1:, :) = right
  end subroutine concat_channels

  subroutine expand_field_channel(field, output)
    real(real32), intent(in) :: field(:,:)
    real(real32), allocatable, intent(out) :: output(:,:,:)

    allocate(output(size(field, 1), 1, size(field, 2)))
    output(:, 1, :) = field
  end subroutine expand_field_channel

  subroutine squeeze_single_channel(tensor, field)
    real(real32), intent(in) :: tensor(:,:,:)
    real(real32), allocatable, intent(out) :: field(:,:)

    allocate(field(size(tensor, 1), size(tensor, 3)))
    field = tensor(:, 1, :)
  end subroutine squeeze_single_channel

  subroutine instance_norm1d_real(input, output, epsilon)
    real(real32), intent(in) :: input(:,:,:)
    real(real32), allocatable, intent(out) :: output(:,:,:)
    real(real32), intent(in), optional :: epsilon

    integer :: batch_idx, channel_idx
    real(real32) :: eps_value, mean_value, variance_value

    eps_value = 1.0e-5_real32
    if (present(epsilon)) eps_value = epsilon

    allocate(output(size(input, 1), size(input, 2), size(input, 3)))
    do batch_idx = 1, size(input, 3)
       do channel_idx = 1, size(input, 2)
          mean_value = sum(input(:, channel_idx, batch_idx)) / real(size(input, 1), real32)
          variance_value = sum((input(:, channel_idx, batch_idx) - mean_value) ** 2) / real(size(input, 1), real32)
          output(:, channel_idx, batch_idx) = (input(:, channel_idx, batch_idx) - mean_value) / sqrt(variance_value + eps_value)
       end do
    end do
  end subroutine instance_norm1d_real

  subroutine extend_dirichlet_field(field, bc_left, bc_right, extended)
    real(real32), intent(in) :: field(:,:)
    real(real32), intent(in) :: bc_left(:)
    real(real32), intent(in) :: bc_right(:)
    real(real32), allocatable, intent(out) :: extended(:,:)

    integer :: grid_size

    grid_size = size(field, 1)
    allocate(extended(grid_size + 2, size(field, 2)))
    extended(1, :) = bc_left
    extended(2:grid_size + 1, :) = field
    extended(grid_size + 2, :) = bc_right
  end subroutine extend_dirichlet_field

  subroutine replicate_pad_field(field, padded)
    real(real32), intent(in) :: field(:,:)
    real(real32), allocatable, intent(out) :: padded(:,:)

    integer :: grid_size

    grid_size = size(field, 1)
    allocate(padded(grid_size + 2, size(field, 2)))
    padded(1, :) = field(1, :)
    padded(2:grid_size + 1, :) = field
    padded(grid_size + 2, :) = field(grid_size, :)
  end subroutine replicate_pad_field

  subroutine second_difference_replicate(field, sec_diff)
    real(real32), intent(in) :: field(:,:)
    real(real32), allocatable, intent(out) :: sec_diff(:,:)

    real(real32), allocatable :: padded(:,:)
    integer :: grid_size

    grid_size = size(field, 1)
    call replicate_pad_field(field, padded)
    allocate(sec_diff(grid_size, size(field, 2)))
    sec_diff = padded(3:grid_size + 2, :) - 2.0_real32 * field + padded(1:grid_size, :)
  end subroutine second_difference_replicate

  subroutine build_unit_interval(num_points, xi, boundary_mask)
    integer, intent(in) :: num_points
    real(real32), allocatable, intent(out) :: xi(:)
    real(real32), allocatable, intent(out) :: boundary_mask(:)

    integer :: point_idx

    allocate(xi(num_points), boundary_mask(num_points))
    do point_idx = 1, num_points
       xi(point_idx) = real(point_idx - 1, real32) / real(max(1, num_points - 1), real32)
    end do

    boundary_mask = 1.0_real32
    if (num_points >= 1) boundary_mask(1) = 0.0_real32
    if (num_points >= 2) boundary_mask(num_points) = 0.0_real32
  end subroutine build_unit_interval

end module cattaneo_lno_athena_runtime_utils