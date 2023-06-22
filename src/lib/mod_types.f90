module custom_types
  use constants, only: real12
  implicit none

  type clip_type
     logical :: l_min_max, l_norm
     real(real12) :: min, max, norm
  end type clip_type

  type neuron_type
     real(real12) :: output
     real(real12) :: delta, delta_batch
     real(real12), allocatable, dimension(:) :: weight, weight_incr
  end type neuron_type

  type network_type
     type(neuron_type), allocatable, dimension(:) :: neuron
  end type network_type

  type convolution_type
     integer :: kernel_size
     integer :: stride
     real(real12) :: delta
     real(real12) :: bias
     !! DO THE WEIGHTS NEED TO BE DIFFERENT PER INPUT CHANNEL?
     !! IF SO, 3 DIMENSIONS. IF NOT, 2 DIMENSIONS
     real(real12), allocatable, dimension(:,:) :: weight, weight_incr
     !real(real12), allocatable, dimension(:,:,:) :: output
  end type convolution_type


  private

  public :: clip_type
  public :: network_type
  public :: convolution_type

end module custom_types
