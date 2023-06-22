module nn_types
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

  private

  public :: clip_type

  public :: network_type

end module nn_types
