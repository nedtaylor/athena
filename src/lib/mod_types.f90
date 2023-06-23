module custom_types
  use constants, only: real12
  implicit none

!!!------------------------------------------------------------------------
!!! gradient clipping type
!!!------------------------------------------------------------------------
  type clip_type
     logical :: l_min_max, l_norm
     real(real12) :: min, max, norm
  end type clip_type

!!!------------------------------------------------------------------------
!!! neural network neuron type
!!!------------------------------------------------------------------------
  type neuron_type
     real(real12) :: output
     real(real12) :: delta, delta_batch
     real(real12), allocatable, dimension(:) :: weight, weight_incr
  end type neuron_type

!!!------------------------------------------------------------------------
!!! fully connected network layer type
!!!------------------------------------------------------------------------
  type network_type
     type(neuron_type), allocatable, dimension(:) :: neuron
  end type network_type

!!!------------------------------------------------------------------------
!!! convolution layer type
!!!------------------------------------------------------------------------
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

!!!------------------------------------------------------------------------
!!! activation (transfer) function base type
!!!------------------------------------------------------------------------
  type, abstract :: activation_type
     real(real12) :: scale
   contains
     procedure (activation_function), deferred :: activate
     procedure (derivative_function), deferred :: differentiate
  end type activation_type
  
  !! interface for activation function
  !!-----------------------------------------------------------------------
  abstract interface
     function activation_function(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), intent(in) :: val
       real(real12) :: output
     end function activation_function
  end interface

  !! interface for derivative function
  !!-----------------------------------------------------------------------
  abstract interface
     function derivative_function(this, val) result(output)
       import activation_type, real12
       class(activation_type), intent(in) :: this
       real(real12), intent(in) :: val
       real(real12) :: output
     end function derivative_function
  end interface


  private

  public :: clip_type
  public :: network_type
  public :: convolution_type
  public :: activation_type

end module custom_types
