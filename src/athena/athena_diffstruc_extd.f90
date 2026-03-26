module athena__diffstruc_extd
  !! Module for extended differential structure types for Athena
  use coreutils, only: real32
  use diffstruc, only: array_type
  use athena__misc_types, only: facets_type
  implicit none


  private

  public :: array_ptr_type
  public :: add_layers, concat_layers
  public :: add_bias
  public :: full_relu, full_tanh, full_softmax
  public :: piecewise, softmax, swish
  public :: huber
  public :: avgpool1d, avgpool2d, avgpool3d
  public :: maxpool1d, maxpool2d, maxpool3d
  public :: pad1d, pad2d, pad3d
  public :: merge_over_channels
  public :: batchnorm_array_type, batchnorm, batchnorm_inference
  public :: conv1d, conv2d, conv2d_relu, conv3d
  public :: kipf_propagate, kipf_update
  public :: duvenaud_propagate, duvenaud_update
  public :: gno_kernel_eval, gno_aggregate
  public :: lno_encode, lno_decode, elem_scale
  public :: ono_encode, ono_decode


  type, extends(array_type) :: batchnorm_array_type
     real(real32), dimension(:), allocatable :: mean
     real(real32), dimension(:), allocatable :: variance
     real(real32) :: epsilon
  end type batchnorm_array_type


!-------------------------------------------------------------------------------
! Array container types
!-------------------------------------------------------------------------------
  type :: array_ptr_type
     type(array_type), pointer :: array(:,:)
  end type array_ptr_type

  ! Operator interfaces
  !-----------------------------------------------------------------------------
  interface add_layers
     module function add_array_ptr(a, idx1, idx2) result(c)
       type(array_ptr_type), dimension(:), intent(in) :: a
       integer, intent(in) :: idx1, idx2
       type(array_type), pointer :: c
     end function add_array_ptr
  end interface

  interface concat_layers
     module function concat_array_ptr(a, idx1, idx2, dim) result(c)
       type(array_ptr_type), dimension(:), intent(in) :: a
       integer, intent(in) :: idx1, idx2, dim
       type(array_type), pointer :: c
     end function concat_array_ptr
  end interface
!-------------------------------------------------------------------------------


!-------------------------------------------------------------------------------
! Activation functions and other operations
!-------------------------------------------------------------------------------
  interface
     module function add_bias(input, bias, dim, dim_act_on_shape) result(output)
       class(array_type), intent(in), target :: input
       class(array_type), intent(in), target :: bias
       integer, intent(in) :: dim
       logical, intent(in), optional :: dim_act_on_shape
       type(array_type), pointer :: output
     end function add_bias

     module function full_relu(input, kernel, bias) result(output)
       type(array_type), intent(in), target :: input
       type(array_type), intent(in), target :: kernel
       type(array_type), intent(in), target :: bias
       type(array_type), pointer :: output
     end function full_relu

     module function full_tanh(input, kernel, bias) result(output)
       type(array_type), intent(in), target :: input
       type(array_type), intent(in), target :: kernel
       type(array_type), intent(in), target :: bias
       type(array_type), pointer :: output
     end function full_tanh

     module function full_softmax(input, kernel, bias) result(output)
       type(array_type), intent(in), target :: input
       type(array_type), intent(in), target :: kernel
       type(array_type), intent(in), target :: bias
       type(array_type), pointer :: output
     end function full_softmax
  end interface

  interface piecewise
     module function piecewise_array(input, gradient, limit) result( output )
       class(array_type), intent(in), target :: input
       real(real32), intent(in) :: gradient
       real(real32), intent(in) :: limit
       type(array_type), pointer :: output
     end function piecewise_array
  end interface

  interface softmax
     module function softmax_array(input, dim) result(output)
       class(array_type), intent(in), target :: input
       integer, intent(in) :: dim
       type(array_type), pointer :: output
     end function softmax_array
  end interface

  interface swish
     module function swish_array(input, beta) result(output)
       class(array_type), intent(in), target :: input
       real(real32), intent(in) :: beta
       type(array_type), pointer :: output
     end function swish_array
  end interface
!-------------------------------------------------------------------------------


!-------------------------------------------------------------------------------
! Loss functions
!-------------------------------------------------------------------------------
  interface huber
     module function huber_array(delta, gamma) result( output )
       class(array_type), intent(in), target :: delta
       real(real32), intent(in) :: gamma
       type(array_type), pointer :: output
     end function huber_array
  end interface
!-------------------------------------------------------------------------------


!-------------------------------------------------------------------------------
! Layer operations
!-------------------------------------------------------------------------------
  interface
     module function avgpool1d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, intent(in) :: pool_size
       integer, intent(in) :: stride
       type(array_type), pointer :: output
     end function avgpool1d

     module function avgpool2d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, dimension(2), intent(in) :: pool_size
       integer, dimension(2), intent(in) :: stride
       type(array_type), pointer :: output
     end function avgpool2d

     module function avgpool3d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, dimension(3), intent(in) :: pool_size
       integer, dimension(3), intent(in) :: stride
       type(array_type), pointer :: output
     end function avgpool3d
  end interface

  interface
     module function maxpool1d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, intent(in) :: pool_size
       integer, intent(in) :: stride
       type(array_type), pointer :: output
     end function maxpool1d

     module function maxpool2d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, dimension(2), intent(in) :: pool_size
       integer, dimension(2), intent(in) :: stride
       type(array_type), pointer :: output
     end function maxpool2d

     module function maxpool3d(input, pool_size, stride) result(output)
       type(array_type), intent(in), target :: input
       integer, dimension(3), intent(in) :: pool_size
       integer, dimension(3), intent(in) :: stride
       type(array_type), pointer :: output
     end function maxpool3d
  end interface

  interface
     module function pad1d(input, facets, pad_size, imethod) result(output)
       type(array_type), intent(in), target :: input
       type(facets_type), intent(in) :: facets
       integer, intent(in) :: pad_size
       integer, intent(in) :: imethod
       type(array_type), pointer :: output
     end function pad1d

     module function pad2d(input, facets, pad_size, imethod) result(output)
       type(array_type), intent(in), target :: input
       type(facets_type), dimension(2), intent(in) :: facets
       integer, dimension(2), intent(in) :: pad_size
       integer, intent(in) :: imethod
       type(array_type), pointer :: output
     end function pad2d

     module function pad3d(input, facets, pad_size, imethod) result(output)
       type(array_type), intent(in), target :: input
       type(facets_type), dimension(3), intent(in) :: facets
       integer, dimension(3), intent(in) :: pad_size
       integer, intent(in) :: imethod
       type(array_type), pointer :: output
     end function pad3d
  end interface

  interface merge_over_channels
     module function merge_scalar_over_channels(tsource, fsource, mask) result(output)
       class(array_type), intent(in), target :: tsource
       real(real32), intent(in) :: fsource
       logical, dimension(:,:), intent(in) :: mask
       type(array_type), pointer :: output
     end function merge_scalar_over_channels
  end interface

  interface
     module function batchnorm( &
          input, params, momentum, mean, variance, epsilon &
     ) result( output )
       class(array_type), intent(in), target :: input
       class(array_type), intent(in), target :: params
       real(real32), intent(in) :: momentum
       real(real32), dimension(:), intent(in) :: mean
       real(real32), dimension(:), intent(in) :: variance
       real(real32), intent(in) :: epsilon
       type(batchnorm_array_type), pointer :: output
     end function batchnorm

     module function batchnorm_inference( &
          input, params, mean, variance, epsilon &
     ) result( output )
       class(array_type), intent(in), target :: input
       class(array_type), intent(in), target :: params
       real(real32), dimension(:), intent(in) :: mean
       real(real32), dimension(:), intent(in) :: variance
       real(real32), intent(in) :: epsilon
       type(batchnorm_array_type), pointer :: output
     end function batchnorm_inference
  end interface

  interface
     module function conv1d(input, kernel, stride, dilation) result(output)
       type(array_type), intent(in), target :: input
       type(array_type), intent(in), target :: kernel
       integer, intent(in) :: stride
       integer, intent(in) :: dilation
       type(array_type), pointer :: output
     end function conv1d

     module function conv2d(input, kernel, stride, dilation) result(output)
       type(array_type), intent(in), target :: input
       type(array_type), intent(in), target :: kernel
       integer, dimension(2), intent(in) :: stride
       integer, dimension(2), intent(in) :: dilation
       type(array_type), pointer :: output
     end function conv2d

     module function conv2d_relu(input, kernel, bias, stride, dilation) result(output)
       type(array_type), intent(in), target :: input
       type(array_type), intent(in), target :: kernel
       type(array_type), intent(in), target :: bias
       integer, dimension(2), intent(in) :: stride
       integer, dimension(2), intent(in) :: dilation
       type(array_type), pointer :: output
     end function conv2d_relu

     module function conv3d(input, kernel, stride, dilation) result(output)
       type(array_type), intent(in), target :: input
       type(array_type), intent(in), target :: kernel
       integer, dimension(3), intent(in) :: stride
       integer, dimension(3), intent(in) :: dilation
       type(array_type), pointer :: output
     end function conv3d
  end interface

  interface
     module function kipf_propagate(vertex_features, adj_ia, adj_ja) result(c)
       !! Propagate values from one autodiff array to another
       class(array_type), intent(in), target :: vertex_features
       integer, dimension(:), intent(in) :: adj_ia
       integer, dimension(:,:), intent(in) :: adj_ja
       type(array_type), pointer :: c
     end function kipf_propagate

     module function kipf_update(a, weight, adj_ia) result(c)
       !! Update the message passing layer
       class(array_type), intent(in), target :: a
       class(array_type), intent(in), target :: weight
       integer, dimension(:), intent(in) :: adj_ia
       type(array_type), pointer :: c
     end function kipf_update
  end interface

  interface
     module function duvenaud_propagate( &
          vertex_features, edge_features, adj_ia, adj_ja &
     ) result(c)
       !! Duvenaud message passing function
       class(array_type), intent(in), target :: vertex_features
       class(array_type), intent(in), target :: edge_features
       integer, dimension(:), intent(in) :: adj_ia
       integer, dimension(:,:), intent(in) :: adj_ja
       type(array_type), pointer :: c
     end function duvenaud_propagate

     module function duvenaud_update( &
          a, weight, adj_ia, min_degree, max_degree &
     ) result(c)
       !! Duvenaud update function
       class(array_type), intent(in), target :: a
       class(array_type), intent(in), target :: weight
       integer, dimension(:), intent(in) :: adj_ia
       integer, intent(in) :: min_degree, max_degree
       type(array_type), pointer :: c
     end function duvenaud_update
  end interface

  interface
     module function gno_kernel_eval( &
          coords, kernel_params, adj_ia, adj_ja, &
          coord_dim, kernel_hidden, F_in, F_out &
     ) result(c)
       !! Evaluate GNO kernel MLP on every edge
       class(array_type), intent(in), target :: coords
       class(array_type), intent(in), target :: kernel_params
       integer, dimension(:), intent(in) :: adj_ia
       integer, dimension(:,:), intent(in) :: adj_ja
       integer, intent(in) :: coord_dim, kernel_hidden, F_in, F_out
       type(array_type), pointer :: c
     end function gno_kernel_eval

     module function gno_aggregate( &
          features, edge_kernels, adj_ia, adj_ja, F_in, F_out &
     ) result(c)
       !! Aggregate neighbour messages using per-edge kernels
       class(array_type), intent(in), target :: features
       class(array_type), intent(in), target :: edge_kernels
       integer, dimension(:), intent(in) :: adj_ia
       integer, dimension(:,:), intent(in) :: adj_ja
       integer, intent(in) :: F_in, F_out
       type(array_type), pointer :: c
     end function gno_aggregate
  end interface

  interface
     module function lno_encode( &
          input, poles, num_inputs, num_modes &
     ) result(c)
       !! Encode input via Laplace basis: E(mu) @ u
       class(array_type), intent(in), target :: input
       class(array_type), intent(in), target :: poles
       integer, intent(in) :: num_inputs, num_modes
       type(array_type), pointer :: c
     end function lno_encode

     module function lno_decode( &
          spectral, poles, num_outputs, num_modes &
     ) result(c)
       !! Decode via Laplace basis: D(mu) @ spectral
       class(array_type), intent(in), target :: spectral
       class(array_type), intent(in), target :: poles
       integer, intent(in) :: num_outputs, num_modes
       type(array_type), pointer :: c
     end function lno_decode
  end interface

  interface
     module function elem_scale(input, scale) result(c)
       !! Element-wise multiply: out[i,s] = input[i,s] * scale[i,1]
       !! Correctly handles non-sample-dependent scale vectors.
       class(array_type), intent(in), target :: input
       class(array_type), intent(in), target :: scale
       type(array_type), pointer :: c
     end function elem_scale
  end interface

  interface
     module function ono_encode( &
          input, basis_weights, num_inputs, num_basis &
     ) result(c)
       !! Encode via orthogonal basis: Q(B)^T @ u
       class(array_type), intent(in), target :: input
       class(array_type), intent(in), target :: basis_weights
       integer, intent(in) :: num_inputs, num_basis
       type(array_type), pointer :: c
     end function ono_encode

     module function ono_decode( &
          mixed, basis_weights, num_inputs, num_basis &
     ) result(c)
       !! Decode via orthogonal basis: Q(B) @ mixed
       class(array_type), intent(in), target :: mixed
       class(array_type), intent(in), target :: basis_weights
       integer, intent(in) :: num_inputs, num_basis
       type(array_type), pointer :: c
     end function ono_decode
  end interface
!-------------------------------------------------------------------------------

end module athena__diffstruc_extd
