!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! definition of the abstract base layer type, from which all other layers ...
!!! ... are derived
!!! module includes the following public abstract derived types:
!!! - base_layer_type      - abstract type for all layers
!!! - pool_layer_type      - abstract type for spatial pooling layers
!!! - drop_layer_type      - abstract type for dropout layers
!!! - learnable_layer_type - abstract type for layers with learnable parameters
!!! - conv_layer_type      - abstract type for spatial convolutional layers
!!! - batch_layer_type     - abstract type for batch normalisation layers
!!!##################
!!! base_layer_type includes the following procedures:
!!! - set_shape            - set the input shape of the layer
!!! - get_num_params       - get the number of parameters in the layer
!!! - print                - print the layer to a file
!!! - get_output           - get the output of the layer
!!! - init                 - initialise the layer
!!! - set_batch_size       - set the batch size of the layer
!!! - forward              - forward pass of layer
!!! - backward             - backward pass of layer
!!!##################
!!! learnable_layer_type includes the following unique procedures:
!!! - layer_reduction      - reduce the layer to a single value
!!! - layer_merge          - merge the layer with another layer
!!! - get_params           - get the learnable parameters of the layer
!!! - set_params           - set the learnable parameters of the layer
!!! - get_gradients        - get the gradients of the layer
!!! - set_gradients        - set the gradients of the layer
!!!#############################################################################
module base_layer
  use constants, only: real32
  use clipper, only: clip_type
  use custom_types, only: activation_type, array_type
  implicit none

  private

  public :: base_layer_type
  public :: pool_layer_type
  public :: drop_layer_type
  public :: learnable_layer_type
  public :: conv_layer_type
  public :: batch_layer_type

!!!-----------------------------------------------------------------------------
!!! layer abstract type
!!!-----------------------------------------------------------------------------
  type, abstract :: base_layer_type !! give it parameterised values?
     integer :: id
     integer :: batch_size = 0
     integer :: input_rank = 0
     logical :: inference = .false.
     character(:), allocatable :: name
     character(4) :: type = 'base'
     class(array_type), allocatable :: output, di
     integer, allocatable, dimension(:) :: input_shape
  contains
     procedure, pass(this) :: set_shape => set_shape_base
     procedure, pass(this) :: get_num_params => get_num_params_base
     procedure, pass(this) :: print => print_base
     procedure, pass(this) :: get_output => get_output_base
     procedure(initialise), deferred, pass(this) :: init
     procedure(set_batch_size), deferred, pass(this) :: set_batch_size
     procedure(forward), deferred, pass(this) :: forward
     procedure(backward), deferred, pass(this) :: backward
     procedure(read_layer), deferred, pass(this) :: read
     !! NO NEED FOR DEFERRED PRODECURES
     !! instead, make this a generic type that just has a set of interfaces for (module) procedures that call 1D, 3D, and 4D forms
     !! Use subroutines because output data is trickier for function tricker to handle
     !! Use a general train subroutine that is called by the main model, which internally goes through forward and backward passes
     !! Input sizes have to be 1D, 3D, or 4D (any 2D data is simply 3D with num_channels=1)
     !! Output sizes defined by user
     !! For every forward, just pass in the whole previous layer container
     !! ... reverse for backward
     !! In each layer container, you know what size you are expecting for the input, so just take that based on a select type (or of a previous?)
  end type base_layer_type

  interface
     !!-------------------------------------------------------------------------
     !! print layer to file (do nothing for a base layer)
     !!-------------------------------------------------------------------------
     !! this = (T, in) base_layer_type
     !! file = (I, in) file name
     module subroutine print_base(this, file)
       class(base_layer_type), intent(in) :: this
       character(*), intent(in) :: file
     end subroutine print_base

     !!-------------------------------------------------------------------------
     !! setup input layer shape
     !!-------------------------------------------------------------------------
     !! this        = (T, inout) base_layer_type
     !! input_shape = (I, in) input shape
     module subroutine set_shape_base(this, input_shape)
       class(base_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
     end subroutine set_shape_base

     !!-------------------------------------------------------------------------
     !! get outputs from layer
     !!-------------------------------------------------------------------------
     !! this   = (T, in) layer_type
     !! output = (R, out) output values
     pure module subroutine get_output_base(this, output)
       class(base_layer_type), intent(in) :: this
       real(real32), allocatable, dimension(..), intent(out) :: output
     end subroutine get_output_base
  end interface


  abstract interface
     !!-------------------------------------------------------------------------
     !! initialise layer
     !!-------------------------------------------------------------------------
     !! this        = (T, inout) base_layer_type
     !! input_shape = (I, in) input shape
     !! batch_size  = (I, in) batch size
     !! verbose     = (I, in) verbosity level
     module subroutine initialise(this, input_shape, batch_size, verbose)
       class(base_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine initialise

     !!-------------------------------------------------------------------------
     !! set batch size
     !!-------------------------------------------------------------------------
     !! this       = (T, inout) base_layer_type
     !! batch_size = (I, in) batch size
     !! verbose    = (I, in) verbosity level
     module subroutine set_batch_size(this, batch_size, verbose)
       class(base_layer_type), intent(inout) :: this
       integer, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine set_batch_size
  end interface

  abstract interface
     !!-------------------------------------------------------------------------
     !! get number of parameters in layer
     !!-------------------------------------------------------------------------
     !! this = (T, in) layer_type
     pure module function get_num_params(this) result(num_params)
       class(base_layer_type), intent(in) :: this
       integer :: num_params
     end function get_num_params

     !!-------------------------------------------------------------------------
     !! forward pass of layer
     !!-------------------------------------------------------------------------
     !! this  = (T, in) layer_type
     !! input = (R, in) input data
     pure module subroutine forward(this, input)
       class(base_layer_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
     end subroutine forward

     !!-------------------------------------------------------------------------
     !! backward pass of layer
     !!-------------------------------------------------------------------------
     !! this     = (T, in) layer_type
     !! input    = (R, in) input data
     !! gradient = (R, in) gradient data
     pure module subroutine backward(this, input, gradient)
       class(base_layer_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
       real(real32), dimension(..), intent(in) :: gradient
     end subroutine backward
  end interface

  abstract interface
     !!-------------------------------------------------------------------------
     !! read layer from file
     !!-------------------------------------------------------------------------
     !! this  = (T, inout) base_layer_type
     !! unit  = (I, in) file unit
     !! verbose = (I, in) verbosity level
     module subroutine read_layer(this, unit, verbose)
       class(base_layer_type), intent(inout) :: this
       integer, intent(in) :: unit
       integer, optional, intent(in) :: verbose
     end subroutine read_layer
  end interface


!!!-----------------------------------------------------------------------------
!!! pooling derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: pool_layer_type
     !! strd = stride (step)
     !! pool = pool
     integer, allocatable, dimension(:) :: pool, strd
     integer :: num_channels
  end type pool_layer_type


!!!-----------------------------------------------------------------------------
!!! dropout derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: drop_layer_type
     !! rate = 1 - keep_prob   -- typical = 0.05-0.25
     real(real32) :: rate = 0.1_real32
   contains
     procedure(generate_mask), deferred, pass(this) :: generate_mask
  end type drop_layer_type

  abstract interface
     !!--------------------------------------------------------------------------
     !! get number of parameters in layer
     !!--------------------------------------------------------------------------
     !! this = (T, in) drop_layer_type
     subroutine generate_mask(this)
       import :: drop_layer_type
       class(drop_layer_type), intent(inout) :: this
     end subroutine generate_mask
  end interface


!!!-----------------------------------------------------------------------------
!!! learnable derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: learnable_layer_type
     integer :: num_params = 0
     character(len=14) :: kernel_initialiser='', bias_initialiser=''
     class(activation_type), allocatable :: transfer
   contains
     procedure(layer_reduction), deferred, pass(this) :: reduce
     procedure(layer_merge), deferred, pass(this) :: merge
     procedure(get_params), deferred, pass(this) :: get_params
     procedure(set_params), deferred, pass(this) :: set_params
     procedure(get_gradients), deferred, pass(this) :: get_gradients
     procedure(set_gradients), deferred, pass(this) :: set_gradients
  end type learnable_layer_type

  abstract interface
     !!-------------------------------------------------------------------------
     !! reduce two layers to a single value
     !!-------------------------------------------------------------------------
     !! this = (T, io) layer_type
     !! rhs  = (T, in) layer_type
     subroutine layer_reduction(this, rhs)
       import :: learnable_layer_type
       class(learnable_layer_type), intent(inout) :: this
       class(learnable_layer_type), intent(in) :: rhs
     end subroutine layer_reduction

     !!-------------------------------------------------------------------------
     !! merge two layers
     !!-------------------------------------------------------------------------
     !! this  = (T, io) layer_type
     !! input = (T, in) layer_type
     subroutine layer_merge(this, input)
       import :: learnable_layer_type
       class(learnable_layer_type), intent(inout) :: this
       class(learnable_layer_type), intent(in) :: input
     end subroutine layer_merge

     !!-------------------------------------------------------------------------
     !! get learnable parameters of layer
     !!-------------------------------------------------------------------------
     !! this  = (T, in) layer_type
     !! param = (R, out) learnable parameters
     pure function get_params(this) result(params)
       import :: learnable_layer_type, real32
       class(learnable_layer_type), intent(in) :: this
       real(real32), dimension(this%num_params) :: params
     end function get_params

     !!-------------------------------------------------------------------------
     !! set learnable parameters of layer
     !!-------------------------------------------------------------------------
     !! this  = (T, io) layer_type
     !! param = (R, in) learnable parameters
     subroutine set_params(this, params)
       import :: learnable_layer_type, real32
       class(learnable_layer_type), intent(inout) :: this
       real(real32), dimension(this%num_params), intent(in) :: params
     end subroutine set_params

     !!-------------------------------------------------------------------------
     !! get parameter gradients of layer
     !!-------------------------------------------------------------------------
     !! this        = (T, in) layer_type
     !! clip_method = (T, in) clip method
     !! gradients   = (R, out) parameter gradients
     pure function get_gradients(this, clip_method) result(gradients)
       import :: learnable_layer_type, real32, clip_type
       class(learnable_layer_type), intent(in) :: this
       type(clip_type), optional, intent(in) :: clip_method
       real(real32), dimension(this%num_params) :: gradients
     end function get_gradients

     !!-------------------------------------------------------------------------
     !! set learnable parameters of layer
     !!-------------------------------------------------------------------------
     !! this      = (T, io) layer_type
     !! gradients = (R, in) learnable parameters
     subroutine set_gradients(this, gradients)
       import :: learnable_layer_type, real32
       class(learnable_layer_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: gradients
     end subroutine set_gradients
  end interface

!!!-----------------------------------------------------------------------------
!!! convolution extended derived type
!!!-----------------------------------------------------------------------------
    type, abstract, extends(learnable_layer_type) :: conv_layer_type
       !! knl = kernel
       !! stp = stride (step)
       !! hlf = half
       !! pad = pad
       !! cen = centre
       !! output%shape = dimension (height, width, depth)
       logical :: calc_input_gradients = .true.
       integer :: num_channels
       integer :: num_filters
       integer, allocatable, dimension(:) :: knl, stp, hlf, pad, cen
       real(real32), allocatable, dimension(:) :: bias
       real(real32), allocatable, dimension(:,:) :: db  ! bias gradient
     contains
       procedure, pass(this) :: get_num_params => get_num_params_conv
    end type conv_layer_type


!!!-----------------------------------------------------------------------------
!!! batch extended derived type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(learnable_layer_type) :: batch_layer_type
     !! gamma = scale factor (learnable)
     !! beta = shift factor (learnable)
     !! dg = gradient of gamma
     !! db = gradient of beta
     !! mean = mean of each feature (not learnable)
     !! variance = variance of each feature (not learnable)
     !! NOTE: if momentum = 0, mean and variance batch-dependent values
     !! NOTE: if momentum > 0, mean and variance are running averages
     !! NED: NEED TO KEEP TRACK OF EXPONENTIAL MOVING AVERAGE (EMA)
     !!   ... FOR INFERENCE
     integer :: num_channels
     real(real32) :: norm
     real(real32) :: momentum = 0.99_real32
     real(real32) :: epsilon = 0.001_real32
     real(real32) :: gamma_init_mean = 1._real32, gamma_init_std = 0.01_real32
     real(real32) :: beta_init_mean  = 0._real32, beta_init_std  = 0.01_real32
     character(len=14) :: moving_mean_initialiser='', &
          moving_variance_initialiser=''
     real(real32), allocatable, dimension(:) :: mean, variance !! not learnable
     real(real32), allocatable, dimension(:) :: gamma, beta !! learnable
     real(real32), allocatable, dimension(:) :: dg, db !! learnable
   contains
     procedure, pass(this) :: get_num_params => get_num_params_batch
     procedure, pass(this) :: get_params => get_params_batch
     procedure, pass(this) :: set_params => set_params_batch
     procedure, pass(this) :: get_gradients => get_gradients_batch
     procedure, pass(this) :: set_gradients => set_gradients_batch
  end type batch_layer_type



  interface
     pure module function get_num_params_base(this) result(num_params)
       class(base_layer_type), intent(in) :: this
       integer :: num_params
     end function get_num_params_base
     pure module function get_gradients_batch(this, clip_method) &
          result(gradients)
       class(batch_layer_type), intent(in) :: this
       type(clip_type), optional, intent(in) :: clip_method
       real(real32), dimension(this%num_params) :: gradients
     end function get_gradients_batch
     pure module function get_num_params_batch(this) result(num_params)
       class(batch_layer_type), intent(in) :: this
       integer :: num_params
     end function get_num_params_batch
     module subroutine set_gradients_batch(this, gradients)
       class(batch_layer_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: gradients
     end subroutine set_gradients_batch
     pure module function get_params_batch(this) result(params)
       class(batch_layer_type), intent(in) :: this
       real(real32), dimension(this%num_params) :: params
     end function get_params_batch
     module subroutine set_params_batch(this, params)
       class(batch_layer_type), intent(inout) :: this
       real(real32), dimension(this%num_params), intent(in) :: params
     end subroutine set_params_batch
     pure module function get_num_params_conv(this) result(num_params)
       class(conv_layer_type), intent(in) :: this
       integer :: num_params
     end function get_num_params_conv
  end interface


end module base_layer
!!!#############################################################################
