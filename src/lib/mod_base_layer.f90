module athena__base_layer
  !! Module containing the abstract base layer type
  !!
  !! This module contains the abstract base layer type, from which all other
  !! layers are derived. The module also contains the abstract derived types
  !! for the following layer types:
  !! - padding
  !! - pooling
  !! - dropout
  !! - learnable
  !! - convolutional
  !! - batch normalisation
  !!
  !! The following procedures are based on code from the neural-fortran library
  !! https://github.com/modern-fortran/neural-fortran/blob/main/src/nf/nf_layer.f90
  use coreutils, only: real32
  use athena__clipper, only: clip_type
  use athena__misc_types, only: activation_type, initialiser_type, facets_type, &
       onnx_attribute_type, onnx_node_type, onnx_initialiser_type
  use diffstruc, only: array_type
  use athena__diffstruc_extd, only: array_ptr_type
  use graphstruc, only: graph_type
  implicit none

  private

  public :: base_layer_type
  public :: pad_layer_type
  public :: pool_layer_type
  public :: drop_layer_type
  public :: learnable_layer_type
  public :: conv_layer_type
  public :: batch_layer_type
  public :: merge_layer_type

!-------------------------------------------------------------------------------
! layer abstract type
!-------------------------------------------------------------------------------
  type, abstract :: base_layer_type
     !! Type for base layer, from which all other layers are derived
     integer :: id
     !! Unique identifier
     integer :: batch_size = 0
     !! Batch size
     integer :: input_rank = 0
     !! Rank of input data
     integer :: output_rank = 0
     !! Rank of output data
     logical :: inference = .false.
     !! Inference mode
     logical :: use_graph_input = .false.
     !! Use graph input
     logical :: use_graph_output = .false.
     !! Use graph output
     character(:), allocatable :: name
     !! Layer name
     character(4) :: type = 'base'
     !! Layer type
     character(20) :: subtype = repeat(" ",20)
     type(graph_type), allocatable, dimension(:) :: graph
     !! Graph structure of input data
     logical :: consistent_sample_shape = .true. !! ONLY FALSE FOR GRAPHS
     !! Boolean whether the layer has a consistent sample shape
     class(array_type), allocatable, dimension(:,:) :: output
     !! Output
     integer, allocatable, dimension(:) :: input_shape
     !! Input shape
     integer, allocatable, dimension(:) :: output_shape
     !! Output shape
   contains
     procedure, pass(this) :: set_rank => set_rank_base
     !! Set the input and output ranks of the layer
     procedure, pass(this) :: set_shape => set_shape_base
     !! Set the input shape of the layer
     procedure, pass(this) :: get_num_params => get_num_params_base
     !! Get the number of parameters in the layer
     procedure, pass(this) :: print => print_base
     !! Print the layer to a file with additional information
     procedure, pass(this) :: print_to_unit => print_to_unit_base
     !! Print the layer to a unit
     procedure, pass(this) :: get_attributes => get_attributes_base
     !! Get the attributes of the layer (for ONNX export)
     procedure, pass(this) :: extract_output => extract_output_base
     !! Extract the output of the layer as a standard real array
     procedure(initialise), deferred, pass(this) :: init
     !! Initialise the layer
     procedure(set_batch_size), deferred, pass(this) :: set_batch_size
     !! Set the batch size of the layer

     procedure, pass(this) :: forward => forward_base
     !! Forward pass of layer
     procedure, pass(this) :: forward_eval => forward_eval_base
     !! Forward pass of layer and return output for evaluation

     procedure, pass(this) :: nullify_graph => nullify_graph_base
     !! Nullify the forward pass data of the layer to free memory


     !! Forward pass of layer using derived array_type
     procedure(read_layer), deferred, pass(this) :: read
     !! Read layer from file
     procedure, pass(this) :: build_from_onnx => build_from_onnx_base
     !! Build layer from ONNX node and initialiser
     procedure, pass(this) :: set_ptrs
     !! Set pointers to layer data
     procedure, pass(this), private :: set_ptrs_hyperparams
     !! Set pointers to hyperparameters
     procedure, pass(this) :: set_graph => set_graph_base
     !! Set the graph structure of the input data !! this is adjacency and edge weighting
  end type base_layer_type

  interface
     module subroutine print_base(this, file, unit, print_header_footer)
       !! Print the layer to a file with additional information
       class(base_layer_type), intent(in) :: this
       !! Instance of the layer
       character(*), optional, intent(in) :: file
       !! File name
       integer, optional, intent(in) :: unit
       !! Unit number
       logical, optional, intent(in) :: print_header_footer
       !! Boolean whether to print header and footer
     end subroutine print_base

     module subroutine print_to_unit_base(this, unit)
       !! Print the layer to a file
       class(base_layer_type), intent(in) :: this
       !! Instance of the layer
       integer, intent(in) :: unit
       !! File unit
     end subroutine print_to_unit_base

     module function get_attributes_base(this) result(attributes)
       !! Get the attributes of the layer (for ONNX export)
       class(base_layer_type), intent(in) :: this
       !! Instance of the layer
       type(onnx_attribute_type), allocatable, dimension(:) :: attributes
       !! Attributes of the layer
     end function get_attributes_base

     module subroutine set_rank_base(this, input_rank, output_rank)
       !! Set the input and output ranks of the layer
       class(base_layer_type), intent(inout) :: this
       !! Instance of the layer
       integer, intent(in) :: input_rank
       !! Input rank
       integer, intent(in) :: output_rank
       !! Output rank
     end subroutine set_rank_base

     module subroutine set_shape_base(this, input_shape)
       !! Set the input shape of the layer
       class(base_layer_type), intent(inout) :: this
       !! Instance of the layer
       integer, dimension(:), intent(in) :: input_shape
       !! Input shape
     end subroutine set_shape_base

     module subroutine extract_output_base(this, output)
       !! Extract the output of the layer as a standard real array
       class(base_layer_type), intent(in) :: this
       !! Instance of the layer
       real(real32), dimension(..), allocatable, intent(out) :: output
       !! Output values
     end subroutine extract_output_base

     module subroutine set_ptrs(this)
       !! Set pointers to layer data
       class(base_layer_type), intent(inout), target :: this
       !! Instance of the layer
     end subroutine set_ptrs

     module subroutine set_ptrs_hyperparams(this)
       !! Set pointers to hyperparameters
       class(base_layer_type), intent(inout), target :: this
       !! Instance of the layer
     end subroutine set_ptrs_hyperparams
  end interface


  interface
     module subroutine initialise(this, input_shape, batch_size, verbose)
       !! Initialise the layer
       class(base_layer_type), intent(inout) :: this
       !! Instance of the layer
       integer, dimension(:), intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine initialise

     module subroutine set_batch_size(this, batch_size, verbose)
       !! Set the batch size of the layer
       class(base_layer_type), intent(inout), target :: this
       !! Instance of the layer
       integer, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine set_batch_size
  end interface

  interface
     pure module function get_num_params(this) result(num_params)
       !! Get number of parameters in layer
       class(base_layer_type), intent(in) :: this
       !! Instance of the layer
       integer :: num_params
       !! Number of parameters
     end function get_num_params
  end interface

  interface
     module subroutine forward_base(this, input)
       !! Forward pass of layer
       class(base_layer_type), intent(inout) :: this
       !! Instance of the layer
       class(array_type), dimension(:,:), intent(in) :: input
       !! Input data
     end subroutine forward_base

     module function forward_eval_base(this, input) result(output)
       !! Forward pass of layer and return output for evaluation
       class(base_layer_type), intent(inout), target :: this
       !! Instance of the layer
       class(array_type), dimension(:,:), intent(in) :: input
       !! Input data
       type(array_type), pointer :: output(:,:)
       !! Output data
     end function forward_eval_base

     module subroutine set_graph_base(this, graph)
       !! Set the graph structure of the input data
       class(base_layer_type), intent(inout) :: this
       !! Instance of the layer
       type(graph_type), dimension(:), intent(in) :: graph
       !! Graph structure of input data
     end subroutine set_graph_base

     module subroutine nullify_graph_base(this)
       !! Nullify the forward pass data of the layer to free memory
       class(base_layer_type), intent(inout) :: this
       !! Instance of the layer
     end subroutine nullify_graph_base
  end interface

  interface
     module subroutine read_layer(this, unit, verbose)
       !! Read layer from file
       class(base_layer_type), intent(inout) :: this
       !! Instance of the layer
       integer, intent(in) :: unit
       !! File unit
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine read_layer

     module subroutine build_from_onnx_base(this, node, initialisers, verbose)
       !! Build layer from ONNX node
       class(base_layer_type), intent(inout) :: this
       !! Instance of the layer
       type(onnx_node_type), intent(in) :: node
       !! ONNX node
       type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
       !! ONNX initialisers
       integer, intent(in) :: verbose
       !! Verbosity level
     end subroutine build_from_onnx_base
  end interface


  type, abstract, extends(base_layer_type) :: pad_layer_type
     !! Type for padding layers
     integer :: num_channels
     !! Number of channels
     integer :: imethod = 0
     !! Method for padding
     integer, allocatable, dimension(:) :: pad
     !! Padding size
     character(len=20) :: method = 'valid'
     !! Padding method
     integer, allocatable, dimension(:,:) :: orig_bound, dest_bound
     !! Original and destination bounds
     type(facets_type), dimension(:), allocatable :: facets
     !! Facets of the layer
   contains
     procedure, pass(this) :: init => init_pad
     !! Initialise the layer
     procedure, pass(this) :: print_to_unit => print_to_unit_pad
     !! Print layer to unit
  end type pad_layer_type

  interface
     module subroutine print_to_unit_pad(this, unit)
       !! Print layer to unit
       class(pad_layer_type), intent(in) :: this
       !! Instance of the layer
       integer, intent(in) :: unit
       !! File unit
     end subroutine print_to_unit_pad

     module subroutine init_pad(this, input_shape, batch_size, verbose)
       class(pad_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine init_pad
  end interface


  type, abstract, extends(base_layer_type) :: pool_layer_type
     !! Type for pooling layers
     integer, allocatable, dimension(:) :: pool, strd
     !! Pooling and stride sizes
     integer :: num_channels
     !! Number of channels
   contains
     procedure, pass(this) :: init => init_pool
     !! Initialise the layer
     procedure, pass(this) :: print_to_unit => print_to_unit_pool
     !! Print layer to unit
     procedure, pass(this) :: get_attributes => get_attributes_pool
     !! Get the attributes of the layer (for ONNX export)
  end type pool_layer_type

  interface
     module subroutine print_to_unit_pool(this, unit)
       !! Print layer to unit
       class(pool_layer_type), intent(in) :: this
       !! Instance of the layer
       integer, intent(in) :: unit
       !! File unit
     end subroutine print_to_unit_pool

     module subroutine init_pool(this, input_shape, batch_size, verbose)
       class(pool_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine init_pool

     module function get_attributes_pool(this) result(attributes)
       !! Get the attributes of the layer (for ONNX export)
       class(pool_layer_type), intent(in) :: this
       !! Instance of the layer
       type(onnx_attribute_type), allocatable, dimension(:) :: attributes
       !! Attributes of the layer
     end function get_attributes_pool
  end interface


  type, abstract, extends(base_layer_type) :: drop_layer_type
     !! Type for dropout layers
     real(real32) :: rate = 0.1_real32
     !! Dropout rate, rate = 1 - keep_prob -- typical = 0.05-0.25
   contains
     procedure(generate_mask), deferred, pass(this) :: generate_mask
     !! Generate dropout mask
  end type drop_layer_type

  abstract interface
     subroutine generate_mask(this)
       !! Generate dropout mask
       import :: drop_layer_type
       class(drop_layer_type), intent(inout) :: this
       !! Instance of the layer
     end subroutine generate_mask
  end interface


  type, abstract, extends(base_layer_type) :: merge_layer_type
     !! Type for merge layers (i.e. add, multiply, concatenate)
     integer :: merge_mode = 1
     !! Integer code for fundamental merge method
     !! 1 = pointwise
     !! 2 = concatenate
     !! 3 = reduction
     !! 4 = parametric (NOT IMPLEMENTED)
     character(len=20) :: method
     !! Merge method
     integer :: num_input_layers = 0
     !! Number of input layers
     integer, allocatable, dimension(:) :: input_layer_ids
     !! IDs of input layers
   contains
     procedure(combine_merge), deferred, pass(this) :: combine
     !! Merge two layers (forward)
     procedure(calc_input_shape), deferred, pass(this) :: calc_input_shape
     !! Calculate input shape based on shapes of input layers
  end type merge_layer_type

  interface
     module subroutine combine_merge(this, input_list)
       !! Combine two layers (forward)
       class(merge_layer_type), intent(inout) :: this
       !! Instance of the layer
       type(array_ptr_type), dimension(:), intent(in) :: input_list
       !! Input values
     end subroutine combine_merge

     module function calc_input_shape(this, input_shapes) result(input_shape)
       !! Calculate input shape based on shapes of input layers
       class(merge_layer_type), intent(in) :: this
       !! Instance of the layer
       integer, dimension(:,:), intent(in) :: input_shapes
       !! Input shapes
       integer, allocatable, dimension(:) :: input_shape
       !! Calculated input shape
     end function calc_input_shape
  end interface

  type, abstract, extends(base_layer_type) :: learnable_layer_type
     !! Type for layers with learnable parameters
     integer :: num_params = 0
     !! Number of learnable parameters
     logical :: has_bias = .false.
     !! Layer has bias
     integer, allocatable, dimension(:,:) :: weight_shape
     !! Shape of weights
     integer, allocatable, dimension(:) :: bias_shape
     !! Shape of biases
     real(real32), allocatable, dimension(:) :: params
     type(array_type), allocatable, dimension(:) :: params_array
     !! Learnable parameters
     real(real32), allocatable, dimension(:,:) :: dp, db
     !! Gradients of parameters and biases
     character(len=14) :: kernel_initialiser='', bias_initialiser=''
     class(initialiser_type), allocatable :: kernel_init, bias_init
     !! Initialisers for kernel and bias
     class(activation_type), allocatable :: transfer
     !! Activation function
   contains
     procedure, pass(this) :: get_params => get_params
     !! Get learnable parameters of layer
     procedure, pass(this) :: set_params => set_params
     !! Set learnable parameters of layer
     procedure, pass(this) :: get_gradients => get_gradients
     !! Get parameter gradients of layer
     procedure, pass(this) :: set_gradients => set_gradients
     !! Set learnable parameters of layer

     procedure, pass(this) :: reduce => reduce_learnable
     !! Merge another learnable layer into this one
     procedure :: add_t_t => add_learnable
     !! Add two layers
     generic :: operator(+) => add_t_t
     !! Operator overloading for addition
  end type learnable_layer_type

  interface
     module subroutine reduce_learnable(this, input)
       !! Merge another learnable layer into this one
       class(learnable_layer_type), intent(inout) :: this
       !! Instance of the layer
       class(learnable_layer_type), intent(in) :: input
       !! Other layer to merge
     end subroutine reduce_learnable

     module function add_learnable(a, b) result(output)
       !! Add two layers
       class(learnable_layer_type), intent(in) :: a, b
       !! Instances of the layers
       class(learnable_layer_type), allocatable :: output
       !! Output layer
     end function add_learnable
  end interface

  interface
     pure module function get_params(this) result(params)
       !! Get learnable parameters of layer
       class(learnable_layer_type), intent(in) :: this
       !! Instance of the layer
       real(real32), dimension(this%num_params) :: params
       !! Learnable parameters
     end function get_params

     module subroutine set_params(this, params)
       !! Set learnable parameters of layer
       class(learnable_layer_type), intent(inout) :: this
       !! Instance of the layer
       real(real32), dimension(this%num_params), intent(in) :: params
       !! Learnable parameters
     end subroutine set_params

     pure module function get_gradients(this, clip_method) result(gradients)
       !! Get parameter gradients of layer
       class(learnable_layer_type), intent(in) :: this
       !! Instance of the layer
       type(clip_type), optional, intent(in) :: clip_method
       !! Clip method
       real(real32), dimension(this%num_params) :: gradients
       !! Parameter gradients
     end function get_gradients

     module subroutine set_gradients(this, gradients)
       !! Set learnable parameters of layer
       class(learnable_layer_type), intent(inout) :: this
       !! Instance of the layer
       real(real32), dimension(..), intent(in) :: gradients
       !! Learnable parameters
     end subroutine set_gradients
  end interface

  type, abstract, extends(learnable_layer_type) :: conv_layer_type
     integer :: num_channels
     !! Number of channels
     integer :: num_filters
     !! Number of filters
     integer, allocatable, dimension(:) :: knl, stp, pad, dil
     !! Kernel, stride, and padding sizes
     integer, allocatable, dimension(:) :: hlf, cen
     !! Half and centre sizes
     real(real32), pointer :: bias(:) => null()
     !! Bias pointer
     class(pad_layer_type), allocatable :: pad_layer
     !! Optional preprocess padding layer
     class(array_type), allocatable :: di_padded
     !! Padded input gradients
   contains
     procedure, pass(this) :: get_num_params => get_num_params_conv
     !! Get the number of parameters in the layer
     procedure, pass(this) :: init => init_conv
     !! Initialise the layer
     procedure, pass(this) :: get_attributes => get_attributes_conv
     !! Get the attributes of the layer (for ONNX export)
  end type conv_layer_type


  type, abstract, extends(learnable_layer_type) :: batch_layer_type
     !! Type for batch normalisation layers
     integer :: num_channels
     !! Number of channels
     real(real32) :: norm
     !! Normalisation factor
     real(real32) :: momentum = 0.99_real32
     !! Momentum factor
     !! NOTE: if momentum = 0, mean and variance batch-dependent values
     !! NOTE: if momentum > 0, mean and variance are running averages
     real(real32) :: epsilon = 0.001_real32
     !! Epsilon factor
     real(real32) :: gamma_init_mean = 1._real32, gamma_init_std = 0.01_real32
     !! Initialisation parameters for gamma
     real(real32) :: beta_init_mean  = 0._real32, beta_init_std  = 0.01_real32
     !! Initialisation parameters for beta
     character(len=14) :: moving_mean_initialiser='', &
          moving_variance_initialiser=''
     !! Initialisers for moving mean and variance
     real(real32), allocatable, dimension(:) :: mean, variance
     !! Mean and variance (not learnable)
   contains
     procedure, pass(this) :: get_num_params => get_num_params_batch
     !! Get the number of parameters in the layer
     procedure, pass(this) :: init => init_batch
     !! Initialise the layer
     procedure, pass(this) :: get_attributes => get_attributes_batch
     !! Get the attributes of the layer (for ONNX export)
  end type batch_layer_type

  interface
     pure module function get_num_params_base(this) result(num_params)
       class(base_layer_type), intent(in) :: this
       integer :: num_params
     end function get_num_params_base

     pure module function get_num_params_batch(this) result(num_params)
       class(batch_layer_type), intent(in) :: this
       integer :: num_params
     end function get_num_params_batch

     pure module function get_num_params_conv(this) result(num_params)
       class(conv_layer_type), intent(in) :: this
       integer :: num_params
     end function get_num_params_conv

     module subroutine init_conv(this, input_shape, batch_size, verbose)
       class(conv_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine init_conv

     module function get_attributes_conv(this) result(attributes)
       class(conv_layer_type), intent(in) :: this
       type(onnx_attribute_type), allocatable, dimension(:) :: attributes
     end function get_attributes_conv

     module subroutine init_batch(this, input_shape, batch_size, verbose)
       class(batch_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine init_batch

     module function get_attributes_batch(this) result(attributes)
       class(batch_layer_type), intent(in) :: this
       !! Instance of the layer
       type(onnx_attribute_type), allocatable, dimension(:) :: attributes
       !! Attributes of the layer
     end function get_attributes_batch
  end interface


end module athena__base_layer
