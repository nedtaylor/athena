module athena__network
  !! Module containing the network class used to define a neural network
  !!
  !! This module contains the types and interfaces for the network class used
  !! to define a neural network.
  !! The network class is used to define a neural network with overloaded
  !! procedures for training, testing, predicting, and updating the network.
  !! The network class is also used to define the network structure and
  !! compile the network with an optimiser, loss function, and accuracy
  !! function.
  use coreutils, only: real32
  use graphstruc, only: graph_type
  use athena__metrics, only: metric_dict_type
  use athena__optimiser, only: base_optimiser_type
  use athena__loss, only: base_loss_type
  use athena__accuracy, only: comp_acc_func => compute_accuracy_function
  use athena__base_layer, only: base_layer_type
  use diffstruc, only: array_type
  use athena__misc_types, only: &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use athena__container_layer, only: container_layer_type
  use athena__diffstruc_extd, only: array_ptr_type
  implicit none


  private

  public :: network_type


  type :: network_type
     !! Type for defining a neural network with overloaded procedures
     character(len=:), allocatable :: name
     !! Name of the network
     real(real32) :: accuracy_val, loss_val
     !! Accuracy and loss of the network
     integer :: batch_size = 0
     !! Batch size
     integer :: epoch = 0
     !! Epoch number
     integer :: num_layers = 0
     !! Number of layers
     integer :: num_outputs = 0
     !! Number of outputs
     integer :: num_params = 0
     !! Number of parameters
     logical :: use_graph_input = .false.
     !! Boolean flag for graph input
     logical :: use_graph_output = .false.
     !! Boolean flag for graph output
     class(base_optimiser_type), allocatable :: optimiser
     !! Optimiser for the network
     class(base_loss_type), allocatable :: loss
     !! Loss method for the network
     type(metric_dict_type), dimension(2) :: metrics
     !! Metrics for the network
     type(container_layer_type), allocatable, dimension(:) :: model
     !! Model layers
     character(len=:), allocatable :: loss_method, accuracy_method
     !! Loss and accuracy method names
     procedure(comp_acc_func), nopass, pointer :: get_accuracy => null()
     !! Pointer to accuracy function
     integer, dimension(:), allocatable :: vertex_order
     !! Order of vertices
     integer, dimension(:), allocatable :: root_vertices, leaf_vertices
     !! Root and output vertices
     type(graph_type) :: auto_graph
     !! Graph structure for the network

     ! Pre-computed forward pass navigation (populated during compile)
     integer, dimension(:), allocatable :: fwd_layer_id
     !! Layer ID for each vertex in forward order
     integer, dimension(:), allocatable :: fwd_num_inputs
     !! Number of input layers for each vertex in forward order
     integer, dimension(:), allocatable :: fwd_parent_id
     !! Parent layer ID for single-input vertices
     integer, dimension(:), allocatable :: fwd_layer_type
     !! Layer type: 0=input, 1=merge, 2=default

     ! Pre-computed parameter segment layout (populated during compile)
     integer :: param_num_segments = 0
     !! Number of parameter segments
     integer, dimension(:), allocatable :: param_seg_layer
     !! Layer index for each parameter segment
     integer, dimension(:), allocatable :: param_seg_pidx
     !! Param index within that layer for each segment
     integer, dimension(:), allocatable :: param_seg_start
     !! Start offset in flat parameter array
     integer, dimension(:), allocatable :: param_seg_end
     !! End offset in flat parameter array

     type(array_type), dimension(:,:), allocatable :: input_array
     !! Input array for the network
     type(graph_type), dimension(:,:), allocatable :: input_graph
     !! Input graph for the network
     type(array_type), dimension(:,:), allocatable :: expected_array
     !! Expected output array for the network
   contains
     procedure, pass(this) :: print
     !! Print the network to file
     procedure, pass(this) :: print_summary
     !! Print a summary of the network architecture
     procedure, pass(this) :: read
     !! Read the network from a file
     procedure, pass(this), private :: read_network_settings
     !! Read network settings from a file
     procedure, pass(this), private :: read_optimiser_settings
     !! Read optimiser settings from a file
     procedure, pass(this) :: build_from_onnx
     !! Build network from ONNX nodes and initialisers
     procedure, pass(this) :: add
     !! Add a layer to the network
     procedure, pass(this) :: reset
     !! Reset the network
     procedure, pass(this) :: compile
     !! Compile the network
     procedure, pass(this) :: set_batch_size
     !! Set batch size
     procedure, pass(this) :: set_metrics
     !! Set network metrics
     procedure, pass(this) :: set_loss
     !! Set network loss method
     procedure, pass(this) :: set_accuracy
     !! Set network accuracy method
     procedure, pass(this) :: reset_state
     !! Reset hidden state of recurrent layers

     procedure, pass(this) :: save_input => save_input_to_network
     !! Convert and save polymorphic input to array or graph
     procedure, pass(this) :: save_output => save_output_to_network
     !! Convert and save polymorphic output to array or graph

     procedure, pass(this) :: layer_from_id
     !! Get the layer of the network from its ID

     procedure, pass(this) :: train
     !! Train the network
     procedure, pass(this) :: test
     !! Test the network

     procedure, pass(this) :: predict_real
     !! Return predicted results from supplied inputs using the trained network
     procedure, pass(this) :: predict_array_from_real
     !! Return predicted results as array from supplied inputs using the trained network
     procedure, pass(this) :: predict_graph1d, predict_graph2d
     !! Return predicted results from supplied inputs using the trained network (graph input)
     procedure, pass(this) :: predict_array
     !! Predict array type output for a generic input
     procedure, pass(this) :: predict_generic
     !! Predict generic type output for a generic input
     generic :: predict => &
          predict_real, predict_graph1d, predict_graph2d, &
          predict_array, predict_array_from_real
     !! Predict function for different input types


     procedure, pass(this), private :: dfs
     !! Depth first search
     procedure, pass(this), private :: build_vertex_order
     !! Generate vertex order
     procedure, pass(this), private :: build_root_vertices
     !! Calculate root vertices
     procedure, pass(this), private :: build_leaf_vertices
     !! Calculate output vertices

     procedure, pass(this) :: reduce => network_reduction
     !! Reduce two networks down to one (i.e. add two networks - parallel)
     procedure, pass(this) :: copy => network_copy
     !! Copy a network

     procedure, pass(this) :: get_num_params
     !! Get number of learnable parameters in the network
     procedure, pass(this) :: get_params
     !! Get learnable parameters
     procedure, pass(this) :: set_params
     !! Set learnable parameters
     procedure, pass(this) :: get_gradients
     !! Get gradients of learnable parameters
     procedure, pass(this) :: set_gradients
     !! Set learnable parameter gradients
     procedure, pass(this) :: reset_gradients
     !! Reset learnable parameter gradients
     procedure, pass(this) :: get_output
     !! Get the output of the network
     procedure, pass(this) :: get_output_shape
     !! Get the output shape of the network
     procedure, pass(this) :: extract_output => extract_output_real
     !! Extract network output as real array (only works for single output layer models)

     procedure, pass(this) :: forward => forward_generic2d
     !! Forward pass for generic 2D input
     procedure, pass(this) :: forward_eval
     !! Forward pass and return pointer to output (only works for single output layer models)
     procedure, pass(this) :: accuracy_eval
     !! Get the accuracy for the output
     procedure, pass(this) :: loss_eval
     !! Get the loss for the output
     procedure, pass(this) :: update
     !! Update the learnable parameters of the network based on gradients

     procedure, pass(this) :: nullify_graph
     !! Nullify graph data in the network to free memory

     procedure, pass(this) :: post_epoch_hook
     !! Called after each training epoch; override in derived types for custom
     !! per-epoch callbacks (e.g. logging to Weights & Biases).
  end type network_type

  interface network_type
     !! Interface for setting up the network (network initialisation)
     module function network_setup( &
          layers, &
          optimiser, loss_method, accuracy_method, &
          metrics, batch_size &
     ) result(network)
       !! Set up the network
       type(container_layer_type), dimension(:), intent(in) :: layers
       !! Layers
       class(base_optimiser_type), optional, intent(in) :: optimiser
       !! Optimiser
       class(*), optional, intent(in) :: loss_method
       !! Loss method
       character(*), optional, intent(in) :: accuracy_method
       !! Accuracy method
       class(*), dimension(..), optional, intent(in) :: metrics
       !! Metrics
       integer, optional, intent(in) :: batch_size
       !! Batch size
       type(network_type) :: network
       !! Instance of the network
     end function network_setup
  end interface network_type

  interface
     !! Interface for printing the network to file
     module subroutine print(this, file)
       !! Print the network to file
       class(network_type), intent(in) :: this
       !! Instance of the network
       character(*), intent(in) :: file
       !! File name
     end subroutine print

     !! Interface for printing a summary of the network
     module subroutine print_summary(this)
       !! Print a summary of the network architecture
       class(network_type), intent(in) :: this
       !! Instance of the network
     end subroutine print_summary

     !! Interface for reading the network from a file
     module subroutine read(this, file)
       !! Read the network from a file
       class(network_type), intent(inout) :: this
       !! Instance of the network
       character(*), intent(in) :: file
       !! File name
     end subroutine read

     !! Interface for reading network settings from a file
     module subroutine read_network_settings(this, unit)
       !! Read network settings from a file
       class(network_type), intent(inout) :: this
       !! Instance of the network
       integer, intent(in) :: unit
       !! Unit number for input
     end subroutine read_network_settings

     !! Interface for reading optimiser settings from a file
     module subroutine read_optimiser_settings(this, unit)
       !! Read optimiser settings from a file
       class(network_type), intent(inout) :: this
       !! Instance of the network
       integer, intent(in) :: unit
       !! Unit number for input
     end subroutine read_optimiser_settings

     !! Interface for building network from ONNX nodes and initialisers
     module subroutine build_from_onnx( &
          this, nodes, initialisers, inputs, value_info, verbose &
     )
       !! Build network from ONNX nodes and initialisers
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(onnx_node_type), dimension(:), intent(in) :: nodes
       !! Array of ONNX nodes
       type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
       !! Array of ONNX initialisers
       type(onnx_tensor_type), dimension(:), intent(in) :: inputs
       !! Array of ONNX input tensors
       type(onnx_tensor_type), dimension(:), intent(in) :: value_info
       !! Array of ONNX value info tensors
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine build_from_onnx

     !! Interface for adding a layer to the network
     module subroutine add(this, layer, input_list, output_list, operator)
       !! Add a layer to the network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       class(base_layer_type), intent(in) :: layer
       !! Layer to add
       integer, dimension(:), intent(in), optional :: input_list, output_list
       !! Input and output list
       class(*), optional, intent(in) :: operator
       !! Operator
     end subroutine add

     !! Interface for resetting the network
     module subroutine reset(this)
       !! Reset the network
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine reset

     !! Interface for compiling the network
     module subroutine compile( &
          this, optimiser, loss_method, accuracy_method, &
          metrics, batch_size, verbose &
     )
       !! Compile the network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       class(base_optimiser_type), optional, intent(in) :: optimiser
       !! Optimiser
       class(*), optional, intent(in) :: loss_method
       !! Loss method
       character(*), optional, intent(in) :: accuracy_method
       !! Accuracy method
       class(*), dimension(..), optional, intent(in) :: metrics
       !! Metrics
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine compile

     !! Interface for setting batch size
     module subroutine set_batch_size(this, batch_size)
       !! Set batch size
       class(network_type), intent(inout) :: this
       !! Instance of the network
       integer, intent(in) :: batch_size
       !! Batch size
     end subroutine set_batch_size

     !! Interface for setting network metrics
     module subroutine set_metrics(this, metrics)
       !! Set network metrics
       class(network_type), intent(inout) :: this
       !! Instance of the network
       class(*), dimension(..), intent(in) :: metrics
       !! Metrics
     end subroutine set_metrics

     !! Interface for setting network loss method
     module subroutine set_loss(this, loss_method, verbose)
       !! Set network loss method
       class(network_type), intent(inout) :: this
       !! Instance of the network
       class(*), intent(in) :: loss_method
       !! Loss method
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine set_loss

     !! Interface for setting network accuracy method
     module subroutine set_accuracy(this, accuracy_method, verbose)
       !! Set network accuracy method
       class(network_type), intent(inout) :: this
       !! Instance of the network
       character(*), intent(in) :: accuracy_method
       !! Accuracy method
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine set_accuracy

     !! Interface for resetting state of recurrent layers
     module subroutine reset_state(this)
       !! Reset hidden state of recurrent layers
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine reset_state

     !! Interface for saving input to network
     module function save_input_to_network( this, input ) result(num_samples)
       !! Convert and save polymorphic input to array or graph
       class(network_type), intent(inout) :: this
       !! Instance of network
       class(*), dimension(..), intent(in) :: input
       !! Input
       integer :: num_samples
       !! Number of samples
     end function save_input_to_network

     !! Interface for saving output to network
     module subroutine save_output_to_network( this, output )
       !! Convert and save polymorphic output to array or graph
       class(network_type), intent(inout) :: this
       !! Instance of network
       class(*), dimension(:,:), intent(in) :: output
       !! Output
     end subroutine save_output_to_network

     module function layer_from_id(this, id) result(layer)
       !! Get the layer of the network from its ID
       class(network_type), intent(in), target :: this
       !! Instance of the network
       integer, intent(in) :: id
       !! Layer ID
       class(base_layer_type), pointer :: layer
       !! Layer pointer
     end function layer_from_id


     !! Interface for training the network
     module subroutine train( &
          this, input, output, num_epochs, batch_size, &
          plateau_threshold, shuffle_batches, batch_print_step, verbose, &
          print_precision, scientific_print &
     )
       !! Train the network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       class(*), dimension(..), intent(in) :: input
       !! Input data
       class(*), dimension(:,:), intent(in) :: output
       !! Expected output data (data labels)
       integer, intent(in) :: num_epochs
       !! Number of epochs to train for
       integer, optional, intent(in) :: batch_size
       !! Batch size (DEPRECATED)
       real(real32), optional, intent(in) :: plateau_threshold
       !! Threshold for checking learning plateau
       logical, optional, intent(in) :: shuffle_batches
       !! Shuffle batch order
       integer, optional, intent(in) :: batch_print_step
       !! Print step for batch
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       integer, optional, intent(in) :: print_precision
       !! Number of decimal places to print for training metrics
       logical, optional, intent(in) :: scientific_print
       !! Whether to print training metrics in scientific notation
     end subroutine train

     !! Interface for testing the network
     module subroutine test(this, input, output, verbose)
       !! Test the network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       class(*), dimension(..), intent(in) :: input
       !! Input data
       class(*), dimension(:,:), intent(in) :: output
       !! Expected output data (data labels)
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine test

     !! Interface for returning predicted results from supplied inputs
     !! using the trained network
     module function predict_real(this, input, verbose) result(output)
       !! Get predicted results from supplied inputs using the trained network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       real(real32), dimension(..), intent(in) :: input
       !! Input data
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       real(real32), dimension(:,:), allocatable :: output
       !! Predicted output data
     end function predict_real

     module function predict_array_from_real( &
          this, input, output_as_array, verbose &
     ) result(output)
       !! Get predicted results as array from supplied inputs using the trained network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       class(*), dimension(..), intent(in) :: input
       !! Input data
       logical, intent(in) :: output_as_array
       !! Whether to output as array
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(array_type), dimension(:,:), allocatable :: output
       !! Predicted output data as array
     end function predict_array_from_real

     !! Interface for returning predicted results from supplied inputs
     !! using the trained network (graph input)
     module function predict_graph1d(this, input, verbose) result(output)
       !! Get predicted results from supplied inputs using the trained network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(graph_type), dimension(:), intent(in) :: input
       !! Input data
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(graph_type), dimension(size(this%leaf_vertices),size(input)) :: &
            output
       !! Predicted output data
     end function predict_graph1d
     module function predict_graph2d(this, input, verbose) result(output)
       !! Get predicted results from supplied inputs using the trained network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(graph_type), dimension(:,:), intent(in) :: input
       !! Input data
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(graph_type), dimension(size(this%leaf_vertices),size(input, 2)) :: &
            output
       !! Predicted output data
     end function predict_graph2d

     module function predict_array( this, input, verbose ) &
          result(output)
       !! Predict the output for a generic input
       class(network_type), intent(inout) :: this
       !! Instance of network
       class(array_type), dimension(..), intent(in) :: input
       !! Input graph
       integer, intent(in), optional :: verbose
       !! Verbosity level
       type(array_type), dimension(:,:), allocatable :: output
     end function predict_array

     module function predict_generic( this, input, verbose, output_as_graph ) &
          result(output)
       !! Predict the output for a generic input
       class(network_type), intent(inout) :: this
       !! Instance of network
       class(*), dimension(:,:), intent(in) :: input
       !! Input graph
       integer, intent(in), optional :: verbose
       !! Verbosity level
       logical, intent(in), optional :: output_as_graph
       !! Boolean whether to output as graph
       class(*), dimension(:,:), allocatable :: output
     end function predict_generic

     !! Interface for updating the learnable parameters of the network
     !! based on gradients
     module subroutine update(this)
       !! Update the learnable parameters of the network based on gradients
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine update

     !! Interface for generating vertex order
     module subroutine build_vertex_order(this)
       !! Generate vertex order
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine build_vertex_order

     !! Interface for depth first search
     recursive module subroutine dfs( &
          this, vertex_index, visited, order, order_index &
     )
       !! Depth first search
       class(network_type), intent(in) :: this
       !! Instance of the network
       integer, intent(in) :: vertex_index
       !! Vertex index
       logical, dimension(this%auto_graph%num_vertices), intent(inout) :: &
            visited
       !! Visited vertices
       integer, dimension(this%auto_graph%num_vertices), intent(inout) :: order
       !! Order of vertices
       integer, intent(inout) :: order_index
       !! Index of order
     end subroutine dfs

     !! Interface for calculating root vertices
     module subroutine build_root_vertices(this)
       !! Calculate root vertices
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine build_root_vertices

     !! Interface for calculating output vertices
     module subroutine build_leaf_vertices(this)
       !! Calculate output vertices
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine build_leaf_vertices

     !! Interface for reducing two networks down to one
     !! (i.e. add two networks - parallel)
     module subroutine network_reduction(this, source)
       !! Reduce two networks down to one (i.e. add two networks - parallel)
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(network_type), intent(in) :: source
       !! Source network
     end subroutine network_reduction

     !! Interface for copying a network
     module subroutine network_copy(this, source)
       !! Copy a network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(network_type), intent(in), target :: source
       !! Source network
     end subroutine network_copy

     !! Interface for getting number of learnable parameters in the network
     pure module function get_num_params(this) result(num_params)
       !! Get number of learnable parameters in the network
       class(network_type), intent(in) :: this
       !! Instance of the network
       integer :: num_params
       !! Number of parameters
     end function get_num_params

     !! Interface for getting learnable parameters
     pure module function get_params(this) result(params)
       !! Get learnable parameters
       class(network_type), intent(in) :: this
       !! Instance of the network
       real(real32), dimension(this%num_params) :: params
       !! Learnable parameters
     end function get_params

     !! Interface for setting learnable parameters
     module subroutine set_params(this, params)
       !! Set learnable parameters
       class(network_type), intent(inout) :: this
       !! Instance of the network
       real(real32), dimension(this%num_params), intent(in) :: params
       !! Learnable parameters
     end subroutine set_params

     !! Interface for getting gradients of learnable parameters
     pure module function get_gradients(this) result(gradients)
       !! Get gradients of learnable parameters
       class(network_type), intent(in) :: this
       !! Instance of the network
       real(real32), dimension(this%num_params) :: gradients
       !! Gradients
     end function get_gradients

     !! Interface for setting learnable parameter gradients
     module subroutine set_gradients(this, gradients)
       !! Set learnable parameter gradients
       class(network_type), intent(inout) :: this
       !! Instance of the network
       real(real32), dimension(..), intent(in) :: gradients
       !! Gradients
     end subroutine set_gradients

     !! Interface for resetting learnable parameter gradients
     module subroutine reset_gradients(this)
       !! Reset learnable parameter gradients
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine reset_gradients

     module function get_output(this) result(output)
       class(network_type), intent(in) :: this
       !! Instance of the network
       type(array_type), dimension(:,:), allocatable :: output
       !! Output
     end function get_output

     module function get_output_shape(this) result(output_shape)
       class(network_type), intent(in) :: this
       !! Instance of the network
       integer, dimension(2) :: output_shape
       !! Output shape
     end function get_output_shape

     module subroutine extract_output_real(this, output)
       class(network_type), intent(in) :: this
       !! Instance of network
       real(real32), dimension(..), allocatable, intent(out) :: output
       !! Output
     end subroutine extract_output_real

     module function accuracy_eval(this, output, start_index, end_index) &
          result(accuracy)
       !! Get the accuracy for the output
       class(network_type), intent(in) :: this
       !! Instance of network
       class(*), dimension(:,:), intent(in) :: output
       !! Output
       integer, intent(in) :: start_index, end_index
       !! Start and end batch indices
       real(real32) :: accuracy
       !! Accuracy value
     end function accuracy_eval

     module function loss_eval(this, start_index, end_index) result(loss)
       !! Get the loss for the output
       ! Arguments
       class(network_type), intent(inout), target :: this
       !! Instance of network
       integer, intent(in) :: start_index, end_index
       !! Start and end batch indices

       type(array_type), pointer :: loss
     end function loss_eval

     !! Interface for forward pass
     module subroutine forward_generic2d(this, input)
       !! Forward pass for generic 2D input
       class(network_type), intent(inout), target :: this
       !! Instance of the network
       class(*), dimension(:,:), intent(in) :: input
       !! Input data
     end subroutine forward_generic2d

     module function forward_eval(this, input) result(output)
       !! Forward pass evaluation
       class(network_type), intent(inout), target :: this
       !! Instance of the network
       class(*), dimension(:,:), intent(in) :: input
       !! Input data
       type(array_type), pointer :: output(:,:)
       !! Output data
     end function forward_eval

     module function forward_eval_multi(this, input) result(output)
       !! Forward pass evaluation for multiple outputs
       class(network_type), intent(inout), target :: this
       !! Instance of the network
       class(*), dimension(:,:), intent(in) :: input
       !! Input data
       type(array_ptr_type), pointer :: output(:)
       !! Output data
     end function forward_eval_multi

     module subroutine nullify_graph(this)
       !! Nullify graph data in the network to free memory
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine nullify_graph

     module subroutine post_epoch_hook(this, epoch, loss, accuracy)
       !! Hook called after each training epoch.
       !! The default implementation is a no-op; override in a derived type to
       !! add custom per-epoch behaviour (e.g. W&B metric logging).
       class(network_type), intent(inout) :: this
       !! Instance of the network
       integer, intent(in) :: epoch
       !! Current epoch number (1-based)
       real(real32), intent(in) :: loss
       !! Mean loss over the epoch
       real(real32), intent(in) :: accuracy
       !! Mean accuracy over the epoch
     end subroutine post_epoch_hook
  end interface

  interface get_sample
#ifdef __flang__
     module function get_sample_flang( &
          input, start_index, end_index, batch_size &
     ) result(sample)
       !! Get a sample from a rank
       implicit none
       ! Arguments
       integer, intent(in) :: start_index, end_index
       !! Start and end indices
       integer, intent(in) :: batch_size
       !! Batch size
       real(real32), dimension(..), intent(in) :: input
       !! Input array
       ! Local variables
       real(real32), allocatable :: sample(:,:)
       !! Sample array
     end function get_sample_flang
#else
     module function get_sample_ptr( &
          input, start_index, end_index, batch_size &
     ) result(sample_ptr)
       !! Get a sample from a rank
       implicit none
       ! Arguments
       integer, intent(in) :: start_index, end_index
       !! Start and end indices
       integer, intent(in) :: batch_size
       !! Batch size
       real(real32), dimension(..), intent(in), target :: input
       !! Input array
       ! Local variables
       real(real32), pointer :: sample_ptr(:,:)
       !! Pointer to sample
     end function get_sample_ptr
#endif
     module function get_sample_array( &
          input, start_index, end_index, batch_size, as_graph&
     ) result(sample)
       !! Get sample for mixed input
       integer, intent(in) :: start_index, end_index
       !! Start and end indices
       integer, intent(in) :: batch_size
       !! Batch size
       class(array_type), dimension(:,:), intent(in) :: input
       !! Input array
       logical, intent(in) :: as_graph
       !! Boolean whether to treat the input as a graph
       type(array_type), dimension(:,:), allocatable :: sample
       !! Sample array
     end function get_sample_array
     module function get_sample_graph1d( &
          input, start_index, end_index, batch_size &
     ) result(sample)
       !! Get sample for graph input
       integer, intent(in) :: start_index, end_index
       !! Start and end indices
       integer, intent(in) :: batch_size
       !! Batch size
       class(graph_type), dimension(:), intent(in) :: input
       !! Input array
       type(graph_type), dimension(1, batch_size) :: sample
       !! Sample array
     end function get_sample_graph1d
     module function get_sample_graph2d( &
          input, start_index, end_index, batch_size &
     ) result(sample)
       !! Get sample for graph input
       integer, intent(in) :: start_index, end_index
       !! Start and end indices
       integer, intent(in) :: batch_size
       !! Batch size
       class(graph_type), dimension(:,:), intent(in) :: input
       !! Input array
       type(graph_type), dimension(size(input,1), batch_size) :: sample
       !! Sample array
     end function get_sample_graph2d
  end interface get_sample

end module athena__network
