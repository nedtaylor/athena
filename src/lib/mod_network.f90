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
  use athena__constants, only: real32
  use graphstruc, only: graph_type
  use athena__metrics, only: metric_dict_type
  use athena__optimiser, only: base_optimiser_type
  use athena__loss, only: &
       comp_loss_func => compute_loss_function, &
       comp_loss_deriv => compute_loss_derivative
  use athena__accuracy, only: comp_acc_func => compute_accuracy_function
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: array_type, array2d_type
  use athena__container_layer, only: container_layer_type
  implicit none


  private

  public :: network_type


  type :: network_type
     !! Type for defining a neural network with overloaded procedures
     real(real32) :: accuracy, loss
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
     type(metric_dict_type), dimension(2) :: metrics
     !! Metrics for the network
     type(container_layer_type), allocatable, dimension(:) :: model
     !! Model layers
     procedure(comp_loss_func), nopass, pointer :: get_loss => null()
     !! Pointer to loss function
     procedure(comp_loss_deriv), nopass, pointer :: get_loss_deriv => null()
     !! Pointer to loss derivative function
     procedure(comp_acc_func), nopass, pointer :: get_accuracy => null()
     !! Pointer to accuracy function
     integer, dimension(:), allocatable :: vertex_order
     !! Order of vertices
     integer, dimension(:), allocatable :: root_vertices, output_vertices
     !! Root and output vertices
     integer, dimension(:,:,:), allocatable :: io_map
     !! Input-output map
     type(graph_type), private :: auto_graph
     !! Graph structure for the network
   contains
     procedure, pass(this) :: print
     !! Print the network to file
     procedure, pass(this) :: read
     !! Read the network from a file
     procedure, pass(this) :: export_onnx
     !! Export the network to ONNX format
     procedure, pass(this) :: write_onnx_initializers
     !! Write ONNX initializers
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
     procedure, pass(this) :: train
     !! Train the network
     procedure, pass(this) :: test
     !! Test the network
     procedure, pass(this) :: predict_1d
     !! Return predicted results from supplied inputs using the trained network
     procedure, pass(this) :: predict_graph
     !! Return predicted results from supplied inputs using the trained network (graph input)
     generic :: predict => predict_1d, predict_graph
     !! Predict function for different input types
     procedure, pass(this) :: update
     !! Update the learnable parameters of the network based on gradients
     procedure, pass(this), private :: generate_vertex_order
     !! Generate vertex order
     procedure, pass(this), private :: dfs
     !! Depth first search
     procedure, pass(this), private :: calculate_root_vertices
     !! Calculate root vertices
     procedure, pass(this), private :: calculate_output_vertices
     !! Calculate output vertices
     procedure, pass(this), private :: calculate_io_map
     !! Calculate input-output map
     procedure, pass(this), private :: get_input_real_autodiff
     !! Get the input of a layer via autodiff
     procedure, pass(this), private :: get_gradient_real_autodiff
     !! Get the gradient of a layer via autodiff
     procedure, pass(this), private :: get_input_graph_autodiff
     !! Get the input of a layer via autodiff (graph input)
     procedure, pass(this), private :: get_gradient_graph_autodiff
     !! Get the gradient of a layer via autodiff (graph input)
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
     procedure, pass(this) :: forward_real
     !! Forward pass for real input
     procedure, pass(this) :: backward_real
     !! Backward pass for real input
     procedure, pass(this) :: forward_derived
     !! Forward pass for derived input
     procedure, pass(this) :: backward_derived
     !! Backward pass for derived input
     procedure, pass(this) :: forward_graph
     !! Forward pass for graph input
     procedure, pass(this) :: backward_graph
     !! Backward pass for graph input
     procedure, pass(this) :: backward_mixed
     generic :: forward => forward_real, forward_derived, forward_graph
     !! Generic for forward propagation
     generic :: backward => backward_real, backward_derived, backward_graph
     !! Generic for backward propagation
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
       character(*), optional, intent(in) :: loss_method, accuracy_method
       !! Loss method and accuracy method
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

     !! Interface for reading the network from a file
     module subroutine read(this, file)
       !! Read the network from a file
       class(network_type), intent(inout) :: this
       !! Instance of the network
       character(*), intent(in) :: file
       !! File name
     end subroutine read

     !! Interface for exporting the network to ONNX format
     module subroutine export_onnx(this, file)
       !! Export the network to ONNX format
       class(network_type), intent(in) :: this
       !! Instance of the network
       character(*), intent(in) :: file
       !! File name
     end subroutine export_onnx

     module subroutine write_onnx_initializers(this, unit, idx, prefix)
       !! Write ONNX initializers
       class(network_type), intent(in) :: this
       !! Instance of the network
       integer, intent(in) :: unit
       !! Unit number for output
       integer, intent(in) :: idx
       !! Index of the layer
       character(*), intent(in) :: prefix
       !! Prefix for the node name (default is 'node_')
     end subroutine write_onnx_initializers

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
       class(base_optimiser_type), intent(in) :: optimiser
       !! Optimiser
       character(*), optional, intent(in) :: loss_method, accuracy_method
       !! Loss method and accuracy method
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
       character(*), intent(in) :: loss_method
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

     !! Interface for training the network
     module subroutine train( &
          this, input, output, num_epochs, batch_size, &
          plateau_threshold, shuffle_batches, batch_print_step, verbose &
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
     module function predict_1d(this, input, verbose) result(output)
       !! Get predicted results from supplied inputs using the trained network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       real(real32), dimension(..), intent(in) :: input
       !! Input data
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       real(real32), dimension(:,:), allocatable :: output
       !! Predicted output data
     end function predict_1d

     !! Interface for returning predicted results from supplied inputs
     !! using the trained network (graph input)
     module function predict_graph(this, input, verbose) result(output)
       !! Get predicted results from supplied inputs using the trained network
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(graph_type), dimension(:,:), intent(in) :: input
       !! Input data
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(graph_type), dimension(size(input,dim=1),size(this%output_vertices)) :: &
            output
       !! Predicted output data
     end function predict_graph

     !! Interface for updating the learnable parameters of the network
     !! based on gradients
     module subroutine update(this)
       !! Update the learnable parameters of the network based on gradients
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine update

     !! Interface for generating vertex order
     module subroutine generate_vertex_order(this)
       !! Generate vertex order
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine generate_vertex_order

     !! Interface for depth first search
     module recursive subroutine dfs( &
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
     module subroutine calculate_root_vertices(this)
       !! Calculate root vertices
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine calculate_root_vertices

     !! Interface for calculating output vertices
     module subroutine calculate_output_vertices(this)
       !! Calculate output vertices
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine calculate_output_vertices

     !! Interface for calculating input-output map
     module subroutine calculate_io_map(this)
       !! Calculate input-output map
       class(network_type), intent(inout) :: this
       !! Instance of the network
     end subroutine calculate_io_map

     !! Interface for getting the input of a layer via autodiff
     pure module subroutine get_input_real_autodiff(this, idx, input)
       !! Get the input of a layer via autodiff
       class(network_type), intent(in) :: this
       !! Instance of the network
       integer, intent(in) :: idx
       !! Index
       real(real32), allocatable, dimension(:,:), intent(out) :: input
       !! Input
     end subroutine get_input_real_autodiff

     !! Interface for getting the input of a layer via autodiff (graph input)
     module subroutine get_input_graph_autodiff(this, idx, input)
       !! Get the input of a layer via autodiff
       class(network_type), intent(in) :: this
       !! Instance of the network
       integer, intent(in) :: idx
       !! Index
       type(array2d_type), dimension(2,this%batch_size), intent(inout) :: input
       !! Input
     end subroutine get_input_graph_autodiff

     !! Interface for getting the gradient of a layer via autodiff
     pure module subroutine get_gradient_real_autodiff(this, idx, gradient)
       !! Get the gradient of a layer via autodiff
       class(network_type), intent(in) :: this
       !! Instance of the network
       integer, intent(in) :: idx
       !! Index
       real(real32), allocatable, dimension(:,:), intent(out) :: gradient
       !! Gradient
     end subroutine get_gradient_real_autodiff

     !! Interface for getting the gradient of a layer via autodiff (graph input)
     pure module subroutine get_gradient_graph_autodiff(this, idx, gradient)
       !! Get the gradient of a layer via autodiff
       class(network_type), intent(in) :: this
       !! Instance of the network
       integer, intent(in) :: idx
       !! Index
       type(array2d_type), &
            dimension(2,this%batch_size), intent(inout) :: gradient
       !! Gradient
     end subroutine get_gradient_graph_autodiff

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

     !! Interface for forward pass
     pure module subroutine forward_real(this, input)
       !! Forward pass for real input
       class(network_type), intent(inout) :: this
       !! Instance of the network
       real(real32), dimension(..), intent(in) :: input
       !! Input data
     end subroutine forward_real
     pure module subroutine forward_derived(this, input)
       !! Forward pass for derived input
       class(network_type), intent(inout) :: this
       !! Instance of the network
       class(array_type), dimension(size(this%root_vertices)), intent(in) :: &
            input
       !! Input data
     end subroutine forward_derived
     module subroutine forward_graph(this, input)
       !! Forward pass for derived input
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(graph_type), dimension(size(this%root_vertices),this%batch_size), &
            intent(in) :: input
       !! Input data
     end subroutine forward_graph

     !! Interface for backward pass
     pure module subroutine backward_real(this, output)
       !! Backward pass
       class(network_type), intent(inout) :: this
       !! Instance of the network
       real(real32), dimension(:,:), intent(in) :: output
       !! Output data
     end subroutine backward_real
     pure module subroutine backward_derived(this, output)
       !! Backward pass
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(array2d_type), dimension(:), intent(in) :: output
       !! Output data
     end subroutine backward_derived
     module subroutine backward_graph(this, output)
       !! Forward pass for derived input
       class(network_type), intent(inout) :: this
       !! Instance of network
       class(graph_type), dimension(size(this%output_vertices),this%batch_size), &
            intent(in) :: output
       !! Output data
     end subroutine backward_graph
     module subroutine backward_mixed(this, output)
       !! Forward pass for derived input
       class(network_type), intent(inout) :: this
       !! Instance of the network
       type(array2d_type), dimension(:,:), intent(in) :: output
       !! Output data
     end subroutine backward_mixed
  end interface

  interface get_sample
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
     module function get_sample_derived( &
          input, start_index, end_index, batch_size &
     ) result(sample)
       !! Get sample for derived input
       integer, intent(in) :: start_index, end_index
       !! Start and end indices
       integer, intent(in) :: batch_size
       !! Batch size
       class(array_type), dimension(:), intent(in), target :: input
       !! Input array
       type(array2d_type), dimension(size(input,1)) :: sample
       !! Sample array
     end function get_sample_derived
     module function get_sample_mixed( &
          input, start_index, end_index, batch_size &
     ) result(sample)
       !! Get sample for mixed input
       integer, intent(in) :: start_index, end_index
       !! Start and end indices
       integer, intent(in) :: batch_size
       !! Batch size
       class(array_type), dimension(:,:), intent(in), target :: input
       !! Input array
       type(array2d_type), dimension(size(input,1), batch_size) :: sample
    !! Sample array
     end function get_sample_mixed
     module function get_sample_graph( &
          input, start_index, end_index, batch_size &
     ) result(sample)
       !! Get sample for graph input
       integer, intent(in) :: start_index, end_index
       !! Start and end indices
       integer, intent(in) :: batch_size
       !! Batch size
       class(graph_type), dimension(:,:), intent(in), target :: input
       !! Input array
       type(graph_type), dimension(size(input,dim=1),batch_size) :: sample
       !! Sample array
     end function get_sample_graph
  end interface get_sample

end module athena__network
