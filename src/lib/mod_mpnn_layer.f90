module athena__mpnn_layer
  !! Module containing the types and interfacees of a message passing layer
  use athena__constants, only: real32
  use graphstruc, only: graph_type
  use athena__base_layer, only: learnable_layer_type
  use athena__clipper, only: clip_type
  implicit none


  private

  public :: mpnn_layer_type, feature_type, method_container_type
  public :: message_phase_type, readout_phase_type
  public :: update_message
  public :: get_output_readout


!-------------------------------------------------------------------------------
! Message passing layer
!-------------------------------------------------------------------------------
  type, extends(learnable_layer_type) :: mpnn_layer_type
     !! Type for message passing layer with overloaded procedures
     !!
     !! This derived type contains the implementation of a message passing
     !! layer. These are useful for graph neural networks and other models
     !! that require message passing.
     !! For graphs, the terms there are two common terms used seemingly
     !! interchangeably in the literature:
     !!   - vertex/node - the individual elements in the graph
     !!   - edge - the connections between the nodes
     !! Here, we use the term vertex to refer to the individual elements
     !! in the graph and edge to refer to the connections between vertices.
     integer :: num_vertex_features
     !! Number of vertex features
     integer :: num_edge_features
     !! Number of edge features
     integer :: num_time_steps
     !! Number of time steps
     integer :: num_outputs
     !! Number of outputs
     type(graph_type), dimension(:), allocatable :: graph
     !! Graph structure, temporary input for forward and backward passes
     !!
     !! The dimension of the array is (batch_size)
     !! The graph structure is a derived type that contains the information
     !! about the graph. This includes the adjacency matrix, the edge list,
     !! and the vertex list.
     class(method_container_type), allocatable :: method
     !! Method container
     !!
     !! This is a derived type that contains the information about the method
     !! used for the message passing neural network. This includes the message
     !! and readout phases.
     !  real(real32), dimension(:,:), allocatable :: output
     !  real(real32), dimension(:,:), allocatable :: di
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_mpnn
     !! Set the hyperparameters for message passing layer
     procedure, pass(this) :: init => init_mpnn
     !! Initialise message passing layer
     procedure, pass(this) :: set_batch_size => set_batch_size_mpnn
     !! Set the batch size for message passing layer
     procedure, pass(this) :: print => print_mpnn
     !! Print the message passing layer
     procedure, pass(this) :: read => read_mpnn
     !! Read the message passing layer

     procedure, pass(this) :: reduce => layer_reduction
     !! Reduce message passing layer
     procedure, pass(this) :: merge => layer_merge
     !! Merge message passing layer
     procedure, pass(this) :: get_num_params => get_num_params_mpnn
     !! Get the number of learnable parameters for message passing layer
     procedure, pass(this) :: get_params => get_params_mpnn
     !! Get the learnable parameters for message passing layer
     procedure, pass(this) :: set_params => set_params_mpnn
     !! Set the learnable parameters for message passing layer
     procedure, pass(this) :: get_gradients => get_gradients_mpnn
     !! Get the gradients for message passing layer
     procedure, pass(this) :: set_gradients => set_gradients_mpnn
     !! Set the gradients for message passing layer

     procedure, pass(this) :: set_graph
     !! Set the graph for message passing layer (i.e. current inputs)
     procedure, pass(this) :: forward => forward_rank
     !! Forward pass for message passing layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward pass for message passing layer
  end type mpnn_layer_type

  ! Interface for setting up the MPNN layer
  !-----------------------------------------------------------------------------
  interface mpnn_layer_type
     !! Interface for setting up the MPNN layer
     module function layer_setup( &
          method, &
          num_features, num_time_steps, num_outputs, batch_size, &
          verbose &
     ) result(layer)
       !! Set up the MPNN layer
       !!! MAKE THESE ASSUMED RANK
       class(method_container_type), intent(in) :: method
       !! Method container
       integer, dimension(2), intent(in) :: num_features
       !! Number of features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(mpnn_layer_type) :: layer
       !! Instance of the message passing layer
     end function layer_setup
  end interface mpnn_layer_type

  ! Interface for handling the message passing layer parameters
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for handling learnable parameters and gradients
     pure module function get_num_params_mpnn(this) result(num_params)
       !! Get the number of learnable parameters for the message passing layer
       class(mpnn_layer_type), intent(in) :: this
       !! Instance of the message passing layer
       integer :: num_params
       !! Number of learnable parameters
     end function get_num_params_mpnn

     pure module function get_params_mpnn(this) result(params)
       !! Get the learnable parameters for the message passing layer
       class(mpnn_layer_type), intent(in) :: this
       !! Instance of the message passing layer
       real(real32), dimension(this%num_params) :: params
       !! Parameters
     end function get_params_mpnn

     pure module subroutine set_params_mpnn(this, params)
       !! Set the learnable parameters for the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       real(real32), dimension(this%num_params), intent(in) :: params
       !! Parameters
     end subroutine set_params_mpnn

     pure module function get_gradients_mpnn(this, clip_method) &
          result(gradients)
       !! Get the gradients for the message passing layer
       class(mpnn_layer_type), intent(in) :: this
       !! Instance of the message passing layer
       type(clip_type), optional, intent(in) :: clip_method
       !! Clip method
       real(real32), dimension(this%num_params) :: gradients
       !! Gradients
     end function get_gradients_mpnn

     pure module subroutine set_gradients_mpnn(this, gradients)
       !! Set the gradients for the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       real(real32), dimension(..), intent(in) :: gradients
       !! Gradients
     end subroutine set_gradients_mpnn
  end interface

  ! Interface for reducing and merging layers
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for reducing and merging layers
     module subroutine layer_reduction(this, rhs)
       !! Reduce the layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       class(learnable_layer_type), intent(in) :: rhs
       !! Instance of the learnable layer (expects a message passing layer)
     end subroutine layer_reduction

     module subroutine layer_merge(this, input)
       !! Merge the layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       class(learnable_layer_type), intent(in) :: input
       !! Instance of the learnable layer (expects a message passing layer)
     end subroutine layer_merge
  end interface

  ! Interface for handling forward and backward passes
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for handling forward and backward passes of the MPNN
     pure module subroutine forward_rank(this, input)
       !! Forward pass for the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       real(real32), dimension(..), intent(in) :: input
       !! Input
     end subroutine forward_rank

     pure module subroutine backward_rank(this, input, gradient)
       !! Backward pass for the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       real(real32), dimension(..), intent(in) :: input
       !! Input
       real(real32), dimension(..), intent(in) :: gradient
       !! Gradient
     end subroutine backward_rank

     pure module subroutine forward_graph(this, graph)
       !! Forward pass for the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
       !! Graph structure
     end subroutine forward_graph

     pure module subroutine backward_graph(this, graph, gradient)
       !! Backward pass for the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
       !! Graph structure
       real(real32), dimension( &
            this%output%shape(1), &
            this%batch_size &
       ), intent(in) :: gradient
       !! Gradient
     end subroutine backward_graph
  end interface

  ! Interface for handling graphs and outputs
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for handling graphs and outputs, and initialising the layer
     module subroutine set_graph(this, graph)
       !! Set the graph for the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
       !! Graph structure
     end subroutine set_graph
     module subroutine print_mpnn(this, file)
       !! Print the message passing layer
       class(mpnn_layer_type), intent(in) :: this
       !! Instance of the message passing layer
       character(*), intent(in) :: file
       !! File to print to
     end subroutine print_mpnn
     module subroutine read_mpnn(this, unit, verbose)
       !! Read the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       integer, intent(in) :: unit
       !! Unit to read from
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine read_mpnn
     module subroutine init_mpnn(this, input_shape, batch_size, verbose)
       !! Initialise the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       integer, dimension(:), intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine init_mpnn
     module subroutine set_batch_size_mpnn(this, batch_size, verbose)
       !! Set the batch size for the message passing layer
       class(mpnn_layer_type), intent(inout), target :: this
       !! Instance of the message passing layer
       integer, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine set_batch_size_mpnn
     module subroutine set_hyperparams_mpnn( &
          this, method, num_features, num_time_steps, num_outputs, verbose &
     )
       !! Set the hyperparameters for the message passing layer
       class(mpnn_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       class(method_container_type), intent(in) :: method
       !! Method container
       integer, dimension(2), intent(in) :: num_features
       !! Number of features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine set_hyperparams_mpnn
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Method container
!-------------------------------------------------------------------------------
  type, abstract :: method_container_type
     !! Type for the method container
     !!
     !! This derived type contains the implementation of the message passing
     !! neural network. This includes the message and readout phases.
     integer :: num_outputs
     !! Number of outputs
     integer :: num_time_steps
     !! Number of time steps
     integer, dimension(2) :: num_features
     !! Number of features
     !! element 1 is vertex features, element 2 is edge features
     integer :: batch_size
     !! Batch size
     class(message_phase_type), dimension(:), allocatable :: message
     !! Message phase
     !! elements are the message phases for each time step
     class(readout_phase_type), allocatable :: readout
     !! Readout phase
   contains
     procedure(init_method), deferred, pass(this) :: init
     !! Initialise the method container
     procedure(set_batch_size_method), deferred, pass(this) :: set_batch_size
     !! Set the batch size for the method container
  end type method_container_type

  ! Interface for initialising the method container
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for initialising the method container
     module subroutine init_method(this, &
          num_vertex_features, num_edge_features, num_time_steps, &
          output_shape, batch_size, verbose &
     )
       class(method_container_type), intent(inout) :: this
       !! Instance of the method container
       integer, intent(in) :: num_vertex_features
       !! Number of vertex features
       integer, intent(in) :: num_edge_features
       !! Number of edge features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, dimension(1), intent(in) :: output_shape
       !! Output shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine init_method
     module subroutine set_batch_size_method(this, batch_size, verbose)
       class(method_container_type), intent(inout) :: this
       !! Instance of the method container
       integer, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine set_batch_size_method
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Feature type
!-------------------------------------------------------------------------------
  type :: feature_type
     !! Type for the feature
     real(real32), dimension(:,:), allocatable :: val
     !! Value of the feature
   contains
     procedure :: add_t_t => feature_add
     !! Add two features
     procedure :: multiply_t_t => feature_multiply
     !! Multiply two features
     generic :: operator(+) => add_t_t
     !! Overload the addition operator
     generic :: operator(*) => multiply_t_t
     !! Overload the multiplication operator
  end type feature_type

  ! Interface for combining features
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for combining layers
     elemental module function feature_add(a, b) result(output)
       !! Add two features
       class(feature_type), intent(in) :: a, b
       !! Features
       type(feature_type) :: output
       !! Output
     end function feature_add

     elemental module function feature_multiply(a, b) result(output)
       !! Multiply two features
       class(feature_type), intent(in) :: a, b
       !! Features
       type(feature_type) :: output
       !! Output
     end function feature_multiply
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Abstract phase type (extended by message and readout phases)
!-------------------------------------------------------------------------------
  type, abstract :: base_phase_type
     !! Type for the base phase
     !!
     !! This derived type contains the basic information for a phase in the
     !! message passing network. The two available phases are the message phase
     !! and the readout phase.
     !!! HAVE LOGICAL THAT STATES WHETHER IT IS LEARNABLE ???
     integer :: num_inputs
     !! Number of inputs
     integer :: num_outputs
     !! Number of outputs
     integer :: batch_size
     !! Batch size
     integer :: num_params
     !! Number of learnable parameters
     type(feature_type), dimension(:), allocatable :: feature
     !! Features
     !! elements are the features for each batch size
     type(feature_type), dimension(:), allocatable :: di
     !! Derivative of the features
   contains
     procedure, pass(this) :: get_num_params => get_phase_num_params
     !! Get the number of learnable parameters for the phase
     procedure, pass(this) :: get_params => get_phase_params
     !! Get the learnable parameters for the phase
     procedure, pass(this) :: set_params => set_phase_params
     !! Set the learnable parameters for the phase
     procedure, pass(this) :: get_gradients => get_phase_gradients
     !! Get the gradients for the phase
     procedure, pass(this) :: set_gradients => set_phase_gradients
     !! Set the gradients for the phase
     procedure, pass(this) :: set_shape => set_phase_shape
     !! Set the shape for the phase
  end type base_phase_type

  ! Interface for handling the base phase parameters
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for handling learnable parameters and gradients of the phases
     pure module function get_phase_num_params(this) result(num_params)
       !! Get the number of learnable parameters for the phase
       class(base_phase_type), intent(in) :: this
       !! Instance of the phase
       integer :: num_params
       !! Number of learnable parameters
     end function get_phase_num_params

     pure module function get_phase_params(this) result(params)
       !! Get the learnable parameters for the phase
       class(base_phase_type), intent(in) :: this
       !! Instance of the phase
       real(real32), dimension(this%num_params) :: params
       !! Parameters
     end function get_phase_params

     pure module subroutine set_phase_params(this, params)
       !! Set the learnable parameters for the phase
       class(base_phase_type), intent(inout) :: this
       !! Instance of the phase
       real(real32), dimension(this%num_params), intent(in) :: params
       !! Parameters
     end subroutine set_phase_params
  end interface

  ! Interface for handling the base phase gradients
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for handling gradients of the phases
     pure module function get_phase_gradients(this, clip_method) &
          result(gradients)
       !! Get the gradients for the phase
       class(base_phase_type), intent(in) :: this
       !! Instance of the phase
       type(clip_type), optional, intent(in) :: clip_method
       !! Clip method
       real(real32), dimension(this%num_params) :: gradients
       !! Gradients
     end function get_phase_gradients

     pure module subroutine set_phase_gradients(this, gradients)
       !! Set the gradients for the phase
       class(base_phase_type), intent(inout) :: this
       !! Instance of the phase
       real(real32), dimension(..), intent(in) :: gradients
       !! Gradients
     end subroutine set_phase_gradients
  end interface

  ! Interface for setting the shape of the phase
  !-----------------------------------------------------------------------------
  interface
     !! Interface for setting the shape of a phase
     module subroutine set_phase_shape(this, shape)
       !! Set the shape of the phase
       class(base_phase_type), intent(inout) :: this
       !! Instance of the phase
       integer, dimension(:), intent(in) :: shape
       !! Shape of the phase
     end subroutine set_phase_shape
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Message phase
!-------------------------------------------------------------------------------
  type, extends(base_phase_type), abstract :: message_phase_type
     !! Type for the message phase
     integer :: num_message_features
     !! Number of message features
     logical :: use_message = .true.
     !! Use message
     type(feature_type), dimension(:), allocatable :: message
     !! Message
   contains
     procedure(update_message), deferred, pass(this) :: update
     !! Update the message
     procedure(calculate_partials_message), deferred, pass(this) :: &
          calculate_partials
     !! Calculate the partials
  end type message_phase_type

  ! Interface for updating the message
  !-----------------------------------------------------------------------------
  abstract interface
     !! interface for the message and phase forward and backward passes
     pure module subroutine update_message(this, input, graph)
       !! Update the message
       class(message_phase_type), intent(inout) :: this
       !! Instance of the message phase
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       !! Input features
       !!   dimensions: (feature, vertex, batch_size)
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
       !! Graph structure
     end subroutine update_message

     pure module subroutine calculate_partials_message( &
          this, input, gradient, graph &
     )
       !! Calculate the partials
       class(message_phase_type), intent(inout) :: this
       !! Instance of the message phase
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       !! Input features
       !!   dimensions: (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: gradient
       !! Derivative of the output
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
       !! Graph structure
     end subroutine calculate_partials_message
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Readout phase
!-------------------------------------------------------------------------------
  type, extends(base_phase_type), abstract :: readout_phase_type
     !! Type for the readout phase
   contains
     procedure(get_output_readout), deferred, pass(this) :: get_output
     !! Get the output
     procedure(calculate_partials_readout), deferred, pass(this) :: &
          calculate_partials
     !! Calculate the partials
  end type readout_phase_type

  ! Interface for updating the readout
  !-----------------------------------------------------------------------------
  abstract interface
     !! Interface for the readout phase forward and backward passes
     pure module subroutine get_output_readout(this, input, output)
       !! Get the output
       class(readout_phase_type), intent(inout) :: this
       !! Instance of the readout phase
       class(message_phase_type), dimension(:), intent(in) :: input
       !! Input features
       real(real32), dimension(this%num_outputs, this%batch_size), &
            intent(out) :: output
       !! Output
     end subroutine get_output_readout

     pure module subroutine calculate_partials_readout(this, input, gradient)
       !! Calculate the partials
       class(readout_phase_type), intent(inout) :: this
       !! Instance of the readout phase
       class(message_phase_type), dimension(:), intent(in) :: input
       !! Input features
       real(real32), dimension(this%num_outputs, this%batch_size), &
            intent(in) :: gradient
       !! Derivative of the output
     end subroutine calculate_partials_readout
  end interface
!-------------------------------------------------------------------------------



end module athena__mpnn_layer
