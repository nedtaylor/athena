module athena__msgpass_layer
  !! Module containing the types and interfaces of a message passing layer
  use athena__constants, only: real32
  use graphstruc, only: graph_type
  use athena__base_layer, only: learnable_layer_type
  use athena__clipper, only: clip_type
  use athena__misc_types, only: array_type, array2d_type
  implicit none


  private

  public :: msgpass_layer_type


!-------------------------------------------------------------------------------
! Message passing layer
!-------------------------------------------------------------------------------
  type, abstract, extends(learnable_layer_type) :: msgpass_layer_type
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
     integer, dimension(:), allocatable :: num_vertex_features
     !! Number of vertex features for each time step
     integer, dimension(:), allocatable :: num_edge_features
     !! Number of edge features for each time step
     integer :: num_time_steps
     !! Number of time steps
     integer :: num_output_vertex_features
     !! Number of output vertex features
     integer :: num_output_edge_features
     !! Number of output edge features
     integer :: num_outputs
     !! Number of outputs (if output is not graph structure)

     integer, dimension(:), allocatable :: num_params_msg
     !! Number of learnable parameters for each message
     integer :: num_params_readout
     !! Number of learnable parameters for the readout

     type(array_type), dimension(:,:), allocatable :: vertex_features
     !! Vertex features for each time step
     type(array_type), dimension(:,:), allocatable :: edge_features
     !! Edge features for each time step
     type(array_type), dimension(:,:), allocatable :: message
     !! Message for each time step
     type(array_type), dimension(:,:), allocatable :: z
     !! Non-transformed message for each time step
   contains
     !  procedure, pass(this) :: set_hyperparams => set_hyperparams_msgpass
     !  !! Set the hyperparameters for message passing layer
     procedure, pass(this) :: init => init_msgpass
     !! Initialise message passing layer
     procedure, pass(this) :: set_batch_size => set_batch_size_msgpass
     !! Set the batch size for message passing layer
     ! procedure, pass(this) :: print => print_msgpass
     ! !! Print the message passing layer
     ! procedure, pass(this) :: read => read_msgpass
     ! !! Read the message passing layer
     procedure, pass(this) :: set_graph => set_graph_msgpass



     ! procedure, pass(this) :: reduce => layer_reduction
     ! !! Reduce message passing layer
     ! procedure, pass(this) :: merge => layer_merge
     ! !! Merge message passing layer
     procedure, pass(this) :: get_num_params => get_num_params_msgpass
     !! Get the number of learnable parameters for message passing layer
     procedure, pass(this) :: get_params => get_params_msgpass
     !! Get the learnable parameters for message passing layer
     procedure, pass(this) :: set_params => set_params_msgpass
     !! Set the learnable parameters for message passing layer
     procedure, pass(this) :: set_param_pointers => set_param_pointers_msgpass
     !! Set the pointers to the learnable parameters for message passing layer
     ! procedure, pass(this) :: get_gradients => get_gradients_msgpass
     ! !! Get the gradients for message passing layer
     ! procedure, pass(this) :: set_gradients => set_gradients_msgpass
     ! !! Set the gradients for message passing layer

     ! procedure, pass(this) :: forward => forward_rank
     ! !! Forward pass for message passing layer
     ! procedure, pass(this) :: backward => backward_rank
     ! !! Backward pass for message passing layer

     procedure, pass(this) :: forward_derived => forward_derived_msgpass
     !! Forward pass for message passing layer
     procedure, pass(this) :: backward_derived => backward_derived_msgpass
     !! Backward pass for message passing layer


     procedure(update_message_msgpass), deferred, pass(this) :: update_message
     !! Update the message
     procedure(update_readout_msgpass), deferred, pass(this) :: update_readout
     !! Update the readout

     procedure(backward_message_msgpass), deferred, pass(this) :: &
          backward_message
     !! Calculate the partials of the message
     procedure(backward_readout_msgpass), deferred, pass(this) :: &
          backward_readout
     !! Calculate the partials of the readout
  end type msgpass_layer_type

  ! Interface for setting up the MPNN layer
  !-----------------------------------------------------------------------------
  interface msgpass_layer_type
     !! Interface for setting up the MPNN layer
     module function layer_setup( &
          num_features, num_time_steps, batch_size, &
          verbose &
     ) result(layer)
       !! Set up the MPNN layer
       !!! MAKE THESE ASSUMED RANK
       integer, dimension(2), intent(in) :: num_features
       !! Number of features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       class(msgpass_layer_type), allocatable :: layer
       !! Instance of the message passing layer
     end function layer_setup
  end interface msgpass_layer_type

  ! Interface for handling the message passing layer parameters
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for handling learnable parameters and gradients
     pure module function get_num_params_msgpass(this) result(num_params)
       !! Get the number of learnable parameters for the message passing layer
       class(msgpass_layer_type), intent(in) :: this
       !! Instance of the message passing layer
       integer :: num_params
       !! Number of learnable parameters
     end function get_num_params_msgpass

     pure module function get_params_msgpass(this) result(params)
       !! Get the learnable parameters for the message passing layer
       class(msgpass_layer_type), intent(in) :: this
       !! Instance of the message passing layer
       real(real32), dimension(this%num_params) :: params
       !! Parameters
     end function get_params_msgpass

     pure module subroutine set_params_msgpass(this, params)
       !! Set the learnable parameters for the message passing layer
       class(msgpass_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       real(real32), dimension(this%num_params), intent(in) :: params
       !! Parameters
     end subroutine set_params_msgpass

     module subroutine set_param_pointers_msgpass(this)
       !! Set the pointers to the learnable parameters
       class(msgpass_layer_type), intent(inout), target :: this
       !! Instance of the message passing layer
     end subroutine set_param_pointers_msgpass



     module subroutine set_graph_msgpass(this, graph)
       !! Set the graph structure of the input data
       class(msgpass_layer_type), intent(inout) :: this
       !! Instance of the layer
       type(graph_type), dimension(:), intent(in) :: graph
       !! Graph structure of input data
     end subroutine set_graph_msgpass


     ! pure module function get_gradients_msgpass(this, clip_method) &
     !      result(gradients)
     !   !! Get the gradients for the message passing layer
     !   class(msgpass_layer_type), intent(in) :: this
     !   !! Instance of the message passing layer
     !   type(clip_type), optional, intent(in) :: clip_method
     !   !! Clip method
     !   real(real32), dimension(this%num_params) :: gradients
     !   !! Gradients
     ! end function get_gradients_msgpass

     ! pure module subroutine set_gradients_msgpass(this, gradients)
     !   !! Set the gradients for the message passing layer
     !   class(msgpass_layer_type), intent(inout) :: this
     !   !! Instance of the message passing layer
     !   real(real32), dimension(..), intent(in) :: gradients
     !   !! Gradients
     ! end subroutine set_gradients_msgpass
  end interface

  ! ! Interface for reducing and merging layers
  ! !-----------------------------------------------------------------------------
  ! interface
  !    !! Interfaces for reducing and merging layers
  !    module subroutine layer_reduction(this, rhs)
  !      !! Reduce the layer
  !      class(msgpass_layer_type), intent(inout) :: this
  !      !! Instance of the message passing layer
  !      class(learnable_layer_type), intent(in) :: rhs
  !      !! Instance of the learnable layer (expects a message passing layer)
  !    end subroutine layer_reduction

  !    module subroutine layer_merge(this, input)
  !      !! Merge the layer
  !      class(msgpass_layer_type), intent(inout) :: this
  !      !! Instance of the message passing layer
  !      class(learnable_layer_type), intent(in) :: input
  !      !! Instance of the learnable layer (expects a message passing layer)
  !    end subroutine layer_merge
  ! end interface

  ! Interface for handling forward and backward passes
  !-----------------------------------------------------------------------------
  interface
     ! !! Interfaces for handling forward and backward passes of the MPNN
     ! module subroutine forward_rank(this, input)
     !   !! Forward pass for the message passing layer
     !   class(msgpass_layer_type), intent(inout) :: this
     !   !! Instance of the message passing layer
     !   real(real32), dimension(..), intent(in) :: input
     !   !! Input
     ! end subroutine forward_rank

     ! module subroutine backward_rank(this, input, gradient)
     !   !! Backward pass for the message passing layer
     !   class(msgpass_layer_type), intent(inout) :: this
     !   !! Instance of the message passing layer
     !   real(real32), dimension(..), intent(in) :: input
     !   !! Input
     !   real(real32), dimension(..), intent(in) :: gradient
     !   !! Gradient
     ! end subroutine backward_rank

     module subroutine forward_derived_msgpass(this, input)
       !! Forward pass for the message passing layer
       class(msgpass_layer_type), intent(inout) :: this
       !! Instance of the layer type
       class(array_type), dimension(:,:), intent(in) :: input
       !! Input data (i.e. vertex and edge features)
     end subroutine forward_derived_msgpass

     module subroutine backward_derived_msgpass(this, input, gradient)
       !! Backward pass for the message passing layer
       class(msgpass_layer_type), intent(inout) :: this
       !! Instance of the layer type
       class(array_type), dimension(:,:), intent(in) :: input
       !! Input data (i.e. vertex and edge features)
       class(array_type), dimension(:,:), intent(in) :: gradient
       !! Gradient data
     end subroutine backward_derived_msgpass
  end interface

  ! Interface for handling graphs and outputs
  !-----------------------------------------------------------------------------
  interface
     !! Interfaces for handling graphs and outputs, and initialising the layer
     ! module subroutine print_msgpass(this, file)
     !   !! Print the message passing layer
     !   class(msgpass_layer_type), intent(in) :: this
     !   !! Instance of the message passing layer
     !   character(*), intent(in) :: file
     !   !! File to print to
     ! end subroutine print_msgpass
     ! module subroutine read_msgpass(this, unit, verbose)
     !   !! Read the message passing layer
     !   class(msgpass_layer_type), intent(inout) :: this
     !   !! Instance of the message passing layer
     !   integer, intent(in) :: unit
     !   !! Unit to read from
     !   integer, optional, intent(in) :: verbose
     !   !! Verbosity level
     ! end subroutine read_msgpass
     module subroutine init_msgpass(this, input_shape, batch_size, verbose)
       !! Initialise the message passing layer
       class(msgpass_layer_type), intent(inout) :: this
       !! Instance of the message passing layer
       integer, dimension(:), intent(in) :: input_shape
       !! Input shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine init_msgpass
     module subroutine set_batch_size_msgpass(this, batch_size, verbose)
       !! Set the batch size for the message passing layer
       class(msgpass_layer_type), intent(inout), target :: this
       !! Instance of the message passing layer
       integer, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
     end subroutine set_batch_size_msgpass
     !  module subroutine set_hyperparams_msgpass( &
     !       this, num_features, num_time_steps, num_outputs, verbose &
     !  )
     !    !! Set the hyperparameters for the message passing layer
     !    class(msgpass_layer_type), intent(inout) :: this
     !    !! Instance of the message passing layer
     !    integer, dimension(2), intent(in) :: num_features
     !    !! Number of features
     !    integer, intent(in) :: num_time_steps
     !    !! Number of time steps
     !    integer, intent(in) :: num_outputs
     !    !! Number of outputs
     !    integer, optional, intent(in) :: verbose
     !    !! Verbosity level
     !  end subroutine set_hyperparams_msgpass
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!



  interface
     !! interface for the message forward and backward passes
     module subroutine update_message_msgpass(this, input)
       !! Update the message
       class(msgpass_layer_type), intent(inout), target :: this
       !! Instance of the message passing layer
       class(array_type), dimension(:,:), intent(in) :: input
       !! Input data (i.e. vertex and edge features)
     end subroutine update_message_msgpass

     module subroutine backward_message_msgpass( &
          this, input, gradient &
     )
       !! Calculate the partials
       class(msgpass_layer_type), intent(inout), target :: this
       !! Instance of the message passing layer
       class(array_type), dimension(:,:), intent(in) :: input
       !! Input data (i.e. vertex and edge features)
       class(array_type), dimension(:,:), intent(in) :: gradient
       !! Gradient data
     end subroutine backward_message_msgpass
  end interface

  interface
     !! interface for the readout forward and backward passes
     module subroutine update_readout_msgpass(this)
       !! Update the message
       class(msgpass_layer_type), intent(inout), target :: this
       !! Instance of the message passing layer
     end subroutine update_readout_msgpass

     module subroutine backward_readout_msgpass(this, gradient)
       !! Calculate the partials
       class(msgpass_layer_type), intent(inout), target :: this
       !! Instance of the message passing layer
       class(array_type), dimension(:,:), intent(in) :: gradient
       !! Gradient data
     end subroutine backward_readout_msgpass
  end interface



end module athena__msgpass_layer
