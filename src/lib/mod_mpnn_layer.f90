!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a message passing neural network
!!!#############################################################################
module mpnn_layer
  use constants, only: real12
  use custom_types, only: graph_type
  use base_layer, only: learnable_layer_type
  use clipper, only: clip_type
  implicit none
  

  private

  public :: mpnn_layer_type, feature_type, method_container_type
  public :: state_method_type, message_method_type, readout_method_type
  public :: update_state
  public :: update_message
  public :: get_output_readout


  type, extends(learnable_layer_type) :: mpnn_layer_type
     type(graph_type), dimension(:), allocatable :: graph
     class(method_container_type), allocatable :: method
     real(real12), dimension(:,:), allocatable :: output
     !real(real12), dimension(:,:,:), allocatable :: di
   contains
     procedure, pass(this) :: init => init_mpnn
     procedure, pass(this) :: get_output => get_output_mpnn
     procedure, pass(this) :: set_batch_size => set_batch_size_mpnn

     procedure, pass(this) :: reduce => layer_reduction
     procedure, pass(this) :: merge => layer_merge
     procedure, pass(this) :: get_num_params => get_num_params_mpnn
     procedure, pass(this) :: get_params => get_params_mpnn
     procedure, pass(this) :: set_params => set_params_mpnn
     procedure, pass(this) :: get_gradients => get_gradients_mpnn
     procedure, pass(this) :: set_gradients => set_gradients_mpnn

     procedure, pass(this) :: set_graph
     procedure, pass(this) :: forward => forward_rank
     procedure, pass(this) :: backward => backward_rank
  end type mpnn_layer_type

  type, abstract :: method_container_type
    integer :: num_outputs
    integer :: num_time_steps
    !! each dimension of num_features is for vertex and edge
    integer, dimension(2) :: num_features
    !! state and message dimension is (time_step)
    class(message_method_type), dimension(:), allocatable :: message
    class(state_method_type), dimension(:), allocatable :: state
    class(readout_method_type), allocatable :: readout
   contains
    procedure(init_method), deferred, pass(this) :: init
  end type method_container_type

  type :: feature_type
     real(real12), dimension(:,:), allocatable :: val
   contains
     ! t = type, r = real, i = int
     procedure :: add_t_t => feature_add
     procedure :: multiply_t_t => feature_multiply
     generic :: operator(+) => add_t_t
     generic :: operator(*) => multiply_t_t
  end type feature_type


  type, abstract :: base_method_type
     !!! HAVE LOGICAL THAT STATES WHETHER IT IS LEARNABLE ???
     integer :: num_inputs
     integer :: num_outputs
     integer :: batch_size
     !! feature has dimensions (batch_size)
     type(feature_type), dimension(:), allocatable :: feature
     type(feature_type), dimension(:), allocatable :: di
   contains
     procedure, pass(this) :: get_num_params => get_method_num_params
     procedure, pass(this) :: get_params => get_method_params
     procedure, pass(this) :: set_params => set_method_params
     procedure, pass(this) :: get_gradients => get_method_gradients
     procedure, pass(this) :: set_gradients => set_method_gradients
     procedure, pass(this) :: set_shape => set_method_shape
  end type base_method_type


  type, extends(base_method_type), abstract :: message_method_type
   contains
     procedure(update_message), deferred, pass(this) :: update
     procedure(calculate_partials_message), deferred, pass(this) :: calculate_partials
  end type message_method_type

  type, extends(base_method_type), abstract :: state_method_type
   contains
     procedure(update_state), deferred, pass(this) :: update
     procedure(calculate_partials_state), deferred, pass(this) :: calculate_partials
  end type state_method_type

  type, extends(base_method_type), abstract :: readout_method_type
   contains
     procedure(get_output_readout), deferred, pass(this) :: get_output
     procedure(calculate_partials_readout), deferred, pass(this) :: calculate_partials
  end type readout_method_type


  abstract interface
     pure module subroutine update_message(this, input, graph)
       class(message_method_type), intent(inout) :: this
       !! input features has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine update_message

     pure module subroutine calculate_partials_message(this, input, gradient, graph)
       class(message_method_type), intent(inout) :: this
       !! hidden features has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(feature_type), dimension(this%batch_size), intent(in) :: gradient
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine calculate_partials_message


     pure module subroutine update_state(this, input, graph)
       class(state_method_type), intent(inout) :: this
       !! input has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine update_state

     pure module subroutine calculate_partials_state(this, input, gradient, graph)
       class(state_method_type), intent(inout) :: this
       !! hidden features has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(feature_type), dimension(this%batch_size), intent(in) :: gradient
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine calculate_partials_state


     pure module subroutine get_output_readout(this, input, output)
       class(readout_method_type), intent(inout) :: this
       class(state_method_type), dimension(:), intent(in) :: input
       real(real12), dimension(this%num_outputs, this%batch_size), intent(out) :: output
     end subroutine get_output_readout

     pure module subroutine calculate_partials_readout(this, input, gradient)
       class(readout_method_type), intent(inout) :: this
       class(state_method_type), dimension(:), intent(in) :: input
       real(real12), dimension(this%num_outputs, this%batch_size), intent(in) :: gradient
     end subroutine calculate_partials_readout
  end interface


  interface
    elemental module function feature_add(a, b) result(output)
      class(feature_type), intent(in) :: a, b
      type(feature_type) :: output
    end function feature_add

    elemental module function feature_multiply(a, b) result(output)
      class(feature_type), intent(in) :: a, b
      type(feature_type) :: output
    end function feature_multiply

    module subroutine layer_reduction(this, rhs)
      class(mpnn_layer_type), intent(inout) :: this
      class(learnable_layer_type), intent(in) :: rhs
    end subroutine layer_reduction
  
    module subroutine layer_merge(this, input)
      class(mpnn_layer_type), intent(inout) :: this
      class(learnable_layer_type), intent(in) :: input
    end subroutine layer_merge
  end interface

  interface
    pure module function get_num_params_mpnn(this) result(num_params)
      class(mpnn_layer_type), intent(in) :: this
      integer :: num_params
    end function get_num_params_mpnn

    pure module function get_params_mpnn(this) result(params)
      class(mpnn_layer_type), intent(in) :: this
      real(real12), allocatable, dimension(:) :: params
    end function get_params_mpnn

    pure module subroutine set_params_mpnn(this, params)
      class(mpnn_layer_type), intent(inout) :: this
      real(real12), dimension(:), intent(in) :: params
    end subroutine set_params_mpnn

    pure module function get_gradients_mpnn(this, clip_method) result(gradients)
      class(mpnn_layer_type), intent(in) :: this
      type(clip_type), optional, intent(in) :: clip_method
      real(real12), allocatable, dimension(:) :: gradients
    end function get_gradients_mpnn

    pure module subroutine set_gradients_mpnn(this, gradients)
      class(mpnn_layer_type), intent(inout) :: this
      real(real12), dimension(..), intent(in) :: gradients
    end subroutine set_gradients_mpnn
  end interface

  interface
    pure module function get_method_num_params(this) result(num_params)
      class(base_method_type), intent(in) :: this
      integer :: num_params
    end function get_method_num_params

    pure module function get_method_params(this) result(params)
      class(base_method_type), intent(in) :: this
      real(real12), allocatable, dimension(:) :: params
    end function get_method_params

    pure module subroutine set_method_params(this, params)
      class(base_method_type), intent(inout) :: this
      real(real12), dimension(:), intent(in) :: params
    end subroutine set_method_params

    pure module function get_method_gradients(this, clip_method) result(gradients)
      class(base_method_type), intent(in) :: this
      type(clip_type), optional, intent(in) :: clip_method
      real(real12), allocatable, dimension(:) :: gradients
    end function get_method_gradients

    pure module subroutine set_method_gradients(this, gradients)
      class(base_method_type), intent(inout) :: this
      real(real12), dimension(..), intent(in) :: gradients      
    end subroutine set_method_gradients
  end interface

  interface
    module subroutine set_method_shape(this, shape)
      class(base_method_type), intent(inout) :: this
      integer, dimension(:), intent(in) :: shape
    end subroutine set_method_shape
  end interface

  interface
    pure module subroutine forward_rank(this, input)
      class(mpnn_layer_type), intent(inout) :: this
      real(real12), dimension(..), intent(in) :: input
    end subroutine forward_rank

    pure module subroutine backward_rank(this, input, gradient)
      class(mpnn_layer_type), intent(inout) :: this
      real(real12), dimension(..), intent(in) :: input
      real(real12), dimension(..), intent(in) :: gradient
    end subroutine backward_rank

    pure module subroutine forward_graph(this, graph)
      class(mpnn_layer_type), intent(inout) :: this
      type(graph_type), dimension(this%batch_size), intent(in) :: graph
    end subroutine forward_graph

    pure module subroutine backward_graph(this, graph, gradient)
      class(mpnn_layer_type), intent(inout) :: this
      type(graph_type), dimension(this%batch_size), intent(in) :: graph
      real(real12), dimension( &
           this%output_shape(1), &
           this%batch_size &
      ), intent(in) :: gradient
    end subroutine backward_graph
  end interface


  interface
     module subroutine set_graph(this, graph)
       class(mpnn_layer_type), intent(inout) :: this
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine set_graph
     pure module subroutine get_output_mpnn(this, output)
       class(mpnn_layer_type), intent(in) :: this
       real(real12), allocatable, dimension(..), intent(out) :: output
     end subroutine get_output_mpnn
     module subroutine set_batch_size_mpnn(this, batch_size, verbose)
       class(mpnn_layer_type), intent(inout) :: this
       integer, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine set_batch_size_mpnn
     module subroutine init_mpnn(this, input_shape, batch_size, verbose)
       class(mpnn_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine init_mpnn
  end interface


  interface
    module subroutine init_method(this, input_shape, output_shape, batch_size)
      class(method_container_type), intent(inout) :: this
      integer, dimension(3), intent(in) :: input_shape
      integer, dimension(1), intent(in) :: output_shape
      integer, intent(in) :: batch_size
    end subroutine init_method
  end interface


  interface mpnn_layer_type
     module function layer_setup( &
          method, &
          num_features, num_time_steps, num_outputs, batch_size &
      ) result(layer)
       !! MAKE THESE ASSUMED RANK
       class(method_container_type), intent(in) :: method
       integer, dimension(2), intent(in) :: num_features
       integer, intent(in) :: num_time_steps
       integer, intent(in) :: num_outputs
       integer, optional, intent(in) :: batch_size
       type(mpnn_layer_type) :: layer
     end function layer_setup
  end interface mpnn_layer_type


end module mpnn_layer
!!!#############################################################################