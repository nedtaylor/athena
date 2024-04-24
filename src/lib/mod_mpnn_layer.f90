!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a message passing neural network
!!!#############################################################################
module mpnn_layer
  use constants, only: real12
  use custom_types, only: graph_type
  implicit none
  

  private

  public :: mpnn_layer_type, feature_type
  public :: state_method_type, message_method_type, readout_method_type
  public :: state_update, get_state_differential
  public :: message_update, get_message_differential
  public :: get_readout_output, get_readout_differential


  type :: mpnn_layer_type
     integer :: num_features
     integer :: num_time_steps
     integer :: num_outputs
     integer :: batch_size
     !! state and message dimension is (time_step)
     class(message_method_type), dimension(:), allocatable :: message
     class(state_method_type), dimension(:), allocatable :: state
     class(readout_method_type), allocatable :: readout
     real(real12), dimension(:,:), allocatable :: output
     !real(real12), dimension(:,:,:), allocatable :: di
   contains
     procedure, pass(this) :: forward
     procedure, pass(this) :: backward
  end type mpnn_layer_type


  type :: feature_type
     real(real12), dimension(:,:), allocatable :: val
   contains
     ! t = type, r = real, i = int
     procedure :: add_t_t => feature_add
     procedure :: multiply_t_t => feature_multiply
     generic :: operator(+) => add_t_t
     generic :: operator(*) => multiply_t_t
  end type feature_type


   type, abstract :: message_method_type
     integer :: num_features
     integer :: batch_size
     !! feature has dimensions (feature, vertex)
     type(feature_type), dimension(:), allocatable :: feature
     type(feature_type), dimension(:), allocatable :: di
   contains
     procedure(message_update), deferred, pass(this) :: update
     procedure(get_message_differential), deferred, pass(this) :: get_differential
     procedure(calculate_message_partials), deferred, pass(this) :: calculate_partials
  end type message_method_type

  type, abstract :: state_method_type
     integer :: num_features
     integer :: batch_size
     !! feature has dimensions (feature, vertex)
     type(feature_type), dimension(:), allocatable :: feature
     type(feature_type), dimension(:), allocatable :: di
   contains
     procedure(state_update), deferred, pass(this) :: update
     procedure(get_state_differential), deferred, pass(this) :: get_differential
     procedure(calculate_state_partials), deferred, pass(this) :: calculate_partials
  end type state_method_type

  type, abstract :: readout_method_type
     integer :: batch_size
     integer :: num_outputs
     type(feature_type), dimension(:), allocatable :: di
   contains
     procedure(get_readout_output), deferred, pass(this) :: get_output
     procedure(get_readout_differential), deferred, pass(this) :: get_differential
     procedure(calculate_readout_partials), deferred, pass(this) :: calculate_partials
  end type readout_method_type


  abstract interface
     module subroutine message_update(this, input, graph)
       class(message_method_type), intent(inout) :: this
       !! input features has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine message_update

     pure module function get_message_differential(this, input, graph) result(output)
       class(message_method_type), intent(in) :: this
       !! input features has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
       type(feature_type), dimension(this%batch_size) :: output
     end function get_message_differential

     module subroutine calculate_message_partials(this, input, gradient, graph)
       class(message_method_type), intent(inout) :: this
       !! hidden features has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(feature_type), dimension(this%batch_size), intent(in) :: gradient
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine calculate_message_partials


     module subroutine state_update(this, input, graph)
       class(state_method_type), intent(inout) :: this
       !! input has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine state_update

     pure module function get_state_differential(this, input, graph) result(output)
       class(state_method_type), intent(in) :: this
       !! input has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
       type(feature_type), dimension(this%batch_size) :: output
     end function get_state_differential

     module subroutine calculate_state_partials(this, input, gradient, graph)
       class(state_method_type), intent(inout) :: this
       !! hidden features has dimensions (feature, vertex, batch_size)
       type(feature_type), dimension(this%batch_size), intent(in) :: input
       type(feature_type), dimension(this%batch_size), intent(in) :: gradient
       type(graph_type), dimension(this%batch_size), intent(in) :: graph
     end subroutine calculate_state_partials


     pure module function get_readout_output(this, input) result(output)
       class(readout_method_type), intent(in) :: this
       class(state_method_type), dimension(:), intent(in) :: input
       real(real12), dimension(:,:), allocatable :: output
     end function get_readout_output

     pure module function get_readout_differential(this, input) result(output)
       class(readout_method_type), intent(in) :: this
       class(state_method_type), dimension(:), intent(in) :: input
       type(feature_type), dimension(this%batch_size) :: output
     end function get_readout_differential

     module subroutine calculate_readout_partials(this, input, gradient)
       class(readout_method_type), intent(inout) :: this
       class(state_method_type), dimension(:), intent(in) :: input
       real(real12), dimension(:,:), intent(in) :: gradient
     end subroutine calculate_readout_partials

  end interface

  interface
    module subroutine forward(this, graph)
      class(mpnn_layer_type), intent(inout) :: this
      type(graph_type), dimension(this%batch_size), intent(in) :: graph
    end subroutine forward

    module subroutine backward(this, graph, gradient)
      class(mpnn_layer_type), intent(inout) :: this
      type(graph_type), dimension(this%batch_size), intent(in) :: graph
      real(real12), dimension( &
           this%readout%num_outputs, &
           this%batch_size &
      ), intent(in) :: gradient
    end subroutine backward
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
  end interface
  

  interface mpnn_layer_type
     module function layer_setup( &
          message_method, state_method, readout_method, &
          num_features, num_time_steps, num_outputs, batch_size &
      ) result(layer)
       !! MAKE THESE ASSUMED RANK
       class(message_method_type), intent(in) :: message_method
       class(state_method_type), intent(in) :: state_method
       class(readout_method_type), intent(in) :: readout_method
       integer, intent(in) :: num_features
       integer, intent(in) :: num_time_steps
       integer, intent(in) :: num_outputs
       integer, optional, intent(in) :: batch_size
       type(mpnn_layer_type) :: layer
     end function layer_setup
  end interface mpnn_layer_type


end module mpnn_layer
!!!#############################################################################