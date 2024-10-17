!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains the network class, which is used to define a neural network
!!! module contains the following derived types:
!!! - network_type
!!!##################
!!! network_type contains the following procedures:
!!! - print           - print the network to file
!!! - read            - read the network from a file
!!! - add             - add a layer to the network
!!! - reset           - reset the network
!!! - compile         - compile the network
!!! - set_batch_size  - set batch size
!!! - set_metrics     - set network metrics
!!! - set_loss        - set network loss method
!!! - train           - train the network
!!! - test            - test the network
!!! - predict         - return predicted results from supplied inputs using ...
!!!                     ... the trained network
!!! - update          - update the learnable parameters of the network based ...
!!!                     ... on gradients
!!! - reduce          - reduce two networks down to one ...
!!!                     ... (i.e. add two networks - parallel)
!!! - copy            - copy a network
!!! - get_num_params  - get number of learnable parameters in the network
!!! - get_params      - get learnable parameters
!!! - set_params      - set learnable parameters
!!! - get_gradients   - get gradients of learnable parameters
!!! - set_gradients   - set learnable parameter gradients
!!! - reset_gradients - reset learnable parameter gradients
!!! - forward         - forward pass
!!! - backward        - backward pass
!!!#############################################################################
module network
  use constants, only: real32
  use graphstruc, only: graph_type
  use metrics, only: metric_dict_type
  use optimiser, only: base_optimiser_type
  use loss, only: &
       comp_loss_func => compute_loss_function, &
       comp_loss_deriv => compute_loss_derivative
  use accuracy, only: comp_acc_func => compute_accuracy_function
  use base_layer, only: base_layer_type
  use container_layer, only: container_layer_type
  implicit none

  private

  public :: network_type


  type :: network_type
     real(real32) :: accuracy, loss
     integer :: batch_size = 0
     integer :: num_layers = 0
     integer :: num_outputs = 0
     integer :: num_params = 0
     class(base_optimiser_type), allocatable :: optimiser
     type(metric_dict_type), dimension(2) :: metrics
     type(container_layer_type), allocatable, dimension(:) :: model
     procedure(comp_loss_func), nopass, pointer :: get_loss => null()
     procedure(comp_loss_deriv), nopass, pointer :: get_loss_deriv => null()
     procedure(comp_acc_func), nopass, pointer :: get_accuracy => null()
     integer, dimension(:), allocatable :: vertex_order
     integer, dimension(:), allocatable :: root_vertices, output_vertices
     type(graph_type(directed=.true.)), private :: auto_graph
   contains
     procedure, pass(this) :: print
     procedure, pass(this) :: read
     procedure, pass(this) :: add
     procedure, pass(this) :: reset
     procedure, pass(this) :: compile
     procedure, pass(this) :: set_batch_size
     procedure, pass(this) :: set_metrics
     procedure, pass(this) :: set_loss
     procedure, pass(this) :: set_accuracy
     procedure, pass(this) :: train
     procedure, pass(this) :: test
     procedure, pass(this) :: predict => predict_1d
     procedure, pass(this) :: update

     procedure, pass(this), private :: generate_vertex_order
     procedure, pass(this), private :: dfs
     procedure, pass(this), private :: calculate_root_vertices
     procedure, pass(this), private :: calculate_output_vertices
     procedure, pass(this), private :: get_input_autodiff
     procedure, pass(this), private :: get_gradient_autodiff

     procedure, pass(this) :: reduce => network_reduction
     procedure, pass(this) :: copy => network_copy

     procedure, pass(this) :: get_num_params
     procedure, pass(this) :: get_params
     procedure, pass(this) :: set_params
     procedure, pass(this) :: get_gradients
     procedure, pass(this) :: set_gradients
     procedure, pass(this) :: reset_gradients

     procedure, pass(this) :: forward => forward_1d
     procedure, pass(this) :: backward => backward_1d
  end type network_type

  interface network_type
     !!-------------------------------------------------------------------------
     !! setup the network (network initialisation)
     !!-------------------------------------------------------------------------
     !! layers      = (T, in) layer container
     !! optimiser   = (T, in, opt) optimiser
     !! loss_method = (S, in, opt) loss method
     !! metrics     = (*, in, opt) metrics, either string or metric_dict_type
     !! batch_size  = (I, in, opt) batch size
     module function network_setup( &
          layers, &
          optimiser, loss_method, accuracy_method, &
          metrics, batch_size) result(network)
       type(container_layer_type), dimension(:), intent(in) :: layers
       class(base_optimiser_type), optional, intent(in) :: optimiser
       character(*), optional, intent(in) :: loss_method, accuracy_method
       class(*), dimension(..), optional, intent(in) :: metrics
       integer, optional, intent(in) :: batch_size
       type(network_type) :: network
     end function network_setup
  end interface network_type

  interface
     !!-------------------------------------------------------------------------
     !! print the network to file
     !!-------------------------------------------------------------------------
     !! this = (T, in) network type
     !! file = (I, in) file name
     module subroutine print(this, file)
       class(network_type), intent(in) :: this
       character(*), intent(in) :: file
     end subroutine print

     !!-------------------------------------------------------------------------
     !! read the network from a file
     !!-------------------------------------------------------------------------
     !! this = (T, io) network type
     !! file = (I, in) file name
     module subroutine read(this, file)
       class(network_type), intent(inout) :: this
       character(*), intent(in) :: file
     end subroutine read

     !!-------------------------------------------------------------------------
     !! add a layer to the network
     !!-------------------------------------------------------------------------
     !! this  = (T, io) network type
     !! layer = (I, in) layer to add
     module subroutine add(this, layer, input_list, output_list)
       class(network_type), intent(inout) :: this
       class(base_layer_type), intent(in) :: layer
       integer, dimension(:), intent(in), optional :: input_list, output_list
     end subroutine add

     !!-------------------------------------------------------------------------
     !! reset the network
     !!-------------------------------------------------------------------------
     !! this = (T, io) network type
     module subroutine reset(this)
       class(network_type), intent(inout) :: this
     end subroutine reset

     !!-------------------------------------------------------------------------
     !! compile the network
     !!-------------------------------------------------------------------------
     !! this        = (T, io) network type
     !! optimiser   = (T, in) optimiser
     !! loss_method = (S, in, opt) loss method
     !! metrics     = (*, in, opt) metrics, either string or metric_dict_type
     !! batch_size  = (I, in, opt) batch size
     !! verbose     = (I, in, opt) verbosity level
     module subroutine compile(this, optimiser, loss_method, accuracy_method, &
          metrics, batch_size, verbose)
       class(network_type), intent(inout) :: this
       class(base_optimiser_type), intent(in) :: optimiser
       character(*), optional, intent(in) :: loss_method, accuracy_method
       class(*), dimension(..), optional, intent(in) :: metrics
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
     end subroutine compile

     !!-------------------------------------------------------------------------
     !! set batch size
     !!-------------------------------------------------------------------------
     !! this       = (T, io) network type
     !! batch_size = (I, in) batch size to use
     module subroutine set_batch_size(this, batch_size)
       class(network_type), intent(inout) :: this
       integer, intent(in) :: batch_size
     end subroutine set_batch_size

     !!-------------------------------------------------------------------------
     !! set network metrics
     !!-------------------------------------------------------------------------
     !! this    = (T, io) network type
     !! metrics = (*, in) metrics to use
     module subroutine set_metrics(this, metrics)
       class(network_type), intent(inout) :: this
       class(*), dimension(..), intent(in) :: metrics
     end subroutine set_metrics

     !!-------------------------------------------------------------------------
     !! set network loss method
     !!-------------------------------------------------------------------------
     !! this        = (T, io) network type
     !! loss_method = (S, in) loss method to use
     !! verbose     = (I, in, opt) verbosity level
     module subroutine set_loss(this, loss_method, verbose)
       class(network_type), intent(inout) :: this
       character(*), intent(in) :: loss_method
       integer, optional, intent(in) :: verbose
     end subroutine set_loss

     !!-------------------------------------------------------------------------
     !! set network accuracy method
     !!-------------------------------------------------------------------------
     !! this        = (T, io) network type
     !! accuracy_method = (S, in) accuracy method to use
     !! verbose     = (I, in, opt) verbosity level
     module subroutine set_accuracy(this, accuracy_method, verbose)
       class(network_type), intent(inout) :: this
       character(*), intent(in) :: accuracy_method
       integer, optional, intent(in) :: verbose
     end subroutine set_accuracy

     !!-------------------------------------------------------------------------
     !! train the network
     !!-------------------------------------------------------------------------
     !! this              = (T, io) network type
     !! input             = (R, in) input data
     !! output            = (*, in) expected output data (data labels)
     !! num_epochs        = (I, in) number of epochs to train for
     !! batch_size        = (I, in, opt) batch size (DEPRECATED)
     !! addit_input       = (R, in, opt) additional input data
     !! addit_layer       = (I, in, opt) layer to insert additional input data
     !! plateau_threshold = (R, in, opt) threshold for checking learning plateau
     !! shuffle_batches   = (B, in, opt) shuffle batch order
     !! batch_print_step  = (I, in, opt) print step for batch
     !! verbose           = (I, in, opt) verbosity level
     module subroutine train(this, input, output, num_epochs, batch_size, &
         addit_input, addit_layer, &
         plateau_threshold, shuffle_batches, batch_print_step, verbose)
       class(network_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
       class(*), dimension(:,:), intent(in) :: output
       integer, intent(in) :: num_epochs
       integer, optional, intent(in) :: batch_size !! deprecated
       real(real32), dimension(:,:), optional, intent(in) :: addit_input
       integer, optional, intent(in) :: addit_layer
       real(real32), optional, intent(in) :: plateau_threshold
       logical, optional, intent(in) :: shuffle_batches
       integer, optional, intent(in) :: batch_print_step
       integer, optional, intent(in) :: verbose
     end subroutine train

     !!-------------------------------------------------------------------------
     !! test the network
     !!-------------------------------------------------------------------------
     !! this        = (T, io) network type
     !! input       = (R, in) input data
     !! output      = (*, in) expected output data (data labels)
     !! addit_input = (R, in, opt) additional input data
     !! addit_layer = (I, in, opt) layer to insert additional input data
     !! verbose     = (I, in, opt) verbosity level
     module subroutine test(this, input, output, &
          verbose)
       class(network_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
       class(*), dimension(:,:), intent(in) :: output
       integer, optional, intent(in) :: verbose
     end subroutine test

     !!-------------------------------------------------------------------------
     !! return predicted results from supplied inputs using the trained network
     !!-------------------------------------------------------------------------
     !! this        = (T, in) network type
     !! input       = (R, in) input data
     !! addit_input = (R, in, opt) additional input data
     !! addit_layer = (I, in, opt) layer to insert additional input data
     !! verbose     = (I, in, opt) verbosity level
     !! output      = (R, out) predicted output data
     module function predict_1d(this, input, &
          verbose) result(output)
       class(network_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
       integer, optional, intent(in) :: verbose
       real(real32), dimension(:,:), allocatable :: output
     end function predict_1d

     !!-------------------------------------------------------------------------
     !! update the learnable parameters of the network based on gradients
     !!-------------------------------------------------------------------------
     !! this = (T, io) network type
     module subroutine update(this)
       class(network_type), intent(inout) :: this
     end subroutine update

     !!-------------------------------------------------------------------------
     !! get layer order
     !!-------------------------------------------------------------------------
     !! this  = (T, in) network type
     module subroutine generate_vertex_order(this)
       class(network_type), intent(inout) :: this
     end subroutine generate_vertex_order

     !!-------------------------------------------------------------------------
     !! depth first search
     !!-------------------------------------------------------------------------
     !! this  = (T, in) network type
     !! vertex_index = (I, in) vertex index
     !! visited = (L, in) visited vertices
     !! order = (I, io) order of vertices
     !! order_index = (I, io) index of order
     module recursive subroutine dfs( &
          this, vertex_index, visited, order, order_index &
     )
       class(network_type), intent(in) :: this
       integer, intent(in) :: vertex_index
       logical, dimension(this%auto_graph%num_vertices), intent(inout) :: visited
       integer, dimension(this%auto_graph%num_vertices), intent(inout) :: order
       integer, intent(inout) :: order_index
     end subroutine dfs

     !!-------------------------------------------------------------------------
     !! calculate root vertices
     !!-------------------------------------------------------------------------
     !! this = (T, in) network type
      module subroutine calculate_root_vertices(this)
        class(network_type), intent(inout) :: this
      end subroutine calculate_root_vertices

     !!-------------------------------------------------------------------------
     !! calculate output vertices
     !!-------------------------------------------------------------------------
     !! this = (T, in) network type
      module subroutine calculate_output_vertices(this)
        class(network_type), intent(inout) :: this
      end subroutine calculate_output_vertices

     !!-------------------------------------------------------------------------
     !! get the input of a layer via autodiff
     !!-------------------------------------------------------------------------
     pure module subroutine get_input_autodiff(this, idx, input)
       class(network_type), intent(in) :: this
       integer, intent(in) :: idx
       real(real32), allocatable, dimension(:,:), intent(out) :: input
     end subroutine get_input_autodiff

     !!-------------------------------------------------------------------------
     !! get the gradient of a layer via autodiff
     !!-------------------------------------------------------------------------
     pure module subroutine get_gradient_autodiff(this, idx, gradient)
       class(network_type), intent(in) :: this
       integer, intent(in) :: idx
       real(real32), allocatable, dimension(:,:), intent(out) :: gradient
     end subroutine get_gradient_autodiff

     !!-------------------------------------------------------------------------
     !! reduce two networks down to one (i.e. add two networks - parallel)
     !!-------------------------------------------------------------------------
     !! this   = (T, io) network type, resultant network of the reduction
     !! source = (T, in) network type
     module subroutine network_reduction(this, source)
       class(network_type), intent(inout) :: this
       type(network_type), intent(in) :: source
     end subroutine network_reduction

     !!-------------------------------------------------------------------------
     !! copy a network
     !!-------------------------------------------------------------------------
     !! this   = (T, io) network type, resultant network of the copy
     !! source = (T, in) network type
     module subroutine network_copy(this, source)
       class(network_type), intent(inout) :: this
       type(network_type), intent(in) :: source
     end subroutine network_copy

     !!-------------------------------------------------------------------------
     !! get number of learnable parameters in the network
     !!-------------------------------------------------------------------------
     !! this       = (T, in) network type
     !! num_params = (I, out) number of parameters
     pure module function get_num_params(this) result(num_params)
       class(network_type), intent(in) :: this
       integer :: num_params
     end function get_num_params

     !!-------------------------------------------------------------------------
     !! get learnable parameters
     !!-------------------------------------------------------------------------
     !! this   = (T, in) network type
     !! params = (R, out) learnable parameters
     pure module function get_params(this) result(params)
       class(network_type), intent(in) :: this
       real(real32), dimension(this%num_params) :: params
     end function get_params

     !!-------------------------------------------------------------------------
     !! set learnable parameters
     !!-------------------------------------------------------------------------
     !! this    = (T, io) network type
     !! params  = (R, in) learnable parameters
     !! verbose = (I, in, opt) verbosity level
     module subroutine set_params(this, params)
       class(network_type), intent(inout) :: this
       real(real32), dimension(this%num_params), intent(in) :: params
     end subroutine set_params

     !!-------------------------------------------------------------------------
     !! get gradients of learnable parameters
     !!-------------------------------------------------------------------------
     !! this      = (T, in) network type
     !! gradients = (R, out) gradients
     pure module function get_gradients(this) result(gradients)
       class(network_type), intent(in) :: this
       real(real32), dimension(this%num_params) :: gradients
     end function get_gradients

     !!-------------------------------------------------------------------------
     !! set learnable parameter gradients
     !!-------------------------------------------------------------------------
     !! this      = (T, io) network type
     !! gradients = (R, in) gradients
     !! verbose   = (I, in, opt) verbosity level
     module subroutine set_gradients(this, gradients)
       class(network_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: gradients
     end subroutine set_gradients

     !!-------------------------------------------------------------------------
     !! reset learnable parameter gradients
     !!-------------------------------------------------------------------------
     !! this    = (T, io) network type
     !! verbose = (I, in, opt) verbosity level
     !!-------------------------------------------------------------------------
     module subroutine reset_gradients(this)
       class(network_type), intent(inout) :: this
     end subroutine reset_gradients

     !!-------------------------------------------------------------------------
     !! forward pass
     !!-------------------------------------------------------------------------
     !! this        = (T, io) network type
     !! input       = (R, in) input data
     !! addit_input = (R, in, opt) additional input data
     !! layer       = (I, in, opt) layer to insert additional input data
     pure module subroutine forward_1d(this, input)
       class(network_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
     end subroutine forward_1d

     !!-------------------------------------------------------------------------
     !! backward pass
     !!-------------------------------------------------------------------------
     !! this        = (T, io) network type
     !! output      = (R, in) output data
     pure module subroutine backward_1d(this, output)
       class(network_type), intent(inout) :: this
       real(real32), dimension(:,:), intent(in) :: output
     end subroutine backward_1d
  end interface

end module network
!!!#############################################################################
