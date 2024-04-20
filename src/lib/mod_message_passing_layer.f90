!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a 1D convolutional layer
!!!#############################################################################
module mpnn_module
  use constants, only: real12
  use custom_types, only: initialiser_type, graph_type
  implicit none
  

  private

  public :: mpnn_type

  type :: mpnn_type
     real(real12), dimension(:), allocatable :: message
   contains
     procedure(message_function), pass(this), pointer :: message_function => null()
     procedure, pass(this) :: message_function
     procedure, pass(this) :: update_function
     procedure, pass(this) :: readout_function
     procedure, pass(this) :: forward
  end type mpnn_type

  
!!!-----------------------------------------------------------------------------
!!! interface for layer set up
!!!-----------------------------------------------------------------------------
  ! interface mpnn_type
  !    module function layer_setup( &
  !         input_shape, batch_size, &
  !         num_filters, kernel_size, stride, padding, &
  !         activation_function, activation_scale, &
  !         kernel_initialiser, bias_initialiser, &
  !         calc_input_gradients) result(layer)
  !      integer, dimension(:), optional, intent(in) :: input_shape
  !      integer, optional, intent(in) :: batch_size
  !      integer, optional, intent(in) :: num_filters
  !      integer, dimension(..), optional, intent(in) :: kernel_size
  !      integer, dimension(..), optional, intent(in) :: stride
  !      real(real12), optional, intent(in) :: activation_scale
  !      character(*), optional, intent(in) :: activation_function, &
  !           kernel_initialiser, bias_initialiser, padding
  !      logical, optional, intent(in) :: calc_input_gradients
  !      type(conv1d_layer_type) :: layer
  !    end function layer_setup
  ! end interface mpnn_type


  abstract interface
     !! compute the loss function
     !! predicted = (R, in) predicted values
     !! expected  = (R, in) expected values
     !! output    = (R, in) loss function
     pure subroutine message_function(this, graph)
       implicit none
       class(mpnn_type), intent(in) :: this
       type(graph_type), intent(in) :: graph
     end subroutine message_function
  end interface
  


contains

!!!#############################################################################
!!! message function
!!!#############################################################################
  pure subroutine convolutional_message_function(this, graph)
    implicit none
    class(mpnn_type), intent(in) :: this
    type(graph_type), intent(in) :: graph

    do i = 1, graph%num_nodes
       do j = 1, graph%num_nodes
          this%message(i,:) = this%message(i,:) + &
               [ graph%node(j)%feature, graph%edge(i,j)%feature ]
    this%message = graph%node()


  end subroutine convolutional_message_function


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward(this, graph)
    implicit none
    class(mpnn_type), intent(inout) :: this
    type(graph_type), intent(in) :: graph


    call this%message(graph)
    call this%update(graph)

    call this%readout()

  end subroutine forward
!!!#############################################################################


end module mpnn_module
!!!#############################################################################
