submodule(athena__msgpass_layer) athena__msgpass_layer_submodule
  !! Submodule containing implementations for a message passing layer
  implicit none



contains

!###############################################################################
  pure module function get_num_params_msgpass(this) result(num_params)
    !! Get the number of learnable parameters in the layer
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(in) :: this
    !! Instance of the layer type
    integer :: num_params
    !! Number of learnable parameters

    ! Local variables
    integer :: t
    !! Time step

    num_params = sum(this%num_params_msg) + this%num_params_readout
  end function get_num_params_msgpass
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       num_features, num_time_steps, &
       verbose &
  ) result(layer)
    !! Procedure to set up the layer
    implicit none

    ! Arguments
    integer, dimension(2), intent(in) :: num_features
    !! Number of features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    class(msgpass_layer_type), allocatable :: layer
    !! Instance of the layer type

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


  end function layer_setup
!###############################################################################


!###############################################################################
  ! module subroutine set_hyperparams_msgpass( &
  !      this, &
  !      num_features, num_time_steps, num_outputs, verbose &
  ! )
  !   !! Set the hyperparameters for the layer
  !   implicit none

  !   ! Arguments
  !   class(msgpass_layer_type), intent(inout) :: this
  !   !! Instance of the layer type
  !   integer, dimension(2), intent(in) :: num_features
  !   !! Number of features
  !   integer, intent(in) :: num_time_steps
  !   !! Number of time steps
  !   integer, intent(in) :: num_outputs
  !   !! Number of output features
  !   integer, optional, intent(in) :: verbose
  !   !! Verbosity level


  !   this%name = 'msgpass'
  !   this%type = 'msgp'
  !   this%input_rank = 1
  !   this%num_outputs = num_outputs
  !   this%num_time_steps = num_time_steps
  !   this%num_vertex_features = num_features(1)
  !   this%num_edge_features = num_features(2)

  ! end subroutine set_hyperparams_msgpass
!###############################################################################


!###############################################################################
  module subroutine init_msgpass(this, input_shape, verbose)
    !! Initialise the layer
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout) :: this
    !! Instance of the layer type
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Time step
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: num_params_message
    !! Number of parameters in the message
    integer :: num_params_readout
    !! Number of parameters in the readout


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    this%input_shape = [ 1 ] !input_shape
    ! this%output_shape = this%num_outputs
    !if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)

  end subroutine init_msgpass
!###############################################################################


!###############################################################################
  module subroutine set_graph_msgpass(this, graph)
    !! Set the graph structure of the input data
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout) :: this
    !! Instance of the layer
    type(graph_type), dimension(:), intent(in) :: graph
    !! Graph structure of input data

    ! Local variables
    integer :: s
    !! Loop indices

    if(allocated(this%graph))then
       if(size(this%graph).ne.size(graph))then
          deallocate(this%graph)
          allocate(this%graph(size(graph)))
       end if
    else
       allocate(this%graph(size(graph)))
    end if
    do s = 1, size(graph)
       this%graph(s)%adj_ia = graph(s)%adj_ia
       this%graph(s)%adj_ja = graph(s)%adj_ja
       this%graph(s)%edge_weights = graph(s)%edge_weights
       this%graph(s)%num_edges = graph(s)%num_edges
       this%graph(s)%num_vertices = graph(s)%num_vertices
    end do

  end subroutine set_graph_msgpass
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module subroutine forward_msgpass(this, input)
    !! Forward propagation for the layer
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout) :: this
    !! Instance of the layer type
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data (i.e. vertex and edge features)


    call this%update_message(input)
    call this%update_readout()

  end subroutine forward_msgpass
!###############################################################################

end submodule athena__msgpass_layer_submodule
