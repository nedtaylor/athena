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
       num_features, num_time_steps, batch_size, &
       verbose &
  ) result(layer)
    !! Procedure to set up the layer
    implicit none

    ! Arguments
    integer, dimension(2), intent(in) :: num_features
    !! Number of features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, optional, intent(in) :: batch_size
    !! Batch size
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
  module subroutine init_msgpass(this, input_shape, batch_size, verbose)
    !! Initialise the layer
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout) :: this
    !! Instance of the layer type
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Time step
    integer :: num_params_message
    !! Number of parameters in the message
    integer :: num_params_readout
    !! Number of parameters in the readout


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if (present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    this%input_shape = [ 1 ] !input_shape
    ! if(allocated(this%output))then
    !    if(this%output%allocated) call this%output%deallocate()
    ! end if
    ! this%output = array2d_type()
    ! this%output_shape = this%num_outputs
    !if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_msgpass
!###############################################################################


!###############################################################################
  module subroutine set_batch_size_msgpass(this, batch_size, verbose)
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout), target :: this
    !! Instance of the layer type
    integer, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       allocate(this%output(2, this%batch_size))
    end if

  end subroutine set_batch_size_msgpass
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
    integer :: s, t
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

!     if(this%use_graph_input)then
!        if(allocated(this%output))then
!           do s = 1, size(graph)
!              if(this%output(1,s)%allocated) &
!                   call this%output(1,s)%deallocate()
!              if(this%output(2,s)%allocated) &
!                   call this%output(2,s)%deallocate()
!              call this%output(1,s)%allocate( &
!                   [ &
!                        this%num_output_vertex_features, &
!                        this%graph(s)%num_vertices &
!                   ] &
!              )
!              call this%output(2,s)%allocate( &
!                   [ &
!                        this%num_output_edge_features, &
!                        this%graph(s)%num_vertices &
!                   ] &
!              )
!           end do
!        end if
!     end if

!     do s = 1, size(graph)
!        if(this%vertex_features(0,s)%allocated) &
!             call this%vertex_features(0,s)%deallocate()
!        if(this%edge_features(0,s)%allocated) &
!             call this%edge_features(0,s)%deallocate()
!        call this%vertex_features(0,s)%allocate( &
!             [ this%num_vertex_features(0), this%graph(s)%num_vertices ] &
!        )
!        call this%edge_features(0,s)%allocate( &
!             [ this%num_edge_features(0), this%graph(s)%num_edges ] &
!        )
!        do t = 1, this%num_time_steps
!           if(this%vertex_features(t,s)%allocated) &
!                call this%vertex_features(t,s)%deallocate()
!           if(this%edge_features(t,s)%allocated) &
!                call this%edge_features(t,s)%deallocate()
!           if(this%message(t,s)%allocated) &
!                call this%message(t,s)%deallocate()
!           if(this%z(t,s)%allocated) &
!                call this%z(t,s)%deallocate()
!           call this%vertex_features(t,s)%allocate( &
!                [ this%num_vertex_features(t), this%graph(s)%num_vertices ] &
!           )
!           call this%edge_features(t,s)%allocate( &
!                [ this%num_edge_features(t), this%graph(s)%num_edges ] &
!           )
!           call this%message(t,s)%allocate( &
!                [ this%num_vertex_features(t-1), this%graph(s)%num_vertices ] &
!           )
!           call this%z(t,s)%allocate( &
!                [ this%num_vertex_features(t), this%graph(s)%num_vertices ] &
!           )
!        end do
!     end do


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
