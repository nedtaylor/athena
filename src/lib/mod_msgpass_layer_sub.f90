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
!-------------------------------------------------------------------------------
  pure module function get_params_msgpass(this) result(params)
    !! Get the learnable parameters in the layer
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(in) :: this
    !! Instance of the layer type
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    params = this%params
  end function get_params_msgpass
!-------------------------------------------------------------------------------
  pure subroutine set_params_msgpass(this, params)
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout) :: this
    !! Instance of the layer type
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    ! Local variables
    integer :: t
    !! Time step
    integer :: istart, iend
    !! Start and end indices

    this%params = params
  end subroutine set_params_msgpass
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


  module subroutine set_param_pointers_msgpass(this)
    !! Set the pointers to the learnable parameters
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
  end subroutine set_param_pointers_msgpass
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
    ! Allocateparameters
    !---------------------------------------------------------------------------
    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(this%num_params), source=0._real32)
    if(allocated(this%dp)) deallocate(this%dp)
    allocate( &
         this%dp(sum(this%num_params_msg), this%batch_size), source=0._real32 &
    )
    if(allocated(this%db)) deallocate(this%db)
    allocate( &
         this%db(this%num_params_readout, this%batch_size), source=0._real32 &
    )


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_msgpass
!###############################################################################


!###############################################################################
  subroutine set_batch_size_msgpass(this, batch_size, verbose)
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
       ! if(.not.allocated(this%output)) this%output = array2d_type()
       ! if(this%output%allocated) call this%output%deallocate(keep_shape=.true.)
       ! call this%output%allocate( &
       !      [ &
       !           this%output%shape(1), &
       !           this%batch_size &
       !      ], &
       !      source=0._real32 &
       ! )
       ! call this%method%init( &
       !      this%num_vertex_features, this%num_edge_features, &
       !      this%num_time_steps, &
       !      this%output%shape, this%batch_size &
       ! )
       ! if(.not.allocated(this%di)) this%di = array2d_type()
       ! if(this%di%allocated) call this%di%deallocate()
       ! call this%di%allocate( &
       !      [1, this%batch_size], &
       !      source=0._real32 &
       ! )

       call this%set_param_pointers()
    end if

  end subroutine set_batch_size_msgpass
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

    if(this%use_graph_input)then
       if(allocated(this%output))then
          do s = 1, size(graph)
             if(this%output(1,s)%allocated) &
                  call this%output(1,s)%deallocate()
             if(this%output(2,s)%allocated) &
                  call this%output(2,s)%deallocate()
             if(this%di(1,s)%allocated) &
                  call this%di(1,s)%deallocate()
             if(this%di(2,s)%allocated) &
                  call this%di(2,s)%deallocate()
             call this%output(1,s)%allocate( &
                  [ &
                       this%num_output_vertex_features, &
                       this%graph(s)%num_vertices &
                  ] &
             )
             call this%output(2,s)%allocate( &
                  [ &
                       this%num_output_edge_features, &
                       this%graph(s)%num_vertices &
                  ] &
             )
             call this%di(1,s)%allocate( &
                  [ &
                       this%num_vertex_features, &
                       this%graph(s)%num_vertices &
                  ] &
             )
             call this%di(2,s)%allocate( &
                  [ &
                       this%num_edge_features, &
                       this%graph(s)%num_edges &
                  ] &
             )
          end do
       end if
       call this%set_ptrs()
    end if

    do s = 1, size(graph)
       if(this%vertex_features(0,s)%allocated) &
            call this%vertex_features(0,s)%deallocate()
       if(this%edge_features(0,s)%allocated) &
            call this%edge_features(0,s)%deallocate()
       call this%vertex_features(0,s)%allocate( &
            [ this%num_vertex_features, this%graph(s)%num_vertices ] &
       )
       call this%edge_features(0,s)%allocate( &
            [ this%num_edge_features, this%graph(s)%num_edges ] &
       )
       do t = 1, this%num_time_steps
          if(this%vertex_features(t,s)%allocated) &
               call this%vertex_features(t,s)%deallocate()
          if(this%edge_features(t,s)%allocated) &
               call this%edge_features(t,s)%deallocate()
          if(this%message(t,s)%allocated) &
               call this%message(t,s)%deallocate()
          if(this%z(t,s)%allocated) &
               call this%z(t,s)%deallocate()
          call this%vertex_features(t,s)%allocate( &
               [ this%num_vertex_features(t), this%graph(s)%num_vertices ] &
          )
          call this%edge_features(t,s)%allocate( &
               [ this%num_edge_features(t), this%graph(s)%num_edges ] &
          )
          call this%message(t,s)%allocate( &
               [ this%num_vertex_features(t-1), this%graph(s)%num_vertices ] &
          )
          call this%z(t,s)%allocate( &
               [ this%num_vertex_features(t), this%graph(s)%num_vertices ] &
          )
       end do
    end do


  end subroutine set_graph_msgpass

!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure module subroutine forward_derived_msgpass(this, input)
    !! Forward propagation for the layer
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout) :: this
    !! Instance of the layer type
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data (i.e. vertex and edge features)

    call this%update_message(input)
    call this%update_readout()

  end subroutine forward_derived_msgpass
!###############################################################################


!###############################################################################
  pure module subroutine backward_derived_msgpass(this, input, gradient)
    !! Backward propagation for the layer
    implicit none

    ! Arguments
    class(msgpass_layer_type), intent(inout) :: this
    !! Instance of the layer type
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data (i.e. vertex and edge features)
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient data

    ! Local variables
    integer :: s, t
    !! Batch index, time step


    ! df/dv_c = h(M_c) * df/dM_y

    ! M_y = sum_c v_c * h(M_c)     message for output y
    ! h()                          hidden function


    this%dp = 0._real32
    if(allocated(this%db)) this%db = 0._real32
    call this%backward_readout(gradient)
    call this%backward_message(input, gradient)

  end subroutine backward_derived_msgpass
!###############################################################################

end submodule athena__msgpass_layer_submodule
