submodule(athena__mpnn_layer) athena__mpnn_layer_submodule
  !! Submodule containing implementations for a message passing layer
  use athena__misc_types, only: array2d_type
  implicit none



contains

!###############################################################################
  elemental module function feature_add(a, b) result(output)
    !! Procedure to add two features

    ! Arguments
    class(feature_type), intent(in) :: a, b
    !! Instances of the feature type
    type(feature_type) :: output
    !! Output feature

    !allocate(output%val(size(a%val,1), size(a%val,2)))
    output%val = a%val + b%val
  end function feature_add

  elemental module function feature_multiply(a, b) result(output)
    !! Procedure to multiply two features

    ! Arguments
    class(feature_type), intent(in) :: a, b
    !! Instances of the feature type
    type(feature_type) :: output
    !! Output feature

    !allocate(output%val(size(a%val,1), size(a%val,2)))
    output%val = a%val * b%val
  end function feature_multiply
!###############################################################################


!###############################################################################
  module subroutine layer_reduction(this, rhs)
    !! Procedure to reduce two layers
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    class(learnable_layer_type), intent(in) :: rhs
    !! Instance of the learnable layer type

    !! NOT YET IMPLEMENTED
  end subroutine layer_reduction

  module subroutine layer_merge(this, input)
    !! Procedure to merge two layers
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    class(learnable_layer_type), intent(in) :: input
    !! Instance of the learnable layer type

    !! NOT YET IMPLEMENTED
  end subroutine layer_merge
!###############################################################################


!############################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!############################################################################!!!



!###############################################################################
  pure module function get_num_params_mpnn(this) result(num_params)
    !! Get the number of learnable parameters in the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(in) :: this
    !! Instance of the layer type
    integer :: num_params
    !! Number of learnable parameters

    ! Local variables
    integer :: t
    !! Time step

    num_params = 0
    do t = 1, this%method%num_time_steps
       num_params = num_params + this%method%message(t)%get_num_params()
    end do
    num_params = num_params + this%method%readout%get_num_params()
  end function get_num_params_mpnn
!-------------------------------------------------------------------------------
  pure module function get_params_mpnn(this) result(params)
    !! Get the learnable parameters in the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(in) :: this
    !! Instance of the layer type
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    ! Local variables
    integer :: t
    !! Time step
    integer :: istart, iend
    !! Start and end indices

    istart = 1
    do t = 1, this%method%num_time_steps
       iend = istart + this%method%message(t)%num_params - 1
       params(istart:iend) = this%method%message(t)%get_params()
       istart = iend + 1
    end do
    params(istart:) = this%method%readout%get_params()
  end function get_params_mpnn
!-------------------------------------------------------------------------------
  pure subroutine set_params_mpnn(this, params)
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    ! Local variables
    integer :: t
    !! Time step
    integer :: istart, iend
    !! Start and end indices

    istart = 1
    do t = 1, this%method%num_time_steps
       iend = istart + this%method%message(t)%num_params - 1
       if(iend.gt.istart-1) &
            call this%method%message(t)%set_params(params(istart:iend))
       istart = iend + 1
    end do
    iend = istart + this%method%readout%num_params - 1
    if(iend.gt.istart-1) &
         call this%method%readout%set_params(params(istart:iend))

  end subroutine set_params_mpnn
!###############################################################################


!###############################################################################
  pure module function get_gradients_mpnn(this, clip_method) result(gradients)
    !! Get the gradients of the learnable parameters in the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(in) :: this
    !! Instance of the layer type
    type(clip_type), optional, intent(in) :: clip_method
    !! Instance of the clipping type
    real(real32), dimension(this%num_params) :: gradients
    !! Gradients of the learnable parameters

    ! Local variables
    integer :: t
    !! Time step
    integer :: istart, iend
    !! Start and end indices

    istart = 1
    if(present(clip_method))then
       do t = 1, this%method%num_time_steps
          iend = istart + this%method%message(t)%num_params - 1
          gradients(istart:iend) = &
               this%method%message(t)%get_gradients(clip_method)
          istart = iend + 1
       end do
       gradients(istart:) = this%method%readout%get_gradients(clip_method)
    else
       do t = 1, this%method%num_time_steps
          iend = istart + this%method%message(t)%num_params - 1
          gradients(istart:iend) = &
               this%method%message(t)%get_gradients()
          istart = iend + 1
       end do
       gradients(istart:) = this%method%readout%get_gradients()
    end if
  end function get_gradients_mpnn
!-------------------------------------------------------------------------------
  pure module subroutine set_gradients_mpnn(this, gradients)
    !! Set the gradients of the learnable parameters in the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients of the learnable parameters

    ! Local variables
    integer :: t
    !! Time step

    do t = 1, this%method%num_time_steps
       call this%method%message(t)%set_gradients(gradients)
    end do
    call this%method%readout%set_gradients(gradients)
  end subroutine set_gradients_mpnn
!###############################################################################


!###############################################################################
  pure module function get_phase_num_params(this) result(num_params)
    !! Get the number of parameters in the phase
    implicit none

    ! Arguments
    class(base_phase_type), intent(in) :: this
    !! Instance of the base phase type
    integer :: num_params
    !! Number of parameters

    num_params = 0
  end function get_phase_num_params
!-------------------------------------------------------------------------------
  pure module function get_phase_params(this) result(params)
    !! Get the parameters in the phase
    implicit none

    ! Arguments
    class(base_phase_type), intent(in) :: this
    !! Instance of the base phase type
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    ! Nothing to do as no parameters in base phase type
  end function get_phase_params
!-------------------------------------------------------------------------------
  pure module subroutine set_phase_params(this, params)
    !! Set the parameters in the phase
    implicit none

    ! Arguments
    class(base_phase_type), intent(inout) :: this
    !! Instance of the base phase type
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    ! Nothing to do as no parameters in base phase type
  end subroutine set_phase_params
!###############################################################################


!###############################################################################
  pure module function get_phase_gradients(this, clip_method) result(gradients)
    !! Get the gradients of the parameters in the base phase
    implicit none

    ! Arguments
    class(base_phase_type), intent(in) :: this
    !! Instance of the base phase type
    type(clip_type), optional, intent(in) :: clip_method
    !! Instance of the clipping type
    real(real32), dimension(this%num_params) :: gradients
    !! Gradients of the parameters

    ! Nothing to do as no parameters in base phase type
  end function get_phase_gradients
!-------------------------------------------------------------------------------
  pure module subroutine set_phase_gradients(this, gradients)
    !! Set the gradients of the parameters in the base phase
    implicit none

    ! Arguments
    class(base_phase_type), intent(inout) :: this
    !! Instance of the base phase type
    real(real32), dimension(..), intent(in) :: gradients

    ! Nothing to do as no parameters in base phase type
  end subroutine set_phase_gradients
!-------------------------------------------------------------------------------
  module subroutine set_phase_shape(this, shape)
    !! Set the shape of the phase
    implicit none

    ! Arguments
    class(base_phase_type), intent(inout) :: this
    !! Instance of the base phase type
    integer, dimension(:), intent(in) :: shape
    !! Shape of the phase

    ! Local variables
    integer :: s
    !! Batch index

    do s = 1, this%batch_size
       if(allocated(this%feature(s)%val)) deallocate(this%feature(s)%val)
       allocate(this%feature(s)%val(this%num_outputs, shape(s)))
    end do

  end subroutine set_phase_shape
!###############################################################################


!############################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!############################################################################!!!



!###############################################################################
  pure module subroutine forward_rank(this, input)
    !! Forward propagation for the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    real(real32), dimension(..), intent(in) :: input
    !! Input to the layer

    call forward_graph(this, this%graph)
  end subroutine forward_rank
!###############################################################################


!###############################################################################
  pure subroutine backward_rank(this, input, gradient)
    !! Backward propagation for the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    real(real32), dimension(..), intent(in) :: input
    !! Input to the layer
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient of the layer

    select rank(gradient); rank(2)
       call backward_graph(this, this%graph, gradient)
    end select
  end subroutine backward_rank
!###############################################################################


!############################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!############################################################################!!!


!###############################################################################
  module function layer_setup( &
       method, &
       num_features, num_time_steps, num_outputs, batch_size, &
       verbose &
  ) result(layer)
    !! Procedure to set up the layer
    implicit none

    ! Arguments
    class(method_container_type), intent(in) :: method
    !! Instance of the method container type
    integer, dimension(2), intent(in) :: num_features
    !! Number of features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, intent(in) :: num_outputs
    !! Number of output features
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(mpnn_layer_type) :: layer
    !! Instance of the layer type

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    if(present(verbose)) verbose_ = verbose

    !!--------------------------------------------------------------------------
    !! set hyperparameters
    !!--------------------------------------------------------------------------
    call layer%set_hyperparams( &
         method = method, &
         num_features = num_features, &
         num_time_steps = num_time_steps, &
         num_outputs = num_outputs, &
         verbose = verbose_ &
    )

    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if (present(batch_size)) then
       layer%batch_size = batch_size
    else
       layer%batch_size = 1
    end if


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    call layer%init( &
         input_shape = [ num_features(1), num_features(2), num_time_steps ], &
         batch_size = layer%batch_size &
    )

  end function layer_setup
!###############################################################################


!###############################################################################
  module subroutine set_hyperparameters_mpnn( &
       this, method, &
       num_features, num_time_steps, num_outputs, verbose &
  )
    !! Set the hyperparameters for the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    class(method_container_type), intent(in) :: method
    !! Instance of the method container type
    integer, dimension(2), intent(in) :: num_features
    !! Number of features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, intent(in) :: num_outputs
    !! Number of output features
    integer, optional, intent(in) :: verbose
    !! Verbosity level


    this%name = 'mpnn'
    this%type = 'mpnn'
    this%input_rank = 1
    this%num_outputs = num_outputs
    this%num_time_steps = num_time_steps
    this%num_vertex_features = num_features(1)
    this%num_edge_features = num_features(2)
    allocate(this%method, source=method)

  end subroutine set_hyperparameters_mpnn
!###############################################################################


!###############################################################################
  module subroutine init_mpnn(this, input_shape, batch_size, verbose)
    !! Initialise the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%input_shape = input_shape
    if(allocated(this%output))then
       if(this%output%allocated) call this%output%deallocate()
    end if
    this%output = array2d_type()
    this%output%shape = this%num_outputs
    !if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_params = this%get_num_params()

    if (present(batch_size)) this%batch_size = batch_size

    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_mpnn
!###############################################################################


!###############################################################################
  subroutine set_batch_size_mpnn(this, batch_size, verbose)
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout), target :: this
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
       if(.not.allocated(this%output)) this%output = array2d_type()
       if(this%output%allocated) call this%output%deallocate(keep_shape=.true.)
       call this%output%allocate( &
            [ &
                 this%output%shape(1), &
                 this%batch_size &
            ], &
            source=0._real32 &
       )
       call this%method%init( &
            this%num_vertex_features, this%num_edge_features, &
            this%num_time_steps, &
            this%output%shape, this%batch_size &
       )
    end if

  end subroutine set_batch_size_mpnn
!###############################################################################


!###############################################################################
  module subroutine set_graph(this, graph)
    !! Set the graph for the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    !! Graph to copy to the layer

    ! Local variables
    integer :: v, s, t
    !! Vertex, batch index, time step
    integer, dimension(:), allocatable :: shape
    !! Shape of the graph

    if(allocated(this%graph)) deallocate(this%graph)
    this%graph = graph

    shape = graph(:)%num_vertices

    do t = 0, this%method%num_time_steps, 1
       call this%method%message(t)%set_shape(shape)
       if(t.eq.0)then
          do s = 1, this%batch_size
             do v = 1, graph(s)%num_vertices
                this%method%message(0)%feature(s)%val(:,v) = &
                     graph(s)%vertex(v)%feature
             end do
          end do
       end if
    end do

    call this%method%readout%set_shape(shape)

  end subroutine set_graph
!###############################################################################


!############################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!############################################################################!!!


!###############################################################################
  subroutine print_mpnn(this, file)
    !! Print the layer to a file
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(in) :: this
    !! Instance of the layer type
    character(*), intent(in) :: file
    !! Filename

    ! Local variables
    integer :: i, unit
    !! Unit number


    ! Open file with new unit
    !---------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'("MPNN")')
    write(unit,'(3X,"NUM_FEATURES = ",2(1X,I0))') &
         this%num_vertex_features, this%num_edge_features
    write(unit,'(3X,"NUM_TIME_STEPS = ",I0)') this%num_time_steps
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%output%shape(1)
    write(unit,'(3X,"METHOD = ",A)') trim(this%name)
    ! MIGHT CHANGE TO this%method%name


    ! Write fully connected weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("PHASES")')
    write(unit,'(" MESSAGE")')
    !!! NEED TO WRITE MESSAGE LAYERS
    write(unit,'(" END MESSAGE")')
    write(unit,'(" READOUT")')
    !!! NEED TO WRITE MESSAGE LAYERS
    write(unit,'(" END READOUT")')
    write(unit,'("END PHASES")')
    write(unit,'("END MPNN")')

    ! Close unit
    !---------------------------------------------------------------------------
    close(unit)

  end subroutine print_mpnn
!###############################################################################


!###############################################################################
  module subroutine read_mpnn(this, unit, verbose)
    !! Read the layer from a file
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Read layer from file
    !---------------------------------------------------------------------------
    ! call this%method%read(unit, verbose)
    write(0,*) "NOT YET IMPLEMENTED"
    stop 1

  end subroutine read_mpnn
!###############################################################################


!############################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!############################################################################!!!


!###############################################################################
  pure module subroutine forward_graph(this, graph)
    !! Forward propagation for the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    !! Graph to forward propagate

    ! Local variables
    integer :: v, s, t
    !! Vertex, batch index, time step

    do t = 1, this%method%num_time_steps, 1
       call this%method%message(t)%update( &
            this%method%message(t-1)%feature, &
            graph &
       )
    end do

    select type(output => this%output)
    type is (array2d_type)
       call this%method%readout%get_output( &
            this%method%message, output%val &
       )
    end select

  end subroutine forward_graph
!###############################################################################


!###############################################################################
  pure module subroutine backward_graph(this, graph, gradient)
    !! Backward propagation for the layer
    implicit none

    ! Arguments
    class(mpnn_layer_type), intent(inout) :: this
    !! Instance of the layer type
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    !! Graph to backward propagate
    real(real32), dimension( &
         this%output%shape(1), &
         this%batch_size &
    ), intent(in) :: gradient
    !! Gradient values

    ! Local variables
    integer :: s, t
    !! Batch index, time step


    ! df/dv_c = h(M_c) * df/dM_y

    ! M_y = sum_c v_c * h(M_c)     message for output y
    ! h()                          hidden function

    call this%method%readout%calculate_partials( &
         input = this%method%message, &
         gradient = gradient &
    )

    call this%method%message(this%method%num_time_steps)%calculate_partials( &
         input = this%method%message(this%method%num_time_steps-1)%feature, &
         gradient = this%method%readout%di, &
         graph = graph &
    )

    do t = this%method%num_time_steps - 1, 1, -1
       call this%method%message(t)%calculate_partials( &
            input = this%method%message(t-1)%feature, &
            gradient = this%method%message(t+1)%di, &
            graph = graph &
       )
    end do

  end subroutine backward_graph
!###############################################################################

end submodule athena__mpnn_layer_submodule
