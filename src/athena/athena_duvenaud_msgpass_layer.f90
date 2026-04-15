module athena__duvenaud_msgpass_layer
  !! Module implementing Duvenaud message passing for molecular graphs
  !!
  !! This module implements the graph neural network architecture from
  !! Duvenaud et al. (2015) for learning on molecular graphs with both
  !! vertex (node) and edge features.
  !!
  !! Mathematical operation (per time step t):
  !! \[ h_v^{(t+1)} = \sigma\left( h_v^{(t)} + \sum_{u \in \mathcal{N}(v)} M(h_v^{(t)}, h_u^{(t)}, e_{vu}) \right) \]
  !!
  !! Graph readout (aggregation to fixed-size vector):
  !! \[ h_{\text{graph}} = \sigma_{\text{readout}}\left( \sum_{d=1}^D \sum_{v:\deg(v)=d} W_d h_v^{(T)} \right) \]
  !!
  !! where \( M \) is a learned message function, \( \sigma \) is activation function,
  !! \( \mathcal{N}(v) \) are neighbors of \( v \), \( e_{vu} \) are edge features, \( W_d \) are
  !! degree-specific weight matrices, and \( D \) is max vertex degree.
  !!
  !! Reference: Duvenaud et al. (2015), NeurIPS
  use coreutils, only: real32
  use graphstruc, only: graph_type
  use athena__misc_types, only: base_actv_type, base_init_type, onnx_attribute_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use diffstruc, only: array_type, sum, matmul, operator(+)
  use athena__base_layer, only: base_layer_type
  use athena__msgpass_layer, only: msgpass_layer_type
  use athena__diffstruc_extd, only: duvenaud_propagate, duvenaud_update
  implicit none


  private

  public :: duvenaud_msgpass_layer_type
  public :: read_duvenaud_msgpass_layer


!-------------------------------------------------------------------------------
! Message passing layer
!-------------------------------------------------------------------------------
  type, extends(msgpass_layer_type) :: duvenaud_msgpass_layer_type

     integer :: min_vertex_degree = 1
     integer :: max_vertex_degree = 0
     !! Maximum vertex degree

     class(base_actv_type), allocatable :: activation_readout
     !! Activation function
     type(array_type), allocatable, dimension(:,:) :: z
     type(array_type), allocatable, dimension(:,:) :: z_readout
     !! Input gradients

   contains
     procedure, pass(this) :: get_num_params => get_num_params_duvenaud
     !! Get the number of parameters for the message passing layer
     procedure, pass(this) :: get_attributes => get_attributes_duvenaud
     !! Get the attributes of the layer (for ONNX export)
     procedure, pass(this) :: set_hyperparams => set_hyperparams_duvenaud
     !! Set the hyperparameters for the message passing layer
     procedure, pass(this) :: init => init_duvenaud
     !! Initialise the message passing layer
     procedure, pass(this) :: print_to_unit => print_to_unit_duvenaud
     ! !! Print the message passing layer
     procedure, pass(this) :: read => read_duvenaud
     !! Read the message passing layer

     procedure, pass(this) :: set_graph => set_graph_duvenaud
     !! Set the graph for the message passing layer

     procedure, pass(this) :: update_message => update_message_duvenaud
     !! Update the message

     procedure, pass(this) :: update_readout => update_readout_duvenaud
     !! Update the readout

     procedure, pass(this) :: emit_onnx_nodes => emit_onnx_nodes_duvenaud
     !! Emit ONNX JSON nodes for Duvenaud GNN layer
     procedure, pass(this) :: emit_onnx_graph_inputs => &
          emit_onnx_graph_inputs_duvenaud
     !! Emit graph input tensor declarations for Duvenaud GNN layer

     final :: finalise_duvenaud
     !! Finalise the message passing layer
  end type duvenaud_msgpass_layer_type

  ! Interface for setting up the MPNN layer
  !-----------------------------------------------------------------------------
  interface duvenaud_msgpass_layer_type
     !! Interface for setting up the MPNN layer
     module function layer_setup( &
          num_vertex_features, num_edge_features, num_time_steps, &
          max_vertex_degree, &
          num_outputs, &
          min_vertex_degree, &
          message_activation, &
          readout_activation, &
          kernel_initialiser, &
          verbose &
     ) result(layer)
       !! Set up the message passing layer
       integer, dimension(:), intent(in) :: num_vertex_features
       !! Number of vertex features
       integer, dimension(:), intent(in) :: num_edge_features
       !! Number of edge features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, intent(in) :: max_vertex_degree
       !! Maximum vertex degree
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: min_vertex_degree
       !! Minimum vertex degree
       class(*), optional, intent(in) :: message_activation, &
            readout_activation
       !! Message and readout activation functions
       character(*), optional, intent(in) :: kernel_initialiser
       !!! Kernel initialiser
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(duvenaud_msgpass_layer_type) :: layer
       !! Instance of the message passing layer
     end function layer_setup
  end interface duvenaud_msgpass_layer_type

  character(len=*), parameter :: default_message_actv_name = "sigmoid"
  character(len=*), parameter :: default_readout_actv_name = "softmax"



contains

!###############################################################################
  function get_attributes_duvenaud(this) result(attributes)
    !! Get the attributes of the Duvenaud message passing layer (for ONNX export)
    !!
    !! Exports hyperparameters needed to reconstruct the layer architecture:
    !!   - num_time_steps: number of message passing iterations
    !!   - min_vertex_degree, max_vertex_degree: degree bucket range
    !!   - num_vertex_features: vertex feature dimensions per time step
    !!   - num_edge_features: edge feature dimensions per time step
    !!   - num_outputs: readout output dimension
    !!   - message_activation, readout_activation: activation function names
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(in) :: this
    !! Instance of the layer
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Attributes of the layer

    ! Local variables
    integer :: t
    !! Time-step index
    character(256) :: buffer
    !! Buffer for integer-to-string conversion

    allocate(attributes(7))

    write(buffer, '(I0)') this%num_time_steps
    attributes(1) = onnx_attribute_type( &
         name='num_time_steps', type='int', val=trim(buffer))

    write(buffer, '(I0)') this%min_vertex_degree
    attributes(2) = onnx_attribute_type( &
         name='min_vertex_degree', type='int', val=trim(buffer))

    write(buffer, '(I0)') this%max_vertex_degree
    attributes(3) = onnx_attribute_type( &
         name='max_vertex_degree', type='int', val=trim(buffer))

    buffer = ''
    do t = 0, this%num_time_steps
       if(t .eq. 0)then
          write(buffer, '(I0)') this%num_vertex_features(t)
       else
          write(buffer, '(A," ",I0)') trim(buffer), this%num_vertex_features(t)
       end if
    end do
    attributes(4) = onnx_attribute_type( &
         name='num_vertex_features', type='ints', val=trim(buffer))

    buffer = ''
    do t = 0, this%num_time_steps
       if(t .eq. 0)then
          write(buffer, '(I0)') this%num_edge_features(t)
       else
          write(buffer, '(A," ",I0)') trim(buffer), this%num_edge_features(t)
       end if
    end do
    attributes(5) = onnx_attribute_type( &
         name='num_edge_features', type='ints', val=trim(buffer))

    write(buffer, '(I0)') this%num_outputs
    attributes(6) = onnx_attribute_type( &
         name='num_outputs', type='int', val=trim(buffer))

    attributes(7) = onnx_attribute_type( &
         name='message_activation', type='string', &
         val=trim(this%activation%name))

  end function get_attributes_duvenaud
!###############################################################################


!###############################################################################
  subroutine finalise_duvenaud(this)
    !! Finalise the message passing layer
    implicit none

    ! Arguments
    type(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output_shape)) deallocate(this%output_shape)
    if(allocated(this%output)) deallocate(this%output)

  end subroutine finalise_duvenaud
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure function get_num_params_duvenaud(this) result(num_params)
    !! Get the number of parameters for the message passing layer
    !!
    !! This function calculates the number of parameters for the message passing
    !! layer.
    !! This procedure is based on code from the neural-fortran library
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(in) :: this
    !! Instance of the message passing layer
    integer :: num_params
    !! Number of parameters

    num_params = ( this%num_vertex_features(0) + this%num_edge_features(0) ) * &
         this%num_vertex_features(0) * &
         ( this%max_vertex_degree - this%min_vertex_degree + 1 ) * &
         this%num_time_steps + &
         this%num_vertex_features(0) * this%num_outputs * this%num_time_steps

  end function get_num_params_duvenaud
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       num_vertex_features ,num_edge_features, num_time_steps, &
       max_vertex_degree, &
       num_outputs, &
       min_vertex_degree, &
       message_activation, &
       readout_activation, &
       kernel_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the message passing layer
    use athena__initialiser, only: initialiser_setup
    use athena__activation, only: activation_setup
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, dimension(:), intent(in) :: num_edge_features
    !! Number of edge features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, intent(in) :: max_vertex_degree
    !! Maximum vertex degree
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: min_vertex_degree
    !! Minimum vertex degree
    class(*), optional, intent(in) :: message_activation, &
         readout_activation
    !! Message and readout activation functions
    character(*), optional, intent(in) :: kernel_initialiser
    !!! Kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    type(duvenaud_msgpass_layer_type) :: layer
    !! Instance of the message passing layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    class(base_actv_type), allocatable :: message_activation_ , readout_activation_
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser_
    !! Kernel and bias initialisers
    integer :: min_vertex_degree_ = 1
    !! Minimum vertex degree

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation functions
    !---------------------------------------------------------------------------
    if(present(message_activation))then
       message_activation_ = activation_setup(message_activation)
    else
       message_activation_ = activation_setup(default_message_actv_name)
    end if
    if(present(readout_activation))then
       readout_activation_ = activation_setup(readout_activation)
    else
       readout_activation_ = activation_setup(default_readout_actv_name)
    end if


    !---------------------------------------------------------------------------
    ! Set minimum vertex degree
    !---------------------------------------------------------------------------
    if(present(min_vertex_degree)) min_vertex_degree_ = min_vertex_degree
    if(max_vertex_degree.lt.min_vertex_degree_)then
       write(0,*) "Error: max_vertex_degree < min_vertex_degree"
       return
    end if


    !---------------------------------------------------------------------------
    ! Define weights (kernels) and biases initialisers
    !---------------------------------------------------------------------------
    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_vertex_features = num_vertex_features, &
         num_edge_features = num_edge_features, &
         min_vertex_degree = min_vertex_degree_, &
         max_vertex_degree = max_vertex_degree, &
         num_time_steps = num_time_steps, &
         num_outputs = num_outputs, &
         message_activation = message_activation_, &
         readout_activation = readout_activation_, &
         kernel_initialiser = kernel_initialiser_, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    call layer%init(input_shape=[ &
         layer%num_vertex_features(0), &
         layer%num_edge_features(0) &
    ])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_duvenaud( &
       this, &
       num_vertex_features, num_edge_features, &
       min_vertex_degree, &
       max_vertex_degree, &
       num_time_steps, &
       num_outputs, &
       message_activation, &
       readout_activation, &
       kernel_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for the message passing layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, dimension(:), intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, dimension(:), intent(in) :: num_edge_features
    !! Number of edge features
    integer, intent(in) :: min_vertex_degree
    !! Minimum vertex degree
    integer, intent(in) :: max_vertex_degree
    !! Maximum vertex degree
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, intent(in) :: num_outputs
    !! Number of outputs
    class(base_actv_type), allocatable, intent(in) :: &
         message_activation, &
         readout_activation
    !! Message and readout activation functions
    class(base_init_type), allocatable, intent(in) :: kernel_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Loop index
    character(len=256) :: buffer


    this%name = 'duvenaud'
    this%type = 'msgp'
    this%input_rank = 2
    this%output_rank = 1
    this%min_vertex_degree = min_vertex_degree
    this%max_vertex_degree = max_vertex_degree
    this%num_time_steps = num_time_steps
    this%num_outputs = num_outputs
    if(allocated(this%num_vertex_features)) &
         deallocate(this%num_vertex_features)
    if(allocated(this%num_edge_features)) &
         deallocate(this%num_edge_features)
    if(size(num_vertex_features, 1) .eq. 1)then
       allocate( &
            this%num_vertex_features(0:num_time_steps), &
            source = num_vertex_features(1) &
       )
    elseif(size(num_vertex_features, 1) .eq. num_time_steps + 1)then
       allocate( &
            this%num_vertex_features(0:this%num_time_steps), &
            source = num_vertex_features &
       )
    else
       write(*,*) "Error: num_vertex_features must be a scalar or a vector of &
            &length num_time_steps + 1"
       stop
    end if
    if(size(num_edge_features, 1) .eq. 1)then
       allocate( &
            this%num_edge_features(0:num_time_steps), &
            source = num_edge_features(1) &
       )
    elseif(size(num_edge_features, 1) .eq. num_time_steps + 1)then
       allocate( &
            this%num_edge_features(0:this%num_time_steps), &
            source = num_edge_features &
       )
    else
       write(*,*) "Error: num_edge_features must be a scalar or a vector of &
            &length num_time_steps + 1"
       stop
    end if
    this%use_graph_input = .true.
    this%use_graph_output = .false.
    if(allocated(this%activation)) deallocate(this%activation)
    if(allocated(this%activation_readout)) deallocate(this%activation_readout)
    if(.not.allocated(message_activation))then
       this%activation = activation_setup(default_message_actv_name)
    else
       allocate( this%activation, source=message_activation )
    end if
    if(.not.allocated(readout_activation))then
       this%activation_readout = activation_setup(default_readout_actv_name)
    else
       allocate(this%activation_readout, source=readout_activation)
    end if
    if(allocated(this%kernel_init)) deallocate(this%kernel_init)
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(this%activation%name)
       this%kernel_init = initialiser_setup(buffer)
    else
       allocate(this%kernel_init, source=kernel_initialiser)
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("DUVENAUD message activation function: ",A)') &
               trim(this%activation%name)
          write(*,'("DUVENAUD readout activation function: ",A)') &
               trim(this%activation_readout%name)
          write(*,'("DUVENAUD kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
       end if
    end if

    if(allocated(this%num_params_msg)) deallocate(this%num_params_msg)
    allocate(this%num_params_msg(1:this%num_time_steps))
    do t = 1, this%num_time_steps
       this%num_params_msg(t) = &
            ( this%num_vertex_features(t-1) + this%num_edge_features(0) ) * &
            this%num_vertex_features(t) * &
            ( this%max_vertex_degree - this%min_vertex_degree + 1 )
    end do
    this%num_params_readout = &
         sum( this%num_vertex_features * this%num_outputs )

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output_shape)) deallocate(this%output_shape)

  end subroutine set_hyperparams_duvenaud
!###############################################################################


!###############################################################################
  subroutine init_duvenaud(this, input_shape, verbose)
    !! Initialise the message passing layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Loop index
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape([input_shape])
    this%output_shape = [this%num_outputs]
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate weight, weight steps (velocities), output, and activation
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(3,2*this%num_time_steps))
    allocate(this%params(this%num_time_steps*2))
    do t = 1, this%num_time_steps
       this%weight_shape(:,t) = [ &
            this%num_vertex_features(t), &
            this%num_vertex_features(t-1) + this%num_edge_features(0), &
            this%max_vertex_degree - this%min_vertex_degree + 1 &
       ]
       this%weight_shape(:,t+this%num_time_steps) = &
            [ this%num_outputs, this%num_vertex_features(t), 1 ]
       call this%params(t)%allocate( [ this%weight_shape(:,t), 1 ] )
       call this%params(t+this%num_time_steps)%allocate( &
            [ this%weight_shape(:2,t+this%num_time_steps), 1 ] &
       )
       call this%params(t)%set_requires_grad(.true.)
       this%params(t)%fix_pointer = .true.
       this%params(t)%is_temporary = .false.
       this%params(t)%is_sample_dependent = .false.
       this%params(t)%indices = [ this%min_vertex_degree, this%max_vertex_degree ]
       call this%params(t+this%num_time_steps)%set_requires_grad(.true.)
       this%params(t+this%num_time_steps)%fix_pointer = .true.
       this%params(t+this%num_time_steps)%is_temporary = .false.
       this%params(t+this%num_time_steps)%is_sample_dependent = .false.
    end do


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    do t = 1, this%num_time_steps, 1
       call this%kernel_init%initialise( &
            this%params(t)%val(:,1), &
            fan_in = this%num_vertex_features(t-1) + this%num_edge_features(0), &
            fan_out = this%num_vertex_features(t), &
            spacing = [ this%num_vertex_features(t-1) ] &
       )
       call this%kernel_init%initialise( &
            this%params(t+this%num_time_steps)%val(:,1), &
            fan_in = sum(this%num_vertex_features), &
            fan_out = this%num_outputs, &
            spacing = this%num_vertex_features &
       )
    end do


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))
    if(allocated(this%z)) deallocate(this%z)

  end subroutine init_duvenaud
!###############################################################################


!##############################################################################!
  subroutine set_graph_duvenaud(this, graph)
    !! Set the graph structure of the input data
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
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
       if(any(this%graph(s)%adj_ja(1,:).gt.this%graph(s)%num_vertices))then
          write(*,*) "Error: graph adjacency matrix has indices greater than &
               &the number of vertices", s, &
               this%graph(s)%num_vertices
          write(*,*) "Adjacency matrix indices: ", this%graph(s)%adj_ja
          stop
       end if
    end do

  end subroutine set_graph_duvenaud
!##############################################################################!


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_duvenaud(this, unit)
    !! Print kipf message passing layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(in) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: unit
    !! Filename

    ! Local variables
    integer :: t
    !! Loop index
    character(100) :: fmt
    !! Format string


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"NUM_TIME_STEPS = ",I0)') this%num_time_steps
    write(fmt,'("(3X,""NUM_VERTEX_FEATURES ="",",I0,"(1X,I0))")') &
         this%num_time_steps + 1
    write(unit,fmt) this%num_vertex_features
    write(fmt,'("(3X,""NUM_EDGE_FEATURES ="",",I0,"(1X,I0))")') &
         this%num_time_steps + 1
    write(unit,fmt) this%num_edge_features

    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit, identifier='MESSAGE')
    end if
    if(this%activation_readout%name .ne. 'none')then
       call this%activation_readout%print_to_unit(unit, identifier='READOUT')
    end if


    ! Write learned parameters
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params(t)%val(:,1)
    end do
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params(t+this%num_time_steps)%val(:,1)
    end do
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_duvenaud
!###############################################################################


!###############################################################################
  subroutine read_duvenaud(this, unit, verbose)
    !! Read the message passing layer
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: unit
    !! Unit to read from
    integer, optional, intent(in) :: verbose
    !! Verbosity level
  end subroutine read_duvenaud
!###############################################################################


!###############################################################################
  function read_duvenaud_msgpass_layer(unit, verbose) result(layer)
    !! Read duvenaud message passing layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the message passing layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source = duvenaud_msgpass_layer_type( &
         num_vertex_features = [ 0 ], &
         num_edge_features = [ 0 ], &
         num_time_steps = 1, &
         max_vertex_degree = 1, &
         num_outputs = 1 &
    ))
    call layer%read(unit, verbose=verbose_)

  end function read_duvenaud_msgpass_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!##############################################################################!
  subroutine update_message_duvenaud(this, input)
    !! Update the message
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in), target :: input
    !! Input to the message passing layer

    ! Local variables
    integer :: s, t
    !! Batch index, time step
    logical :: has_activation
    type(array_type), pointer :: ptr1, ptr2, ptr3, ptr_edge, ptr_params
    !! Pointers to arrays


    if(allocated(this%z))then
       if(size(this%z,2).ne.size(input,2))then
          deallocate(this%z)
          allocate(this%z(this%num_time_steps,size(input,2)))
       end if
    else
       allocate(this%z(this%num_time_steps,size(input,2)))
    end if


    if(.not.allocated(this%activation))then
       has_activation = .false.
    else
       if(trim(this%activation%name).eq."none")then
          has_activation = .true.
       else
          has_activation = .true.
       end if
    end if
    do s = 1, size(input,2)
       ptr1 => input(1,s)
       ptr_edge => input(2,s)
       do t = 1, this%num_time_steps
          ptr2 => duvenaud_propagate( &
               ptr1, ptr_edge, &
               this%graph(s)%adj_ia, this%graph(s)%adj_ja &
          )

          ptr_params => this%params(t)
          ptr3 => duvenaud_update( &
               ptr2, ptr_params, &
               this%graph(s)%adj_ia, &
               this%min_vertex_degree, this%max_vertex_degree &
          )
          if(has_activation)then
             ptr3 => this%activation%apply( ptr3 )
          end if
          call this%z(t,s)%zero_grad()
          call this%z(t,s)%assign_and_deallocate_source(ptr3)
          this%z(t,s)%is_temporary = .false.
          ptr1 => this%z(t,s)
       end do
    end do

  end subroutine update_message_duvenaud
!###############################################################################


!##############################################################################!
  subroutine update_readout_duvenaud(this)
    !! Update the readout
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer

    ! Local variables
    integer :: s, t, batch_size
    !! Loop indices
    type(array_type), pointer :: ptr1, ptr2, ptr3, ptr_params, ptr_z


    batch_size = size(this%z,2)
    call this%output(1,1)%zero_grad()
    do t = 1, this%num_time_steps, 1
       do s = 1, batch_size, 1
          ptr_params => this%params(t+this%num_time_steps)
          ptr_z => this%z(t,s)
          ptr1 => matmul( &
               ptr_params, &
               ptr_z &
          )
          ptr2 => this%activation_readout%apply( ptr1 )
          if(t.eq.1.and.s.eq.1)then
             ptr3 => &
                  sum( ptr2, dim = 2, new_dim_index=s, new_dim_size=batch_size )
          else
             ptr3 => ptr3 + &
                  sum( ptr2, dim = 2, new_dim_index=s, new_dim_size=batch_size )
          end if
       end do
    end do
    call this%output(1,1)%assign_and_deallocate_source(ptr3)
    this%output(1,1)%is_temporary = .false.

  end subroutine update_readout_duvenaud
!###############################################################################


!###############################################################################
  subroutine emit_onnx_nodes_duvenaud( &
       this, prefix, &
       nodes, num_nodes, max_nodes, &
       inits, num_inits, max_inits, &
       input_name, is_last_layer, format &
  )
    !! Emit ONNX JSON nodes for Duvenaud GNN layer
    !!
    !! Decomposes the Duvenaud message passing layer into standard ONNX ops:
    !!   Gather, Concat, ScatterElements, MatMul, Sigmoid/activation,
    !!   Softmax, ReduceSum, Add, Div, Clip, Sub, etc.
    !!
    !! This override is called by write_onnx instead of the standard
    !! node emission logic, making the ONNX export extensible for new
    !! GNN layer types.
    use athena__onnx_msgpass_utils, only: emit_output_identity
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(in) :: this
    !! Instance of the layer
    character(*), intent(in) :: prefix
    !! Node name prefix (e.g. "node_2")
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Accumulator for ONNX nodes
    integer, intent(inout) :: num_nodes
    !! Current number of nodes
    integer, intent(in) :: max_nodes
    !! Maximum capacity
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    !! Accumulator for ONNX initialisers
    integer, intent(inout) :: num_inits
    !! Current number of initialisers
    integer, intent(in) :: max_inits
    !! Maximum capacity
    character(*), optional, intent(in) :: input_name
    !! Unused sequential input name
    logical, optional, intent(in) :: is_last_layer
    !! Unused last-layer flag
    integer, optional, intent(in) :: format
    !! Unused export format selector

    ! Local variables
    integer :: t
    character(128) :: cur_vertex_name, readout_accum

    ! Must be called with vertex_input, edge_input etc. already set
    ! These are stored in the node's input naming convention
    ! prefix is e.g. "node_2", inputs come from the calling context

    ! ===== Emit message passing time steps =====
    do t = 1, this%num_time_steps
       call emit_duvenaud_timestep( &
            prefix, t, &
            this%num_vertex_features(t-1), this%num_edge_features(0), &
            this%num_vertex_features(t), &
            this%min_vertex_degree, this%max_vertex_degree, &
            this%params(t)%val(:,1), &
            this%activation%name, &
            nodes, num_nodes, max_nodes, &
            inits, num_inits, max_inits, &
            cur_vertex_name &
       )
    end do

    ! ===== Emit readout =====
    call emit_duvenaud_readout_impl( &
         prefix, this, &
         nodes, num_nodes, max_nodes, &
         inits, num_inits, max_inits, &
         readout_accum &
    )

    ! The readout output becomes the layer output for downstream layers.
    call emit_output_identity( &
         prefix, trim(readout_accum), this%activation%name, &
         nodes, num_nodes)

  end subroutine emit_onnx_nodes_duvenaud
!###############################################################################


!###############################################################################
  subroutine emit_onnx_graph_inputs_duvenaud( &
       this, prefix, &
       graph_inputs, num_inputs &
  )
    !! Emit graph input tensor declarations for Duvenaud GNN layer
    !!
    !! Adds: vertex features, edge features, edge_index [3, ncsr], degree
    use athena__onnx_msgpass_utils, only: emit_msgpass_graph_inputs
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(in) :: this
    !! Instance of the layer
    character(*), intent(in) :: prefix
    !! Input name prefix (e.g. "input_1")
    type(onnx_tensor_type), intent(inout), dimension(:) :: graph_inputs
    !! Accumulator for graph input tensor declarations
    integer, intent(inout) :: num_inputs
    !! Current number of graph input declarations

    call emit_msgpass_graph_inputs( &
         prefix, this%input_shape, graph_inputs, num_inputs)

  end subroutine emit_onnx_graph_inputs_duvenaud
!###############################################################################


!###############################################################################
  subroutine emit_duvenaud_timestep( &
       prefix, t, nv_in, ne_in, nv_out, &
       min_degree, max_degree, weight_data, activation_name, &
       nodes, num_nodes, max_nodes, &
       inits, num_inits, max_inits, vertex_out)
    !! Emit ONNX nodes for one Duvenaud message passing time step.
    use athena__onnx_utils, only: emit_node, emit_constant_int64, &
         emit_activation_node
    use athena__onnx_msgpass_utils, only: get_timestep_output_name, &
         emit_edge_index_component, emit_scatter_aggregator
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    integer, intent(in) :: t
    integer, intent(in) :: nv_in, ne_in, nv_out
    integer, intent(in) :: min_degree, max_degree
    real(real32), intent(in) :: weight_data(:)
    character(*), intent(in) :: activation_name
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    integer, intent(inout) :: num_nodes
    integer, intent(in) :: max_nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits
    integer, intent(in) :: max_inits
    character(128), intent(out) :: vertex_out

    ! Local variables
    character(128) :: tp, tmp1, tmp2, tmp3
    character(128) :: vertex_in, edge_in, edge_index_in, degree_in
    character(128) :: src_idx, edge_idx, target_idx
    character(128) :: msg_name, aggr_name, sq_out
    character(len=*), parameter :: onnx_axis0_attr = &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]'
    character(len=*), parameter :: onnx_concat_axis1_attr = &
         '        "attribute": [{"name": "axis", "i": "1", "type": "INT"}]'

    write(tp, '(A,"_t",I0)') trim(prefix), t

    ! Input tensor names follow the convention set during write_onnx.
    ! For t=1 the vertex input comes from the previous layer, while edge,
    ! edge_index, and degree are always rooted at the original graph input.
    write(vertex_in, '(A,"_vertex_in")') trim(prefix)
    write(edge_in, '(A,"_edge_in")') trim(prefix)
    write(edge_index_in, '(A,"_edge_index_in")') trim(prefix)
    write(degree_in, '(A,"_degree_in")') trim(prefix)
    if(t .gt. 1)then
       call get_timestep_output_name( &
            prefix, t-1, activation_name, '_sq_out', '_sq', vertex_in)
    end if

    ! --- Step 1: Extract source and edge-feature indices from edge_index ---
    write(tmp1, '(A,"_idx0")') trim(tp)
    call emit_constant_int64(trim(tmp1), [0], [1], &
         nodes, num_nodes, inits, num_inits)
    write(tmp2, '(A,"_idx1")') trim(tp)
    call emit_constant_int64(trim(tmp2), [1], [1], &
         nodes, num_nodes, inits, num_inits)
    write(tmp3, '(A,"_idx2")') trim(tp)
    call emit_constant_int64(trim(tmp3), [2], [1], &
         nodes, num_nodes, inits, num_inits)

    call emit_edge_index_component( &
         tp, edge_index_in, trim(tmp1), 'src', src_idx, nodes, num_nodes)
    call emit_edge_index_component( &
         tp, edge_index_in, trim(tmp2), 'eidx', edge_idx, nodes, num_nodes)
    call emit_edge_index_component( &
         tp, edge_index_in, trim(tmp3), 'tgt', target_idx, nodes, num_nodes)

    ! --- Step 2: Gather source vertex features and edge features ---
    write(tmp1, '(A,"_src_feat")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_vfeat', &
         trim(tmp1), onnx_axis0_attr, nodes, num_nodes, &
         in1=trim(vertex_in), in2=trim(src_idx))

    write(tmp2, '(A,"_edge_feat")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_efeat', &
         trim(tmp2), onnx_axis0_attr, nodes, num_nodes, &
         in1=trim(edge_in), in2=trim(edge_idx))

    ! --- Step 3: Concat source vertex + edge features ---
    write(msg_name, '(A,"_msg")') trim(tp)
    call emit_node('Concat', trim(tp)//'_concat_msg', &
         trim(msg_name), onnx_concat_axis1_attr, nodes, num_nodes, &
         in1=trim(tmp1), in2=trim(tmp2))

    ! --- Step 4: Scatter-add to aggregate messages per target vertex ---
    call emit_scatter_aggregator( &
         tp, vertex_in, target_idx, msg_name, nv_in + ne_in, &
         nodes, num_nodes, inits, num_inits, aggr_name)

    ! --- Step 5: Degree-specific weight application ---
    call emit_duvenaud_degree_update( &
         tp, degree_in, min_degree, max_degree, nv_in + ne_in, nv_out, &
         weight_data, aggr_name, nodes, num_nodes, inits, num_inits, sq_out)

    ! --- Step 6: Activation ---
    if(trim(activation_name) .ne. 'none')then
       call emit_activation_node(activation_name, trim(tp)//'_sq', &
            trim(sq_out), nodes, num_nodes, max_nodes)
       vertex_out = trim(nodes(num_nodes)%outputs(1))
    else
       vertex_out = trim(sq_out)
    end if

  end subroutine emit_duvenaud_timestep
!###############################################################################


!###############################################################################
  subroutine emit_duvenaud_degree_update( &
       tp, degree_in, min_degree, max_degree, feature_dim, nv_out, &
       weight_data, aggr_in, nodes, num_nodes, inits, num_inits, sq_out)
    !! Emit the degree-dependent weight selection and update block.
    use athena__onnx_utils, only: emit_node, emit_squeeze_node, &
         emit_constant_int64, emit_constant_float
    use athena__onnx_msgpass_utils, only: emit_weight_initialiser_3d
    implicit none

    ! Arguments
    character(*), intent(in) :: tp, degree_in, aggr_in
    integer, intent(in) :: min_degree, max_degree, feature_dim, nv_out
    real(real32), intent(in) :: weight_data(:)
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    integer, intent(inout) :: num_nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits
    character(128), intent(out) :: sq_out

    ! Local variables
    character(128) :: min_deg_name, max_deg_name, deg_float
    character(128) :: deg_clip, deg_idx_float, deg_idx
    character(128) :: weight_name, weight_sel, deg_us
    character(128) :: aggr_norm, aggr_us, matmul_out
    character(128) :: axes1_name, axes2_name
    character(len=*), parameter :: onnx_axis0_attr = &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]'
    character(len=*), parameter :: onnx_cast_float_attr = &
         '        "attribute": [{"name": "to", "i": "1", "type": "INT"}]'
    character(len=*), parameter :: onnx_cast_int64_attr = &
         '        "attribute": [{"name": "to", "i": "7", "type": "INT"}]'

    ! Clip degree to the supported bucket interval.
    write(min_deg_name, '(A,"_min_deg")') trim(tp)
    call emit_constant_float(trim(min_deg_name), &
         [ real(min_degree, real32) ], [1], &
         nodes, num_nodes, inits, num_inits)

    write(max_deg_name, '(A,"_max_deg")') trim(tp)
    call emit_constant_float(trim(max_deg_name), &
         [ real(max_degree, real32) ], [1], &
         nodes, num_nodes, inits, num_inits)

    write(deg_float, '(A,"_deg_f")') trim(tp)
    call emit_node('Cast', trim(tp)//'_cast_deg', &
         trim(deg_float), onnx_cast_float_attr, nodes, num_nodes, &
         in1=trim(degree_in))

    write(deg_clip, '(A,"_deg_clip")') trim(tp)
    call emit_node('Clip', trim(tp)//'_clip_deg', &
         trim(deg_clip), '', nodes, num_nodes, &
         in1=trim(deg_float), in2=trim(min_deg_name), in3=trim(max_deg_name))

    ! Shift clipped degrees so they can index the weight bank from zero.
    write(deg_idx_float, '(A,"_deg_idx_f")') trim(tp)
    call emit_node('Sub', trim(tp)//'_sub_mindeg', &
         trim(deg_idx_float), '', nodes, num_nodes, &
         in1=trim(deg_clip), in2=trim(min_deg_name))

    write(deg_idx, '(A,"_deg_idx")') trim(tp)
    call emit_node('Cast', trim(tp)//'_cast_degidx', &
         trim(deg_idx), onnx_cast_int64_attr, nodes, num_nodes, &
         in1=trim(deg_idx_float))

    ! Store the degree-specific weight bank as a 3D initialiser.
    write(weight_name, '(A,"_W")') trim(tp)
    call emit_weight_initialiser_3d( &
         trim(weight_name), max_degree - min_degree + 1, &
         nv_out, feature_dim, weight_data, inits, num_inits)

    write(weight_sel, '(A,"_W_sel")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_W', &
         trim(weight_sel), onnx_axis0_attr, nodes, num_nodes, &
         in1=trim(weight_name), in2=trim(deg_idx))

    ! Divide by degree and reshape for batched MatMul.
    write(axes1_name, '(A,"_us_ax1_deg")') trim(tp)
    call emit_constant_int64(trim(axes1_name), [1], [1], &
         nodes, num_nodes, inits, num_inits)

    write(deg_us, '(A,"_deg_us")') trim(tp)
    call emit_node('Unsqueeze', trim(tp)//'_us_deg', &
         trim(deg_us), '', nodes, num_nodes, &
         in1=trim(deg_clip), in2=trim(axes1_name))

    write(aggr_norm, '(A,"_aggr_norm")') trim(tp)
    call emit_node('Div', trim(tp)//'_div_deg', &
         trim(aggr_norm), '', nodes, num_nodes, &
         in1=trim(aggr_in), in2=trim(deg_us))

    write(axes2_name, '(A,"_us_ax2")') trim(tp)
    call emit_constant_int64(trim(axes2_name), [2], [1], &
         nodes, num_nodes, inits, num_inits)

    write(aggr_us, '(A,"_aggr_us")') trim(tp)
    call emit_node('Unsqueeze', trim(tp)//'_us_aggr', &
         trim(aggr_us), '', nodes, num_nodes, &
         in1=trim(aggr_norm), in2=trim(axes2_name))

    write(matmul_out, '(A,"_matmul_out")') trim(tp)
    call emit_node('MatMul', trim(tp)//'_matmul', &
         trim(matmul_out), '', nodes, num_nodes, &
         in1=trim(weight_sel), in2=trim(aggr_us))

    write(sq_out, '(A,"_sq_out")') trim(tp)
    call emit_squeeze_node(trim(tp)//'_sq_mm', &
         trim(matmul_out), trim(axes2_name), trim(sq_out), &
         nodes, num_nodes)

  end subroutine emit_duvenaud_degree_update
!###############################################################################


!###############################################################################
  subroutine emit_duvenaud_readout_step( &
       prefix, activation_name, t, nv, no, weight_data, &
       nodes, num_nodes, inits, num_inits, step_sum)
    !! Emit one Duvenaud readout timestep.
    !!
    !! This expands to the timestep readout projection, the readout softmax,
    !! and the reduction over nodes before the timestep contributions are added.
    use athena__onnx_utils, only: emit_node, emit_constant_int64
    use athena__onnx_msgpass_utils, only: get_timestep_output_name, &
         emit_weight_initialiser_2d
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    character(*), intent(in) :: activation_name
    integer, intent(in) :: t, nv, no
    real(real32), intent(in) :: weight_data(:)
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    integer, intent(inout) :: num_nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits
    character(128), intent(out) :: step_sum

    ! Local variables
    character(128) :: tp, z_name, weight_name, z_transpose
    character(128) :: matmul_out, softmax_out, axis1_name
    character(len=*), parameter :: onnx_softmax_axis0_attr = &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]'
    character(len=*), parameter :: onnx_transpose_10_attr = &
         '        "attribute": [{"name": "perm", "ints": ["1", "0"], ' // &
         '"type": "INTS"}]'
    character(len=*), parameter :: onnx_reduce_sum_attr = &
         '        "attribute": [{"name": "keepdims", "i": "0", ' // &
         '"type": "INT"}]'

    write(tp, '(A,"_ro_t",I0)') trim(prefix), t
    call get_timestep_output_name( &
         prefix, t, activation_name, '_sq_out', '_sq', z_name)

    ! Store the readout matrix for timestep t as an ONNX initialiser.
    write(weight_name, '(A,"_R")') trim(tp)
    call emit_weight_initialiser_2d( &
         trim(weight_name), no, nv, weight_data, inits, num_inits)

    ! Transpose node features before multiplying by the readout matrix.
    write(z_transpose, '(A,"_zt")') trim(tp)
    call emit_node('Transpose', trim(tp)//'_transpose_z', &
         trim(z_transpose), onnx_transpose_10_attr, nodes, num_nodes, &
         in1=trim(z_name))

    write(matmul_out, '(A,"_Rz")') trim(tp)
    call emit_node('MatMul', trim(tp)//'_matmul_R', &
         trim(matmul_out), '', nodes, num_nodes, &
         in1=trim(weight_name), in2=trim(z_transpose))

    ! Softmax and ReduceSum reproduce the ATHENA readout accumulation.
    write(softmax_out, '(A,"_sm")') trim(tp)
    call emit_node('Softmax', trim(tp)//'_softmax', &
         trim(softmax_out), onnx_softmax_axis0_attr, nodes, num_nodes, &
         in1=trim(matmul_out))

    write(axis1_name, '(A,"_ax1")') trim(tp)
    call emit_constant_int64(trim(axis1_name), [1], [1], &
         nodes, num_nodes, inits, num_inits)

    write(step_sum, '(A,"_sum")') trim(tp)
    call emit_node('ReduceSum', trim(tp)//'_reducesum', &
         trim(step_sum), onnx_reduce_sum_attr, nodes, num_nodes, &
         in1=trim(softmax_out), in2=trim(axis1_name))

  end subroutine emit_duvenaud_readout_step
!###############################################################################


!###############################################################################
  subroutine emit_duvenaud_readout_impl( &
       prefix, layer, &
       nodes, num_nodes, max_nodes, &
       inits, num_inits, max_inits, &
       readout_output &
  )
    !! Emit ONNX nodes for Duvenaud readout
    use athena__onnx_utils, only: emit_node, emit_constant_int64
    implicit none
    character(*), intent(in) :: prefix
    class(duvenaud_msgpass_layer_type), intent(in) :: layer
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    integer, intent(inout) :: num_nodes
    integer, intent(in) :: max_nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits
    integer, intent(in) :: max_inits
    character(128), intent(out) :: readout_output

    ! Local variables
    integer :: t
    character(128) :: tmp1, prev_accum, step_sum

    do t = 1, layer%num_time_steps
       call emit_duvenaud_readout_step( &
            prefix, layer%activation%name, t, &
            layer%num_vertex_features(t), layer%num_outputs, &
            layer%params(t + layer%num_time_steps)%val(:,1), &
            nodes, num_nodes, inits, num_inits, step_sum)

       ! Accumulate across timesteps
       if(t .eq. 1)then
          prev_accum = trim(step_sum)
       else
          write(tmp1, '(A,"_ro_t",I0,"_accum")') trim(prefix), t
          call emit_node('Add', trim(tmp1)//'_node', &
               trim(tmp1), '', nodes, num_nodes, &
               in1=trim(prev_accum), in2=trim(step_sum))
          prev_accum = trim(tmp1)
       end if
    end do

    ! Unsqueeze to add batch dimension: [no] → [1, no]
    write(tmp1, '(A,"_ro_ax0")') trim(prefix)
    call emit_constant_int64(trim(tmp1), [0], [1], &
         nodes, num_nodes, inits, num_inits)
    write(readout_output, '(A,"_readout")') trim(prefix)
    call emit_node('Unsqueeze', trim(prefix)//'_us_readout', &
         trim(readout_output), '', nodes, num_nodes, &
         in1=trim(prev_accum), in2=trim(tmp1))

  end subroutine emit_duvenaud_readout_impl
!###############################################################################

end module athena__duvenaud_msgpass_layer
