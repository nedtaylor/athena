module athena__duvenaud_msgpass_layer
  !! Module containing the types and interfacees of a message passing layer
  use coreutils, only: real32
  use graphstruc, only: graph_type
  use athena__misc_types, only: base_actv_type, base_init_type
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

     class(base_actv_type), allocatable :: transfer_readout
     !! Activation function
     type(array_type), allocatable, dimension(:,:) :: z
     type(array_type), allocatable, dimension(:,:) :: z_readout
     !! Input gradients

   contains
     procedure, pass(this) :: get_num_params => get_num_params_duvenaud
     !! Get the number of parameters for the message passing layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_duvenaud
     !! Set the hyperparameters for the message passing layer
     procedure, pass(this) :: init => init_duvenaud
     !! Initialise the message passing layer
     procedure, pass(this) :: set_batch_size => set_batch_size_duvenaud
     !! Set batch size
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
          batch_size, &
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
       integer, optional, intent(in) :: batch_size
       !! Batch size
       character(*), optional, intent(in) :: message_activation, &
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



contains

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
       batch_size, &
       message_activation, &
       readout_activation, &
       kernel_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the message passing layer
    use athena__initialiser, only: initialiser_setup
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
    integer, optional, intent(in) :: batch_size
    !! Batch size
    character(*), optional, intent(in) :: message_activation, &
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
    real(real32) :: &
         message_scale = 1._real32, &
         readout_scale = 1._real32
    !! Activation scale
    character(len=10) :: &
         message_activation_ = "sigmoid", &
         readout_activation_ = "softmax"
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser_
    !! Kernel and bias initialisers
    integer :: min_vertex_degree_ = 1
    !! Minimum vertex degree

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(message_activation)) &
         message_activation_ = message_activation
    if(present(readout_activation)) &
         readout_activation_ = readout_activation
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
    ! Initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


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
    character(*), intent(in) :: &
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
    if(size(num_vertex_features, 1) .eq. 1) then
       allocate( &
            this%num_vertex_features(0:num_time_steps), &
            source = num_vertex_features(1) &
       )
    elseif(size(num_vertex_features, 1) .eq. num_time_steps + 1) then
       allocate( &
            this%num_vertex_features(0:this%num_time_steps), &
            source = num_vertex_features &
       )
    else
       write(*,*) "Error: num_vertex_features must be a scalar or a vector of &
            &length num_time_steps + 1"
       stop
    end if
    if(size(num_edge_features, 1) .eq. 1) then
       allocate( &
            this%num_edge_features(0:num_time_steps), &
            source = num_edge_features(1) &
       )
    elseif(size(num_edge_features, 1) .eq. num_time_steps + 1) then
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
    if(allocated(this%transfer)) deallocate(this%transfer)
    if(allocated(this%transfer_readout)) deallocate(this%transfer_readout)
    allocate(this%transfer, &
         source = activation_setup(message_activation))
    allocate(this%transfer_readout, &
         source = activation_setup(readout_activation))
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(message_activation)
       this%kernel_init = initialiser_setup(buffer)
    else
       this%kernel_init = kernel_initialiser
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("DUVENAUD message activation function: ",A)') &
               trim(this%transfer%name)
          write(*,'("DUVENAUD readout activation function: ",A)') &
               trim(this%transfer_readout%name)
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
  subroutine init_duvenaud(this, input_shape, batch_size, verbose)
    !! Initialise the message passing layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Loop index
    integer :: verbose_ = 0
    !! Verbosity level
    real(real32) :: mean, std
    !! Mean and standard deviation of the parameters


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape([input_shape])
    this%output_shape = [this%num_outputs]
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate weight, weight steps (velocities), output, and activation
    !---------------------------------------------------------------------------
    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(this%num_params), source=0._real32)
    allocate(this%weight_shape(3,2*this%num_time_steps))
    allocate(this%params_array(this%num_time_steps*2))
    do t = 1, this%num_time_steps
       this%weight_shape(:,t) = [ &
            this%num_vertex_features(t), &
            this%num_vertex_features(t-1) + this%num_edge_features(0), &
            this%max_vertex_degree - this%min_vertex_degree + 1 &
       ]
       this%weight_shape(:,t+this%num_time_steps) = &
            [ this%num_outputs, this%num_vertex_features(t), 1 ]
       call this%params_array(t)%allocate( [ this%weight_shape(:,t), 1 ] )
       call this%params_array(t+this%num_time_steps)%allocate( &
            [ this%weight_shape(:2,t+this%num_time_steps), 1 ] &
       )
       call this%params_array(t)%set_requires_grad(.true.)
       this%params_array(t)%fix_pointer = .true.
       this%params_array(t)%is_temporary = .false.
       this%params_array(t)%is_sample_dependent = .false.
       this%params_array(t)%indices = [ this%min_vertex_degree, this%max_vertex_degree ]
       call this%params_array(t+this%num_time_steps)%set_requires_grad(.true.)
       this%params_array(t+this%num_time_steps)%fix_pointer = .true.
       this%params_array(t+this%num_time_steps)%is_temporary = .false.
       this%params_array(t+this%num_time_steps)%is_sample_dependent = .false.
    end do


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    do t = 1, this%num_time_steps, 1
       call this%kernel_init%initialise( &
            this%params_array(t)%val(:,1), &
            fan_in = this%num_vertex_features(t-1) + this%num_edge_features(0), &
            fan_out = this%num_vertex_features(t), &
            spacing = [ this%num_vertex_features(t-1) ] &
       )
       call this%kernel_init%initialise( &
            this%params_array(t+this%num_time_steps)%val(:,1), &
            fan_in = sum(this%num_vertex_features), &
            fan_out = this%num_outputs, &
            spacing = this%num_vertex_features &
       )
    end do
    ! write the standard deviation of the params values
    if(verbose_.gt.0)then
       mean = sum(this%params(:sum(this%num_params_msg))) / &
            real(sum(this%num_params_msg), kind=real32)
       std = sqrt(sum((this%params(:sum(this%num_params_msg)) - mean)**2) / &
            real(sum(this%num_params_msg), kind=real32))
       write(*,*) "Initialised message parameters with mean = ", mean, &
            " and std = ", std
       mean = sum(this%params(sum(this%num_params_msg)+1:)) / &
            real(this%num_params_readout, kind=real32)
       std = sqrt(sum((this%params(sum(this%num_params_msg)+1:) - mean)**2) / &
            real(this%num_params_readout, kind=real32))
       write(*,*) "Initialised readout parameters with mean = ", mean, &
            " and std = ", std
    end if


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_duvenaud
!###############################################################################


!###############################################################################
  subroutine set_batch_size_duvenaud(this, batch_size, verbose)
    !! Set the batch size for message passing layer
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout), target :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0


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
       allocate(this%output(1,1))
       if(allocated(this%z)) deallocate(this%z)
       allocate(this%z(this%num_time_steps,this%batch_size))
    end if

  end subroutine set_batch_size_duvenaud
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

    write(unit,'(3X,"MESSAGE_ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"READOUT_ACTIVATION = ",A)') trim(this%transfer_readout%name)


    ! Write learned parameters
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params_array(t)%val(:,1)
    end do
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params_array(t+this%num_time_steps)%val(:,1)
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


    if(.not.allocated(this%transfer))then
       has_activation = .false.
    else
       if(trim(this%transfer%name).eq."none")then
          has_activation = .true.
       else
          has_activation = .true.
       end if
    end if
    do s = 1, this%batch_size
       ptr1 => input(1,s)
       ptr_edge => input(2,s)
       do t = 1, this%num_time_steps
          ptr2 => duvenaud_propagate( &
               ptr1, ptr_edge, &
               this%graph(s)%adj_ia, this%graph(s)%adj_ja &
          )

          ptr_params => this%params_array(t)
          ptr3 => duvenaud_update( &
               ptr2, ptr_params, &
               this%graph(s)%adj_ia, &
               this%min_vertex_degree, this%max_vertex_degree &
          )
          if(has_activation)then
             ptr3 => this%transfer%activate( ptr3 )
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
    integer :: s, t
    !! Loop indices
    type(array_type), pointer :: ptr1, ptr2, ptr3, ptr_params, ptr_z


    call this%output(1,1)%zero_grad()
    do t = 1, this%num_time_steps, 1
       do s = 1, this%batch_size
          ptr_params => this%params_array(t+this%num_time_steps)
          ptr_z => this%z(t,s)
          ptr1 => matmul( &
               ptr_params, &
               ptr_z &
          )
          ptr2 => this%transfer_readout%activate( ptr1 )
          if(t.eq.1.and.s.eq.1)then
             ptr3 => &
                  sum( ptr2, dim = 2, new_dim_index=s, new_dim_size=this%batch_size )
          else
             ptr3 => ptr3 + &
                  sum( ptr2, dim = 2, new_dim_index=s, new_dim_size=this%batch_size )
          end if
       end do
    end do
    call this%output(1,1)%assign_and_deallocate_source(ptr3)
    this%output(1,1)%is_temporary = .false.

  end subroutine update_readout_duvenaud
!###############################################################################

end module athena__duvenaud_msgpass_layer
