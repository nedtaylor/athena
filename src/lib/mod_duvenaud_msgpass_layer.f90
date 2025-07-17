module athena__duvenaud_msgpass_layer
  !! Module containing the types and interfacees of a message passing layer
  use athena__constants, only: real32
  use graphstruc, only: graph_type
  use athena__misc_types, only: activation_type, initialiser_type, &
       array_type, array2d_type
  use athena__base_layer, only: base_layer_type
  use athena__msgpass_layer, only: msgpass_layer_type
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

     class(activation_type), allocatable :: transfer_readout
     !! Activation function
     type(array2d_type), allocatable, dimension(:,:) :: di_msg, di_readout, &
          z_readout
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

     procedure, pass(this) :: forward => forward_rank
     !! Forward pass for message passing layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward pass for message passing layer

     procedure, pass(this) :: update_message => update_message_duvenaud
     !! Update the message
     procedure, pass(this) :: backward_message => backward_message_duvenaud
     !! Backward pass for the message phase

     procedure, pass(this) :: update_readout => update_readout_duvenaud
     !! Update the readout
     procedure, pass(this) :: backward_readout => backward_readout_duvenaud
     !! Backward pass for the readout phase

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
          message_activation_function, message_activation_scale, &
          readout_activation_function, readout_activation_scale, &
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
       real(real32), optional, intent(in) :: message_activation_scale, &
            readout_activation_scale
       !! Message and readout activation scales
       character(*), optional, intent(in) :: message_activation_function, &
            readout_activation_function
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
    if(allocated(this%di)) deallocate(this%di)
    if(allocated(this%di_msg)) deallocate(this%di_msg)
    if(allocated(this%di_readout)) deallocate(this%di_readout)
    if(allocated(this%z)) deallocate(this%z)
    if(allocated(this%z_readout)) deallocate(this%z_readout)

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
  subroutine forward_rank(this, input)
    !! Forward pass for message
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    real(real32), dimension(..), intent(in) :: input
    !! Input to the message passing layer

  end subroutine forward_rank
!###############################################################################


!###############################################################################
  subroutine backward_rank(this, input, gradient)
    !! Backward pass for message
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    real(real32), dimension(..), intent(in) :: input
    !! Input to the message passing layer
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

  end subroutine backward_rank
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
       message_activation_function, message_activation_scale, &
       readout_activation_function, readout_activation_scale, &
       kernel_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the message passing layer
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
    real(real32), optional, intent(in) :: message_activation_scale, &
         readout_activation_scale
    !! Message and readout activation scales
    character(*), optional, intent(in) :: message_activation_function, &
         readout_activation_function
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
         message_activation_function_ = "sigmoid", &
         readout_activation_function_ = "softmax"
    !! Activation function
    integer :: min_vertex_degree_ = 1
    !! Minimum vertex degree

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(message_activation_function)) &
         message_activation_function_ = message_activation_function
    if(present(message_activation_scale)) message_scale = message_activation_scale
    if(present(readout_activation_function)) &
         readout_activation_function_ = readout_activation_function
    if(present(readout_activation_scale)) readout_scale = readout_activation_scale
    if(present(min_vertex_degree)) min_vertex_degree_ = min_vertex_degree
    if(max_vertex_degree.lt.min_vertex_degree_)then
       write(0,*) "Error: max_vertex_degree < min_vertex_degree"
       return
    end if


    !---------------------------------------------------------------------------
    ! Define weights (kernels) and biases initialisers
    !---------------------------------------------------------------------------
    if(present(kernel_initialiser)) layer%kernel_initialiser =kernel_initialiser


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
         message_activation_function = message_activation_function_, &
         message_activation_scale = message_scale, &
         readout_activation_function = readout_activation_function_, &
         readout_activation_scale = readout_scale, &
         kernel_initialiser = layer%kernel_initialiser, &
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
       message_activation_function, message_activation_scale, &
       readout_activation_function, readout_activation_scale, &
       kernel_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for the message passing layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser
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
         message_activation_function, &
         readout_activation_function
    !! Message and readout activation functions
    real(real32), optional, intent(in) :: &
         message_activation_scale, &
         readout_activation_scale
    !! Message and readout activation scales
    character(*), optional, intent(in) :: kernel_initialiser
    !! Kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Loop index

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
         source = activation_setup(message_activation_function, &
              message_activation_scale))
    allocate(this%transfer_readout, &
         source = activation_setup(readout_activation_function, &
              readout_activation_scale))
    if(trim(kernel_initialiser).eq.'') &
         this%kernel_initialiser = &
              get_default_initialiser(message_activation_function)
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("DUVENAUD message activation function: ",A)') &
               trim(message_activation_function)
          write(*,'("DUVENAUD readout activation function: ",A)') &
               trim(readout_activation_function)
          write(*,'("DUVENAUD kernel initialiser: ",A)') &
               trim(this%kernel_initialiser)
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
    class(initialiser_type), allocatable :: initialiser_
    !! Initialiser


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
    allocate(this%weight_shape(2,this%num_time_steps))
    do t = 1, this%num_time_steps
       this%weight_shape(:,t) = [ &
            this%num_vertex_features(t), &
            this%num_vertex_features(t-1) + this%num_edge_features(0) &
       ]
       this%weight_shape(:,t+this%num_time_steps) = &
            [ this%num_outputs, this%num_vertex_features(t) ]
    end do


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%kernel_initialiser))
    do t = 1, this%num_time_steps, 1
       call initialiser_%initialise( &
            this%params( &
                 sum(this%num_params_msg(1:t-1)) + 1: &
                 sum(this%num_params_msg(1:t)) &
            ), &
            fan_in = this%num_vertex_features(t-1) + this%num_edge_features(0), &
            fan_out = this%num_vertex_features(t), &
            spacing = [ this%num_vertex_features(t-1) ] &
       )
    end do
    call initialiser_%initialise( &
         this%params(sum(this%num_params_msg)+1:), &
         fan_in = sum(this%num_vertex_features), &
         fan_out = this%num_outputs, &
         spacing = this%num_vertex_features &
    )
    deallocate(initialiser_)
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
       allocate(this%output(1,1), source=array2d_type())
       !! output val arrays are allocated in set_graph
       ! call this%output(1,1)%allocate( &
       !      [this%num_outputs, this%batch_size], &
       !      source=0._real32 &
       ! )
       if(allocated(this%z)) deallocate(this%z)
       allocate( &
            this%z(this%num_time_steps, this%batch_size), &
            source=array2d_type() &
       )
       ! select type(output => this%output(1,1))
       ! type is (array2d_type)
       !    allocate( this%z, source = output%val )
       ! end select
       if(allocated(this%dp)) deallocate(this%dp)
       allocate( &
            this%dp( &
                 this%num_params, &
                 this%batch_size &
            ), source=0._real32 &
       )
       if(allocated(this%di)) deallocate(this%di)
       allocate(this%di(2,this%batch_size), source=array2d_type())

       if(allocated(this%di_msg)) deallocate(this%di_msg)
       allocate(this%di_msg(this%num_time_steps, this%batch_size))

       if(allocated(this%di_readout)) deallocate(this%di_readout)
       allocate(this%di_readout(this%num_time_steps, this%batch_size))

       if(allocated(this%z_readout)) deallocate(this%z_readout)
       allocate(this%z_readout(this%num_time_steps, this%batch_size))

       !! input val arrays are allocated in set_graph
    end if


    if(allocated(this%vertex_features)) deallocate(this%vertex_features)
    allocate(this%vertex_features(0:this%num_time_steps, 1:this%batch_size))
    if(allocated(this%edge_features)) deallocate(this%edge_features)
    allocate(this%edge_features(0:this%num_time_steps, 1:this%batch_size))
    if(allocated(this%message)) deallocate(this%message)
    allocate(this%message(1:this%num_time_steps, 1:this%batch_size))

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
    end do
!     do s = 1, size(graph)
!        call this%graph(s)%copy(graph(s), sparse=.true.)
!     end do

    if(this%use_graph_input)then
       if(allocated(this%output))then
          if(this%output(1,1)%allocated) &
               call this%output(1,1)%deallocate()
          call this%output(1,1)%allocate( &
               [ &
                    this%num_outputs, &
                    size(graph) &
               ] &
          )
          call this%output(1,1)%set_ptr()
          do s = 1, size(graph)
             if(this%di(1,s)%allocated) &
                  call this%di(1,s)%deallocate()
             if(this%di(2,s)%allocated) &
                  call this%di(2,s)%deallocate()
             call this%di(1,s)%allocate( &
                  [ &
                       this%num_vertex_features(0), &
                       this%graph(s)%num_vertices &
                  ] &
             )
             call this%di(2,s)%allocate( &
                  [ &
                       this%num_edge_features(0), &
                       this%graph(s)%num_edges &
                  ] &
             )
             call this%di(1,s)%set_ptr()
             call this%di(2,s)%set_ptr()
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
            [ this%num_vertex_features(0), this%graph(s)%num_vertices ] &
       )
       call this%edge_features(0,s)%allocate( &
            [ this%num_edge_features(0), this%graph(s)%num_edges ] &
       )
       do t = 1, this%num_time_steps, 1
          if(this%vertex_features(t,s)%allocated) &
               call this%vertex_features(t,s)%deallocate()
          if(this%message(t,s)%allocated) &
               call this%message(t,s)%deallocate()
          if(this%z(t,s)%allocated) &
               call this%z(t,s)%deallocate()
          call this%vertex_features(t,s)%allocate( &
               [ this%num_vertex_features(t), this%graph(s)%num_vertices ] &
          )
          call this%message(t,s)%allocate( &
               [ &
                    this%num_vertex_features(t) + this%num_edge_features(0), &
                    this%graph(s)%num_vertices &
               ] &
          )
          if(this%z_readout(t,s)%allocated) &
               call this%z_readout(t,s)%deallocate()
          if(this%z(t,s)%allocated) &
               call this%z(t,s)%deallocate()
          if(this%di_readout(t,s)%allocated) &
               call this%di_readout(t,s)%deallocate()
          if(this%di_msg(t,s)%allocated) &
               call this%di_msg(t,s)%deallocate()
          call this%z(t,s)%allocate( &
               [ this%num_vertex_features(t), this%graph(s)%num_vertices ] &
          )
          call this%z_readout(t,s)%allocate( &
               [ this%num_outputs, this%graph(s)%num_vertices ] &
          )
          call this%di_readout(t,s)%allocate( &
               [ this%num_vertex_features(t), &
                    this%graph(s)%num_vertices ] &
          )
          call this%di_msg(t,s)%allocate( &
               [ this%num_vertex_features(t) + this%num_edge_features(0), &
                    this%graph(s)%num_vertices ] &
          )
       end do
    end do

  end subroutine set_graph_duvenaud
!##############################################################################!


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_duvenaud(this, unit)
    !! Print kipf message passing layer to unit
    use athena__misc, only: to_upper
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
    write(unit,'(3X,"MESSAGE_ACTIVATION_SCALE = ",F0.9)') this%transfer%scale
    write(unit,'(3X,"READOUT_ACTIVATION = ",A)') trim(this%transfer_readout%name)
    write(unit,'(3X,"READOUT_ACTIVATION_SCALE = ",F0.9)') this%transfer_readout%scale


    ! Write learned parameters
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params( &
            sum(this%num_params_msg(1:t-1:1)) + 1 : &
            sum(this%num_params_msg(1:t:1)) &
       )
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
         max_vertex_degree = 0, &
         num_outputs = 1 &
    ))
    call layer%read(unit, verbose=verbose_)

  end function read_duvenaud_msgpass_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!##############################################################################!
  pure subroutine update_message_duvenaud(this, input)
    !! Update the message
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input to the message passing layer

    ! Local variables
    integer :: s, v, e, t
    !! Batch index, vertex index, edge index, time step
    integer :: degree
    !! Degree of the vertex
    real(real32), pointer :: weight(:,:,:)
    !! Pointer to the weight matrix


    do s = 1, this%batch_size
       this%vertex_features(0,s)%val = input(1,s)%val
       this%edge_features(0,s)%val = input(2,s)%val
    end do

    do t = 1, this%num_time_steps
       weight( &
            1:this%num_vertex_features(t), &
            1:this%num_vertex_features(t-1) + this%num_edge_features(0), &
            this%min_vertex_degree:this%max_vertex_degree &
       ) => this%params( &
            sum(this%num_params_msg(1:t-1:1)) + 1 : &
            sum(this%num_params_msg(1:t:1)) &
       )
       do concurrent (s = 1: this%batch_size)
          do v = 1, this%graph(s)%num_vertices
             this%message(t,s)%val(:,v) = 0._real32
             degree = this%graph(s)%adj_ia(v+1) - this%graph(s)%adj_ia(v)
             degree = max( &
                  this%min_vertex_degree, &
                  min(degree, this%max_vertex_degree) &
             )
             do e = this%graph(s)%adj_ia(v), this%graph(s)%adj_ia(v+1) - 1
                if(this%graph(s)%adj_ja(2,e).eq.0)then
                   this%message(t,s)%val(:this%num_vertex_features(t-1),v) = &
                        this%message(t,s)%val(:this%num_vertex_features(t-1),v) + &
                        this%vertex_features(t-1,s)%val(:,v)
                else
                   this%message(t,s)%val(:,v) = &
                        this%message(t,s)%val(:,v) + &
                        [ &
                             this%vertex_features(t-1,s)%val( &
                                  :, &
                                  this%graph(s)%adj_ja(1,e) &
                             ), &
                             this%edge_features(0,s)%val( &
                                  :, &
                                  this%graph(s)%adj_ja(2,e) &
                             ) &
                        ]
                end if
             end do
             this%z(t,s)%val(:,v) = matmul( &
                  weight(:,:,degree), &
                  this%message(t,s)%val(:,v) / degree &
             )
          end do
          this%vertex_features(t,s)%val(:,:) = &
               this%transfer%activate( this%z(t,s)%val(:,:) )
       end do
    end do

  end subroutine update_message_duvenaud
!###############################################################################


!##############################################################################!
  pure subroutine update_readout_duvenaud(this)
    !! Update the readout
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer

    ! Local variables
    integer :: s, v, t
    !! Loop indices
    integer :: num_params_old, num_params_tmp
    !! Number of parameters in the previous and current time step
    real(real32), pointer :: weight(:,:)
    !! Pointer to the weight matrix


    this%output(1,1)%val = 0._real32
    num_params_old = sum(this%num_params_msg)
    do t = 1, this%num_time_steps, 1
       num_params_tmp = this%num_vertex_features(t) * this%num_outputs
       weight( &
            1:this%num_outputs, &
            1:this%num_vertex_features(t) &
       ) => this%params( &
            num_params_old + 1 : num_params_old + num_params_tmp &
       )
       do s = 1, this%batch_size
          do v = 1, this%graph(s)%num_vertices
             this%z_readout(t,s)%val(:,v) = matmul( &
                  weight(:,:), &
                  this%vertex_features(t,s)%val(:,v) &
             )
             this%output(1,1)%val(:,s) = this%output(1,1)%val(:,s) + &
                  this%transfer_readout%activate( this%z_readout(t,s)%val(:,v) )
          end do
       end do
       num_params_old = num_params_old + num_params_tmp
    end do

  end subroutine update_readout_duvenaud
!###############################################################################


!##############################################################################!
  subroutine backward_message_duvenaud(this, input, gradient)
    !! Backward pass for the message phase
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data (i.e. vertex and edge features)
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

    ! Local variables
    integer :: degree
    !! Degree of the vertex
    integer :: t, s, v, e, i, j, idx
    !! Loop indices
    real(real32), dimension(:,:), allocatable :: delta
    !! Delta values for the message phase
    real(real32), pointer :: weight(:,:,:), dw(:,:,:)
    !! Pointer to the weight matrix


    do t = this%num_time_steps, 1, -1
       weight( &
            1:this%num_vertex_features(t), &
            1:this%num_vertex_features(t-1) + this%num_edge_features(0), &
            this%min_vertex_degree:this%max_vertex_degree &
       ) => this%params( &
            sum(this%num_params_msg(1:t-1:1)) + 1 : &
            sum(this%num_params_msg(1:t:1)) &
       )
       do concurrent(s=1:this%batch_size)
          dw( &
               1:this%num_vertex_features(t), &
               1:this%num_vertex_features(t-1) + this%num_edge_features(0), &
               this%min_vertex_degree:this%max_vertex_degree &
          ) => this%dp( &
               sum(this%num_params_msg(1:t-1:1)) + 1 : &
               sum(this%num_params_msg(1:t:1)), s &
          )
          if(t.eq.1)then
             this%di(1,s)%val = 0._real32
             this%di(2,s)%val = 0._real32
          end if

          if(allocated(delta)) deallocate(delta)
          allocate(delta( &
               this%num_vertex_features(t), &
               this%graph(s)%num_vertices &
          ), source = 0._real32)
          delta(:this%num_vertex_features(t),:) = this%di_readout(t,s)%val
          if(t.lt.this%num_time_steps)then
             delta = delta + &
                  this%di_msg(t,s)%val(:this%num_vertex_features(t),:)
          end if
          delta = delta * this%transfer%differentiate(this%z(t,s)%val(:,:))

          ! Partial derivatives of error wrt weights
          ! dE/dW = o/p(l-1) * delta
          do v = 1, this%graph(s)%num_vertices
             ! GET VERTEX DEGREE FOR sparse graph
             degree = this%graph(s)%adj_ia(v+1) - this%graph(s)%adj_ia(v)
             degree = max( &
                  this%min_vertex_degree, &
                  min(degree, this%max_vertex_degree) &
             )
             ! i.e. outer product of the input and delta
             ! sum weights and biases errors to use in batch gradient descent
             do e = this%graph(s)%adj_ia(v), this%graph(s)%adj_ia(v+1) - 1
                !if(this%graph(s)%adj_ja(2,e).eq.0) cycle ! self interaction
                if(this%graph(s)%adj_ja(2,e).eq.0)then
                   do i = 1, this%num_vertex_features(t-1)
                      dw(:,i,degree) = dw(:,i,degree) + &
                           this%vertex_features(t-1,s)%val(i,v) * delta(:,v)
                   end do
                   if(t.eq.1)then
                      this%di(1,s)%val(:,v) = &
                           this%di(1,s)%val(:,v) + &
                           matmul( &
                                delta(:,v), &
                                weight( &
                                     :,:this%num_vertex_features(t-1),degree &
                                ) &
                           )
                   else
                      this%di_msg(t,s)%val(:,v) = &
                           this%di_msg(t,s)%val(:,v) + &
                           matmul(delta(:,v),weight(:,:,degree))
                   end if
                else
                   do j = 1, this%num_vertex_features(t)
                      do i = 1, this%num_vertex_features(t-1)
                         dw(j,i,degree) = dw(j,i,degree) + &
                              this%vertex_features(t-1,s)%val( &
                                   i,this%graph(s)%adj_ja(1,e) &
                              ) * delta(j,v)
                      end do
                      do i = this%num_vertex_features(t-1) + 1, &
                           this%num_vertex_features(t-1) + this%num_edge_features(0)
                         dw(j,i,degree) = dw(j,i,degree) + &
                              this%edge_features(0,s)%val( &
                                   i-this%num_vertex_features(t-1), &
                                   this%graph(s)%adj_ja(2,e) &
                              ) * &
                              delta(j,v)
                      end do
                   end do
                   ! The errors are summed from the delta of the ...
                   ! ... 'child' node * 'child' weight
                   ! dE/dI(l-1) = sum(weight(l) * delta(l))
                   ! this prepares dE/dI for when it is passed into the previous layer
                   if(t.eq.1)then
                      this%di(1,s)%val(:,this%graph(s)%adj_ja(1,e)) = &
                           this%di(1,s)%val(:,this%graph(s)%adj_ja(1,e)) + &
                           matmul( &
                                delta(:,v), &
                                weight( &
                                     :,:this%num_vertex_features(t-1),degree &
                                ) &
                           )
                      this%di(2,s)%val(:,this%graph(s)%adj_ja(2,e)) = &
                           this%di(2,s)%val(:,this%graph(s)%adj_ja(2,e)) + &
                           matmul( &
                                delta(:,v), &
                                weight( &
                                     :,this%num_vertex_features(t-1)+1:,degree &
                                ) &
                           )
                   else
                      this%di_msg(t,s)%val(:,this%graph(s)%adj_ja(1,e)) = &
                           this%di_msg(t,s)%val(:,this%graph(s)%adj_ja(1,e)) + &
                           matmul(delta(:,v),weight(:,:,degree))
                   end if
                end if
             end do
          end do
       end do
    end do

  end subroutine backward_message_duvenaud
!###############################################################################


!##############################################################################!
  subroutine backward_readout_duvenaud(this, gradient)
    !! Backward pass for the readout phase
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

    ! Local variables
    integer :: i, j
    !! Loop indices
    integer :: s, v, t, num_params_old, num_params_tmp
    !! Batch index, vertex index, time step index
    real(real32), dimension(this%num_outputs) :: delta
    !! Delta values for the readout phase
    !! i.e. partial derivatives of the error wrt the hidden features
    real(real32), pointer :: weight(:,:), dw(:,:)
    !! Pointer to the weight matrix


    num_params_old = sum(this%num_params_msg)
    do t = 1, this%num_time_steps, 1
       num_params_tmp = this%num_vertex_features(t) * this%num_outputs
       weight( &
            1:this%num_outputs, &
            1:this%num_vertex_features(t) &
       ) => this%params( &
            num_params_old + 1 : num_params_old + num_params_tmp &
       )
       do concurrent(s=1:this%batch_size)
          dw( &
               1:this%num_outputs, &
               1:this%num_vertex_features(t) &
          ) => this%dp( &
               num_params_old + 1 : num_params_old + num_params_tmp, &
               s &
          )
          ! There is no message passing transfer function
          ! Partial derivatives of error wrt weights
          ! dE/dW = o/p(l-1) * delta
          do v = 1, this%graph(s)%num_vertices

             delta = &
                  gradient(1,1)%val(:,s) * &
                  this%transfer_readout%differentiate( &
                       this%z_readout(t,s)%val(:,v) &
                  )

             do j = 1, this%num_vertex_features(t)
                do i = 1, this%num_outputs
                   dw(i,j) = dw(i,j) + &
                        this%vertex_features(t,s)%val(j,v) * delta(i)
                end do
             end do

             this%di_readout(t,s)%val(:,v) = matmul(delta(:), weight(:,:))
          end do
       end do
       num_params_old = num_params_old + num_params_tmp
    end do

  end subroutine backward_readout_duvenaud
!###############################################################################

end module athena__duvenaud_msgpass_layer
