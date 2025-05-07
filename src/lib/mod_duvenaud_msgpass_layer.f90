module athena__duvenaud_msgpass_layer
  !! Module containing the types and interfacees of a message passing layer
  use athena__constants, only: real32
  use graphstruc, only: graph_type
  use athena__misc_types, only: activation_type, initialiser_type, &
       array_type, array2d_type
  use athena__msgpass_layer, only: msgpass_layer_type
  implicit none


  private

  public :: duvenaud_msgpass_layer_type


!-------------------------------------------------------------------------------
! Message passing layer
!-------------------------------------------------------------------------------
  type, extends(msgpass_layer_type) :: duvenaud_msgpass_layer_type

     integer :: max_vertex_degree = 0
     !! Maximum vertex degree
     real(real32), pointer :: weight_msg(:,:,:,:) => null(), &
          weight_readout(:,:,:) => null()
     !! Weights for the message passing layer

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
          num_features, num_time_steps, &
          max_vertex_degree, &
          num_outputs, &
          batch_size, &
          activation_function, activation_scale, &
          kernel_initialiser, &
          verbose &
     ) result(layer)
       !! Set up the message passing layer
       integer, dimension(2), intent(in) :: num_features
       !! Number of features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, intent(in) :: max_vertex_degree
       !! Maximum vertex degree
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: batch_size
       !! Batch size
       real(real32), optional, intent(in) :: activation_scale
       !! Activation scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser
       !! Activation function and kernel initialiser
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

    if(associated(this%weight_msg)) nullify(this%weight_msg)
    if(associated(this%weight_readout)) nullify(this%weight_readout)
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
         this%max_vertex_degree * &
         this%num_time_steps + &
         this%num_vertex_features(0) * this%num_outputs * this%num_time_steps

  end function get_num_params_duvenaud
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_rank(this, input)
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
  pure subroutine backward_rank(this, input, gradient)
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
       num_features, num_time_steps, &
       max_vertex_degree, &
       num_outputs, &
       batch_size, &
       activation_function, activation_scale, &
       kernel_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the message passing layer
    implicit none

    ! Arguments
    integer, dimension(2), intent(in) :: num_features
    !! Number of features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, intent(in) :: max_vertex_degree
    !! Maximum vertex degree
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: batch_size
    !! Batch size
    real(real32), optional, intent(in) :: activation_scale
    !! Activation scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser
    !! Activation function and kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    type(duvenaud_msgpass_layer_type) :: layer
    !! Instance of the message passing layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    real(real32) :: scale = 1._real32
    !! Activation scale
    character(len=10) :: activation_function_ = "none"
    !! Activation function

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation_function)) activation_function_ = activation_function
    if(present(activation_scale)) scale = activation_scale


    !---------------------------------------------------------------------------
    ! Define weights (kernels) and biases initialisers
    !---------------------------------------------------------------------------
    if(present(kernel_initialiser)) layer%kernel_initialiser =kernel_initialiser


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_vertex_features = num_features(1), &
         num_edge_features = num_features(2), &
         max_vertex_degree = max_vertex_degree, &
         num_time_steps = num_time_steps, &
         num_outputs = num_outputs, &
         activation_function = activation_function_, &
         activation_scale = scale, &
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
    call layer%init(input_shape=[layer%num_vertex_features(1)])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_duvenaud( &
       this, &
       num_vertex_features, num_edge_features, &
       max_vertex_degree, &
       num_time_steps, &
       num_outputs, &
       activation_function, activation_scale, &
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
    integer, intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, intent(in) :: num_edge_features
    !! Number of edge features
    integer, intent(in) :: max_vertex_degree
    !! Maximum vertex degree
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, intent(in) :: num_outputs
    !! Number of outputs
    character(*), intent(in) :: activation_function
    !! Activation function
    real(real32), optional, intent(in) :: activation_scale
    !! Activation scale
    character(*), optional, intent(in) :: kernel_initialiser
    !! Kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = 'duvenaud'
    this%type = 'msgp'
    this%input_rank = 1
    this%max_vertex_degree = max_vertex_degree
    this%num_time_steps = num_time_steps
    this%num_outputs = num_outputs
    if(allocated(this%num_vertex_features)) &
         deallocate(this%num_vertex_features)
    if(allocated(this%num_edge_features)) &
         deallocate(this%num_edge_features)
    allocate( &
         this%num_vertex_features(0:this%num_time_steps), &
         source = num_vertex_features &
    )
    allocate( &
         this%num_edge_features(0:0), &
         source = num_edge_features &
    )
    this%use_graph_input = .true.
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale))
    if(trim(kernel_initialiser).eq.'') &
         this%kernel_initialiser = get_default_initialiser(activation_function)
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("DUVENAUD activation function: ",A)') &
               trim(activation_function)
          write(*,'("DUVENAUD kernel initialiser: ",A)') &
               trim(this%kernel_initialiser)
       end if
    end if

    if(allocated(this%num_params_msg)) deallocate(this%num_params_msg)
    allocate(this%num_params_msg(1:this%num_time_steps))
    this%num_params_msg = &
         ( this%num_vertex_features(0) + this%num_edge_features(0) ) * &
         this%num_vertex_features(0) * &
         this%max_vertex_degree
    this%num_params_readout = &
         this%num_vertex_features(0) * this%num_outputs * &
         this%num_time_steps

    allocate(this%transfer_readout, source=activation_setup('softmax'))

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
    integer :: verbose_ = 0
    !! Verbosity level
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


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%kernel_initialiser))
    call initialiser_%initialise( &
         this%params(1:sum(this%num_params_msg)), &
         fan_in = this%num_vertex_features(0) + this%num_edge_features(0), &
         fan_out = this%num_vertex_features(0), &
         spacing = [ this%num_vertex_features(0) ] &
    )
    call initialiser_%initialise( &
         this%params(sum(this%num_params_msg)+1:this%num_params), &
         fan_in = this%num_vertex_features(0), &
         fan_out = this%num_outputs, &
         spacing = [ this%num_vertex_features(0) ] &
    )
    deallocate(initialiser_)


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
    ! Set weights and biases pointers to params array
    !---------------------------------------------------------------------------
    this%weight_msg( &
         1:this%num_vertex_features(0) + this%num_edge_features(0), &
         1:this%num_vertex_features(0), &
         1:this%max_vertex_degree, &
         1:this%num_time_steps &
    ) => this%params(1:sum(this%num_params_msg))
    this%weight_readout( &
         1:this%num_vertex_features(0), &
         1:this%num_outputs, &
         1:this%num_time_steps &
    ) => this%params(sum(this%num_params_msg)+1:this%num_params)


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
    allocate(this%edge_features(0:0, 1:this%batch_size))
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
       call this%graph(s)%copy(graph(s), sparse=.true.)
    end do

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
            [ this%num_vertex_features, this%graph(s)%num_vertices ] &
       )
       call this%edge_features(0,s)%allocate( &
            [ this%num_edge_features, this%graph(s)%num_edges ] &
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


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!



!##############################################################################!
  pure subroutine update_message_duvenaud(this, input)
    !! Update the message
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input to the message passing layer

    ! Local variables
    integer :: s, v, e, t
    !! Batch index, vertex index, edge index, time step
    integer :: degree
    !! Degree of the vertex


    do s = 1, this%batch_size
       this%vertex_features(0,s)%val = input(1,s)%val
       this%edge_features(0,s)%val = input(2,s)%val
    end do

    do t = 1, this%num_time_steps
       do concurrent (s = 1: this%batch_size)
          do v = 1, this%graph(s)%num_vertices
             degree = this%graph(s)%adj_ia(v+1) - this%graph(s)%adj_ia(v)
             if(degree .gt. this%max_vertex_degree) &
                  degree = this%max_vertex_degree
             if(degree .eq. 0) cycle
             do e = this%graph(s)%adj_ia(v), this%graph(s)%adj_ia(v+1) - 1
                if(this%graph(s)%adj_ja(2,e).eq.0) cycle ! self interaction
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
             end do
             this%z(t,s)%val(:,v) = matmul( &
                  this%message(t,s)%val(:,v), &
                  this%weight_msg(:,:,degree,t) &
             )
          end do
          this%vertex_features(t,s)%val(:,:) = &
               this%transfer%activate( this%z(t,s)%val(:,:) )
       end do
    end do

  end subroutine update_message_duvenaud
!##############################################################################!


!##############################################################################!
  pure subroutine update_readout_duvenaud(this)
    !! Update the readout
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer

    ! Local variables
    integer :: s, v, t
    !! Loop indices


    ! combine %weight and %weight_readout and convert to 1D for each time step
    ! then, in each update_ procedure, use an associate block to convert them
    ! to the appropriate shape for the matmul
    this%output(1,1)%val = 0._real32
    do s = 1, this%batch_size
       do t = 1, this%num_time_steps, 1
          do v = 1, this%graph(s)%num_vertices
             this%z_readout(t,s)%val(:,v) = matmul( &
                  this%vertex_features(t,s)%val(:,v), &
                  this%weight_readout(:,:,t) &
             )
             this%output(1,1)%val(:,s) = this%output(1,1)%val(:,s) + &
                  this%transfer_readout%activate( this%z_readout(t,s)%val(:,v) )
          end do
       end do
    end do

  end subroutine update_readout_duvenaud
!##############################################################################!


!##############################################################################!
  pure subroutine backward_message_duvenaud(this, input, gradient)
    !! Backward pass for the message phase
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data (i.e. vertex and edge features)
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

    integer :: degree
    integer :: t, s, v, i, j, idx
    real(real32), dimension(:,:), allocatable :: delta


    this%dp = 0._real32
    do t = this%num_time_steps, 1, -1
       do concurrent(s=1:this%batch_size)
          ! There is no message passing transfer function
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
             degree = &
                  min( &
                       degree, &!this%graph(s)%vertex(v)%degree, &
                       this%max_vertex_degree &
                  )
             ! i.e. outer product of the input and delta
             ! sum weights and biases errors to use in batch gradient descent
             do concurrent ( &
                  i = 1:this%num_vertex_features(t) + &
                  this%num_edge_features(0), &
                  j = 1:this%num_vertex_features(t) &
             )
                idx = i + &
                     ( &
                          this%num_vertex_features(t) + &
                          this%num_edge_features(0) &
                     ) * &
                     ( (j-1) + this%num_vertex_features(t) * ( &
                          (degree-1) + &
                          this%max_vertex_degree * (t-1) &
                     ) )
                ! ARE WE MISSING THE REST OF delta(:,v)?
                if(i.gt.this%num_vertex_features(t))then
                   this%dp(idx,s) = this%dp(idx,s) + &
                        this%edge_features(0,s)%val( &
                             i-this%num_vertex_features(t),v &
                        ) * &
                        delta(j,v)
                else
                   this%dp(idx,s) = this%dp(idx,s) + &
                        this%vertex_features(t,s)%val(i,v) * delta(j,v)
                end if
             end do
             ! The errors are summed from the delta of the ...
             ! ... 'child' node * 'child' weight
             ! dE/dI(l-1) = sum(weight(l) * delta(l))
             ! this prepares dE/dI for when it is passed into the previous layer
             if(t.eq.1)then
                this%di(1,s)%val(:,v) = &
                     matmul( &
                          this%weight_msg( &
                               :this%num_vertex_features(t),:,degree,t &
                          ), &
                          delta(:,v) &
                     )
                this%di(2,s)%val(:,v) = &
                     matmul( &
                          this%weight_msg( &
                               this%num_vertex_features(t)+1:,:,degree,t &
                          ), &
                          delta(:,v) &
                     )
             else
                this%di_msg(t,s)%val(:,v) = &
                     this%di_msg(t,s)%val(:,v) + &
                     matmul(this%weight_msg(:,:,degree,t), delta(:,v))
             end if
          end do
       end do
    end do

  end subroutine backward_message_duvenaud
!##############################################################################!


!##############################################################################!
  pure subroutine backward_readout_duvenaud(this, gradient)
    !! Backward pass for the readout phase
    implicit none

    ! Arguments
    class(duvenaud_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

    ! Local variables
    integer :: i, j, idx
    !! Loop indices
    integer :: s, v, t, num_features
    !! Batch index, vertex index, time step index
    real(real32), dimension(this%num_outputs) :: delta
    !! Delta values for the readout phase
    !! i.e. partial derivatives of the error wrt the hidden features


    do concurrent(s=1:this%batch_size)
       ! There is no message passing transfer function

       ! Partial derivatives of error wrt weights
       ! dE/dW = o/p(l-1) * delta
       do t = 1, this%num_time_steps, 1
          do v = 1, this%graph(s)%num_vertices

             delta = &
                  gradient(1,s)%val(:,1) * &
                  this%transfer_readout%differentiate( &
                       this%z_readout(t,s)%val(:,v) &
                  )

             do concurrent( &
                  j = 1:this%num_vertex_features(0), &
                  i = 1:this%num_outputs &
             )
                idx = i + (j-1) * this%num_outputs + (t-1) * &
                     this%num_outputs * &
                     this%num_vertex_features(1)
                this%dp(idx,s) = this%dp(idx,s) + &
                     this%vertex_features(t,s)%val(j,v) * delta(i)
             end do

             this%di_readout(t,s)%val(:,v) = &
                  matmul(this%weight_readout(:,:,t), delta(:))
          end do
       end do
    end do

  end subroutine backward_readout_duvenaud
!##############################################################################!

end module athena__duvenaud_msgpass_layer
