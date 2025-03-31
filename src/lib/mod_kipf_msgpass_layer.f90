module athena__kipf_msgpass_layer
  !! Module containing the types and interfacees of a message passing layer
  use athena__constants, only: real32
  use graphstruc, only: graph_type
  use athena__misc_types, only: activation_type, initialiser_type, &
       array_type, array2d_type
  use athena__msgpass_layer, only: msgpass_layer_type
  implicit none


  private

  public :: kipf_msgpass_layer_type


!-------------------------------------------------------------------------------
! Message passing layer
!-------------------------------------------------------------------------------
  type, extends(msgpass_layer_type) :: kipf_msgpass_layer_type

     ! this is for chen 2021 et al
     !  type(array2d_type), dimension(:), allocatable :: edge_weight
     !  !! Weights for the edges
     !  type(array2d_type), dimension(:), allocatable :: vertex_weight
     !  !! Weights for the vertices

     real(real32), pointer :: weight(:,:,:) => null()
     !! Weights for the message passing layer
     real(real32), pointer :: dw(:,:,:,:) => null()
     !! Weight gradients

   contains
     procedure, pass(this) :: get_num_params => get_num_params_kipf
     !! Get the number of parameters for the message passing layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_kipf
     !! Set the hyperparameters for the message passing layer
     procedure, pass(this) :: init => init_kipf
     !! Initialise the message passing layer
     procedure, pass(this) :: set_batch_size => set_batch_size_kipf
     !! Set batch size
     procedure, pass(this) :: read => read_kipf
     !! Read the message passing layer

     procedure, pass(this) :: forward => forward_rank
     !! Forward pass for message passing layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward pass for message passing layer

     procedure, pass(this) :: update_message => update_message_kipf
     !! Update the message
     procedure, pass(this) :: backward_message => backward_message_kipf
     !! Backward pass for the message phase

     procedure, pass(this) :: update_readout => update_readout_kipf
     !! Update the readout
     procedure, pass(this) :: backward_readout => backward_readout_kipf
     !! Backward pass for the readout phase
  end type kipf_msgpass_layer_type

  ! Interface for setting up the MPNN layer
  !-----------------------------------------------------------------------------
  interface kipf_msgpass_layer_type
     !! Interface for setting up the MPNN layer
     module function layer_setup( &
          num_features, num_time_steps, batch_size, &
          activation_function, activation_scale, &
          kernel_initialiser, &
          verbose &
     ) result(layer)
       !! Set up the message passing layer
       integer, dimension(2), intent(in) :: num_features
       !! Number of features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, optional, intent(in) :: batch_size
       !! Batch size
       real(real32), optional, intent(in) :: activation_scale
       !! Activation scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser
       !! Activation function and kernel initialiser
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(kipf_msgpass_layer_type) :: layer
       !! Instance of the message passing layer
     end function layer_setup
  end interface kipf_msgpass_layer_type

contains


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure function get_num_params_kipf(this) result(num_params)
    !! Get the number of parameters for the message passing layer
    !!
    !! This function calculates the number of parameters for the message passing
    !! layer.
    !! This procedure is based on code from the neural-fortran library
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(in) :: this
    !! Instance of the message passing layer
    integer :: num_params
    !! Number of parameters

    num_params = ( this%num_vertex_features(1) ** 2 ) * this%num_time_steps

  end function get_num_params_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine forward_rank(this, input)
    !! Forward pass for message
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
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
    class(kipf_msgpass_layer_type), intent(inout) :: this
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
       num_features, num_time_steps, batch_size, &
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
    integer, optional, intent(in) :: batch_size
    !! Batch size
    real(real32), optional, intent(in) :: activation_scale
    !! Activation scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser
    !! Activation function and kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    type(kipf_msgpass_layer_type) :: layer
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
         num_time_steps = num_time_steps, &
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
    call layer%init(input_shape=[layer%num_vertex_features])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_kipf( &
       this, &
       num_vertex_features, num_edge_features, &
       num_time_steps, &
       activation_function, activation_scale, &
       kernel_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for the message passing layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, intent(in) :: num_edge_features
    !! Number of edge features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    character(*), intent(in) :: activation_function
    !! Activation function
    real(real32), optional, intent(in) :: activation_scale
    !! Activation scale
    character(*), optional, intent(in) :: kernel_initialiser
    !! Kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    this%name = 'kipf'
    this%type = 'msgp'
    this%input_rank = 1
    this%num_vertex_features = num_vertex_features
    this%num_edge_features = num_edge_features
    this%num_time_steps = num_time_steps
    this%use_graph_input = .true.
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale))
    if(trim(kernel_initialiser).eq.'') &
         this%kernel_initialiser = get_default_initialiser(activation_function)
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("KIPF activation function: ",A)') &
               trim(activation_function)
          write(*,'("KIPF kernel initialiser: ",A)') &
               trim(this%kernel_initialiser)
       end if
    end if

  end subroutine set_hyperparams_kipf
!###############################################################################


!###############################################################################
  subroutine init_kipf(this, input_shape, batch_size, verbose)
    !! Initialise the message passing layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
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
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%output_shape = [this%num_vertex_features]
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
         this%params, &
         fan_in = this%num_vertex_features(1), &
         fan_out = this%num_vertex_features(1), &
         spacing = [ this%num_vertex_features(1) ] &
    )
    deallocate(initialiser_)


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_kipf
!###############################################################################


!###############################################################################
  subroutine set_batch_size_kipf(this, batch_size, verbose)
    !! Set the batch size for message passing layer
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout), target :: this
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
    this%weight( &
         1:this%num_vertex_features(1), &
         1:this%num_vertex_features(1), &
         1:this%num_time_steps &
    ) => this%params


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       allocate(this%output(2,this%batch_size), source=array2d_type())
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
       this%dw( &
            1:this%num_vertex_features(1), &
            1:this%num_vertex_features(1), &
            1:this%num_time_steps, &
            1:this%batch_size &
       ) => this%dp
       if(allocated(this%di)) deallocate(this%di)
       allocate(this%di(2,this%batch_size), source=array2d_type())
       !! input val arrays are allocated in set_graph
    end if


    write(*,*) "test0"
    if(allocated(this%vertex_features)) deallocate(this%vertex_features)
    allocate(this%vertex_features(0:this%num_time_steps, 1:this%batch_size))
    write(*,*) "test1"
    if(allocated(this%edge_features)) deallocate(this%edge_features)
    allocate(this%edge_features(0:this%num_time_steps, 1:this%batch_size))
    write(*,*) "test2"
    if(allocated(this%message)) deallocate(this%message)
    allocate(this%message(1:this%num_time_steps, 1:this%batch_size))
    write(*,*) "test3"

  end subroutine set_batch_size_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine read_kipf(this, unit, verbose)
    !! Read the message passing layer
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: unit
    !! Unit to read from
    integer, optional, intent(in) :: verbose
    !! Verbosity level
  end subroutine read_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!



  pure subroutine update_message_kipf(this, input)
    !! Update the message
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input to the message passing layer

    ! Local variables
    integer :: s, v, e, t
    !! Batch index, vertex index, edge index, time step
    real(real32) :: c
    !! Normalisation constant for the message passing

    ! real(real32), dimension(:,:), allocatable :: xe


    do s = 1, this%batch_size
       this%vertex_features(0,s)%val = input(1,s)%val
       this%edge_features(0,s)%val = input(2,s)%val
    end do

    do t = 1, this%num_time_steps
       do concurrent (s = 1: this%batch_size)
          do v = 1, this%graph(s)%num_vertices
             !  allocate( xe(2 * this%num_vertex_features + this%num_edge_features, this%graph(s)%vertex(v)%degree) )
             do e = this%graph(s)%adj_ia(v), this%graph(s)%adj_ia(v+1) - 1

                ! this is from Chen 2021 et al
                ! xe(:,e-this%graph(s)%adj_ia(v)+1) = [ &
                !      ( this%vertex_features(t-1,s)%val(:,v) + this%vertex_features(t-1,s)%val(:,this%graph(s)%adj_ja(1,e)) ) / 2._real32, &
                !      abs( this%vertex_features(t-1,s)%val(:,v) - this%vertex_features(t-1,s)%val(:,this%graph(s)%adj_ja(1,e)) ) / 2._real32, &
                !      this%edge_features(t-1,s)%val(:,e) &
                ! ]

                ! xe(:,e-this%graph(s)%adj_ia(v)+1) = matmul( &
                !      this%edge_weight(t)%val(:,:), &
                !      xe(:,e-this%graph(s)%adj_ia(v)+1) &
                ! )



                if( this%graph(s)%adj_ja(2,e) .eq. 0 )then
                   c = 1._real32
                else
                   c = this%graph(s)%edge_weights(this%graph(s)%adj_ja(2,e))
                end if
                ! fix this for lower memory case, where we don't store the vertices as derived types
                c = c * ( &
                     ( this%graph(s)%adj_ia(v+1) - this%graph(s)%adj_ia(v) ) * &
                     ( &
                          ( this%graph(s)%adj_ia( &
                               this%graph(s)%adj_ja(1,e) + 1 &
                          ) - this%graph(s)%adj_ia( &
                               this%graph(s)%adj_ja(1,e) &
                          ) ) &
                     ) ) ** ( -0.5_real32 )

                ! c = c * ( &
                !      ( this%graph(s)%vertex(v)%degree + 1 ) * &
                !      ( &
                !           this%graph(s)%vertex( &
                !                this%graph(s)%adj_ja(1,e) &
                !           )%degree + 1 &
                !      ) &
                ! ) ** ( -0.5_real32 )
                this%message(t,s)%val(:,v) = &
                     this%message(t,s)%val(:,v) + &
                     c * [ &
                          this%vertex_features(t-1,s)%val( &
                               :, &
                               this%graph(s)%adj_ja(1,e) &
                          ) &
                     ]
             end do
             this%z(t,s)%val(:,v) = matmul( &
                  this%message(t,s)%val(:,v), &
                  this%weight(:,:,t) &
             )
          end do
          this%vertex_features(t,s)%val(:,:) = &
               this%transfer%activate( this%z(t,s)%val(:,:) )
       end do
    end do

  end subroutine update_message_kipf


  pure subroutine update_readout_kipf(this)
    !! Update the readout
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer

    ! Local variables
    integer :: s, v
    !! Loop indices


    do s = 1, this%batch_size
       this%output(1,s)%val = this%vertex_features(this%num_time_steps,s)%val
       this%output(2,s)%val = this%edge_features(this%num_time_steps,s)%val
    end do

  end subroutine update_readout_kipf


  pure subroutine backward_message_kipf(this, input, gradient)
    !! Backward pass for the message phase
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data (i.e. vertex and edge features)
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

  end subroutine backward_message_kipf



  pure subroutine backward_readout_kipf(this, gradient)
    !! Backward pass for the readout phase
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

  end subroutine backward_readout_kipf




end module athena__kipf_msgpass_layer
