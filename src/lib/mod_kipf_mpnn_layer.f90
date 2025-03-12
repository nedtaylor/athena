module athena__kipf_mpnn_layer
  !! Module containing the convolutional message passing layer
  !!
  !! This module contains the implementation of the convolutional message passing
  !! neural network layer. This layer is based on the work of Kipf et al. (2016)
  !! https://openreview.net/pdf?id=SJU4ayYgl
  use athena__constants, only: real32
  use athena__misc, only: outer_product
  use athena__misc_types, only: activation_type, initialiser_type
  use graphstruc, only: graph_type
  use athena__activation, only: activation_setup
  use athena__clipper, only: clip_type
  use athena__initialiser, only: initialiser_setup
  use athena__mpnn_layer, only: &
       mpnn_layer_type, method_container_type, &
       message_phase_type, readout_phase_type, &
       feature_type
  implicit none


  private

  public :: kipf_mpnn_layer_type


!-------------------------------------------------------------------------------
! Type for the convolutional message passing phase
!-------------------------------------------------------------------------------
  type, extends(message_phase_type) :: kipf_message_phase_type
     !! Type for the convolutional message passing phase
     real(real32), pointer :: weight(:,:) => null()
     !! Weight matrix for the convolutional readout phase
     real(real32), pointer :: dw(:,:,:) => null()
     !! Gradient matrix for the convolutional readout phase
     type(feature_type), dimension(:), allocatable :: z
     !! Hidden features for the convolutional message passing phase
     class(activation_type), allocatable :: transfer
     !! Transfer function for the convolutional readout phase
   contains
     procedure :: get_num_params => get_num_params_message_kipf
     !! Get the number of learnable parameters
     procedure :: get_params => get_params_message_kipf
     !! Get the learnable parameters
     procedure :: set_params => set_params_message_kipf
     !! Set the learnable parameters
     procedure :: get_gradients => get_gradients_message_kipf
     !! Get the gradients
     procedure :: set_gradients => set_gradients_message_kipf
     !! Set the gradients
     procedure :: set_shape => set_shape_message_kipf
     !! Set the shape of the phase
     procedure :: update => update_message_kipf
     !! Update the phase
     procedure :: calculate_partials => calculate_partials_message_kipf
     !! Calculate the partials for the phase
  end type kipf_message_phase_type

  ! Interface for the convolutional message passing phase
  !-----------------------------------------------------------------------------
  interface kipf_message_phase_type
     !! Interface for the convolutional message passing phase
     module function message_phase_setup( &
          num_vertex_features, num_edge_features, &
          batch_size &
     ) result(message_phase)
       !! Setup the convolutional message passing phase
       integer, intent(in) :: num_vertex_features
       !! Number of vertex features
       integer, intent(in) :: num_edge_features
       !! Number of edge features
       integer, intent(in) :: batch_size
       !! Batch size
       type(kipf_message_phase_type) :: message_phase
       !! Instance of the convolutional message passing phase
     end function message_phase_setup
  end interface kipf_message_phase_type
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Type for the convolutional readout phase
!-------------------------------------------------------------------------------
  type, extends(readout_phase_type) :: kipf_readout_phase_type
     !! Type for the convolutional readout phase
     real(real32), pointer :: weight(:,:) => null()
     !! Weight matrix for the convolutional readout phase
     real(real32), pointer :: dw(:,:,:) => null()
     !! Gradient matrix for the convolutional readout phase
     type(feature_type), dimension(:,:), allocatable :: z
     !! Hidden features for the convolutional readout phase
     class(activation_type), allocatable :: transfer
     !! Transfer function for the convolutional readout phase
   contains
     procedure :: get_num_params => get_num_params_readout_kipf
     !! Get the number of learnable parameters
     procedure :: get_params => get_params_readout_kipf
     !! Get the learnable parameters
     procedure :: set_params => set_params_readout_kipf
     !! Set the learnable parameters
     procedure :: get_gradients => get_gradients_readout_kipf
     !! Get the gradients
     procedure :: set_gradients => set_gradients_readout_kipf
     !! Set the gradients
     procedure :: set_shape => set_shape_readout_kipf
     !! Set the shape of the phase
     procedure :: get_output => get_output_readout_kipf
     !! Get the output of the phase
     procedure :: calculate_partials => calculate_partials_readout_kipf
     !! Calculate the partials for the phase
  end type kipf_readout_phase_type

  ! Interface for the convolutional readout passing phase
  !-----------------------------------------------------------------------------
  interface kipf_readout_phase_type
     !! Interface for the convolutional readout passing phase
     module function readout_phase_setup( &
          num_inputs, num_outputs, batch_size &
     ) result(readout_phase)
       !! Setup the convolutional readout phase
       integer, intent(in) :: num_inputs
       !! Number of input features
       integer, intent(in) :: num_outputs
       !! Number of output features
       integer, intent(in) :: batch_size
       !! Batch size
       type(kipf_readout_phase_type) :: readout_phase
       !! Instance of the convolutional readout phase
     end function readout_phase_setup
  end interface kipf_readout_phase_type
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Type for the convolutional message passing neural network method container
!-------------------------------------------------------------------------------
  type, extends(method_container_type) :: kipf_method_container_type
     !! Type for the convolutional message passing neural network method container
   contains
     procedure, pass(this) :: init => init_kipf_mpnn_method
     !! Initialise the method container
     procedure, pass(this) :: set_batch_size => set_batch_size_kipf_mpnn_method
     !! Set the batch size
  end type kipf_method_container_type

  ! Interface for the convolutional MPNN method container
  !-----------------------------------------------------------------------------
  interface kipf_method_container_type
     !! Interface for the convolutional MPNN method container
     module function method_setup( &
          num_vertex_features, num_edge_features, num_time_steps, &
          output_shape, &
          batch_size, verbose &
     ) result(method)
       !! Setup the convolutional MPNN method container
       integer, intent(in) :: num_vertex_features
       !! Number of vertex features
       integer, intent(in) :: num_edge_features
       !! Number of edge features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, dimension(1), intent(in) :: output_shape
       !! Output shape
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(kipf_method_container_type) :: method
       !! Instance of the convolutional MPNN method container
     end function method_setup

  end interface kipf_method_container_type
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Type for the convolutional message passing neural network layer
!-------------------------------------------------------------------------------
  type, extends(mpnn_layer_type) :: kipf_mpnn_layer_type
     !! Type for the convolutional message passing neural network layer
   contains
     procedure, pass(this) :: set_hyperparams_extd => set_hyperparams_kipf
     !! Set the hyperparameters
     procedure :: set_param_pointers => set_param_pointers_kipf
     !! Set the pointers to the learnable parameters
     procedure :: backward_graph => backward_graph_kipf
     !! Backward pass for the graph
  end type kipf_mpnn_layer_type

  ! Interface for the convolutional MPNN layer
  !-----------------------------------------------------------------------------
  interface kipf_mpnn_layer_type
     module function layer_setup( &
          num_time_steps, num_features, num_outputs, &
          batch_size, verbose &
     ) result(layer)
       !! Setup the convolutional MPNN layer
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, dimension(2), intent(in) :: num_features
       !! Number of features
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: batch_size
       !! Batch size
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(kipf_mpnn_layer_type) :: layer
       !! Instance of the convolutional MPNN layer
     end function layer_setup
  end interface kipf_mpnn_layer_type
!-------------------------------------------------------------------------------



contains

!###############################################################################
  pure function get_num_params_message_kipf(this) result(num_params)
    !! Get the number of learnable parameters for the message phase
    implicit none

    ! Arguments
    class(kipf_message_phase_type), intent(in) :: this
    !! Instance of the message phase
    integer :: num_params
    !! Number of learnable parameters

    num_params = &
         this%num_message_features * &
         this%num_message_features
  end function get_num_params_message_kipf
!-------------------------------------------------------------------------------
  pure function get_num_params_readout_kipf(this) result(num_params)
    !! Get the number of learnable parameters for the readout phase
    implicit none

    ! Arguments
    class(kipf_readout_phase_type), intent(in) :: this
    !! Instance of the readout phase
    integer :: num_params
    !! Number of learnable parameters

    num_params = 0
  end function get_num_params_readout_kipf
!###############################################################################


!###############################################################################
  pure module function get_params_message_kipf(this) result(params)
    !! Get the learnable parameters for the message phase
    implicit none

    ! Arguments
    class(kipf_message_phase_type), intent(in) :: this
    !! Instance of the message phase
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    ! Local variables
    integer :: t
    !! Time step index

    params = reshape(this%weight, [ size(this%weight) ])
  end function get_params_message_kipf
!-------------------------------------------------------------------------------
  pure module function get_params_readout_kipf(this) result(params)
    !! Get the learnable parameters for the readout phase
    implicit none

    ! Arguments
    class(kipf_readout_phase_type), intent(in) :: this
    !! Instance of the readout phase
    real(real32), dimension(this%num_params) :: params
    !! Learnable parameters

    ! Local variables
    integer :: t
    !! Time step index

    ! params = reshape(this%weight, [ size(this%weight) ])
  end function get_params_readout_kipf
!###############################################################################


!###############################################################################
  pure subroutine set_params_message_kipf(this, params)
    !! Set the learnable parameters for the message phase
    implicit none

    ! Arguments
    class(kipf_message_phase_type), intent(inout) :: this
    !! Instance of the message phase
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    ! Local variables
    integer :: t
    !! Time step index

    this%weight = reshape(params, shape(this%weight))
  end subroutine set_params_message_kipf
!-------------------------------------------------------------------------------
  pure subroutine set_params_readout_kipf(this, params)
    !! Set the learnable parameters for the readout phase
    implicit none

    ! Arguments
    class(kipf_readout_phase_type), intent(inout) :: this
    !! Instance of the readout phase
    real(real32), dimension(this%num_params), intent(in) :: params
    !! Learnable parameters

    ! Local variables
    integer :: t
    !! Time step index

    ! this%weight = reshape(params, shape(this%weight))
  end subroutine set_params_readout_kipf
!###############################################################################


!###############################################################################
  module subroutine set_param_pointers_kipf(this)
    !! Set the pointers to the learnable parameters

    ! Arguments
    class(kipf_mpnn_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer

    ! Local variables
    integer :: i, num_params
    !! Loop index and number of learnable parameters
    class(initialiser_type), allocatable :: initialiser_
    !! Initialiser for the weights

    num_params = 1
    do i = 1, this%num_time_steps
       select type(message => this%method%message(i))
       type is(kipf_message_phase_type)
          message%weight( &
               1:message%num_message_features, &
               1:message%num_message_features &
          ) => this%params(num_params:num_params+message%num_params-1)
          message%dw( &
               1:message%num_message_features, &
               1:message%num_message_features, &
               1:this%batch_size &
          ) => this%dp
          num_params = num_params + message%num_params

          allocate(initialiser_, source=initialiser_setup("he_normal"))
          call initialiser_%initialise(message%weight(:,:), &
               fan_in=message%num_message_features, &
               fan_out=message%num_message_features)
          message%weight = &
               message%weight / (40._real32 * sum(message%weight))
          deallocate(initialiser_)

       end select
    end do
    ! select type(readout => this%method%readout)
    ! type is(kipf_readout_phase_type)
    !    readout%weight( &
    !         1:readout%num_inputs, &
    !         1:readout%num_outputs, &
    !    ) => this%params(num_params:num_params+readout%num_params-1)
    !    readout%dw( &
    !         1:readout%num_inputs, &
    !         1:readout%num_outputs, &
    !         1:this%num_time_steps+1, &
    !         1:this%batch_size &
    !    ) => this%db

    !    allocate(initialiser_, source=initialiser_setup("he_normal"))
    !    call initialiser_%initialise(readout%weight(:,:,:), &
    !         fan_in=readout%num_inputs, &
    !         fan_out=readout%num_outputs &
    !    )
    !    deallocate(initialiser_)
    !    readout%weight = readout%weight / size(readout%weight)

    ! end select

  end subroutine set_param_pointers_kipf
!###############################################################################


!###############################################################################
  pure function get_gradients_message_kipf(this, clip_method) result(gradients)
    !! Get the gradients for the message phase
    implicit none

    ! Arguments
    class(kipf_message_phase_type), intent(in) :: this
    !! Instance of the message phase
    type(clip_type), optional, intent(in) :: clip_method
    !! Instance of the clipper
    real(real32), dimension(this%num_params) :: gradients
    !! Gradients of the learnable parameters

    gradients = reshape(sum(this%dw,dim=3)/this%batch_size, &
         [ size(this%dw,1) * size(this%dw,2) ])

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)
  end function get_gradients_message_kipf
!-------------------------------------------------------------------------------
  pure function get_gradients_readout_kipf(this, clip_method) result(gradients)
    !! Get the gradients for the readout phase
    implicit none

    ! Arguments
    class(kipf_readout_phase_type), intent(in) :: this
    !! Instance of the readout phase
    type(clip_type), optional, intent(in) :: clip_method
    !! Instance of the clipper
    real(real32), dimension(this%num_params) :: gradients
    !! Gradients of the learnable parameters

    ! gradients = reshape(sum(this%dw,dim=4)/this%batch_size, &
    !      [ size(this%dw,1) * size(this%dw,2) * size(this%dw,3) ])

    ! if(present(clip_method)) call clip_method%apply(size(gradients),gradients)
  end function get_gradients_readout_kipf
!###############################################################################


!###############################################################################
  pure subroutine set_gradients_message_kipf(this, gradients)
    !! Set the gradients for the message phase
    implicit none

    ! Arguments
    class(kipf_message_phase_type), intent(inout) :: this
    !! Instance of the message phase
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients of the learnable parameters

    select rank(gradients)
    rank(0)
       this%dw = gradients
    rank(1)
       this%dw = spread(reshape(gradients, shape(this%dw(:,:,1))), 3, &
            this%batch_size)
    end select

  end subroutine set_gradients_message_kipf
!-------------------------------------------------------------------------------
  pure subroutine set_gradients_readout_kipf(this, gradients)
    !! Set the gradients for the readout phase
    implicit none

    ! Arguments
    class(kipf_readout_phase_type), intent(inout) :: this
    !! Instance of the readout phase
    real(real32), dimension(..), intent(in) :: gradients
    !! Gradients of the learnable parameters

    ! select rank(gradients)
    ! rank(0)
    !    this%dw = gradients
    ! rank(1)
    !    this%dw = spread(reshape(gradients, shape(this%dw(:,:,:,1))), 4, &
    !         this%batch_size)
    ! end select

  end subroutine set_gradients_readout_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine set_shape_message_kipf(this, shape)
    !! Set the shape of the message phase
    implicit none

    ! Arguments
    class(kipf_message_phase_type), intent(inout) :: this
    !! Instance of the message phase
    integer, dimension(:), intent(in) :: shape
    !! Shape of the phase

    ! Local variables
    integer :: s
    !! Batch index


    if(this%use_message)then
       do s = 1, this%batch_size
          if(allocated(this%message(s)%val)) deallocate(this%message(s)%val)
          allocate(this%message(s)%val(this%num_message_features, shape(s)))
       end do
    end if

    do s = 1, this%batch_size
       if(allocated(this%feature(s)%val)) deallocate(this%feature(s)%val)
       allocate(this%feature(s)%val(this%num_outputs, shape(s)))

       if(allocated(this%z(s)%val)) deallocate(this%z(s)%val)
       allocate(this%z(s)%val(this%num_outputs, shape(s)))
       if(allocated(this%di(s)%val)) deallocate(this%di(s)%val)
       allocate(this%di(s)%val(this%num_inputs, shape(s)))
    end do

  end subroutine set_shape_message_kipf
!-------------------------------------------------------------------------------
  subroutine set_shape_readout_kipf(this, shape)
    !! Set the shape of the readout phase
    implicit none

    ! Arguments
    class(kipf_readout_phase_type), intent(inout) :: this
    !! Instance of the readout phase
    integer, dimension(:), intent(in) :: shape
    !! Shape of the phase

    ! Local variables
    integer :: s, t
    !! Batch index and time step index

    ! do s = 1, this%batch_size
    !    do t = 0, this%num_time_steps, 1
    !       if(allocated(this%di(this%batch_size * t + s)%val)) &
    !            deallocate(this%di(this%batch_size * t + s)%val)
    !       allocate(this%di(this%batch_size * t + s)%val( &
    !            this%num_inputs, shape(s) &
    !       ))

    !       if(allocated(this%z(t+1,s)%val)) deallocate(this%z(t+1,s)%val)
    !       allocate(this%z(t+1,s)%val(this%num_outputs, shape(s)))
    !    end do
    ! end do

  end subroutine set_shape_readout_kipf
!###############################################################################


!###############################################################################
  module function message_phase_setup( &
       num_vertex_features, num_edge_features, &
       batch_size &
  ) result(message_phase)
    !! Setup the convolutional message passing phase
    implicit none

    ! Arguments
    integer, intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, intent(in) :: num_edge_features
    !! Number of edge features
    integer, intent(in) :: batch_size
    !! Batch size
    type(kipf_message_phase_type) :: message_phase
    !! Instance of the convolutional message passing phase


    message_phase%num_inputs  = num_vertex_features
    message_phase%num_outputs = num_vertex_features
    message_phase%num_message_features = num_vertex_features
    message_phase%batch_size  = batch_size

    allocate(message_phase%message(batch_size))
    allocate(message_phase%feature(batch_size))
    message_phase%num_params = message_phase%get_num_params()

    allocate(message_phase%z(batch_size))
    allocate(message_phase%di(batch_size))

    write(*,*) "setting up transfer function"
    allocate(message_phase%transfer, &
         source=activation_setup("relu", 1._real32))
    write(*,*) "transfer function set up"

  end function message_phase_setup
!-------------------------------------------------------------------------------
  module function readout_phase_setup( &
       num_inputs, num_outputs,batch_size &
  ) result(readout_phase)
    !! Setup the convolutional readout phase
    implicit none

    ! Arguments
    integer, intent(in) :: num_inputs
    !! Number of input features
    integer, intent(in) :: num_outputs
    !! Number of output features
    integer, intent(in) :: batch_size
    !! Batch size
    type(kipf_readout_phase_type) :: readout_phase
    !! Instance of the convolutional readout phase


    readout_phase%num_inputs  = num_inputs
    readout_phase%num_outputs = num_outputs
    readout_phase%batch_size  = batch_size
    readout_phase%num_params = readout_phase%get_num_params()

    ! allocate(readout_phase%dw( &
    !      num_inputs, num_outputs, num_time_steps+1, batch_size))
    ! allocate(readout_phase%z(num_time_steps+1, batch_size))
    ! allocate(readout_phase%di(batch_size * (readout_phase%num_time_steps + 1) ))

    ! write(*,*) "setting up transfer function"
    ! allocate(readout_phase%transfer, &
    !      source=activation_setup("softmax", 1._real32))
    ! write(*,*) "transfer function set up"

  end function readout_phase_setup
!###############################################################################


!###############################################################################
  subroutine init_kipf_mpnn_method( this, &
       num_vertex_features, num_edge_features, num_time_steps, &
       output_shape, batch_size, verbose &
  )
    !! Initialise the convolutional MPNN method container
    implicit none

    ! Arguments
    class(kipf_method_container_type), intent(inout) :: this
    !! Instance of the method container
    integer, intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, intent(in) :: num_edge_features
    !! Number of edge features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, dimension(1), intent(in) :: output_shape
    !! Output shape
    integer, optional, intent(in) :: batch_size
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
    if(present(batch_size)) this%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise input and output shape
    !---------------------------------------------------------------------------
    this%num_features = [num_vertex_features, num_edge_features]
    this%num_time_steps = num_time_steps
    this%num_outputs = output_shape(1)

    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_kipf_mpnn_method
!###############################################################################


!###############################################################################
  subroutine set_batch_size_kipf_mpnn_method(this, batch_size, verbose)
    !! Set the batch size for the convolutional MPNN method container
    implicit none

    ! Arguments
    class(kipf_method_container_type), intent(inout) :: this
    !! Instance of the method container
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


    if(allocated(this%message)) deallocate(this%message)

    allocate(this%message(0:this%num_time_steps), &
         source = kipf_message_phase_type( &
              this%num_features(1), this%num_features(2), &
              batch_size &
         ) &
    )
    this%message(0)%use_message = .false.

    if(allocated(this%readout)) deallocate(this%readout)
    allocate(this%readout, &
         source = kipf_readout_phase_type( &
              this%num_features(1), &
              this%num_outputs, batch_size &
         ) &
    )

  end subroutine set_batch_size_kipf_mpnn_method
!###############################################################################


!###############################################################################
  module function method_setup(num_vertex_features, num_edge_features, &
       num_time_steps, output_shape, &
       batch_size, verbose &
  ) result(method)
    !! Setup the convolutional MPNN method container
    implicit none

    ! Arguments
    integer, intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, intent(in) :: num_edge_features
    !! Number of edge features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, dimension(1), intent(in) :: output_shape
    !! Output shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(kipf_method_container_type) :: method
    !! Instance of the convolutional MPNN method container

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    if(present(verbose)) verbose_ = verbose

    if(present(batch_size)) method%batch_size = batch_size
    if(method%batch_size .gt. 0)then
       call method%init( &
            num_vertex_features, num_edge_features, num_time_steps, &
            output_shape, method%batch_size, verbose_ &
       )
    end if

  end function method_setup
!###############################################################################


!###############################################################################
  module function layer_setup( &
       num_time_steps, &
       num_features, &
       num_outputs, &
       batch_size, verbose &
  ) result(layer)
    !! Setup the convolutional MPNN layer
    implicit none

    ! Arguments
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, dimension(2), intent(in) :: num_features
    !! Number of features
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(kipf_mpnn_layer_type) :: layer
    !! Instance of the convolutional MPNN layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) layer%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams_extd( &
         num_features, num_time_steps, &
         num_outputs, verbose=verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    call layer%init( &
         input_shape = [ num_features(1), num_features(2), num_time_steps ], &
         batch_size = layer%batch_size &
    )

  end function layer_setup
!###############################################################################


!###############################################################################
  module subroutine set_hyperparams_kipf( &
       this, &
       num_features, num_time_steps, &
       num_outputs, verbose &
  )
    !! Set the hyperparameters for the convolutional MPNN layer
    implicit none

    ! Arguments
    class(kipf_mpnn_layer_type), intent(inout) :: this
    !! Instance of the convolutional MPNN layer
    integer, dimension(2), intent(in) :: num_features
    !! Number of features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level


    if(present(verbose)) verbose_ = verbose

    this%name = 'kipf_mpnn'
    this%type = 'mpnn'
    this%input_rank = 1
    this%num_outputs = num_outputs
    this%input_shape = [ 1 ]
    this%num_time_steps = num_time_steps
    this%num_vertex_features = num_features(1)
    this%num_edge_features = num_features(2)
    allocate(this%method, source=kipf_method_container_type( &
         this%num_vertex_features, this%num_edge_features, &
         this%num_time_steps, [ this%num_outputs ], &
         verbose = verbose_ &
    ))

  end subroutine set_hyperparams_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine update_message_kipf(this, input, graph)
    !! Update the message phase
    !!
    !! This subroutine updates the message phase of the convolutional message
    !! passing neural network layer. The message phase is updated by passing
    !! messages between vertices in the graph.
    !! This is effectively the forward pass of the message phase.
    implicit none

    ! Arguments
    class(kipf_message_phase_type), intent(inout) :: this
    !! Instance of the message phase
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    !! Input features
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    !! Graph structure

    ! Local variables
    integer :: s, v, w, e
    !! Batch index, vertex index, neighbour index, edge index
    real(real32) :: c
    !! Normalisation constant for the message passing


    if(this%use_message)then
       do concurrent (s = 1: this%batch_size)
          do v = 1, graph(s)%num_vertices
             this%message(s)%val(:,v) = input(s)%val(:,v)
             do e = graph(s)%adj_ia(v), graph(s)%adj_ia(v+1) - 1
                if( graph(s)%adj_ja(2,e) .eq. 0 )then
                   c = 1._real32
                else
                   c = graph(s)%edge(graph(s)%adj_ja(2,e))%weight
                end if
                c = c * ( &
                     ( graph(s)%vertex(v)%degree + 1 ) * &
                     ( graph(s)%vertex(graph(s)%adj_ja(1,e))%degree + 1 ) &
                ) ** ( -0.5_real32 )
                this%message(s)%val(:,v) = &
                     this%message(s)%val(:,v) + &
                     c * [ input(s)%val(:,graph(s)%adj_ja(1,e)) ]
             end do
             this%z(s)%val(:,v) = matmul( &
                  this%message(s)%val(:,v), &
                  this%weight(:,:) &
             )
          end do
          this%feature(s)%val(:,:) = &
               this%transfer%activate( this%z(s)%val(:,:) )
       end do
    else
       do concurrent (s = 1: this%batch_size)
          do v = 1, graph(s)%num_vertices
             this%z(s)%val(:,v) = matmul( &
                  input(s)%val(:,v), &
                  this%weight(:,:) &
             )
          end do
          this%feature(s)%val(:,:) = &
               this%transfer%activate( this%z(s)%val(:,:) )
       end do
    end if

  end subroutine update_message_kipf
!###############################################################################


!###############################################################################
  pure subroutine calculate_partials_message_kipf(this, input, gradient, graph)
    !! Calculate the partials for the message phase
    !!
    !! This subroutine calculates the partial derivatives of the error with
    !! respect to the learnable parameters of the message phase.
    !! This is effectively the backward pass of the message phase.
    implicit none

    ! Arguments
    class(kipf_message_phase_type), intent(inout) :: this
    !! Instance of the message phase
    type(feature_type), dimension(this%batch_size), intent(in) :: input
    !! Input features
    !! Hidden features has dimensions (feature, vertex, batch_size)
    type(feature_type), dimension(this%batch_size), intent(in) :: gradient
    !! Gradient of the output features
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    !! Graph structure

    ! Local variables
    integer :: s, v
    !! Batch index, vertex index
    real(real32), dimension(:,:), allocatable :: delta
    !! Delta values for the message phase
    !! i.e. partial derivatives of the error wrt the hidden features


    this%dw = 0._real32
    do concurrent(s=1:this%batch_size)
       ! There is no message passing transfer function
       delta = gradient(s)%val(:,:) * &
            this%transfer%differentiate(this%z(s)%val(:,:))

       ! Partial derivatives of error wrt weights
       ! dE/dW = o/p(l-1) * delta
       do v = 1, graph(s)%num_vertices
          ! i.e. outer product of the input and delta
          ! sum weights and biases errors to use in batch gradient descent
          this%dw(:,:,s) = this%dw(:,:,s) + &
               outer_product(input(s)%val(:,v), delta(:,v))
          ! The errors are summed from the delta of the ...
          ! ... 'child' node * 'child' weight
          ! dE/dI(l-1) = sum(weight(l) * delta(l))
          ! this prepares dE/dI for when it is passed into the previous layer
          this%di(s)%val(:this%num_inputs,v) = &
               matmul(this%weight(:this%num_inputs,:), delta(:,v))
       end do
    end do

  end subroutine calculate_partials_message_kipf
!###############################################################################


!###############################################################################
  pure subroutine get_output_readout_kipf(this, input, output)
    !! Get the output of the readout phase
    !!
    !! This subroutine calculates the output of the readout phase of the
    !! convolutional message passing neural network layer.
    !! This is effectively the forward pass of the readout phase.
    implicit none

    ! Arguments
    class(kipf_readout_phase_type), intent(inout) :: this
    !! Instance of the readout phase
    class(message_phase_type), dimension(1), &
         intent(in) :: input
    !! Input features
    real(real32), dimension(this%num_outputs, this%batch_size), &
         intent(out) :: output
    !! Output features

    ! Local variables
    integer :: s, v, t
    !! Batch index, vertex index, time step index


    ! do s = 1, this%batch_size
    !    output(:,s) = 0._real32
    !    do t = 0, this%num_time_steps, 1
    !       do v = 1, size(input(t)%feature(s)%val, 2)
    !          this%z(t+1,s)%val(:,v) = matmul( &
    !               input(t)%feature(s)%val(:,v), &
    !               this%weight(:,:,t+1) &
    !          )
    !          output(:,s) = output(:,s) + &
    !               this%transfer%activate( this%z(t+1,s)%val(:,v) )
    !       end do
    !    end do
    ! end do

  end subroutine get_output_readout_kipf
!###############################################################################


!###############################################################################
  pure subroutine calculate_partials_readout_kipf(this, input, gradient)
    !! Calculate the partials for the readout phase
    !!
    !! This subroutine calculates the partial derivatives of the error with
    !! respect to the learnable parameters of the readout phase.
    !! This is effectively the backward pass of the readout phase.
    implicit none

    ! Arguments
    class(kipf_readout_phase_type), intent(inout) :: this
    !! Instance of the readout phase
    class(message_phase_type), dimension(1), &
         intent(in) :: input
    !! Input features
    real(real32), dimension(this%num_outputs, this%batch_size), &
         intent(in) :: gradient
    !! Gradient of the output features

    ! Local variables
    integer :: s, v, t, num_features
    !! Batch index, vertex index, time step index
    real(real32), dimension(this%num_outputs) :: delta
    !! Delta values for the readout phase
    !! i.e. partial derivatives of the error wrt the hidden features


    ! this%dw = 0._real32
    ! do concurrent(s=1:this%batch_size)
    !    ! There is no message passing transfer function

    !    ! Partial derivatives of error wrt weights
    !    ! dE/dW = o/p(l-1) * delta
    !    do t = 0, this%num_time_steps, 1
    !       do v = 1, size(input(t)%feature(s)%val, 2)

    !          delta = &
    !               gradient(:,s) * &
    !               this%transfer%differentiate(this%z(t+1,s)%val(:,v))

    !          this%dw(:,:,t+1,s) = this%dw(:,:,t+1,s) + &
    !               outer_product(input(t)%feature(s)%val(:,v), delta(:))

    !          this%di(this%batch_size * t + s)%val(:,v) = &
    !               matmul(this%weight(:,:,t+1), delta(:))
    !       end do
    !    end do
    ! end do

  end subroutine calculate_partials_readout_kipf
!###############################################################################


!###############################################################################
  pure module subroutine backward_graph_kipf(this, graph, gradient)
    !! Backward pass for the graph
    implicit none

    ! Arguments
    class(kipf_mpnn_layer_type), intent(inout) :: this
    !! Instance of the convolutional MPNN layer
    type(graph_type), dimension(this%batch_size), intent(in) :: graph
    !! Graph structure
    real(real32), dimension( &
         this%output%shape(1), &
         this%batch_size &
    ), intent(in) :: gradient
    !! Gradient of the output features

    ! Local variables
    integer :: t
    !! Time step index


    !---------------------------------------------------------------------------
    ! Backward pass for the readout phase
    !---------------------------------------------------------------------------
    ! call this%method%readout%calculate_partials( &
    !      input = this%method%message, &
    !      gradient = gradient &
    ! )

    !---------------------------------------------------------------------------
    ! Backward pass for the final time step message phase
    !---------------------------------------------------------------------------
    call this%method%message(this%method%num_time_steps)%calculate_partials( &
         input = this%method%message(this%method%num_time_steps-1)%feature, &
         gradient = this%method%readout%di( 1 : this%batch_size ), &
         graph = graph &
    )

    !---------------------------------------------------------------------------
    ! Backward pass for the remaining time steps message phase
    !---------------------------------------------------------------------------
    do t = this%method%num_time_steps - 1, 1, -1
       call this%method%message(t)%calculate_partials( &
            input = this%method%message(t-1)%feature, &
            gradient = this%method%message(t+1)%di, &
            graph = graph &
       )
    end do

  end subroutine backward_graph_kipf
!###############################################################################

end module athena__kipf_mpnn_layer
