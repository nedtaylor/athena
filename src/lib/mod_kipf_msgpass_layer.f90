module athena__kipf_msgpass_layer
  !! Module containing the types and interfacees of a message passing layer
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use graphstruc, only: graph_type
  use athena__misc_types, only: activation_type, initialiser_type, &
       array_type, array2d_type
  use athena__base_layer, only: base_layer_type
  use athena__msgpass_layer, only: msgpass_layer_type
  implicit none


  private

  public :: kipf_msgpass_layer_type
  public :: read_kipf_msgpass_layer


!-------------------------------------------------------------------------------
! Message passing layer
!-------------------------------------------------------------------------------
  type, extends(msgpass_layer_type) :: kipf_msgpass_layer_type

     ! this is for chen 2021 et al
     !  type(array2d_type), dimension(:), allocatable :: edge_weight
     !  !! Weights for the edges
     !  type(array2d_type), dimension(:), allocatable :: vertex_weight
     !  !! Weights for the vertices

   contains
     procedure, pass(this) :: get_num_params => get_num_params_kipf
     !! Get the number of parameters for the message passing layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_kipf
     !! Set the hyperparameters for the message passing layer
     procedure, pass(this) :: init => init_kipf
     !! Initialise the message passing layer
     procedure, pass(this) :: set_batch_size => set_batch_size_kipf
     !! Set batch size
     procedure, pass(this) :: print_to_unit => print_to_unit_kipf
     !! Print the message passing layer
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
          num_vertex_features, num_edge_features, num_time_steps, batch_size, &
          activation_function, activation_scale, &
          kernel_initialiser, &
          verbose &
     ) result(layer)
       !! Set up the message passing layer
       integer, dimension(:), intent(in) :: num_vertex_features, num_edge_features
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

    ! Local variables
    integer :: t
    !! Loop index

    num_params = 0
    do t = 1, this%num_time_steps
       num_params = num_params + &
            this%num_vertex_features(t-1) * this%num_vertex_features(t)
    end do

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
       num_vertex_features, num_time_steps, batch_size, &
       activation_function, activation_scale, &
       kernel_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the message passing layer
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: num_vertex_features
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
         num_vertex_features = num_vertex_features, &
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
    call layer%init(input_shape=[layer%num_vertex_features(0), 0])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_kipf( &
       this, &
       num_vertex_features, &
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
    integer, dimension(:), intent(in) :: num_vertex_features
    !! Number of vertex features
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

    ! Local variables
    integer :: t
    !! Loop index

    this%name = 'kipf'
    this%type = 'msgp'
    this%input_rank = 2
    this%output_rank = 2
    this%use_graph_output = .true.
    this%num_time_steps = num_time_steps
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
    allocate( this%num_edge_features(0:this%num_time_steps), source = 0 )
    this%use_graph_input = .true.
    if(allocated(this%transfer)) deallocate(this%transfer)
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
    if(allocated(this%num_params_msg)) deallocate(this%num_params_msg)
    allocate(this%num_params_msg(1:this%num_time_steps))
    do t = 1, this%num_time_steps
       this%num_params_msg(t) = &
            this%num_vertex_features(t-1) * this%num_vertex_features(t)
    end do
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output_shape)) deallocate(this%output_shape)

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
    integer :: t
    !! Loop index
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
    this%output_shape = [this%num_vertex_features(this%num_time_steps), 0]
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
    do t = 1, this%num_time_steps
       call initialiser_%initialise( &
            this%params( &
                 sum(this%num_params_msg(1:t-1)) + 1: &
                 sum(this%num_params_msg(1:t)) &
            ), &
            fan_in = this%num_vertex_features(t-1), &
            fan_out = this%num_vertex_features(t), &
            spacing = [ this%num_vertex_features(t) ] &
       )
    end do
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
       if(allocated(this%di)) deallocate(this%di)
       allocate(this%di(2,this%batch_size), source=array2d_type())
       !! input val arrays are allocated in set_graph
    end if


    if(allocated(this%vertex_features)) deallocate(this%vertex_features)
    allocate(this%vertex_features(0:this%num_time_steps, 1:this%batch_size))
    if(allocated(this%edge_features)) deallocate(this%edge_features)
    allocate(this%edge_features(0:this%num_time_steps, 1:this%batch_size))
    if(allocated(this%message)) deallocate(this%message)
    allocate(this%message(1:this%num_time_steps, 1:this%batch_size))

  end subroutine set_batch_size_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_kipf(this, unit)
    !! Print kipf message passing layer to unit
    use athena__misc, only: to_upper
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(in) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: unit
    !! File unit

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

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale


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

  end subroutine print_to_unit_kipf
!###############################################################################


!###############################################################################
  subroutine read_kipf(this, unit, verbose)
    !! Read the message passing layer
    use athena__tools_infile, only: assign_val, assign_vec, get_val
    use athena__misc, only: to_lower, to_upper, icount
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: unit
    !! Unit to read from
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    !! Status of read
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: t, j, k, c, itmp1, num_params_old, num_params_new
    !! Loop variables and temporary integer
    integer :: num_time_steps = 0
    !! Number of time steps
    real(real32) :: activation_scale
    !! Activation scale
    logical :: found_weights = .false.
    !! Flag for found weights
    character(14) :: kernel_initialiser=''
    !! Kernel and bias initialisers
    character(20) :: activation_function
    !! Padding and activation function
    integer, dimension(:), allocatable :: num_vertex_features
    !! Number of vertex and edge features
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    real(real32), allocatable, dimension(:) :: data_list
    !! Data list

    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    tag_loop: do

       ! Check for end of file
       !------------------------------------------------------------------------
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       ! Check for end of layer card
       !------------------------------------------------------------------------
       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("NUM_TIME_STEPS")
          call assign_val(buffer, num_time_steps, itmp1)
       case("NUM_VERTEX_FEATURES")
          itmp1 = icount(get_val(buffer))
          allocate(num_vertex_features(itmp1), source=0)
          call assign_vec(buffer, num_vertex_features, itmp1)
       case("ACTIVATION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
       case("KERNEL_INITIALISER")
          call assign_val(buffer, kernel_initialiser, itmp1)
       case("WEIGHTS")
          found_weights = .true.
          kernel_initialiser = 'zeros'
          exit tag_loop
       case default
          ! Don't look for "e" due to scientific notation of numbers
          ! ... i.e. exponent (E+00)
          if(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             cycle tag_loop
          elseif(tag(:3).eq.'END')then
             cycle tag_loop
          end if
          write(err_msg,'("Unrecognised line in input file: ",A)') &
               trim(adjustl(buffer))
          call stop_program(err_msg)
          return
       end select
    end do tag_loop


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    if(num_time_steps.gt.0 .and. num_time_steps.ne.size(num_vertex_features,1)-1)then
       write(err_msg,'("NUM_TIME_STEPS = ",I0," does not match length of "// &
            &"NUM_VERTEX_FEATURES = ",I0)') num_time_steps, &
            size(num_vertex_features,1)-1
       call stop_program(err_msg)
       return
    end if
    call this%set_hyperparams( &
         num_time_steps = num_time_steps, &
         num_vertex_features = num_vertex_features, &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         kernel_initialiser = kernel_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[this%num_vertex_features(0), 0])


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(.not.found_weights)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       num_params_old = 0
       do t = 1, this%num_time_steps
          num_params_new = this%num_vertex_features(t-1) * this%num_vertex_features(t)
          allocate(data_list((num_params_old + num_params_new)), source=0._real32)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.num_params_new)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          this%params( &
               num_params_old + 1:num_params_old + num_params_new &
          ) = data_list
          num_params_old = num_params_old + num_params_new
          deallocate(data_list)
       end do

       ! Check for end of weights card
       !------------------------------------------------------------------------
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(0,*) trim(adjustl(buffer))
          call stop_program("END WEIGHTS not where expected")
          return
       end if
    end if


    !---------------------------------------------------------------------------
    ! Check for end of layer card
    !---------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_kipf
!###############################################################################


!###############################################################################
  function read_kipf_msgpass_layer(unit, verbose) result(layer)
    !! Read kipf message passing layer from file and return layer
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
    allocate(layer, source = kipf_msgpass_layer_type( &
         num_time_steps = 1, &
         num_vertex_features = [ 0, 0 ] &
    ))
    call layer%read(unit, verbose=verbose_)

  end function read_kipf_msgpass_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure subroutine update_message_kipf(this, input)
    !! Update the message
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input to the message passing layer

    ! Local variables
    integer :: s, v, e, t
    !! Batch index, vertex index, edge index, time step
    real(real32) :: c
    !! Normalisation constant for the message passing
    real(real32), pointer :: weight(:,:)
    !! Pointer to the weight matrix
    ! real(real32), dimension(:,:), allocatable :: xe


    do s = 1, this%batch_size
       this%vertex_features(0,s)%val = input(1,s)%val
       this%edge_features(0,s)%val = input(2,s)%val
    end do

    do t = 1, this%num_time_steps
       weight( &
            1:this%num_vertex_features(t), &
            1:this%num_vertex_features(t-1) &
       ) => this%params( &
            sum(this%num_params_msg(1:t-1:1)) + 1: &
            sum(this%num_params_msg(1:t:1)) &
       )

       do concurrent (s = 1: this%batch_size)
          do v = 1, this%graph(s)%num_vertices
             this%message(t,s)%val(:,v) = 0._real32
             do e = this%graph(s)%adj_ia(v), this%graph(s)%adj_ia(v+1) - 1

                if( this%graph(s)%adj_ja(2,e) .eq. 0 )then
                   c = 1._real32
                else
                   c = this%graph(s)%edge_weights(this%graph(s)%adj_ja(2,e))
                end if
                ! fix this for lower memory case,
                ! where we don't store the vertices as derived types
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
                  weight(:,:), &
                  this%message(t,s)%val(:,v) &
             )
          end do
          this%vertex_features(t,s)%val(:,:) = &
               this%transfer%activate( this%z(t,s)%val(:,:) )
       end do
    end do

  end subroutine update_message_kipf
!###############################################################################


!###############################################################################
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
!###############################################################################


!###############################################################################
  pure subroutine backward_message_kipf(this, input, gradient)
    !! Backward pass for the message phase
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data (i.e. vertex and edge features)
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

    ! Local variables
    integer :: s, v, e, t, u
    !! Batch index, vertex index, edge index, time step, neighbor index
    integer :: from, to
    !! Indices for the weight parameters
    real(real32) :: c
    !! Normalisation constant for the message passing
    real(real32), dimension(:,:), allocatable :: dz
    !! Gradient of the loss with respect to z
    real(real32), dimension(:,:), allocatable :: dv_features
    !! Gradient of the loss with respect to vertex features
    real(real32), pointer :: weight(:,:), dw(:,:)
    !! Pointer to the weight matrix and its gradient


    ! Initialise vertex features gradients at time T
    do s = 1, this%batch_size
       this%di(1,s)%val = gradient(1,s)%val
       this%di(2,s)%val = gradient(2,s)%val
    end do

    ! Backpropagate through time steps
    do t = this%num_time_steps, 1, -1
       from = sum(this%num_params_msg(1:t-1:1)) + 1
       to = sum(this%num_params_msg(1:t:1))
       weight( &
            1:this%num_vertex_features(t), &
            1:this%num_vertex_features(t-1) &
       ) => this%params(from:to:1)
       do s = 1, this%batch_size
          ! Calculate gradient with respect to z at time t
          allocate(dz, mold=this%z(t,s)%val)
          dz = this%transfer%differentiate(this%z(t,s)%val) * this%di(1,s)%val

          ! Calculate gradient with respect to weights
          dw( &
               1:this%num_vertex_features(t), &
               1:this%num_vertex_features(t-1) &
          ) => this%dp( from:to:1, s)
          do v = 1, this%graph(s)%num_vertices
             dw(:,:) = dw(:,:) + &
                  matmul( &
                       reshape(dz(:,v), [size(dz, 1), 1]), &
                       reshape( &
                            this%message(t,s)%val(:,v), &
                            [1, size(this%message(t,s)%val, 1)] &
                       ) &
                  )
          end do

          ! Allocate space for vertex feature gradients
          allocate(dv_features, mold=this%vertex_features(t-1,s)%val)
          dv_features = 0._real32

          ! Backpropagate through message passing
          do v = 1, this%graph(s)%num_vertices
             ! Compute gradients for each vertex
             do e = this%graph(s)%adj_ia(v), this%graph(s)%adj_ia(v+1) - 1
                u = this%graph(s)%adj_ja(1,e)  ! Neighbour vertex index

                ! Compute normalisation constant
                if(this%graph(s)%adj_ja(2,e) .eq. 0) then
                   c = 1._real32
                else
                   c = this%graph(s)%edge_weights(this%graph(s)%adj_ja(2,e))
                end if
                c = c * ( &
                     (this%graph(s)%adj_ia(v+1) - this%graph(s)%adj_ia(v)) * &
                     (this%graph(s)%adj_ia(u+1) - this%graph(s)%adj_ia(u)) &
                ) ** (-0.5_real32)

                ! Add gradient contribution to neighbour
                dv_features(:,u) = dv_features(:,u) + &
                     c * matmul( &
                          dz(:,v), &
                          weight(:,:) &
                     )
             end do
          end do

          ! Update input gradient for prior time step
          this%di(1,s)%val = dv_features

          deallocate(dz, dv_features)
       end do
    end do

  end subroutine backward_message_kipf
!###############################################################################


!###############################################################################
  pure subroutine backward_readout_kipf(this, gradient)
    !! Backward pass for the readout phase
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

    ! Local variables
    integer :: s
    !! Batch index

    ! Pass gradients from output to final vertex/edge features
    do s = 1, this%batch_size
       this%di(1,s)%val = gradient(1,s)%val
       this%di(2,s)%val = gradient(2,s)%val
    end do

  end subroutine backward_readout_kipf
!###############################################################################

end module athena__kipf_msgpass_layer
