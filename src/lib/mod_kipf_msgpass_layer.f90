module athena__kipf_msgpass_layer
  !! Module containing the types and interfaces of a message passing layer
  use coreutils, only: real32, stop_program
  use graphstruc, only: graph_type
  use athena__misc_types, only: base_actv_type, base_init_type
  use diffstruc, only: array_type
  use athena__base_layer, only: base_layer_type
  use athena__msgpass_layer, only: msgpass_layer_type
  use athena__diffstruc_extd, only: kipf_propagate, kipf_update
  use diffstruc, only: matmul
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

     procedure, pass(this) :: update_message => update_message_kipf
     !! Update the message

     procedure, pass(this) :: update_readout => update_readout_kipf
     !! Update the readout
  end type kipf_msgpass_layer_type

  ! Interface for setting up the MPNN layer
  !-----------------------------------------------------------------------------
  interface kipf_msgpass_layer_type
     !! Interface for setting up the MPNN layer
     module function layer_setup( &
          num_vertex_features, num_time_steps, batch_size, &
          activation, &
          kernel_initialiser, &
          verbose &
     ) result(layer)
       !! Set up the message passing layer
       integer, dimension(:), intent(in) :: num_vertex_features
       !! Number of features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       integer, optional, intent(in) :: batch_size
       !! Batch size
       character(*), optional, intent(in) :: activation, &
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
  module function layer_setup( &
       num_vertex_features, num_time_steps, batch_size, &
       activation, &
       kernel_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the message passing layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: num_vertex_features
    !! Number of features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    integer, optional, intent(in) :: batch_size
    !! Batch size
    character(*), optional, intent(in) :: activation, &
         kernel_initialiser
    !! Activation function and kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    type(kipf_msgpass_layer_type) :: layer
    !! Instance of the message passing layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    class(base_actv_type), allocatable :: activation_
    !! Activation function object
    class(base_init_type), allocatable :: kernel_initialiser_
    !! Kernel initialisers

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation)) activation_ = activation_setup(activation)


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
         num_time_steps = num_time_steps, &
         activation = activation_, &
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
    call layer%init(input_shape=[layer%num_vertex_features(0), 0])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_kipf( &
       this, &
       num_vertex_features, &
       num_time_steps, &
       activation, &
       kernel_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for the message passing layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, dimension(:), intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    class(base_actv_type), allocatable, intent(in) :: activation
    !! Activation function
    class(base_init_type), allocatable, intent(in) :: kernel_initialiser
    !! Kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Loop index
    character(len=256) :: buffer


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
       call stop_program( &
            "Error: num_vertex_features must be a scalar or a vector of length &
            &num_time_steps + 1" &
       )
    end if
    allocate( this%num_edge_features(0:this%num_time_steps), source = 0 )
    this%use_graph_input = .true.
    if(.not.allocated(activation))then
       this%activation = activation_setup("none")
    else
       this%activation = activation
    end if
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(this%activation%name)
       this%kernel_init = initialiser_setup(buffer)
    else
       this%kernel_init = kernel_initialiser
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("KIPF activation function: ",A)') &
               trim(this%activation%name)
          write(*,'("KIPF kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
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
    if(allocated(this%weight_shape)) deallocate(this%weight_shape)
    if(allocated(this%bias_shape)) deallocate(this%bias_shape)
    allocate(this%weight_shape(2,this%num_time_steps))
    do t = 1, this%num_time_steps
       this%weight_shape(:,t) = &
            [ this%num_vertex_features(t), this%num_vertex_features(t-1) ]
    end do


    !---------------------------------------------------------------------------
    ! Allocate weight, weight steps (velocities), output, and activation
    !---------------------------------------------------------------------------
    if(allocated(this%params_array)) deallocate(this%params_array)
    allocate(this%params_array(this%num_time_steps))
    do t = 1, this%num_time_steps
       call this%params_array(t)%allocate( &
            array_shape = [ this%weight_shape(:,t), 1 ] &
       )
       call this%params_array(t)%set_requires_grad(.true.)
       this%params_array(t)%is_sample_dependent = .false.
       this%params_array(t)%is_temporary = .false.
       this%params_array(t)%fix_pointer = .true.
    end do


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    do t = 1, this%num_time_steps
       call this%kernel_init%initialise( &
            this%params_array(t)%val(:,1), &
            fan_in = this%num_vertex_features(t-1), &
            fan_out = this%num_vertex_features(t), &
            spacing = [ this%num_vertex_features(t) ] &
       )
    end do


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
       allocate(this%output(2,this%batch_size))
    end if

  end subroutine set_batch_size_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_kipf(this, unit)
    !! Print kipf message passing layer to unit
    use coreutils, only: to_upper
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

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%activation%name)


    ! Write learned parameters
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params_array(t)%val
    end do
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_kipf
!###############################################################################


!###############################################################################
  subroutine read_kipf(this, unit, verbose)
    !! Read the message passing layer
    use athena__tools_infile, only: assign_val, assign_vec, get_val, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
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
    integer :: t, j, k, c, itmp1, iline
    !! Loop variables and temporary integer
    integer :: num_time_steps = 0
    !! Number of time steps
    character(14) :: kernel_initialiser_name=''
    !! Initialisers
    character(20) :: activation_name=''
    !! Activation function name
    class(base_actv_type), allocatable :: activation
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser
    !! Initialisers
    integer, dimension(:), allocatable :: num_vertex_features
    !! Number of vertex and edge features
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    real(real32), allocatable, dimension(:) :: data_list
    !! Data list
    integer :: param_line, final_line
    !! Parameter line number


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    iline = 0
    param_line = 0
    final_line = 0
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
          final_line = iline
          backspace(unit)
          exit tag_loop
       end if
       iline = iline + 1

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
          call assign_val(buffer, activation_name, itmp1)
       case("KERNEL_INITIALISER", "KERNEL_INIT", "KERNEL_INITIALIZER")
          call assign_val(buffer, kernel_initialiser_name, itmp1)
       case("WEIGHTS")
          kernel_initialiser_name = 'zeros'
          param_line = iline
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
    activation = activation_setup(activation_name)
    kernel_initialiser = initialiser_setup(kernel_initialiser_name)


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
         activation = activation, &
         kernel_initialiser = kernel_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[this%num_vertex_features(0), 0])


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       call move(unit, param_line - iline, iostat=stat)
       do t = 1, this%num_time_steps
          allocate(data_list(this%num_params_msg(t)), source=0._real32)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.this%num_params_msg(t))
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          this%params_array(t)%val(:,1) = data_list(1:this%num_params_msg(t))
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


!##############################################################################!
  subroutine update_message_kipf(this, input)
    !! Update the message
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in), target :: input
    !! Input to the message passing layer

    ! Local variables
    integer :: s, t
    !! Batch index, time step
    type(array_type), pointer :: ptr1, ptr2, ptr3
    !! Pointers to arrays


    do s = 1, this%batch_size
       ptr1 => input(1,s)
       do t = 1, this%num_time_steps
          ptr2 => kipf_propagate( &
               ptr1, &
               this%graph(s)%adj_ia, this%graph(s)%adj_ja &
          )

          ! this%z(t,s) = kipf_update( &
          !      this%message(t,s), this%params_array(t), this%graph(s)%adj_ia &
          ! )
          ptr3 => matmul( this%params_array(t), ptr2 )
          ptr1 => this%activation%apply( ptr3 )
       end do
       call this%output(1,s)%zero_grad()
       call this%output(1,s)%assign_and_deallocate_source(ptr1)
       this%output(1,s)%is_temporary = .false.
    end do

  end subroutine update_message_kipf
!###############################################################################


!###############################################################################
  subroutine update_readout_kipf(this)
    !! Update the readout
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer

    ! Local variables
    integer :: s, v
    !! Loop indices


    ! do s = 1, this%batch_size
    !    this%output(1,s)%val = this%vertex_features(this%num_time_steps,s)%val
    !    this%output(2,s)%val = this%edge_features(this%num_time_steps,s)%val
    ! end do

  end subroutine update_readout_kipf
!###############################################################################

end module athena__kipf_msgpass_layer
