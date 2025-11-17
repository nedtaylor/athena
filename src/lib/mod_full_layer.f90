module athena__full_layer
  !! Module containing implementation of a fully connected layer
  !!
  !! This module implements a fully connected (aka dense) layer for a
  !! neural network.
  !! Attribution statement:
  !! The get_num_params procedure is based on code from the
  !! neural-fortran library
  !! https://github.com/modern-fortran/neural-fortran/blob/main/src/nf/nf_layer.f90
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: initialiser_type
  use diffstruc, only: array_type, operator(.mmul.), operator(+)
  implicit none


  private

  public :: full_layer_type
  public :: read_full_layer


  type, extends(learnable_layer_type) :: full_layer_type
     !! Type for fully connected (aka dense) layer with overloaded procedures
     integer :: num_inputs
     !! Number of inputs
     integer :: num_outputs
     !! Number of outputs
     type(array_type), dimension(1) :: z
     !! Temporary arrays for forward propagation
   contains
     procedure, pass(this) :: get_num_params => get_num_params_full
     !! Get the number of parameters for fully connected layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_full
     !! Set the hyperparameters for fully connected layer
     procedure, pass(this) :: init => init_full
     !! Initialise fully connected layer
     procedure, pass(this) :: set_batch_size => set_batch_size_full
     !! Set the batch size for fully connected layer
     procedure, pass(this) :: print_to_unit => print_to_unit_full
     !! Print the layer to a file
     procedure, pass(this) :: read => read_full
     !! Read the layer from a file

     procedure, pass(this) :: nullify_graph => nullify_graph_full

     procedure, pass(this) :: forward_derived => forward_derived_full
     !! Forward propagation derived type handler

     final :: finalise_full
     !! Finalise fully connected layer
  end type full_layer_type

  interface full_layer_type
     !! Interface for setting up the fully connected layer
     module function layer_setup( &
          num_outputs, num_inputs, batch_size, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       !! Setup a fully connected layer
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: num_inputs
       !! Number of inputs
       integer, optional, intent(in) :: batch_size
       !! Batch size
       real(real32), optional, intent(in) :: activation_scale
       !! Activation scale
       character(*), optional, intent(in) :: activation_function
       !! Activation function, kernel initialiser, and bias initialiser
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       !! Kernel and bias initialisers
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(full_layer_type) :: layer
       !! Instance of the fully connected layer
     end function layer_setup
  end interface full_layer_type



contains

!###############################################################################
  subroutine finalise_full(this)
    !! Finalise fully connected layer
    implicit none

    ! Arguments
    type(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine finalise_full
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure function get_num_params_full(this) result(num_params)
    !! Get the number of parameters for fully connected layer
    !!
    !! This function calculates the number of parameters for a fully connected
    !! layer.
    !! This procedure is based on code from the neural-fortran library
    implicit none

    ! Arguments
    class(full_layer_type), intent(in) :: this
    !! Instance of the fully connected layer
    integer :: num_params
    !! Number of parameters

    num_params = ( this%num_inputs + 1 )* this%num_outputs

  end function get_num_params_full
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       num_outputs, num_inputs, &
       batch_size, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    !! Setup a fully connected layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: num_inputs
    !! Number of inputs
    integer, optional, intent(in) :: batch_size
    !! Batch size
    real(real32), optional, intent(in) :: activation_scale
    !! Activation scale
    character(*), optional, intent(in) :: activation_function
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    !! Activation function, kernel initialiser, and bias initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(full_layer_type) :: layer
    !! Instance of the fully connected layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    real(real32) :: scale = 1._real32
    !! Activation scale
    character(len=10) :: activation_function_ = "none"
    !! Activation function
    class(initialiser_type), allocatable :: kernel_initialiser_, bias_initialiser_
    !! Kernel and bias initialisers

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation and derivative functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation_function)) activation_function_ = activation_function
    if(present(activation_scale)) scale = activation_scale


    !---------------------------------------------------------------------------
    ! Define weights (kernels) and biases initialisers
    !---------------------------------------------------------------------------
    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if
    if(present(bias_initialiser))then
       bias_initialiser_ = initialiser_setup(bias_initialiser)
    end if


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_outputs = num_outputs, &
         activation_function = activation_function_, &
         activation_scale = scale, &
         kernel_initialiser = kernel_initialiser_, &
         bias_initialiser = bias_initialiser_, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise batch size
    !---------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_full( &
       this, num_outputs, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for fully connected layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, intent(in) :: num_outputs
    !! Number of outputs
    character(*), intent(in) :: activation_function
    !! Activation function
    real(real32), intent(in) :: activation_scale
    !! Activation scale
    class(initialiser_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    character(len=256) :: buffer


    this%name = "full"
    this%type = "full"
    this%input_rank = 1
    this%output_rank = 1
    this%has_bias = .true.
    this%num_outputs = num_outputs
    if(allocated(this%transfer)) deallocate(this%transfer)
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale) &
    )
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(activation_function)
       this%kernel_init = initialiser_setup(buffer)
    else
       this%kernel_init = kernel_initialiser
    end if
    if(.not.allocated(bias_initialiser))then
       buffer = get_default_initialiser( &
            activation_function, &
            is_bias=.true. &
       )
       this%bias_init = initialiser_setup(buffer)
    else
       this%bias_init = bias_initialiser
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("FULL activation function: ",A)') &
               trim(activation_function)
          write(*,'("FULL kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("FULL bias initialiser: ",A)') &
               trim(this%bias_init%name)
       end if
    end if

  end subroutine set_hyperparams_full
!###############################################################################


!###############################################################################
  subroutine init_full(this, input_shape, batch_size, verbose)
    !! Initialise fully connected layer
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: batch_size
    !! Batch size
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: i
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
    this%num_inputs = this%input_shape(1)
    this%output_shape = [this%num_outputs]
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate weight, weight steps (velocities), output, and activation
    !---------------------------------------------------------------------------
    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(this%num_params), source=0._real32)
    allocate(this%weight_shape(2,1))
    this%weight_shape(:,1) = [ this%num_outputs, this%num_inputs ]
    this%bias_shape = [ this%num_outputs ]

    allocate(this%params_array(2))
    call this%params_array(1)%allocate([this%weight_shape(:,1), 1])
    call this%params_array(1)%set_requires_grad(.true.)
    this%params_array(1)%fix_pointer = .true.
    this%params_array(1)%is_sample_dependent = .false.
    this%params_array(1)%is_temporary = .false.
    call this%params_array(2)%allocate([this%bias_shape, 1])
    call this%params_array(2)%set_requires_grad(.true.)
    this%params_array(2)%fix_pointer = .true.
    this%params_array(2)%is_sample_dependent = .false.
    this%params_array(2)%is_temporary = .false.

    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params_array(1)%val(:,1), &
         fan_in = this%num_inputs + 1, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )

    ! Initialise biases
    !---------------------------------------------------------------------------
    call this%bias_init%initialise( &
         this%params_array(2)%val(:,1), &
         fan_in=this%num_inputs+1, fan_out=this%num_outputs &
    )


    !---------------------------------------------------------------------------
    ! Initialise batch size-dependent arrays
    !---------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_full
!###############################################################################


!###############################################################################
  subroutine set_batch_size_full(this, batch_size, verbose)
    !! Set the batch size for fully connected layer
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout), target :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: i
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
       call this%output(1,1)%allocate( &
            [this%num_outputs, this%batch_size], &
            source=0._real32 &
       )
       if(this%z(1)%allocated) call this%z(1)%deallocate()
       call this%z(1)%allocate( &
            [this%num_outputs, this%batch_size], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_full
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_full(this, unit)
    !! Print fully connected layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(full_layer_type), intent(in) :: this
    !! Instance of the fully connected layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: i
    !! Loop index


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale


    ! Write fully connected weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params_array(1)%val(:,1)
    write(unit,'(5(E16.8E2))') this%params_array(2)%val(:,1)
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_full
!###############################################################################


!###############################################################################
  subroutine read_full(this, unit, verbose)
    !! Read fully connected layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    !! Status of read
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: i, j, k, c, itmp1, iline, num_params
    !! Loop variables and temporary integer
    integer :: num_inputs, num_outputs
    !! Number of inputs and outputs
    real(real32) :: activation_scale
    !! Activation scale
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    !! Initialisers
    character(20) :: activation_function
    !! Activation function
    class(initialiser_type), allocatable :: kernel_initialiser, bias_initialiser
    !! Initialisers
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    integer, dimension(2) :: input_shape
    !! Input shape
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
       case("NUM_INPUTS")
          call assign_val(buffer, num_inputs, itmp1)
       case("NUM_OUTPUTS")
          call assign_val(buffer, num_outputs, itmp1)
       case("ACTIVATION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
       case("KERNEL_INITIALISER", "KERNEL_INIT", "KERNEL_INITIALIZER")
          call assign_val(buffer, kernel_initialiser_name, itmp1)
       case("BIAS_INITIALISER", "BIAS_INIT", "BIAS_INITIALIZER")
          call assign_val(buffer, bias_initialiser_name, itmp1)
       case("WEIGHTS")
          kernel_initialiser_name = 'zeros'
          bias_initialiser_name   = 'zeros'
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
    kernel_initialiser = initialiser_setup(kernel_initialiser_name)
    bias_initialiser = initialiser_setup(bias_initialiser_name)


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    call this%set_hyperparams( &
         num_outputs = num_outputs, &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[num_inputs])


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       call move(unit, param_line - iline, iostat=stat)
       num_params = this%num_inputs * this%num_outputs
       allocate(data_list(num_params), source=0._real32)
       c = 1
       k = 1
       data_concat_loop: do while(c.le.num_params)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop
       this%params_array(1)%val(:,1) = data_list
       deallocate(data_list)
       allocate(data_list(num_outputs), source=0._real32)
       c = 1
       k = 1
       data_concat_loop2: do while(c.le.num_outputs)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop2
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop2
       this%params_array(2)%val(:,1) = data_list(1:num_outputs)
       deallocate(data_list)

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
    call move(unit, final_line - iline, iostat=stat)
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_full
!###############################################################################


!###############################################################################
  function read_full_layer(unit, verbose) result(layer)
    !! Read fully connected layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the fully connected layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=full_layer_type(num_outputs=0))
    call layer%read(unit, verbose=verbose_)

  end function read_full_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_derived_full(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    type(array_type), pointer :: ptr => null()


    ! Generate outputs from weights, biases, and inputs
    !---------------------------------------------------------------------------
    ptr => ( this%params_array(1) .mmul. input(1,1) ) + this%params_array(2)

    ! Apply activation function to activation
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    if(trim(this%transfer%name) .eq. "none") then
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    else
       call this%z(1)%zero_grad()
       call this%z(1)%assign_and_deallocate_source(ptr)
       this%z(1)%is_temporary = .false.
       ptr => this%transfer%activate(this%z(1))
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    end if
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_derived_full
!###############################################################################


!###############################################################################
  subroutine nullify_graph_full(this)
    !! Nullify computation graph for fully connected layer
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer

    call this%output(1,1)%nullify_graph()

  end subroutine nullify_graph_full
!###############################################################################

end module athena__full_layer
