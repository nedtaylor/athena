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
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_node_type, onnx_initialiser_type
  use diffstruc, only: array_type, matmul, operator(+)
  implicit none


  private

  public :: full_layer_type
  public :: read_full_layer, create_from_onnx_full_layer


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
     procedure, pass(this) :: build_from_onnx => build_from_onnx_full
     !! Build fully connected layer from ONNX node and initialiser

     procedure, pass(this) :: forward => forward_full
     !! Forward propagation derived type handler

     final :: finalise_full
     !! Finalise fully connected layer
  end type full_layer_type

  interface full_layer_type
     !! Interface for setting up the fully connected layer
     module function layer_setup( &
          num_outputs, num_inputs, batch_size, use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       !! Setup a fully connected layer
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: num_inputs
       !! Number of inputs
       integer, optional, intent(in) :: batch_size
       !! Batch size
       logical, optional, intent(in) :: use_bias
       !! Whether to use bias
       class(*), optional, intent(in) :: activation
       !! Activation function
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
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    !! Setup a fully connected layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: num_inputs
    !! Number of inputs
    integer, optional, intent(in) :: batch_size
    !! Batch size
    logical, optional, intent(in) :: use_bias
    !! Whether to use bias
    class(*), optional, intent(in) :: activation
    !! Activation function
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    !! Activation function, kernel initialiser, and bias initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(full_layer_type) :: layer
    !! Instance of the fully connected layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    logical :: use_bias_ = .true.
    !! Whether to use bias
    class(base_actv_type), allocatable :: activation_
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser_, bias_initialiser_
    !! Kernel and bias initialisers

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set use_bias
    !---------------------------------------------------------------------------
    if(present(use_bias)) use_bias_ = use_bias


    !---------------------------------------------------------------------------
    ! Set activation functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation))then
       activation_ = activation_setup(activation)
    else
       activation_ = activation_setup("none")
    end if


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
         use_bias = use_bias_, &
         activation = activation_, &
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
       use_bias, &
       activation, &
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
    logical, intent(in) :: use_bias
    !! Whether to use bias
    class(base_actv_type), allocatable, intent(in) :: activation
    !! Activation function
    class(base_init_type), allocatable, intent(in) :: &
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
    this%use_bias = use_bias
    this%num_outputs = num_outputs
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
    if(.not.allocated(bias_initialiser))then
       buffer = get_default_initialiser( &
            this%activation%name, &
            is_bias=.true. &
       )
       this%bias_init = initialiser_setup(buffer)
    else
       this%bias_init = bias_initialiser
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("FULL activation function: ",A)') &
               trim(this%activation%name)
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
    integer :: i, num_inputs
    !! Loop index
    integer :: verbose_ = 0


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

    if(this%use_bias)then
       this%bias_shape = [ this%num_outputs ]
       allocate(this%params_array(2))
    else
       allocate(this%params_array(1))
    end if
    call this%params_array(1)%allocate([this%weight_shape(:,1), 1])
    call this%params_array(1)%set_requires_grad(.true.)
    this%params_array(1)%fix_pointer = .true.
    this%params_array(1)%is_sample_dependent = .false.
    this%params_array(1)%is_temporary = .false.
    num_inputs = this%num_inputs
    if(this%use_bias)then
       num_inputs = this%num_inputs + 1
       call this%params_array(2)%allocate([this%bias_shape, 1])
       call this%params_array(2)%set_requires_grad(.true.)
       this%params_array(2)%fix_pointer = .true.
       this%params_array(2)%is_sample_dependent = .false.
       this%params_array(2)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params_array(1)%val(:,1), &
         fan_in = num_inputs, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )

    ! Initialise biases
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       call this%bias_init%initialise( &
            this%params_array(2)%val(:,1), &
            fan_in = num_inputs, fan_out = this%num_outputs &
       )
    end if


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

    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if


    ! Write fully connected weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params_array(1)%val(:,1)
    if(this%use_bias)then
       write(unit,'(5(E16.8E2))') this%params_array(2)%val(:,1)
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_full
!###############################################################################


!###############################################################################
  subroutine read_full(this, unit, verbose)
    !! Read fully connected layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
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
    logical :: use_bias = .true.
    !! Whether to use bias
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    !! Initialisers
    character(20) :: activation_name=''
    !! Activation function
    class(base_actv_type), allocatable :: activation
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
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
       case("USE_BIAS")
          call assign_val(buffer, use_bias, itmp1)
       case("ACTIVATION")
          iline = iline - 1
          backspace(unit)
          activation = read_activation(unit, iline)
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
         use_bias = use_bias, &
         activation = activation, &
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
       if(use_bias)then
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
       end if

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


!###############################################################################
  subroutine build_from_onnx_full(this, node, initialisers, verbose )
    !! Read ONNX attributes for fully connected layer
    implicit none

    ! Arguments
    class(full_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    type(onnx_node_type), intent(in) :: node
    !! Instance of ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! Instance of ONNX initialiser information
    integer, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    ! Initialise parameters from initialisers
    write(0,*) "WARNING: Weights initialisation from ONNX not yet implemented &
         &for fully connected layer"

    ! call this%set_hyperparams(num_outputs=...)

  end subroutine build_from_onnx_full
!###############################################################################


!###############################################################################
  function create_from_onnx_full_layer(node, initialisers, verbose) result(layer)
    !! Build fully connected layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! Instance of ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! Instance of ONNX initialiser information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 2D convolutional layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=full_layer_type(num_outputs=0))
    call layer%build_from_onnx(node, initialisers, verbose=verbose_)

  end function create_from_onnx_full_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_full(this, input)
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
    if(this%use_bias)then
       ptr => matmul(this%params_array(1), input(1,1) ) + this%params_array(2)
    else
       ptr => matmul(this%params_array(1), input(1,1) )
    end if

    ! Apply activation function to activation
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    if(trim(this%activation%name) .eq. "none") then
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    else
       call this%z(1)%zero_grad()
       call this%z(1)%assign_and_deallocate_source(ptr)
       this%z(1)%is_temporary = .false.
       ptr => this%activation%apply(this%z(1))
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    end if
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_full
!###############################################################################

end module athena__full_layer
