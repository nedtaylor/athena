module athena__recurrent_layer
  !! Module containing implementation of recurrent neural network layers
  !!
  !! This module implements the simple recurrent neural network (RNN) layer,
  !! which is designed to handle sequential data by maintaining a hidden state.
  !!
  !! **Simple RNN layer (equivalent to RNNCell of PyTorch):**
  !! \[
  !! \begin{align}
  !! \mathbf{h}_t &= \sigma(\mathbf{W}_{ih}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h) \\
  !! \mathbf{y}_t &= \mathbf{W}_{ho}\mathbf{h}_t + \mathbf{b}_o
  !! \end{align}
  !! \]
  !!
  !! where:
  !!   - \(\mathbf{x}_t\) is input at time t
  !!   - \(\mathbf{h}_t\) is hidden state at time t
  !!   - \(\sigma\) is the activation function (e.g., tanh, relu)
  !!   - \(\mathbf{W}\) matrices are learnable weights
  !!   - \(\mathbf{b}\) vectors are learnable biases
  !!
  !! Properties:
  !!   - Processes sequential data with temporal dependencies
  !!   - Maintains hidden state across time steps
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_node_type, onnx_initialiser_type
  use diffstruc, only: array_type, matmul, operator(+), operator(*)
  implicit none


  private

  public :: recurrent_layer_type
  public :: read_recurrent_layer


  type, extends(learnable_layer_type) :: recurrent_layer_type
     !! Type for simple RNN layer
     integer :: hidden_size
     !! Size of hidden state
     integer :: input_size
     !! Size of input
     integer :: time_step
     !! Current time step
     type(array_type), pointer :: hidden_state => null()
     !! Hidden state
   contains
     procedure, pass(this) :: get_num_params => get_num_params_recurrent
     procedure, pass(this) :: set_hyperparams => set_hyperparams_recurrent
     procedure, pass(this) :: init => init_recurrent
     procedure, pass(this) :: print_to_unit => print_to_unit_recurrent
     procedure, pass(this) :: read => read_recurrent
     procedure, pass(this) :: forward => forward_recurrent
     procedure, pass(this) :: reset_state => reset_state_recurrent
  end type recurrent_layer_type

  interface recurrent_layer_type
     module function layer_setup( &
          hidden_size, input_size, use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       integer, intent(in) :: hidden_size
       integer, optional, intent(in) :: input_size
       logical, optional, intent(in) :: use_bias
       class(*), optional, intent(in) :: activation
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       integer, optional, intent(in) :: verbose
       type(recurrent_layer_type) :: layer
     end function layer_setup
  end interface recurrent_layer_type



contains

!###############################################################################
  pure function get_num_params_recurrent(this) result(num_params)
    implicit none
    class(recurrent_layer_type), intent(in) :: this
    integer :: num_params

    num_params = &
         this%hidden_size * this%input_size + &  ! W_ih
         this%hidden_size * this%hidden_size     ! W_hh
    if(this%use_bias)then
       num_params = num_params + 2 * this%hidden_size    ! b_h + b_o
    end if

  end function get_num_params_recurrent
!###############################################################################


!###############################################################################
  subroutine reset_state_recurrent(this)
    !! Reset the hidden state of the recurrent layer
    implicit none

    ! Arguments
    class(recurrent_layer_type), intent(inout) :: this
    !! Instance of the recurrent layer

    this%time_step = 0
    if(associated(this%hidden_state))then
       call this%hidden_state%deallocate()
       nullify(this%hidden_state)
    end if

  end subroutine reset_state_recurrent
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       hidden_size, input_size, use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    !! Setup a recurrent layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, intent(in) :: hidden_size
    !! Size of hidden state
    integer, optional, intent(in) :: input_size
    !! Size of input
    logical, optional, intent(in) :: use_bias
    !! Whether to use bias
    class(*), optional, intent(in) :: activation
    !! Activation function
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    !! Activation function, kernel initialiser, and bias initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(recurrent_layer_type) :: layer
    !! Instance of the recurrent layer

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
       activation_ = activation_setup("tanh")
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
         hidden_size = hidden_size, &
         use_bias = use_bias_, &
         activation = activation_, &
         kernel_initialiser = kernel_initialiser_, &
         bias_initialiser = bias_initialiser_, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    if(present(input_size)) call layer%init(input_shape=[input_size])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_recurrent( &
       this, hidden_size, &
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
    class(recurrent_layer_type), intent(inout) :: this
    !! Instance of the recurrent layer
    integer, intent(in) :: hidden_size
    !! Number of hidden units
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


    this%name = "recu"
    this%type = "recurrent"
    this%input_rank = 1
    this%output_rank = 1
    this%use_bias = use_bias
    this%hidden_size = hidden_size
    if(allocated(this%activation)) deallocate(this%activation)
    if(.not.allocated(activation))then
       this%activation = activation_setup("none")
    else
       allocate(this%activation, source=activation)
    end if
    if(allocated(this%kernel_init)) deallocate(this%kernel_init)
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(this%activation%name)
       this%kernel_init = initialiser_setup(buffer)
    else
       allocate(this%kernel_init, source=kernel_initialiser)
    end if
    if(allocated(this%bias_init)) deallocate(this%bias_init)
    if(.not.allocated(bias_initialiser))then
       buffer = get_default_initialiser( &
            this%activation%name, &
            is_bias=.true. &
       )
       this%bias_init = initialiser_setup(buffer)
    else
       allocate(this%bias_init, source=bias_initialiser)
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("RECU activation function: ",A)') &
               trim(this%activation%name)
          write(*,'("RECU kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("RECU bias initialiser: ",A)') &
               trim(this%bias_init%name)
       end if
    end if

  end subroutine set_hyperparams_recurrent
!###############################################################################


!###############################################################################
  subroutine init_recurrent(this, input_shape, verbose)
    !! Initialise the recurrent layer
    implicit none

    ! Arguments
    class(recurrent_layer_type), intent(inout) :: this
    !! Instance of the recurrent layer
    integer, dimension(:), intent(in) :: input_shape
    !! Shape of the input
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: num_inputs
    !! Temporary variable
    integer :: verbose_ = 0


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%input_size = this%input_shape(1)
    this%output_shape = [this%hidden_size]
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate weight, weight steps (velocities), output, and activation
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(2,2))
    this%weight_shape(:,1) = [ this%hidden_size, this%input_size ]
    this%weight_shape(:,2) = [ this%hidden_size, this%hidden_size ]

    if(this%use_bias)then
       this%bias_shape = [ this%hidden_size, this%hidden_size ]
       allocate(this%params(4))
    else
       allocate(this%params(2))
    end if
    call this%params(1)%allocate([this%weight_shape(:,1), 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.
    call this%params(2)%allocate([this%weight_shape(:,2), 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%fix_pointer = .true.
    this%params(2)%is_sample_dependent = .false.
    this%params(2)%is_temporary = .false.

    num_inputs = this%input_size + this%hidden_size
    if(this%use_bias)then
       num_inputs = num_inputs + 2 * this%hidden_size
       call this%params(3)%allocate([this%bias_shape(1), 1])
       call this%params(3)%set_requires_grad(.true.)
       this%params(3)%fix_pointer = .true.
       this%params(3)%is_sample_dependent = .false.
       this%params(3)%is_temporary = .false.
       call this%params(4)%allocate([this%bias_shape(2), 1])
       call this%params(4)%set_requires_grad(.true.)
       this%params(4)%fix_pointer = .true.
       this%params(4)%is_sample_dependent = .false.
       this%params(4)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(1)%val(:,1), &
         fan_in = num_inputs, fan_out = this%hidden_size, &
         spacing = [ this%hidden_size ] &
    )
    call this%kernel_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = num_inputs, fan_out = this%hidden_size, &
         spacing = [ this%hidden_size ] &
    )

    ! Initialise biases
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       call this%bias_init%initialise( &
            this%params(3)%val(:,1), &
            fan_in = num_inputs, fan_out = this%hidden_size &
       )
       call this%bias_init%initialise( &
            this%params(4)%val(:,1), &
            fan_in = num_inputs, fan_out = this%hidden_size &
       )
    end if


    !---------------------------------------------------------------------------
    ! Allocate arrays and initialise time_step
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))
    this%time_step = 0

  end subroutine init_recurrent
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_recurrent(this, unit)
    !! Print recurrent layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(recurrent_layer_type), intent(in) :: this
    !! Instance of the fully connected layer
    integer, intent(in) :: unit
    !! File unit


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"INPUT_SIZE = ",I0)') this%input_size
    write(unit,'(3X,"HIDDEN_SIZE = ",I0)') this%hidden_size

    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if


    ! Write fully connected weights and biases
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)
    write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)
    if(this%use_bias)then
       write(unit,'(5(E16.8E2))') this%params(3)%val(:,1)
       write(unit,'(5(E16.8E2))') this%params(4)%val(:,1)
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_recurrent
!###############################################################################


!###############################################################################
  subroutine read_recurrent(this, unit, verbose)
    !! Read recurrent layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(recurrent_layer_type), intent(inout) :: this
    !! Instance of the recurrent layer
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
    integer :: input_size, hidden_size
    !! Input and hidden sizes
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
       case("INPUT_SIZE", "NUM_INPUTS")
          call assign_val(buffer, input_size, itmp1)
       case("HIDDEN_SIZE", "NUM_OUTPUTS")
          call assign_val(buffer, hidden_size, itmp1)
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
         hidden_size = hidden_size, &
         use_bias = use_bias, &
         activation = activation, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[input_size])


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       call move(unit, param_line - iline, iostat=stat)
       num_params = this%input_size * this%hidden_size
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
       this%params(1)%val(:,1) = data_list
       deallocate(data_list)
       num_params = this%hidden_size * this%hidden_size
       allocate(data_list(num_params), source=0._real32)
       c = 1
       k = 1
       data_concat_loop1: do while(c.le.num_params)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop1
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop1
       this%params(2)%val(:,1) = data_list
       deallocate(data_list)
       if(use_bias)then
          do i = 1, 2
             hidden_size = this%hidden_size
             allocate(data_list(hidden_size), source=0._real32)
             c = 1
             k = 1
             data_concat_loop_bias: do while(c.le.hidden_size)
                read(unit,'(A)',iostat=stat) buffer
                if(stat.ne.0) exit data_concat_loop_bias
                k = icount(buffer)
                read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
                c = c + k
             end do data_concat_loop_bias
             this%params(i+2)%val(:,1) = data_list(1:hidden_size)
             deallocate(data_list)
          end do
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

  end subroutine read_recurrent
!###############################################################################


!###############################################################################
  function read_recurrent_layer(unit, verbose) result(layer)
    !! Read recurrent layer from file and return layer
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
    allocate(layer, source=recurrent_layer_type(hidden_size=0))
    call layer%read(unit, verbose=verbose_)

  end function read_recurrent_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_recurrent(this, input)
    !! Forward propagation
    implicit none

    ! Arguments
    class(recurrent_layer_type), intent(inout) :: this
    !! Instance of the recurrent layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    type(array_type), pointer :: ptr1, ptr2, ptr

    if(.not.associated(this%hidden_state))then
       call this%reset_state()
       allocate(this%hidden_state)
       call this%hidden_state%allocate( &
            [this%hidden_size, size(input(1,1)%val,2)], &
            source = 0._real32 &
       )
       this%hidden_state%is_temporary = .false.
    end if


    ! Generate outputs from weights, biases, and inputs
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       ptr1 => matmul(this%params(1), input(1,1) ) + this%params(3)
       ptr2 => matmul(this%params(2), this%hidden_state ) + this%params(4)
    else
       ptr1 => matmul(this%params(1), input(1,1) )
       ptr2 => matmul(this%params(2), this%hidden_state )
    end if
    ptr => ptr1 + ptr2

    ! Apply activation function to activation
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    if(trim(this%activation%name) .ne. "none")then
       ptr => this%activation%apply(ptr)
    end if
    this%hidden_state => ptr
    call this%output(1,1)%assign_shallow(ptr)
    this%output(1,1)%is_temporary = .false.
    this%time_step = this%time_step + 1

  end subroutine forward_recurrent
!###############################################################################

end module athena__recurrent_layer
