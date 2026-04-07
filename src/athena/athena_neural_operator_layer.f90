module athena__neural_operator_layer
  !! Module containing implementation of a simple neural operator layer
  !!
  !! This module implements a neural operator layer that approximates a
  !! discretized integral operator. The layer combines a standard affine
  !! transform (local component) with a mean-field integral operator
  !! (global/non-local component):
  !!
  !! \[ \mathbf{v} = \sigma\!\left(\mathbf{W}\mathbf{u}
  !!    + \mathbf{w}_k \langle\mathbf{u}\rangle + \mathbf{b}\right) \]
  !!
  !! where:
  !!   - \(\mathbf{u} \in \mathbb{R}^{n_{in}}\) is the input (discretised function)
  !!   - \(\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}\) are the local weights
  !!   - \(\mathbf{w}_k \in \mathbb{R}^{n_{out}}\) are the integral kernel weights
  !!   - \(\langle\mathbf{u}\rangle = \frac{1}{n_{in}}\sum_j u_j\) is the input mean
  !!   - \(\mathbf{b} \in \mathbb{R}^{n_{out}}\) is the bias
  !!   - \(\sigma\) is the activation function
  !!
  !! The global mean \(\langle\mathbf{u}\rangle\) acts as a rank-1 approximation
  !! to a continuous integral operator \(\mathcal{K}[u](x) = \kappa(x)\int u\,\mathrm{d}y\),
  !! where \(\mathbf{w}_k\) discretises \(\kappa\).  Using this layer stacked
  !! in sequence provides a resolution-invariant building block similar to the
  !! graph neural operator family.
  !!
  !! Number of parameters:
  !!   - with bias: \(n_{out}(n_{in} + 1) + n_{out} = n_{out}(n_{in}+2)\)
  !!   - without bias: \(n_{out}(n_{in} + 1)\)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_attribute_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use diffstruc, only: array_type, matmul, mean, operator(+)
  use athena__initialiser_data, only: data_init_type
  implicit none


  private

  public :: neural_operator_layer_type
  public :: read_neural_operator_layer


  type, extends(learnable_layer_type) :: neural_operator_layer_type
     !! Type for a neural operator layer
     integer :: num_inputs
     !! Number of inputs (discretisation points of the input function)
     integer :: num_outputs
     !! Number of outputs (discretisation points of the output function)
     type(array_type), dimension(1) :: z
     !! Temporary array for pre-activation values (forward propagation)
   contains
     procedure, pass(this) :: get_num_params => get_num_params_neural_operator
     !! Get the number of parameters for the neural operator layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_neural_operator
     !! Set the hyperparameters for the neural operator layer
     procedure, pass(this) :: init => init_neural_operator
     !! Initialise the neural operator layer
     procedure, pass(this) :: print_to_unit => print_to_unit_neural_operator
     !! Print the layer to a file
     procedure, pass(this) :: read => read_neural_operator
     !! Read the layer from a file

     procedure, pass(this) :: forward => forward_neural_operator
     !! Forward propagation
     procedure, pass(this) :: get_attributes => get_attributes_neural_operator
     !! Get layer attributes for ONNX export

     final :: finalise_neural_operator
     !! Finalise neural operator layer
  end type neural_operator_layer_type

  interface neural_operator_layer_type
     !! Interface for setting up the neural operator layer
     module function layer_setup( &
          num_outputs, num_inputs, use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       !! Setup a neural operator layer
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: num_inputs
       !! Number of inputs
       logical, optional, intent(in) :: use_bias
       !! Whether to use bias
       class(*), optional, intent(in) :: activation
       !! Activation function
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       !! Kernel and bias initialisers
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(neural_operator_layer_type) :: layer
       !! Instance of the neural operator layer
     end function layer_setup
  end interface neural_operator_layer_type



contains

!###############################################################################
  subroutine finalise_neural_operator(this)
    !! Finalise neural operator layer
    implicit none

    ! Arguments
    type(neural_operator_layer_type), intent(inout) :: this
    !! Instance of the neural operator layer

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine finalise_neural_operator
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure function get_num_params_neural_operator(this) result(num_params)
    !! Get the number of parameters for the neural operator layer
    !!
    !! Parameters consist of:
    !!   - W matrix : num_outputs * num_inputs
    !!   - W_k vector : num_outputs (integral kernel coupling)
    !!   - b vector : num_outputs (if use_bias)
    implicit none

    ! Arguments
    class(neural_operator_layer_type), intent(in) :: this
    !! Instance of the neural operator layer
    integer :: num_params
    !! Number of parameters

    ! W: n_out * n_in, W_k: n_out, b: n_out (if use_bias)
    num_params = this%num_outputs * this%num_inputs + this%num_outputs
    if(this%use_bias) num_params = num_params + this%num_outputs

  end function get_num_params_neural_operator
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       num_outputs, num_inputs, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    !! Setup a neural operator layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: num_inputs
    !! Number of inputs
    logical, optional, intent(in) :: use_bias
    !! Whether to use bias
    class(*), optional, intent(in) :: activation
    !! Activation function
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(neural_operator_layer_type) :: layer
    !! Instance of the neural operator layer

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
    ! Set activation function
    !---------------------------------------------------------------------------
    if(present(activation))then
       activation_ = activation_setup(activation)
    else
       activation_ = activation_setup("none")
    end if


    !---------------------------------------------------------------------------
    ! Define kernel and bias initialisers
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
    ! Initialise layer shape if num_inputs is provided
    !---------------------------------------------------------------------------
    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_neural_operator( &
       this, num_outputs, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for the neural operator layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    ! Arguments
    class(neural_operator_layer_type), intent(inout) :: this
    !! Instance of the neural operator layer
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


    this%name = "neural_operator"
    this%type = "nop"
    this%input_rank = 1
    this%output_rank = 1
    this%use_bias = use_bias
    this%num_outputs = num_outputs
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
       if(allocated(this%bias_init)) deallocate(this%bias_init)
       allocate(this%bias_init, source=bias_initialiser)
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("NEURAL_OPERATOR activation function: ",A)') &
               trim(this%activation%name)
          write(*,'("NEURAL_OPERATOR kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("NEURAL_OPERATOR bias initialiser: ",A)') &
               trim(this%bias_init%name)
       end if
    end if

  end subroutine set_hyperparams_neural_operator
!###############################################################################


!###############################################################################
  subroutine init_neural_operator(this, input_shape, verbose)
    !! Initialise neural operator layer
    implicit none

    ! Arguments
    class(neural_operator_layer_type), intent(inout) :: this
    !! Instance of the neural operator layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: num_inputs
    !! Effective fan-in for initialisation
    integer :: verbose_ = 0


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_inputs = this%input_shape(1)
    this%output_shape = [this%num_outputs]
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate parameters
    !
    ! params(1): W        (n_out x n_in)  - local transform weights
    ! params(2): W_k      (n_out x 1)     - integral kernel coupling weights
    ! params(3): b        (n_out)         - bias  [only when use_bias=.true.]
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(2,1))
    this%weight_shape(:,1) = [ this%num_outputs, this%num_inputs ]

    if(this%use_bias)then
       this%bias_shape = [ this%num_outputs ]
       allocate(this%params(3))
    else
       allocate(this%params(2))
    end if

    ! W: local transform  (n_out x n_in)
    call this%params(1)%allocate([this%weight_shape(:,1), 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.

    ! W_k: integral kernel coupling  (n_out x 1)
    call this%params(2)%allocate([this%num_outputs, 1, 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%fix_pointer = .true.
    this%params(2)%is_sample_dependent = .false.
    this%params(2)%is_temporary = .false.

    num_inputs = this%num_inputs
    if(this%use_bias)then
       num_inputs = this%num_inputs + 1
       call this%params(3)%allocate([this%bias_shape, 1])
       call this%params(3)%set_requires_grad(.true.)
       this%params(3)%fix_pointer = .true.
       this%params(3)%is_sample_dependent = .false.
       this%params(3)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise W with kernel initialiser
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(1)%val(:,1), &
         fan_in = num_inputs, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )

    !---------------------------------------------------------------------------
    ! Initialise W_k with kernel initialiser (smaller scale), treating it as
    ! a rank-1 integral correction so fan_in=1
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = num_inputs, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )

    !---------------------------------------------------------------------------
    ! Initialise bias if used
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       call this%bias_init%initialise( &
            this%params(3)%val(:,1), &
            fan_in = num_inputs, fan_out = this%num_outputs &
       )
    end if


    !---------------------------------------------------------------------------
    ! Allocate output and pre-activation arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine init_neural_operator
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_neural_operator(this, unit)
    !! Print neural operator layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(neural_operator_layer_type), intent(in) :: this
    !! Instance of the neural operator layer
    integer, intent(in) :: unit
    !! File unit


    ! Write hyperparameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs

    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if


    ! Write weights, kernel coupling, and optional bias
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)   ! W
    write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)   ! W_k
    if(this%use_bias)then
       write(unit,'(5(E16.8E2))') this%params(3)%val(:,1) ! b
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_neural_operator
!###############################################################################


!###############################################################################
  subroutine read_neural_operator(this, unit, verbose)
    !! Read neural operator layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(neural_operator_layer_type), intent(inout) :: this
    !! Instance of the neural operator layer
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
    !! Loop variables and temporary integers
    integer :: num_inputs, num_outputs
    !! Number of inputs and outputs
    logical :: use_bias = .true.
    !! Whether to use bias
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    !! Initialiser names
    character(20) :: activation_name=''
    !! Activation function name
    class(base_actv_type), allocatable :: activation
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
    !! Initialisers
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    real(real32), allocatable, dimension(:) :: data_list
    !! Data list
    integer :: param_line, final_line
    !! Parameter line numbers


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
          ! Skip lines that only contain numbers (e.g. scientific notation)
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


    ! Read weights if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in " // trim(this%name) // " not found"
    else
       call move(unit, param_line - iline, iostat=stat)

       ! Read W  (num_inputs * num_outputs elements)
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
       this%params(1)%val(:,1) = data_list
       deallocate(data_list)

       ! Read W_k  (num_outputs elements)
       allocate(data_list(this%num_outputs), source=0._real32)
       c = 1
       k = 1
       data_concat_loop2: do while(c.le.this%num_outputs)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop2
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop2
       this%params(2)%val(:,1) = data_list(1:this%num_outputs)
       deallocate(data_list)

       ! Read b  (num_outputs elements, only if use_bias)
       if(use_bias)then
          allocate(data_list(num_outputs), source=0._real32)
          c = 1
          k = 1
          data_concat_loop3: do while(c.le.num_outputs)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop3
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop3
          this%params(3)%val(:,1) = data_list(1:num_outputs)
          deallocate(data_list)
       end if

       ! Check for END WEIGHTS tag
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

  end subroutine read_neural_operator
!###############################################################################


!###############################################################################
  function read_neural_operator_layer(unit, verbose) result(layer)
    !! Read neural operator layer from file and return as base_layer_type
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Allocated layer instance

    ! Local variables
    integer :: verbose_ = 0

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=neural_operator_layer_type(num_outputs=0))
    call layer%read(unit, verbose=verbose_)

  end function read_neural_operator_layer
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine forward_neural_operator(this, input)
    !! Forward propagation for the neural operator layer
    !!
    !! Computes:
    !!   v = sigma( W * u  +  W_k * mean(u)  +  b )
    !!
    !! where mean(u) is the global mean of the input (scalar per sample),
    !! approximating the integral operator.
    implicit none

    ! Arguments
    class(neural_operator_layer_type), intent(inout) :: this
    !! Instance of the neural operator layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    ! Local variables
    type(array_type), pointer :: ptr, ptr_mean, ptr_kern


    ! Local transform: W · u  →  shape [n_out]
    !---------------------------------------------------------------------------
    ptr => matmul(this%params(1), input(1,1))

    ! Integral (mean-field) term: W_k · mean(u)  →  shape [n_out]
    !   mean(input, dim=1) reduces over all spatial elements, giving a scalar
    !   per batch sample (shape [1]).  matmul then expands W_k ([n_out x 1])
    !   by this scalar to produce a [n_out] correction vector.
    !---------------------------------------------------------------------------
    ptr_mean => mean(input(1,1), dim=1)
    ptr_kern => matmul(this%params(2), ptr_mean)

    ! Combine local + integral terms
    !---------------------------------------------------------------------------
    ptr => ptr + ptr_kern

    ! Add bias if used
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       ptr => ptr + this%params(3)
    end if

    ! Apply activation function
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

  end subroutine forward_neural_operator
!###############################################################################


!###############################################################################
  function get_attributes_neural_operator(this) result(attributes)
    implicit none
    class(neural_operator_layer_type), intent(in) :: this
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes

    character(32) :: buf

    allocate(attributes(4))

    write(buf, '(I0)') this%num_inputs
    attributes(1) = onnx_attribute_type( &
         name='num_inputs', type='int', val=trim(buf))
    write(buf, '(I0)') this%num_outputs
    attributes(2) = onnx_attribute_type( &
         name='num_outputs', type='int', val=trim(buf))
    if(this%use_bias)then
       buf = 'T'
    else
       buf = 'F'
    end if
    attributes(3) = onnx_attribute_type( &
         name='use_bias', type='int', val=trim(buf))
    attributes(4) = onnx_attribute_type( &
         name='activation', type='string', val=trim(this%activation%name))

  end function get_attributes_neural_operator
!###############################################################################

end module athena__neural_operator_layer
