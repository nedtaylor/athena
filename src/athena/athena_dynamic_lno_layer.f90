module athena__dynamic_lno_layer
  !! Module containing implementation of a Laplace Neural Operator layer
  !!
  !! This module implements a Laplace Neural Operator (LNO) layer that
  !! approximates an integral kernel operator in the Laplace-transform domain.
  !! It combines a spectral pathway (encode → spectral mixing → decode)
  !! with a local affine bypass:
  !!
  !! \[ \mathbf{v} = \sigma\!\bigl(
  !!    \underbrace{\mathbf{D}\,\mathbf{R}\,\mathbf{E}\,\mathbf{u}}_{\text{spectral}}
  !!  + \underbrace{\mathbf{W}\,\mathbf{u}}_{\text{local}}
  !!  + \mathbf{b}\bigr) \]
  !!
  !! where:
  !!   - \(\mathbf{u} \in \mathbb{R}^{n_{in}}\) is the discretised input
  !!   - \(\boldsymbol{\mu} \in \mathbb{R}^{M}\) are learnable Laplace-domain
  !!     poles that define dynamic bases via \(H(s)=\sum_n \beta_n/(s-\mu_n)\)
  !!   - \(\mathbf{E}(\boldsymbol{\mu}) \in \mathbb{R}^{M \times n_{in}}\) is
  !!     the pole-residue encoder: \(E_{k,j}=\exp(-\mu_k\,t_j)\),
  !!     \(t_j = (j{-}1)/(n_{in}{-}1)\)
  !!   - \(\boldsymbol{\beta} \in \mathbb{R}^{M}\) are learnable residues;
  !!     \(\mathrm{diag}(\boldsymbol{\beta})\) replaces the former full mixing R
  !!   - \(\mathbf{D}(\boldsymbol{\mu}) \in \mathbb{R}^{n_{out} \times M}\) is
  !!     the pole-residue decoder: \(D_{i,k}=\exp(-\mu_k\,\tau_i)\),
  !!     \(\tau_i = (i{-}1)/(n_{out}{-}1)\)
  !!   - \(\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}\) are the
  !!     local (bypass) weights
  !!   - \(\mathbf{b} \in \mathbb{R}^{n_{out}}\) is the bias
  !!   - \(\sigma\) is the activation function
  !!   - \(M\) = num_modes, the number of spectral poles
  !!
  !! Bases \(\mathbf{E}\) and \(\mathbf{D}\) are rebuilt from the current poles
  !! at every forward call via \texttt{rebuild\_bases}.
  !!
  !! Number of parameters (learnable):
  !!   \(2M + n_{out}\,n_{in}\) without bias,
  !!   \(2M + n_{out}\,n_{in} + n_{out}\) with bias.
  use coreutils, only: real32, stop_program, pi
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_attribute_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use athena__onnx_nop_utils, only: emit_nop_input_transpose, &
       emit_nop_output_tail, emit_float_initialiser, emit_matrix_initialiser
  use diffstruc, only: array_type, matmul, operator(+), operator(*)
  use athena__diffstruc_extd, only: lno_encode, lno_decode, elem_scale
  implicit none


  private

  public :: dynamic_lno_layer_type
  public :: read_dynamic_lno_layer


  type, extends(learnable_layer_type) :: dynamic_lno_layer_type
     !! Type for a pole-residue Laplace Neural Operator layer
     integer :: num_inputs
     !! Number of inputs (discretisation points)
     integer :: num_outputs
     !! Number of outputs (discretisation points)
     integer :: num_modes
     !! Number of Laplace spectral modes
     type(array_type), dimension(1) :: z
     !! Temporary array for pre-activation values
   contains
     procedure, pass(this) :: get_num_params => get_num_params_dynamic_lno
     procedure, pass(this) :: set_hyperparams => set_hyperparams_dynamic_lno
     procedure, pass(this) :: init => init_dynamic_lno
     procedure, pass(this) :: print_to_unit => print_to_unit_dynamic_lno
     procedure, pass(this) :: read => read_dynamic_lno
     procedure, pass(this) :: get_bases => get_bases_dynamic_lno

     procedure, pass(this) :: forward => forward_dynamic_lno
     procedure, pass(this) :: get_attributes => get_attributes_dynamic_lno
     procedure, pass(this) :: emit_onnx_nodes => emit_onnx_nodes_dynamic_lno

     final :: finalise_dynamic_lno
  end type dynamic_lno_layer_type

  interface dynamic_lno_layer_type
     module function layer_setup( &
          num_outputs, num_modes, num_inputs, use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       integer, intent(in) :: num_outputs
       integer, intent(in) :: num_modes
       integer, optional, intent(in) :: num_inputs
       logical, optional, intent(in) :: use_bias
       class(*), optional, intent(in) :: activation
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       integer, optional, intent(in) :: verbose
       type(dynamic_lno_layer_type) :: layer
     end function layer_setup
  end interface dynamic_lno_layer_type



contains

!###############################################################################
  subroutine finalise_dynamic_lno(this)
    !! Finalise the dynamic Laplace neural operator layer
    implicit none

    ! Arguments
    type(dynamic_lno_layer_type), intent(inout) :: this
    !! Layer instance to release

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine finalise_dynamic_lno
!###############################################################################


!###############################################################################
  pure function get_num_params_dynamic_lno(this) result(num_params)
    !! Return the number of learnable parameters for the layer
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(in) :: this
    !! Layer instance
    integer :: num_params
    !! Total number of learnable parameters

    ! mu: num_modes, beta: num_modes, W: n_out * n_in, b: n_out (optional)
    num_params = 2 * this%num_modes + &
         this%num_outputs * this%num_inputs
    if(this%use_bias) num_params = num_params + this%num_outputs

  end function get_num_params_dynamic_lno
!###############################################################################


!###############################################################################
  module function layer_setup( &
       num_outputs, num_modes, num_inputs, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, intent(in) :: num_outputs
    !! Number of output features
    integer, intent(in) :: num_modes
    !! Number of learnable spectral poles
    integer, optional, intent(in) :: num_inputs
    !! Number of input features when known at construction time
    logical, optional, intent(in) :: use_bias
    !! Whether to allocate a bias term
    class(*), optional, intent(in) :: activation
    !! Activation function specification
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    !! Kernel and bias initialiser specifications
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(dynamic_lno_layer_type) :: layer
    !! Constructed dynamic LNO layer

    ! Local variables
    integer :: verbose_ = 0
    !! Effective verbosity level
    logical :: use_bias_ = .true.
    !! Effective bias flag
    class(base_actv_type), allocatable :: activation_
    !! Materialised activation object
    class(base_init_type), allocatable :: kernel_initialiser_, bias_initialiser_
    !! Materialised kernel and bias initialisers

    if(present(verbose)) verbose_ = verbose
    if(present(use_bias)) use_bias_ = use_bias

    if(present(activation))then
       activation_ = activation_setup(activation)
    else
       activation_ = activation_setup("none")
    end if

    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if
    if(present(bias_initialiser))then
       bias_initialiser_ = initialiser_setup(bias_initialiser)
    end if

    call layer%set_hyperparams( &
         num_outputs = num_outputs, &
         num_modes = num_modes, &
         use_bias = use_bias_, &
         activation = activation_, &
         kernel_initialiser = kernel_initialiser_, &
         bias_initialiser = bias_initialiser_, &
         verbose = verbose_ &
    )

    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_dynamic_lno( &
       this, num_outputs, num_modes, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(inout) :: this
    !! Layer instance to configure
    integer, intent(in) :: num_outputs
    !! Number of output features
    integer, intent(in) :: num_modes
    !! Number of learnable spectral poles
    logical, intent(in) :: use_bias
    !! Whether to use a bias term
    class(base_actv_type), allocatable, intent(in) :: activation
    !! Activation function object
    class(base_init_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    !! Kernel and bias initialiser objects
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    character(len=256) :: buffer
    !! Buffer for default initialiser lookup

    this%name = "dynamic_lno"
    this%type = "nop"
    this%input_rank = 1
    this%output_rank = 1
    this%use_bias = use_bias
    this%num_outputs = num_outputs
    this%num_modes = num_modes

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
          write(*,'("dynamic_lno activation: ",A)') &
               trim(this%activation%name)
       end if
    end if

  end subroutine set_hyperparams_dynamic_lno
!###############################################################################


!###############################################################################
  subroutine init_dynamic_lno(this, input_shape, verbose)
    !! Initialise parameter storage and output buffers for the layer
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(inout) :: this
    !! Layer instance to initialise
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape used to infer num_inputs
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: num_inputs, k
    !! Effective fan-in size and pole index
    integer :: verbose_ = 0
    !! Effective verbosity level

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Set shapes
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_inputs = this%input_shape(1)
    this%output_shape = [this%num_outputs]
    this%num_params = this%get_num_params()


    !---------------------------------------------------------------------------
    ! Allocate learnable parameters
    !
    ! params(1): mu  learnable poles        [num_modes]
    ! params(2): beta learnable residues    [num_modes]
    ! params(3): W   local bypass weights   [num_outputs x num_inputs]
    ! params(4): b   bias                   [num_outputs]  (optional)
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(2,3))
    this%weight_shape(:,1) = [ this%num_modes, 1 ]
    this%weight_shape(:,2) = [ this%num_modes, 1 ]
    this%weight_shape(:,3) = [ this%num_outputs, this%num_inputs ]

    if(this%use_bias)then
       this%bias_shape = [ this%num_outputs ]
       allocate(this%params(4))
    else
       allocate(this%params(3))
    end if

    ! mu: learnable poles
    call this%params(1)%allocate([this%num_modes, 1, 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.

    ! beta: learnable residues
    call this%params(2)%allocate([this%num_modes, 1, 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%fix_pointer = .true.
    this%params(2)%is_sample_dependent = .false.
    this%params(2)%is_temporary = .false.

    ! W: local bypass weights
    call this%params(3)%allocate([this%num_outputs, this%num_inputs, 1])
    call this%params(3)%set_requires_grad(.true.)
    this%params(3)%fix_pointer = .true.
    this%params(3)%is_sample_dependent = .false.
    this%params(3)%is_temporary = .false.

    num_inputs = this%num_inputs
    if(this%use_bias)then
       num_inputs = this%num_inputs + 1
       call this%params(4)%allocate([this%bias_shape, 1])
       call this%params(4)%set_requires_grad(.true.)
       this%params(4)%fix_pointer = .true.
       this%params(4)%is_sample_dependent = .false.
       this%params(4)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise learnable parameters
    !
    ! Poles: mu_n = n*pi (matches original fixed Laplace frequencies, gives
    !         the same initial spectral basis as the prior fixed construction)
    ! Residues: kernel initialiser (small random values)
    ! W: kernel initialiser
    ! b: bias initialiser
    !---------------------------------------------------------------------------
    do k = 1, this%num_modes
       this%params(1)%val(k, 1) = real(k, real32) * pi
    end do
    call this%kernel_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = this%num_modes, fan_out = this%num_modes, &
         spacing = [ this%num_modes ] &
    )
    call this%kernel_init%initialise( &
         this%params(3)%val(:,1), &
         fan_in = num_inputs, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )
    if(this%use_bias)then
       call this%bias_init%initialise( &
            this%params(4)%val(:,1), &
            fan_in = num_inputs, fan_out = this%num_outputs &
       )
    end if


    !---------------------------------------------------------------------------
    ! Allocate output arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine init_dynamic_lno
!###############################################################################


!###############################################################################
  function get_bases_dynamic_lno(this) result(bases)
    !! Rebuild the dynamic Laplace encoder/decoder bases from the current
    !! learnable pole values (params(1)).
    !!
    !! Called at the start of each forward pass so that the computation graph
    !! always uses up-to-date poles.  The rebuilt bases are non-tracked
    !! (requires_grad = .false.); gradient flow for the residues beta goes
    !! through the diffstruc * operator, and gradient flow for the bypass
    !! weights W goes through matmul.
    !!
    !!   E(mu)[n,j] = exp(-mu_n * t_j),  t_j = (j-1)/(n_in-1)
    !!   D(mu)[i,n] = exp(-mu_n * tau_i), tau_i = (i-1)/(n_out-1)
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(in) :: this
    !! Layer instance providing pole values
    type(array_type), dimension(2) :: bases
    !! Encoder and decoder basis tensors rebuilt from poles

    ! Local variables
    integer :: j, k, i, idx
    !! Basis-construction loop indices and flattened index
    real(real32) :: s, t
    !! Pole value and normalised coordinate

    !---------------------------------------------------------------------------
    ! Encoder E [num_modes x num_inputs]
    !---------------------------------------------------------------------------
    call bases(1)%allocate( [this%num_modes, this%num_inputs, 1] )
    bases(1)%is_sample_dependent = .false.
    bases(1)%requires_grad = .false.
    bases(1)%fix_pointer = .true.
    bases(1)%is_temporary = .false.

    do j = 1, this%num_inputs
       if(this%num_inputs .gt. 1)then
          t = real(j-1, real32) / real(this%num_inputs-1, real32)
       else
          t = 0.0_real32
       end if
       do k = 1, this%num_modes
          s = this%params(1)%val(k, 1)
          idx = k + (j-1) * this%num_modes
          bases(1)%val(idx, 1) = exp(-s * t)
       end do
    end do

    !---------------------------------------------------------------------------
    ! Decoder D [num_outputs x num_modes]
    !---------------------------------------------------------------------------
    call bases(2)%allocate( [this%num_outputs, this%num_modes, 1] )
    bases(2)%is_sample_dependent = .false.
    bases(2)%requires_grad = .false.
    bases(2)%fix_pointer = .true.
    bases(2)%is_temporary = .false.

    do k = 1, this%num_modes
       s = this%params(1)%val(k, 1)
       do i = 1, this%num_outputs
          if(this%num_outputs .gt. 1)then
             t = real(i-1, real32) / real(this%num_outputs-1, real32)
          else
             t = 0.0_real32
          end if
          idx = i + (k-1) * this%num_outputs
          bases(2)%val(idx, 1) = exp(-s * t)
       end do
    end do

  end function get_bases_dynamic_lno
!###############################################################################


!###############################################################################
  subroutine print_to_unit_dynamic_lno(this, unit)
    !! Print dynamic LNO settings and parameters to a unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(in) :: this
    !! Layer instance to print
    integer, intent(in) :: unit
    !! Output unit number

    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs
    write(unit,'(3X,"NUM_MODES = ",I0)') this%num_modes
    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if

    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)   ! poles
    write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)   ! residues
    write(unit,'(5(E16.8E2))') this%params(3)%val(:,1)   ! W
    if(this%use_bias)then
       write(unit,'(5(E16.8E2))') this%params(4)%val(:,1) ! b
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_dynamic_lno
!###############################################################################


!###############################################################################
  subroutine read_dynamic_lno(this, unit, verbose)
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(inout) :: this
    !! Layer instance to populate from file data
    integer, intent(in) :: unit
    !! Input unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat, verbose_ = 0
    !! I/O status and effective verbosity level
    integer :: j, k, c, itmp1, iline
    !! Loop counters and parser scratch integers
    integer :: num_inputs, num_outputs, num_modes
    !! Parsed layer dimensions
    logical :: use_bias = .true.
    !! Parsed bias flag
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    !! Parsed initialiser names
    class(base_actv_type), allocatable :: activation
    !! Parsed activation object
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
    !! Parsed initialiser objects
    character(256) :: buffer, tag, err_msg
    !! Input buffer, parsed tag and formatted error message
    real(real32), allocatable, dimension(:) :: data_list
    !! Temporary storage for flattened parameter blocks
    integer :: param_line, final_line, num_vals
    !! Weights-section line markers and current block size

    if(present(verbose)) verbose_ = verbose

    iline = 0
    param_line = 0
    final_line = 0
    tag_loop: do
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          final_line = iline
          backspace(unit)
          exit tag_loop
       end if
       iline = iline + 1

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       select case(trim(tag))
       case("NUM_INPUTS")
          call assign_val(buffer, num_inputs, itmp1)
       case("NUM_OUTPUTS")
          call assign_val(buffer, num_outputs, itmp1)
       case("NUM_MODES")
          call assign_val(buffer, num_modes, itmp1)
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

    call this%set_hyperparams( &
         num_outputs = num_outputs, &
         num_modes = num_modes, &
         use_bias = use_bias, &
         activation = activation, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[num_inputs])

    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in " // trim(this%name) // " not found"
    else
       call move(unit, param_line - iline, iostat=stat)

       ! Read poles (num_modes values)
       num_vals = num_modes
       allocate(data_list(num_vals), source=0._real32)
       c = 1
       k = 1
       do while(c.le.num_vals)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do
       this%params(1)%val(:,1) = data_list
       deallocate(data_list)

       ! Read residues (num_modes values)
       num_vals = num_modes
       allocate(data_list(num_vals), source=0._real32)
       c = 1
       k = 1
       do while(c.le.num_vals)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do
       this%params(2)%val(:,1) = data_list
       deallocate(data_list)

       ! Read W (num_outputs * num_inputs values)
       num_vals = num_outputs * num_inputs
       allocate(data_list(num_vals), source=0._real32)
       c = 1
       k = 1
       do while(c.le.num_vals)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do
       this%params(3)%val(:,1) = data_list
       deallocate(data_list)

       ! Read b if use_bias
       if(use_bias)then
          allocate(data_list(num_outputs), source=0._real32)
          c = 1
          k = 1
          do while(c.le.num_outputs)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do
          this%params(4)%val(:,1) = data_list(1:num_outputs)
          deallocate(data_list)
       end if

       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          call stop_program("END WEIGHTS not where expected")
          return
       end if

    end if

    call move(unit, final_line - iline, iostat=stat)
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_dynamic_lno
!###############################################################################


!###############################################################################
  function read_dynamic_lno_layer(unit, verbose) result(layer)
    !! Read a dynamic LNO layer from file and return it
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Input unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Allocated base-layer instance containing the result

    ! Local variables
    integer :: verbose_ = 0
    !! Effective verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=dynamic_lno_layer_type( &
         num_outputs=0, num_modes=1))
    call layer%read(unit, verbose=verbose_)

  end function read_dynamic_lno_layer
!###############################################################################


!###############################################################################
  subroutine forward_dynamic_lno(this, input)
    !! Forward propagation for the Laplace Neural Operator layer
    !!
    !! Computes via pole-residue spectral decomposition:
    !!   v = sigma( D(mu) @ diag(beta) @ E(mu) @ u  +  W @ u  +  b )
    !!
    !! Bases E(mu) and D(mu) are rebuilt from current poles at each call.
    !! The element-wise residue scaling uses the diffstruc * broadcast:
    !!   beta [M,1] * encoded [M,batch] -> [M,batch]
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(inout) :: this
    !! Layer instance to execute
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input batch tensor collection

    ! Local variables
    type(array_type), pointer :: ptr, ptr_spec, ptr_local
    !! Combined output, spectral-path output and local-path output


    ! Spectral pathway:  D(mu) @ diag(beta) @ E(mu) @ u
    ! Uses autodiff-tracked lno_encode/lno_decode for pole gradients
    !---------------------------------------------------------------------------
    ptr_spec => lno_encode(input(1,1), this%params(1), &
         this%num_inputs, this%num_modes)                ! [M, batch]
    ptr_spec => elem_scale(ptr_spec, this%params(2))
    ! [M, batch] residue scaling
    ptr_spec => lno_decode(ptr_spec, this%params(1), &
         this%num_outputs, this%num_modes)               ! [n_out, batch]

    ! Local bypass:  W @ u
    !---------------------------------------------------------------------------
    ptr_local => matmul(this%params(3), input(1,1))     ! [n_out, batch]

    ! Combine
    !---------------------------------------------------------------------------
    ptr => ptr_spec + ptr_local

    ! Add bias
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       ptr => ptr + this%params(4)
    end if

    ! Apply activation
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    if(trim(this%activation%name) .eq. "none")then
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    else
       call this%z(1)%zero_grad()
       call this%z(1)%assign_and_deallocate_source(ptr)
       this%z(1)%is_temporary = .false.
       ptr => this%activation%apply(this%z(1))
       call this%output(1,1)%assign_and_deallocate_source(ptr)
    end if
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_dynamic_lno
!###############################################################################


!###############################################################################
  function get_attributes_dynamic_lno(this) result(attributes)
    !! Return list of dynamic LNO attributes for ONNX export
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(in) :: this
    !! Instance of the dynamic LNO layer
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! List of attributes for ONNX export

    ! Local variables
    character(32) :: buffer
    !! Buffer for integer-to-string conversion

    allocate(attributes(5))

    write(buffer, '(I0)') this%num_inputs
    attributes(1) = onnx_attribute_type( &
         name='num_inputs', type='int', val=trim(buffer))
    write(buffer, '(I0)') this%num_outputs
    attributes(2) = onnx_attribute_type( &
         name='num_outputs', type='int', val=trim(buffer))
    write(buffer, '(I0)') this%num_modes
    attributes(3) = onnx_attribute_type( &
         name='num_modes', type='int', val=trim(buffer))
    if(this%use_bias)then
       buffer = '1'
    else
       buffer = '0'
    end if
    attributes(4) = onnx_attribute_type( &
         name='use_bias', type='int', val=trim(buffer))
    attributes(5) = onnx_attribute_type( &
         name='activation', type='string', val=trim(this%activation%name))

  end function get_attributes_dynamic_lno
!###############################################################################


!###############################################################################
  subroutine emit_onnx_nodes_dynamic_lno( &
       this, prefix, nodes, num_nodes, max_nodes, inits, num_inits, &
       max_inits, input_name, is_last_layer, format)
    !! Emit decomposed standard ONNX nodes for a Dynamic LNO layer.
    !!
    !! Decomposes the forward pass v = sigma(D(mu)*diag(beta)*E(mu)*u + W*u + b)
    !! into: Exp, MatMul, Mul, Add, Transpose, and optional Relu nodes.
    implicit none

    ! Arguments
    class(dynamic_lno_layer_type), intent(in) :: this
    !! Dynamic LNO layer instance
    character(*), intent(in) :: prefix
    !! Layer name prefix (e.g. "layer1")
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    !! Node accumulator
    integer, intent(inout) :: num_nodes
    !! Node counter
    integer, intent(in) :: max_nodes
    !! Node limit
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    !! Initialiser accumulator
    integer, intent(inout) :: num_inits
    !! Initialiser counter
    integer, intent(in) :: max_inits
    !! Initialiser limit
    character(*), optional, intent(in) :: input_name
    !! Name of the input tensor (e.g. "input" or previous layer output)
    logical, optional, intent(in) :: is_last_layer
    !! Whether this is the last layer in the network
    integer, optional, intent(in) :: format
    !! Export format selector

    ! Local variables
    integer :: j, k, idx, n
    real(real32) :: s, t
    real(real32), allocatable :: e_args(:), d_args(:)
    character(128) :: e_args_name, d_args_name, beta_name, w_name, b_name
    character(128) :: exp_e_out, exp_d_out, trans_in_out
    character(128) :: mm_e_out, mul_out, mm_d_out, mm_w_out
    character(128) :: add_out, add_b_out, final_output, output_source
    integer :: format_

    format_ = 1
    if(present(format)) format_ = format
    if(format_ .ne. 2) return
    if(.not.present(input_name)) return
    if(.not.present(is_last_layer)) return

    !--------------------------------------------------------------------------
    ! Build initialiser names
    !--------------------------------------------------------------------------
    write(e_args_name, '(A,".E_args")') trim(prefix)
    write(d_args_name, '(A,".D_args")') trim(prefix)
    write(beta_name, '(A,".beta")') trim(prefix)
    write(w_name, '(A,".W")') trim(prefix)
    write(b_name, '(A,".b")') trim(prefix)

    !--------------------------------------------------------------------------
    ! Build intermediate tensor names
    !--------------------------------------------------------------------------
    write(exp_e_out, '("/",A,"/Exp_output_0")') trim(prefix)
    write(exp_d_out, '("/",A,"/Exp_1_output_0")') trim(prefix)
    write(trans_in_out, '("/",A,"/Transpose_output_0")') trim(prefix)
    write(mm_e_out, '("/",A,"/MatMul_output_0")') trim(prefix)
    write(mul_out, '("/",A,"/Mul_output_0")') trim(prefix)
    write(mm_d_out, '("/",A,"/MatMul_1_output_0")') trim(prefix)
    write(mm_w_out, '("/",A,"/MatMul_2_output_0")') trim(prefix)
    write(add_out, '("/",A,"/Add_output_0")') trim(prefix)
    write(add_b_out, '("/",A,"/Add_1_output_0")') trim(prefix)

    !--------------------------------------------------------------------------
    ! Emit ONNX nodes
    !--------------------------------------------------------------------------
    ! 1. Exp(E_args) -> E [M, n_in]
    num_nodes = num_nodes + 1
    write(nodes(num_nodes)%name, '("/",A,"/Exp")') trim(prefix)
    nodes(num_nodes)%op_type = 'Exp'
    allocate(nodes(num_nodes)%inputs(1))
    nodes(num_nodes)%inputs(1) = trim(e_args_name)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(exp_e_out)
    nodes(num_nodes)%attributes_json = ''

    ! 2. Exp(D_args) -> D [n_out, M]
    num_nodes = num_nodes + 1
    write(nodes(num_nodes)%name, '("/",A,"/Exp_1")') trim(prefix)
    nodes(num_nodes)%op_type = 'Exp'
    allocate(nodes(num_nodes)%inputs(1))
    nodes(num_nodes)%inputs(1) = trim(d_args_name)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(exp_d_out)
    nodes(num_nodes)%attributes_json = ''

    ! 3. Transpose(input) -> x_t [n_in, batch]
    call emit_nop_input_transpose(trim(prefix), trim(input_name), nodes, &
         num_nodes, trim(trans_in_out))

    ! 4. MatMul(E, x_t) -> encoded [M, batch]
    num_nodes = num_nodes + 1
    write(nodes(num_nodes)%name, '("/",A,"/MatMul")') trim(prefix)
    nodes(num_nodes)%op_type = 'MatMul'
    allocate(nodes(num_nodes)%inputs(2))
    nodes(num_nodes)%inputs(1) = trim(exp_e_out)
    nodes(num_nodes)%inputs(2) = trim(trans_in_out)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(mm_e_out)
    nodes(num_nodes)%attributes_json = ''

    ! 5. Mul(beta, encoded) -> scaled [M, batch]
    num_nodes = num_nodes + 1
    write(nodes(num_nodes)%name, '("/",A,"/Mul")') trim(prefix)
    nodes(num_nodes)%op_type = 'Mul'
    allocate(nodes(num_nodes)%inputs(2))
    nodes(num_nodes)%inputs(1) = trim(beta_name)
    nodes(num_nodes)%inputs(2) = trim(mm_e_out)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(mul_out)
    nodes(num_nodes)%attributes_json = ''

    ! 6. MatMul(D, scaled) -> spectral [n_out, batch]
    num_nodes = num_nodes + 1
    write(nodes(num_nodes)%name, '("/",A,"/MatMul_1")') trim(prefix)
    nodes(num_nodes)%op_type = 'MatMul'
    allocate(nodes(num_nodes)%inputs(2))
    nodes(num_nodes)%inputs(1) = trim(exp_d_out)
    nodes(num_nodes)%inputs(2) = trim(mul_out)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(mm_d_out)
    nodes(num_nodes)%attributes_json = ''

    ! 7. MatMul(W, x_t) -> local [n_out, batch]
    num_nodes = num_nodes + 1
    write(nodes(num_nodes)%name, '("/",A,"/MatMul_2")') trim(prefix)
    nodes(num_nodes)%op_type = 'MatMul'
    allocate(nodes(num_nodes)%inputs(2))
    nodes(num_nodes)%inputs(1) = trim(w_name)
    nodes(num_nodes)%inputs(2) = trim(trans_in_out)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(mm_w_out)
    nodes(num_nodes)%attributes_json = ''

    ! 8. Add(spectral, local) -> combined [n_out, batch]
    num_nodes = num_nodes + 1
    write(nodes(num_nodes)%name, '("/",A,"/Add")') trim(prefix)
    nodes(num_nodes)%op_type = 'Add'
    allocate(nodes(num_nodes)%inputs(2))
    nodes(num_nodes)%inputs(1) = trim(mm_d_out)
    nodes(num_nodes)%inputs(2) = trim(mm_w_out)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(add_out)
    nodes(num_nodes)%attributes_json = ''

    ! 9. Add(combined, bias) -> biased [n_out, batch]
    if(this%use_bias)then
       num_nodes = num_nodes + 1
       write(nodes(num_nodes)%name, '("/",A,"/Add_1")') trim(prefix)
       nodes(num_nodes)%op_type = 'Add'
       allocate(nodes(num_nodes)%inputs(2))
       nodes(num_nodes)%inputs(1) = trim(add_out)
       nodes(num_nodes)%inputs(2) = trim(b_name)
       allocate(nodes(num_nodes)%outputs(1))
       nodes(num_nodes)%outputs(1) = trim(add_b_out)
       nodes(num_nodes)%attributes_json = ''
    end if

    if(this%use_bias)then
       output_source = add_b_out
    else
       output_source = add_out
    end if
    call emit_nop_output_tail(trim(prefix), trim(this%activation%name), &
         is_last_layer, trim(output_source), nodes, num_nodes, final_output)

    !--------------------------------------------------------------------------
    ! Emit initialisers
    !--------------------------------------------------------------------------

    ! W: bypass weights [n_out, n_in] in row-major
    n = this%num_outputs * this%num_inputs
    call emit_matrix_initialiser(trim(w_name), this%params(3)%val(:,1), &
         this%num_outputs, this%num_inputs, inits, num_inits)

    ! E_args: -mu*t for encoder [M, n_in] in row-major
    n = this%num_modes * this%num_inputs
    allocate(e_args(n))
    do j = 1, this%num_inputs
       if(this%num_inputs .gt. 1)then
          t = real(j - 1, real32) / real(this%num_inputs - 1, real32)
       else
          t = 0.0_real32
       end if
       do k = 1, this%num_modes
          s = this%params(1)%val(k, 1)
          idx = (k - 1) * this%num_inputs + j
          e_args(idx) = -s * t
       end do
    end do
    call emit_float_initialiser(trim(e_args_name), e_args, &
         [this%num_modes, this%num_inputs], inits, num_inits)
    deallocate(e_args)

    ! D_args: -mu*tau for decoder [n_out, M] in row-major
    n = this%num_outputs * this%num_modes
    allocate(d_args(n))
    do k = 1, this%num_modes
       s = this%params(1)%val(k, 1)
       do j = 1, this%num_outputs
          if(this%num_outputs .gt. 1)then
             t = real(j - 1, real32) / real(this%num_outputs - 1, real32)
          else
             t = 0.0_real32
          end if
          idx = (j - 1) * this%num_modes + k
          d_args(idx) = -s * t
       end do
    end do
    call emit_float_initialiser(trim(d_args_name), d_args, &
         [this%num_outputs, this%num_modes], inits, num_inits)
    deallocate(d_args)

    ! beta: residues [M, 1]
    call emit_float_initialiser(trim(beta_name), this%params(2)%val(:,1), &
         [this%num_modes, 1], inits, num_inits)

    ! b: bias [n_out, 1] (if use_bias)
    if(this%use_bias)then
       call emit_float_initialiser(trim(b_name), this%params(4)%val(:,1), &
            [this%num_outputs, 1], inits, num_inits)
    end if

  end subroutine emit_onnx_nodes_dynamic_lno
!###############################################################################

end module athena__dynamic_lno_layer
