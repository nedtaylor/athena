module athena__spectral_filter_layer
  !! Module containing implementation of a Spectral Filter layer
  !!
  !! This module implements a spectral filtering layer that applies learnable
  !! element-wise weights in the frequency domain using a Discrete Cosine
  !! Transform (DCT-II) basis, followed by an inverse transform back to
  !! physical space.  A local affine bypass is added for expressiveness:
  !!
  !! \[ \mathbf{v} = \sigma\!\bigl(
  !!    \underbrace{\boldsymbol{\Phi}^{-1}\,
  !!    \mathrm{diag}(\mathbf{w}_s)\,
  !!    \boldsymbol{\Phi}\,\mathbf{u}}_{\text{spectral filtering}}
  !!  + \underbrace{\mathbf{W}\,\mathbf{u}}_{\text{local}}
  !!  + \mathbf{b}\bigr) \]
  !!
  !! where:
  !!   - \(\mathbf{u} \in \mathbb{R}^{n_{in}}\) is the input
  !!   - \(\boldsymbol{\Phi} \in \mathbb{R}^{M \times n_{in}}\) is the
  !!     forward DCT basis,
  !!     \(\Phi_{k,j} = \cos\!\bigl(\pi(k{-}1)(j{-}\tfrac12)/n_{in}\bigr)\)
  !!   - \(\mathrm{diag}(\mathbf{w}_s)\) are learnable per-mode spectral
  !!     filter weights (\(\mathbf{w}_s \in \mathbb{R}^M\))
  !!   - \(\boldsymbol{\Phi}^{-1} \in \mathbb{R}^{n_{out} \times M}\) is
  !!     the inverse DCT basis,
  !!     \(\Phi^{-1}_{i,k} = \cos\!\bigl(\pi(k{-}1)(i{-}\tfrac12)/n_{out}\bigr)\)
  !!   - \(\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}\) are the local
  !!     (bypass) weights
  !!   - \(\mathbf{b} \in \mathbb{R}^{n_{out}}\) is the bias
  !!   - \(\sigma\) is the activation function
  !!   - \(M\) = num_modes, the number of retained spectral modes
  !!
  !! Number of parameters (learnable):
  !!   \(M + n_{out}\,n_{in}\) without bias,
  !!   \(M + n_{out}\,n_{in} + n_{out}\) with bias.
  use coreutils, only: real32, stop_program, pi
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use diffstruc, only: array_type, matmul, operator(+), operator(*)
  implicit none


  private

  public :: spectral_filter_layer_type
  public :: read_spectral_filter_layer


  type, extends(learnable_layer_type) :: spectral_filter_layer_type
     !! Type for a spectral filter layer
     integer :: num_inputs
     !! Number of inputs
     integer :: num_outputs
     !! Number of outputs
     integer :: num_modes
     !! Number of retained spectral (cosine) modes
     type(array_type) :: forward_basis
     !! Fixed forward DCT basis Phi [num_modes x num_inputs]
     type(array_type) :: inverse_basis
     !! Fixed inverse DCT basis Phi_inv [num_outputs x num_modes]
     type(array_type), dimension(1) :: z
     !! Temporary array for pre-activation values
   contains
     procedure, pass(this) :: get_num_params => get_num_params_spectral_filter
     procedure, pass(this) :: set_hyperparams => set_hyperparams_spectral_filter
     procedure, pass(this) :: init => init_spectral_filter
     procedure, pass(this) :: print_to_unit => print_to_unit_spectral_filter
     procedure, pass(this) :: read => read_spectral_filter

     procedure, pass(this) :: forward => forward_spectral_filter

     final :: finalise_spectral_filter
  end type spectral_filter_layer_type

  interface spectral_filter_layer_type
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
       type(spectral_filter_layer_type) :: layer
     end function layer_setup
  end interface spectral_filter_layer_type



contains

!###############################################################################
  subroutine finalise_spectral_filter(this)
    !! Finalise the spectral filter layer
    implicit none

    ! Arguments
    type(spectral_filter_layer_type), intent(inout) :: this
    !! Layer instance to release

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(this%z(1)%allocated) call this%z(1)%deallocate()
    if(this%forward_basis%allocated) call this%forward_basis%deallocate()
    if(this%inverse_basis%allocated) call this%inverse_basis%deallocate()

  end subroutine finalise_spectral_filter
!###############################################################################


!###############################################################################
  pure function get_num_params_spectral_filter(this) result(num_params)
    !! Return the number of learnable parameters for the layer
    implicit none

    ! Arguments
    class(spectral_filter_layer_type), intent(in) :: this
    !! Layer instance
    integer :: num_params
    !! Total number of learnable parameters

    ! w_s: num_modes, W: n_out * n_in, b: n_out (optional)
    num_params = this%num_modes + &
         this%num_outputs * this%num_inputs
    if(this%use_bias) num_params = num_params + this%num_outputs

  end function get_num_params_spectral_filter
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
    !! Number of retained spectral modes
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

    type(spectral_filter_layer_type) :: layer
    !! Constructed spectral filter layer

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
  subroutine set_hyperparams_spectral_filter( &
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
    class(spectral_filter_layer_type), intent(inout) :: this
    !! Layer instance to configure
    integer, intent(in) :: num_outputs
    !! Number of output features
    integer, intent(in) :: num_modes
    !! Number of retained spectral modes
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

    this%name = "spectral_filter"
    this%type = "spfl"
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
          write(*,'("SPECTRAL_FILTER activation: ",A)') &
               trim(this%activation%name)
       end if
    end if

  end subroutine set_hyperparams_spectral_filter
!###############################################################################


!###############################################################################
  subroutine init_spectral_filter(this, input_shape, verbose)
    !! Initialise parameter storage, fixed bases and output buffers
    implicit none

    ! Arguments
    class(spectral_filter_layer_type), intent(inout) :: this
    !! Layer instance to initialise
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape used to infer num_inputs
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: num_inputs, j, k, i, idx
    !! Effective fan-in size and basis-construction indices
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
    ! params(1): w_s  spectral filter weights [num_modes]
    ! params(2): W    local bypass weights    [num_outputs x num_inputs]
    ! params(3): b    bias                    [num_outputs]  (optional)
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(2,1))
    this%weight_shape(:,1) = [ this%num_outputs, this%num_inputs ]

    if(this%use_bias)then
       this%bias_shape = [ this%num_outputs ]
       allocate(this%params(3))
    else
       allocate(this%params(2))
    end if

    ! w_s: spectral filter weights (1D, one per mode)
    call this%params(1)%allocate([this%num_modes, 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.

    ! W: local bypass weights
    call this%params(2)%allocate([this%num_outputs, this%num_inputs, 1])
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
    ! Initialise learnable parameters
    ! Spectral weights initialised to 1 (identity filter), local weights
    ! and bias via the chosen initialisers.
    !---------------------------------------------------------------------------
    this%params(1)%val(:,1) = 1.0_real32

    call this%kernel_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = num_inputs, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )
    if(this%use_bias)then
       call this%bias_init%initialise( &
            this%params(3)%val(:,1), &
            fan_in = num_inputs, fan_out = this%num_outputs &
       )
    end if


    !---------------------------------------------------------------------------
    ! Build fixed forward DCT basis Phi [num_modes x num_inputs]
    !   Phi(k,j) = cos( pi * (k-1) * (j - 0.5) / n_in )
    !---------------------------------------------------------------------------
    if(this%forward_basis%allocated) call this%forward_basis%deallocate()
    call this%forward_basis%allocate( &
         [this%num_modes, this%num_inputs, 1])
    this%forward_basis%is_sample_dependent = .false.
    this%forward_basis%requires_grad = .false.
    this%forward_basis%fix_pointer = .true.
    this%forward_basis%is_temporary = .false.

    do j = 1, this%num_inputs
       do k = 1, this%num_modes
          idx = k + (j-1) * this%num_modes
          this%forward_basis%val(idx, 1) = cos( &
               pi * real(k-1, real32) * &
               (real(j, real32) - 0.5_real32) / &
               real(this%num_inputs, real32) &
          )
       end do
    end do


    !---------------------------------------------------------------------------
    ! Build fixed inverse DCT basis Phi_inv [num_outputs x num_modes]
    !   Phi_inv(i,k) = cos( pi * (k-1) * (i - 0.5) / n_out )
    !---------------------------------------------------------------------------
    if(this%inverse_basis%allocated) call this%inverse_basis%deallocate()
    call this%inverse_basis%allocate( &
         [this%num_outputs, this%num_modes, 1])
    this%inverse_basis%is_sample_dependent = .false.
    this%inverse_basis%requires_grad = .false.
    this%inverse_basis%fix_pointer = .true.
    this%inverse_basis%is_temporary = .false.

    do k = 1, this%num_modes
       do i = 1, this%num_outputs
          idx = i + (k-1) * this%num_outputs
          this%inverse_basis%val(idx, 1) = cos( &
               pi * real(k-1, real32) * &
               (real(i, real32) - 0.5_real32) / &
               real(this%num_outputs, real32) &
          )
       end do
    end do


    !---------------------------------------------------------------------------
    ! Allocate output arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine init_spectral_filter
!###############################################################################


!###############################################################################
  subroutine print_to_unit_spectral_filter(this, unit)
    !! Print spectral filter settings and parameters to a unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(spectral_filter_layer_type), intent(in) :: this
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
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)   ! w_s
    write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)   ! W
    if(this%use_bias)then
       write(unit,'(5(E16.8E2))') this%params(3)%val(:,1) ! b
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_spectral_filter
!###############################################################################


!###############################################################################
  subroutine read_spectral_filter(this, unit, verbose)
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(spectral_filter_layer_type), intent(inout) :: this
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
       write(0,*) "WARNING: WEIGHTS card in SPECTRAL_FILTER not found"
    else
       call move(unit, param_line - iline, iostat=stat)

       ! Read w_s (num_modes values)
       allocate(data_list(num_modes), source=0._real32)
       c = 1
       k = 1
       do while(c.le.num_modes)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do
       this%params(1)%val(:,1) = data_list(1:num_modes)
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
       this%params(2)%val(:,1) = data_list
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
          this%params(3)%val(:,1) = data_list(1:num_outputs)
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

  end subroutine read_spectral_filter
!###############################################################################


!###############################################################################
  function read_spectral_filter_layer(unit, verbose) result(layer)
    !! Read a spectral filter layer from file and return it
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
    allocate(layer, source=spectral_filter_layer_type( &
         num_outputs=0, num_modes=1))
    call layer%read(unit, verbose=verbose_)

  end function read_spectral_filter_layer
!###############################################################################


!###############################################################################
  subroutine forward_spectral_filter(this, input)
    !! Forward propagation for the spectral filter layer
    !!
    !! Computes:
    !!   v = sigma( Phi_inv @ diag(w_s) @ Phi @ u  +  W @ u  +  b )
    !!
    !! The element-wise spectral filtering is performed via the * operator
    !! between the spectral weights (non-sample-dependent) and the spectral
    !! coefficients (sample-dependent).
    implicit none

    ! Arguments
    class(spectral_filter_layer_type), intent(inout) :: this
    !! Layer instance to execute
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input batch tensor collection

    ! Local variables
    type(array_type), pointer :: ptr, ptr_spec, ptr_local
    !! Combined output, spectral-path output and local-path output


    ! Spectral pathway:  Phi_inv @ (w_s * (Phi @ u))
    !---------------------------------------------------------------------------
    ptr_spec => matmul(this%forward_basis, input(1,1))   ! [M, batch]
    ptr_spec => this%params(1) * ptr_spec                ! element-wise filter
    ptr_spec => matmul(this%inverse_basis, ptr_spec)     ! [n_out, batch]

    ! Local bypass:  W @ u
    !---------------------------------------------------------------------------
    ptr_local => matmul(this%params(2), input(1,1))      ! [n_out, batch]

    ! Combine
    !---------------------------------------------------------------------------
    ptr => ptr_spec + ptr_local

    ! Add bias
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       ptr => ptr + this%params(3)
    end if

    ! Apply activation
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

  end subroutine forward_spectral_filter
!###############################################################################

end module athena__spectral_filter_layer
