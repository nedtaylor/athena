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
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
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
     type(array_type) :: encoder_basis
     !! Dynamic Laplace encoder basis E [num_modes x num_inputs],
     !! rebuilt from learnable poles at each forward call
     type(array_type) :: decoder_basis
     !! Dynamic Laplace decoder basis D [num_outputs x num_modes],
     !! rebuilt from learnable poles at each forward call
     type(array_type), dimension(1) :: z
     !! Temporary array for pre-activation values
   contains
     procedure, pass(this) :: get_num_params => get_num_params_dynamic_lno
     procedure, pass(this) :: set_hyperparams => set_hyperparams_dynamic_lno
     procedure, pass(this) :: init => init_dynamic_lno
     procedure, pass(this) :: print_to_unit => print_to_unit_dynamic_lno
     procedure, pass(this) :: read => read_dynamic_lno
     procedure, pass(this) :: rebuild_bases => rebuild_bases_dynamic_lno

     procedure, pass(this) :: forward => forward_dynamic_lno

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
    implicit none
    type(dynamic_lno_layer_type), intent(inout) :: this

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(this%z(1)%allocated) call this%z(1)%deallocate()
    if(this%encoder_basis%allocated) call this%encoder_basis%deallocate()
    if(this%decoder_basis%allocated) call this%decoder_basis%deallocate()

  end subroutine finalise_dynamic_lno
!###############################################################################


!###############################################################################
  pure function get_num_params_dynamic_lno(this) result(num_params)
    implicit none
    class(dynamic_lno_layer_type), intent(in) :: this
    integer :: num_params

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

    integer, intent(in) :: num_outputs
    integer, intent(in) :: num_modes
    integer, optional, intent(in) :: num_inputs
    logical, optional, intent(in) :: use_bias
    class(*), optional, intent(in) :: activation
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose

    type(dynamic_lno_layer_type) :: layer

    integer :: verbose_ = 0
    logical :: use_bias_ = .true.
    class(base_actv_type), allocatable :: activation_
    class(base_init_type), allocatable :: kernel_initialiser_, bias_initialiser_

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

    class(dynamic_lno_layer_type), intent(inout) :: this
    integer, intent(in) :: num_outputs
    integer, intent(in) :: num_modes
    logical, intent(in) :: use_bias
    class(base_actv_type), allocatable, intent(in) :: activation
    class(base_init_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose

    character(len=256) :: buffer

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
    implicit none

    class(dynamic_lno_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: verbose

    integer :: num_inputs, k
    integer :: verbose_ = 0

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
    ! Build initial encoder/decoder bases from initial pole values
    !---------------------------------------------------------------------------
    call this%rebuild_bases()


    !---------------------------------------------------------------------------
    ! Allocate output arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine init_dynamic_lno
!###############################################################################


!###############################################################################
  subroutine rebuild_bases_dynamic_lno(this)
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

    class(dynamic_lno_layer_type), intent(inout) :: this

    integer :: j, k, i, idx
    real(real32) :: s, t

    !---------------------------------------------------------------------------
    ! Encoder E [num_modes x num_inputs]
    !---------------------------------------------------------------------------
    if(this%encoder_basis%allocated) call this%encoder_basis%deallocate()
    call this%encoder_basis%allocate( &
         [this%num_modes, this%num_inputs, 1])
    this%encoder_basis%is_sample_dependent = .false.
    this%encoder_basis%requires_grad = .false.
    this%encoder_basis%fix_pointer = .true.
    this%encoder_basis%is_temporary = .false.

    do j = 1, this%num_inputs
       if(this%num_inputs .gt. 1) then
          t = real(j-1, real32) / real(this%num_inputs-1, real32)
       else
          t = 0.0_real32
       end if
       do k = 1, this%num_modes
          s = this%params(1)%val(k, 1)
          idx = k + (j-1) * this%num_modes
          this%encoder_basis%val(idx, 1) = exp(-s * t)
       end do
    end do

    !---------------------------------------------------------------------------
    ! Decoder D [num_outputs x num_modes]
    !---------------------------------------------------------------------------
    if(this%decoder_basis%allocated) call this%decoder_basis%deallocate()
    call this%decoder_basis%allocate( &
         [this%num_outputs, this%num_modes, 1])
    this%decoder_basis%is_sample_dependent = .false.
    this%decoder_basis%requires_grad = .false.
    this%decoder_basis%fix_pointer = .true.
    this%decoder_basis%is_temporary = .false.

    do k = 1, this%num_modes
       s = this%params(1)%val(k, 1)
       do i = 1, this%num_outputs
          if(this%num_outputs .gt. 1) then
             t = real(i-1, real32) / real(this%num_outputs-1, real32)
          else
             t = 0.0_real32
          end if
          idx = i + (k-1) * this%num_outputs
          this%decoder_basis%val(idx, 1) = exp(-s * t)
       end do
    end do

  end subroutine rebuild_bases_dynamic_lno
!###############################################################################


!###############################################################################
  subroutine print_to_unit_dynamic_lno(this, unit)
    use coreutils, only: to_upper
    implicit none

    class(dynamic_lno_layer_type), intent(in) :: this
    integer, intent(in) :: unit

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

    class(dynamic_lno_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: stat, verbose_ = 0
    integer :: j, k, c, itmp1, iline
    integer :: num_inputs, num_outputs, num_modes
    logical :: use_bias = .true.
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    class(base_actv_type), allocatable :: activation
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
    character(256) :: buffer, tag, err_msg
    real(real32), allocatable, dimension(:) :: data_list
    integer :: param_line, final_line, num_vals

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

       ! Rebuild encoder/decoder bases from the loaded pole values
       call this%rebuild_bases()
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
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer
    integer :: verbose_ = 0

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

    class(dynamic_lno_layer_type), intent(inout) :: this
    class(array_type), dimension(:,:), intent(in) :: input

    type(array_type), pointer :: ptr, ptr_spec, ptr_local


    ! Rebuild bases for diagnostics (phi inspection)
    !---------------------------------------------------------------------------
    call this%rebuild_bases()

    ! Spectral pathway:  D(mu) @ diag(beta) @ E(mu) @ u
    ! Uses autodiff-tracked lno_encode/lno_decode for pole gradients
    !---------------------------------------------------------------------------
    ptr_spec => lno_encode(input(1,1), this%params(1), &
         this%num_inputs, this%num_modes)                ! [M, batch]
    ptr_spec => elem_scale(ptr_spec, this%params(2))    ! [M, batch] residue scaling
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

  end subroutine forward_dynamic_lno
!###############################################################################

end module athena__dynamic_lno_layer
