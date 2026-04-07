module athena__orthogonal_nop_block
  !! Module containing implementation of an Orthogonal Neural Operator layer
  !!
  !! This module implements the Orthogonal Neural Operator (ONO) from
  !! "Improved Operator Learning by Orthogonal Attention" (Luo et al., 2024).
  !!
  !! The ONO layer uses an orthogonal attention kernel to approximate the
  !! integral operator. It combines:
  !!   1. A learned orthogonal basis for efficient attention (k << N)
  !!   2. A spectral pathway through the orthogonal basis
  !!   3. A local affine bypass
  !!
  !! The layer computes:
  !! \[
  !!   \mathbf{v} = \sigma\!\bigl(
  !!     \mathbf{W}_V\,\mathbf{\Phi}\,(\mathbf{\Phi}^T\,\mathbf{u})
  !!   + \mathbf{W}\,\mathbf{u}
  !!   + \mathbf{b}\bigr)
  !! \]
  !!
  !! where \(\mathbf{\Phi} \in \mathbb{R}^{n_{in} \times k}\) is obtained
  !! by QR/Gram-Schmidt orthogonalisation of learnable basis weights
  !! \(\mathbf{B}\).
  !!
  !! Parameters (learnable):
  !!   - \(\mathbf{R} \in \mathbb{R}^{k \times k}\) spectral mixing
  !!   - \(\mathbf{B} \in \mathbb{R}^{n_{in} \times k}\) basis (orthogonalised)
  !!   - \(\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}\) bypass
  !!   - \(\mathbf{b} \in \mathbb{R}^{n_{out}}\) bias (optional)
  !!
  !! The spectral path is: decode( R * encode(u) )
  !!   encode = Phi^T @ u       [k, batch]
  !!   mix    = R @ encoded      [k, batch]
  !!   decode = Phi @ mix        [n_in, batch]
  !!
  !! Then a linear projection to the output: W_out @ decode -> [n_out, batch]
  !! Plus bypass: W @ u -> [n_out, batch]
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type
  use diffstruc, only: array_type, matmul, operator(+)
  use athena__diffstruc_extd, only: ono_encode, ono_decode
  implicit none
  public :: read_orthogonal_nop_block


  type, extends(learnable_layer_type) :: orthogonal_nop_block_type
     !! Type for an Orthogonal Neural Operator layer
     integer :: num_inputs = 0
     !! Number of inputs (discretisation points)
     integer :: num_outputs = 0
     !! Number of outputs (discretisation points)
     integer :: num_basis = 0
     !! Number of orthogonal basis functions (k)
     type(array_type), dimension(1) :: z
     !! Temporary array for pre-activation values
   contains
     procedure, pass(this) :: get_num_params => get_num_params_ono
     procedure, pass(this) :: set_hyperparams => set_hyperparams_ono
     procedure, pass(this) :: init => init_ono
     procedure, pass(this) :: print_to_unit => print_to_unit_ono
     procedure, pass(this) :: read => read_ono

     procedure, pass(this) :: forward => forward_ono
     procedure, pass(this) :: get_bases => get_bases_ono
     procedure, pass(this) :: get_orthogonality_metric

     final :: finalise_ono
  end type orthogonal_nop_block_type

  interface orthogonal_nop_block_type
     module function layer_setup( &
          num_outputs, num_basis, &
          num_inputs, use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       integer, intent(in) :: num_outputs
       integer, intent(in) :: num_basis
       integer, optional, intent(in) :: num_inputs
       logical, optional, intent(in) :: use_bias
       class(*), optional, intent(in) :: activation
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       integer, optional, intent(in) :: verbose
       type(orthogonal_nop_block_type) :: layer
     end function layer_setup
  end interface orthogonal_nop_block_type



contains

!###############################################################################
  subroutine finalise_ono(this)
    implicit none
    type(orthogonal_nop_block_type), intent(inout) :: this

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine finalise_ono
!###############################################################################


!###############################################################################
  pure function get_num_params_ono(this) result(num_params)
    implicit none
    class(orthogonal_nop_block_type), intent(in) :: this
    integer :: num_params

    ! R:     num_basis^2         (spectral mixing)
    ! B:     num_inputs * num_basis (basis weights)
    ! W_out: num_outputs * num_inputs (output projection / bypass)
    ! b:     num_outputs (optional)
    num_params = this%num_basis * this%num_basis + &
         this%num_inputs * this%num_basis + &
         this%num_outputs * this%num_inputs
    if(this%use_bias) num_params = num_params + this%num_outputs

  end function get_num_params_ono
!###############################################################################


!###############################################################################
  module function layer_setup( &
       num_outputs, num_basis, &
       num_inputs, use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    integer, intent(in) :: num_outputs
    integer, intent(in) :: num_basis
    integer, optional, intent(in) :: num_inputs
    logical, optional, intent(in) :: use_bias
    class(*), optional, intent(in) :: activation
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose

    type(orthogonal_nop_block_type) :: layer

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
         num_basis = num_basis, &
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
  subroutine set_hyperparams_ono( &
       this, num_outputs, num_basis, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    class(orthogonal_nop_block_type), intent(inout) :: this
    integer, intent(in) :: num_outputs
    integer, intent(in) :: num_basis
    logical, intent(in) :: use_bias
    class(base_actv_type), allocatable, intent(in) :: activation
    class(base_init_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose

    character(len=256) :: buffer

    this%name = "orthogonal_nop"
    this%type = "nop"
    this%input_rank = 1
    this%output_rank = 1
    this%use_bias = use_bias
    this%num_outputs = num_outputs
    this%num_basis = num_basis

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
          write(*,'("ORTHOGONAL_NOP activation: ",A)') &
               trim(this%activation%name)
       end if
    end if

  end subroutine set_hyperparams_ono
!###############################################################################


!###############################################################################
  subroutine init_ono(this, input_shape, verbose)
    implicit none

    class(orthogonal_nop_block_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: verbose

    integer :: num_inputs
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
    ! params(1): R      spectral mixing       [num_basis x num_basis]
    ! params(2): B      basis weights          [num_inputs x num_basis]
    ! params(3): W      bypass/output weights  [num_outputs x num_inputs]
    ! params(4): b      bias                   [num_outputs]  (optional)
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(2,3))
    this%weight_shape(:,1) = [ this%num_basis, this%num_basis ]
    this%weight_shape(:,2) = [ this%num_inputs, this%num_basis ]
    this%weight_shape(:,3) = [ this%num_outputs, this%num_inputs ]

    if(this%use_bias)then
       this%bias_shape = [ this%num_outputs ]
       allocate(this%params(4))
    else
       allocate(this%params(3))
    end if

    num_inputs = this%num_inputs
    if(this%use_bias) num_inputs = this%num_inputs + 1

    ! R: spectral mixing weights
    call this%params(1)%allocate([this%num_basis, this%num_basis, 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.

    ! B: basis weights (stored flat for Gram-Schmidt, but allocated shaped)
    call this%params(2)%allocate([this%num_inputs, this%num_basis, 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%fix_pointer = .true.
    this%params(2)%is_sample_dependent = .false.
    this%params(2)%is_temporary = .false.

    ! W: bypass/output weights
    call this%params(3)%allocate([this%num_outputs, this%num_inputs, 1])
    call this%params(3)%set_requires_grad(.true.)
    this%params(3)%fix_pointer = .true.
    this%params(3)%is_sample_dependent = .false.
    this%params(3)%is_temporary = .false.

    if(this%use_bias)then
       call this%params(4)%allocate([this%bias_shape, 1])
       call this%params(4)%set_requires_grad(.true.)
       this%params(4)%fix_pointer = .true.
       this%params(4)%is_sample_dependent = .false.
       this%params(4)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise learnable parameters
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(1)%val(:,1), &
         fan_in = this%num_basis, fan_out = this%num_basis, &
         spacing = [ this%num_basis ] &
    )
    call this%kernel_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = this%num_inputs, fan_out = this%num_basis, &
         spacing = [ this%num_inputs ] &
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

  end subroutine init_ono
!###############################################################################


!###############################################################################
  function get_bases_ono(this) result(phi)
    !! Orthogonalise the basis matrix B using modified Gram-Schmidt
    implicit none
    class(orthogonal_nop_block_type), intent(in) :: this
    type(array_type) :: phi

    integer :: n, k, i, j
    real(real32), allocatable :: B(:,:), Q(:,:)
    real(real32) :: norm_val, proj

    n = this%num_inputs
    k = this%num_basis

    allocate(B(n, k), Q(n, k))

    ! Reshape B from flat params(2) into [n, k]
    B = reshape(this%params(2)%val(:,1), [n, k])

    ! Modified Gram-Schmidt orthogonalisation
    Q = B
    do j = 1, k
       do i = 1, j - 1
          proj = dot_product(Q(:,i), Q(:,j))
          Q(:,j) = Q(:,j) - proj * Q(:,i)
       end do
       norm_val = sqrt(dot_product(Q(:,j), Q(:,j)))
       if(norm_val .gt. 1.0e-12_real32)then
          Q(:,j) = Q(:,j) / norm_val
       else
          Q(:,j) = 0.0_real32
       end if
    end do

    ! Store phi [n x k]
    call phi%allocate([n, k, 1])
    phi%is_sample_dependent = .false.
    phi%requires_grad = .false.
    phi%fix_pointer = .true.
    phi%is_temporary = .false.
    phi%val(:,1) = reshape(Q, [n * k])

    deallocate(B, Q)

  end function get_bases_ono
!###############################################################################


!###############################################################################
  function get_orthogonality_metric(this) result(metric)
    !! Compute max(|Phi^T @ Phi - I|) as a measure of basis orthogonality
    implicit none
    class(orthogonal_nop_block_type), intent(in) :: this
    real(real32) :: metric

    integer :: n, k, i, j, idx_ij
    real(real32), allocatable :: Q(:,:), QtQ(:,:)
    real(real32) :: val
    type(array_type) :: phi

    n = this%num_inputs
    k = this%num_basis

    allocate(Q(n, k), QtQ(k, k))
    phi = this%get_bases()
    Q = reshape(phi%val(:,1), [n, k])

    ! Compute Q^T @ Q
    QtQ = matmul(transpose(Q), Q)

    ! max(|Q^T Q - I|)
    metric = 0.0_real32
    do j = 1, k
       do i = 1, k
          if(i .eq. j)then
             val = abs(QtQ(i,j) - 1.0_real32)
          else
             val = abs(QtQ(i,j))
          end if
          if(val .gt. metric) metric = val
       end do
    end do

    deallocate(Q, QtQ)

  end function get_orthogonality_metric
!###############################################################################


!###############################################################################
  subroutine print_to_unit_ono(this, unit)
    use coreutils, only: to_upper
    implicit none

    class(orthogonal_nop_block_type), intent(in) :: this
    integer, intent(in) :: unit

    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs
    write(unit,'(3X,"NUM_BASIS = ",I0)') this%num_basis
    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if

    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)   ! R
    write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)   ! B
    write(unit,'(5(E16.8E2))') this%params(3)%val(:,1)   ! W
    if(this%use_bias)then
       write(unit,'(5(E16.8E2))') this%params(4)%val(:,1) ! b
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_ono
!###############################################################################


!###############################################################################
  subroutine read_ono(this, unit, verbose)
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    class(orthogonal_nop_block_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: stat, verbose_ = 0
    integer :: j, k, c, itmp1, iline
    integer :: num_inputs, num_outputs, num_basis
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
       case("NUM_BASIS")
          call assign_val(buffer, num_basis, itmp1)
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
         num_basis = num_basis, &
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

       ! Read R (num_basis^2)
       num_vals = num_basis * num_basis
       allocate(data_list(num_vals), source=0._real32)
       c = 1; k = 1
       do while(c.le.num_vals)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do
       this%params(1)%val(:,1) = data_list
       deallocate(data_list)

       ! Read B (num_inputs * num_basis)
       num_vals = num_inputs * num_basis
       allocate(data_list(num_vals), source=0._real32)
       c = 1; k = 1
       do while(c.le.num_vals)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do
       this%params(2)%val(:,1) = data_list
       deallocate(data_list)

       ! Read W (num_outputs * num_inputs)
       num_vals = num_outputs * num_inputs
       allocate(data_list(num_vals), source=0._real32)
       c = 1; k = 1
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
          c = 1; k = 1
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

  end subroutine read_ono
!###############################################################################


!###############################################################################
  function read_orthogonal_nop_block(unit, verbose) result(layer)
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer
    integer :: verbose_ = 0

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=orthogonal_nop_block_type( &
         num_outputs=0, num_basis=1))
    call layer%read(unit, verbose=verbose_)

  end function read_orthogonal_nop_block
!###############################################################################


!###############################################################################
  subroutine forward_ono(this, input)
    !! Forward propagation for the Orthogonal Neural Operator layer
    !!
    !! Computes:
    !!   encoded = Phi^T @ u          [k, batch]
    !!   mixed   = R @ encoded        [k, batch]
    !!   decoded = Phi @ mixed        [n_in, batch]
    !!   spectral= W @ decoded        [n_out, batch]  (reuse W for output proj)
    !!
    !!   bypass  = W @ u              [n_out, batch]
    !!
    !!   v = sigma( spectral + bypass + b )
    !!
    !! Actually, we separate the spectral and bypass paths clearly:
    !!   spectral path uses the orthogonal basis + R mixing
    !!   bypass path uses W directly on input
    !!   Both project to [n_out] via W (shared) or separate matrices.
    !!
    !! Here we implement:
    !!   spectral = W @ Phi @ R @ Phi^T @ u
    !!   bypass   = W @ u
    !!   v = sigma( spectral + bypass + b )
    !!
    !! Note: W is params(3) [n_out x n_in], shared for both paths
    !! This means: v = sigma( W @ (Phi @ R @ Phi^T @ u + u) + b )
    !!           = sigma( W @ ((Phi @ R @ Phi^T + I) @ u) + b )
    implicit none

    class(orthogonal_nop_block_type), intent(inout) :: this
    class(array_type), dimension(:,:), intent(in) :: input

    type(array_type), pointer :: ptr, ptr_spec, ptr_bypass
    type(array_type), pointer :: ptr_encoded, ptr_mixed, ptr_decoded


    ! Spectral pathway: Phi @ R @ Phi^T @ u
    ! Uses autodiff-tracked ono_encode/ono_decode for basis gradients
    !---------------------------------------------------------------------------

    ! Encode: Q(B)^T @ u  -> [k, batch]
    ptr_encoded => ono_encode(input(1,1), this%params(2), &
         this%num_inputs, this%num_basis)

    ! Mix: R @ encoded   -> [k, batch]
    ptr_mixed => matmul(this%params(1), ptr_encoded)

    ! Decode: Q(B) @ mixed -> [n_in, batch]
    ptr_decoded => ono_decode(ptr_mixed, this%params(2), &
         this%num_inputs, this%num_basis)

    ! Spectral projection: W @ decoded -> [n_out, batch]
    ptr_spec => matmul(this%params(3), ptr_decoded)

    ! Bypass: W @ u -> [n_out, batch]
    ptr_bypass => matmul(this%params(3), input(1,1))

    ! Combine
    ptr => ptr_spec + ptr_bypass

    ! Add bias
    if(this%use_bias)then
       ptr => ptr + this%params(4)
    end if

    ! Apply activation
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

  end subroutine forward_ono
!###############################################################################

end module athena__orthogonal_nop_block
