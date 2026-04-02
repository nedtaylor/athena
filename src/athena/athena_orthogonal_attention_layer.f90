module athena__orthogonal_attention_layer
  !! Module containing implementation of an Orthogonal Attention layer
  !!
  !! This module implements the Orthogonal Attention mechanism from
  !! "Improved Operator Learning by Orthogonal Attention" (Luo et al., 2024).
  !!
  !! Instead of softmax attention, this layer projects queries and keys
  !! onto a learned orthonormal basis of dimension \(k \ll N\), giving
  !! a linear-cost approximation to the attention kernel.
  !!
  !! Given input \(\mathbf{u} \in \mathbb{R}^{n_{in}}\):
  !!
  !! \[
  !!   \mathbf{Q} = \mathbf{W}_Q\,\mathbf{u}, \quad
  !!   \mathbf{K} = \mathbf{W}_K\,\mathbf{u}, \quad
  !!   \mathbf{V} = \mathbf{W}_V\,\mathbf{u}
  !! \]
  !!
  !! The orthogonal basis \(\mathbf{\Phi} \in \mathbb{R}^{n_{in} \times k}\)
  !! is obtained by QR decomposition of learnable weights
  !! \(\mathbf{B} \in \mathbb{R}^{n_{in} \times k}\).
  !!
  !! The attention output is:
  !! \[
  !!   \text{Attn}(\mathbf{u}) = \mathbf{\Phi}\,
  !!     (\mathbf{\Phi}^T \mathbf{Q})^T\,
  !!     (\mathbf{\Phi}^T \mathbf{K})\,
  !!     \mathbf{V}
  !! \]
  !!
  !! The layer output is:
  !! \[
  !!   \mathbf{v} = \sigma\!\bigl(
  !!     \text{Attn}(\mathbf{u}) + \mathbf{W}\,\mathbf{u} + \mathbf{b}
  !!   \bigr)
  !! \]
  !!
  !! Parameters (learnable):
  !!   - \(\mathbf{W}_Q \in \mathbb{R}^{d_k \times n_{in}}\)
  !!   - \(\mathbf{W}_K \in \mathbb{R}^{d_k \times n_{in}}\)
  !!   - \(\mathbf{W}_V \in \mathbb{R}^{n_{out} \times n_{in}}\)
  !!   - \(\mathbf{B}   \in \mathbb{R}^{n_{in} \times k}\)  (basis, orthogonalised)
  !!   - \(\mathbf{W}   \in \mathbb{R}^{n_{out} \times n_{in}}\)  (bypass)
  !!   - \(\mathbf{b}   \in \mathbb{R}^{n_{out}}\)  (optional bias)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type
  use diffstruc, only: array_type, matmul, operator(+), operator(*)
  use athena__diffstruc_extd, only: ono_encode, ono_decode
  implicit none


  private

  public :: orthogonal_attention_layer_type
  public :: read_orthogonal_attention_layer


  type, extends(learnable_layer_type) :: orthogonal_attention_layer_type
     !! Type for an Orthogonal Attention layer
     integer :: num_inputs = 0
     !! Number of input features / discretisation points
     integer :: num_outputs = 0
     !! Number of output features / discretisation points
     integer :: num_basis = 0
     !! Number of orthogonal basis functions (k)
     integer :: key_dim = 0
     !! Dimension of query/key projections (d_k)
     type(array_type), dimension(1) :: z
     !! Temporary array for pre-activation values
   contains
     procedure, pass(this) :: get_num_params => get_num_params_ono_attn
     procedure, pass(this) :: set_hyperparams => set_hyperparams_ono_attn
     procedure, pass(this) :: init => init_ono_attn
     procedure, pass(this) :: print_to_unit => print_to_unit_ono_attn
     procedure, pass(this) :: read => read_ono_attn

     procedure, pass(this) :: forward => forward_ono_attn
     procedure, pass(this) :: get_bases => get_bases_ono_attn

     final :: finalise_ono_attn
  end type orthogonal_attention_layer_type

  interface orthogonal_attention_layer_type
     module function layer_setup( &
          num_outputs, num_basis, key_dim, &
          num_inputs, use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       integer, intent(in) :: num_outputs
       integer, intent(in) :: num_basis
       integer, optional, intent(in) :: key_dim
       integer, optional, intent(in) :: num_inputs
       logical, optional, intent(in) :: use_bias
       class(*), optional, intent(in) :: activation
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       integer, optional, intent(in) :: verbose
       type(orthogonal_attention_layer_type) :: layer
     end function layer_setup
  end interface orthogonal_attention_layer_type



contains

!###############################################################################
  subroutine finalise_ono_attn(this)
    implicit none
    type(orthogonal_attention_layer_type), intent(inout) :: this

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine finalise_ono_attn
!###############################################################################


!###############################################################################
  pure function get_num_params_ono_attn(this) result(num_params)
    implicit none
    class(orthogonal_attention_layer_type), intent(in) :: this
    integer :: num_params

    ! W_Q: key_dim * num_inputs
    ! W_K: key_dim * num_inputs
    ! W_V: num_outputs * num_inputs
    ! B:   num_inputs * num_basis  (basis weights to orthogonalise)
    ! W:   num_outputs * num_inputs (bypass)
    ! b:   num_outputs (optional)
    num_params = this%key_dim * this%num_inputs + &     ! W_Q
         this%key_dim * this%num_inputs + &              ! W_K
         this%num_outputs * this%num_inputs + &          ! W_V
         this%num_inputs * this%num_basis + &            ! B
         this%num_outputs * this%num_inputs              ! W
    if(this%use_bias) num_params = num_params + this%num_outputs

  end function get_num_params_ono_attn
!###############################################################################


!###############################################################################
  module function layer_setup( &
       num_outputs, num_basis, key_dim, &
       num_inputs, use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    integer, intent(in) :: num_outputs
    integer, intent(in) :: num_basis
    integer, optional, intent(in) :: key_dim
    integer, optional, intent(in) :: num_inputs
    logical, optional, intent(in) :: use_bias
    class(*), optional, intent(in) :: activation
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose

    type(orthogonal_attention_layer_type) :: layer

    integer :: verbose_ = 0
    integer :: key_dim_
    logical :: use_bias_ = .true.
    class(base_actv_type), allocatable :: activation_
    class(base_init_type), allocatable :: kernel_initialiser_, bias_initialiser_

    if(present(verbose)) verbose_ = verbose
    if(present(use_bias)) use_bias_ = use_bias
    key_dim_ = num_basis
    if(present(key_dim)) key_dim_ = key_dim

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
         key_dim = key_dim_, &
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
  subroutine set_hyperparams_ono_attn( &
       this, num_outputs, num_basis, key_dim, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    class(orthogonal_attention_layer_type), intent(inout) :: this
    integer, intent(in) :: num_outputs
    integer, intent(in) :: num_basis
    integer, intent(in) :: key_dim
    logical, intent(in) :: use_bias
    class(base_actv_type), allocatable, intent(in) :: activation
    class(base_init_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose

    character(len=256) :: buffer

    this%name = "orthogonal_attention"
    this%type = "nop"
    this%input_rank = 1
    this%output_rank = 1
    this%use_bias = use_bias
    this%num_outputs = num_outputs
    this%num_basis = num_basis
    this%key_dim = key_dim

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
          write(*,'("ORTHOGONAL_ATTENTION activation: ",A)') &
               trim(this%activation%name)
       end if
    end if

  end subroutine set_hyperparams_ono_attn
!###############################################################################


!###############################################################################
  subroutine init_ono_attn(this, input_shape, verbose)
    implicit none

    class(orthogonal_attention_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: verbose

    integer :: num_inputs, idx, nparams
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
    ! params(1): W_Q  query projection   [key_dim x num_inputs]
    ! params(2): W_K  key projection     [key_dim x num_inputs]
    ! params(3): W_V  value projection   [num_outputs x num_inputs]
    ! params(4): B    basis weights       [num_inputs x num_basis]
    ! params(5): W    bypass weights      [num_outputs x num_inputs]
    ! params(6): b    bias                [num_outputs]  (optional)
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(2,5))
    this%weight_shape(:,1) = [ this%key_dim, this%num_inputs ]
    this%weight_shape(:,2) = [ this%key_dim, this%num_inputs ]
    this%weight_shape(:,3) = [ this%num_outputs, this%num_inputs ]
    this%weight_shape(:,4) = [ this%num_inputs, this%num_basis ]
    this%weight_shape(:,5) = [ this%num_outputs, this%num_inputs ]

    if(this%use_bias)then
       this%bias_shape = [ this%num_outputs ]
       allocate(this%params(6))
    else
       allocate(this%params(5))
    end if

    num_inputs = this%num_inputs
    if(this%use_bias) num_inputs = this%num_inputs + 1

    ! W_Q
    call this%params(1)%allocate([this%key_dim, this%num_inputs, 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.

    ! W_K
    call this%params(2)%allocate([this%key_dim, this%num_inputs, 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%fix_pointer = .true.
    this%params(2)%is_sample_dependent = .false.
    this%params(2)%is_temporary = .false.

    ! W_V
    call this%params(3)%allocate([this%num_outputs, this%num_inputs, 1])
    call this%params(3)%set_requires_grad(.true.)
    this%params(3)%fix_pointer = .true.
    this%params(3)%is_sample_dependent = .false.
    this%params(3)%is_temporary = .false.

    ! B (basis weights)
    call this%params(4)%allocate([this%num_inputs, this%num_basis, 1])
    call this%params(4)%set_requires_grad(.true.)
    this%params(4)%fix_pointer = .true.
    this%params(4)%is_sample_dependent = .false.
    this%params(4)%is_temporary = .false.

    ! W (bypass)
    call this%params(5)%allocate([this%num_outputs, this%num_inputs, 1])
    call this%params(5)%set_requires_grad(.true.)
    this%params(5)%fix_pointer = .true.
    this%params(5)%is_sample_dependent = .false.
    this%params(5)%is_temporary = .false.

    ! b (bias, optional)
    if(this%use_bias)then
       call this%params(6)%allocate([this%bias_shape, 1])
       call this%params(6)%set_requires_grad(.true.)
       this%params(6)%fix_pointer = .true.
       this%params(6)%is_sample_dependent = .false.
       this%params(6)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise learnable parameters
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(1)%val(:,1), &
         fan_in = this%num_inputs, fan_out = this%key_dim, &
         spacing = [ this%key_dim ] &
    )
    call this%kernel_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = this%num_inputs, fan_out = this%key_dim, &
         spacing = [ this%key_dim ] &
    )
    call this%kernel_init%initialise( &
         this%params(3)%val(:,1), &
         fan_in = num_inputs, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )
    call this%kernel_init%initialise( &
         this%params(4)%val(:,1), &
         fan_in = this%num_inputs, fan_out = this%num_basis, &
         spacing = [ this%num_inputs ] &
    )
    call this%kernel_init%initialise( &
         this%params(5)%val(:,1), &
         fan_in = num_inputs, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )
    if(this%use_bias)then
       call this%bias_init%initialise( &
            this%params(6)%val(:,1), &
            fan_in = num_inputs, fan_out = this%num_outputs &
       )
    end if


    !---------------------------------------------------------------------------
    ! Allocate output arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))
    if(this%z(1)%allocated) call this%z(1)%deallocate()

  end subroutine init_ono_attn
!###############################################################################


!###############################################################################
  function get_bases_ono_attn(this) result(phi)
    !! Orthogonalise the basis matrix B using modified Gram-Schmidt
    implicit none
    class(orthogonal_attention_layer_type), intent(inout) :: this
    type(array_type) :: phi

    integer :: n, k, i, j
    real(real32), allocatable :: B(:,:), Q(:,:)
    real(real32) :: norm_val, proj

    n = this%num_inputs
    k = this%num_basis

    allocate(B(n, k), Q(n, k))

    ! Reshape B from flat params(4) into [n, k]
    B = reshape(this%params(4)%val(:,1), [n, k])

    ! Modified Gram-Schmidt orthogonalisation
    Q = B
    do j = 1, k
       ! Subtract projections of previous orthogonal vectors
       do i = 1, j - 1
          proj = dot_product(Q(:,i), Q(:,j))
          Q(:,j) = Q(:,j) - proj * Q(:,i)
       end do
       ! Normalise
       norm_val = sqrt(dot_product(Q(:,j), Q(:,j)))
       if(norm_val .gt. 1.0e-12_real32)then
          Q(:,j) = Q(:,j) / norm_val
       else
          Q(:,j) = 0.0_real32
       end if
    end do

    ! Store in phi as a fixed array_type
    call phi%allocate([n, k, 1])
    phi%is_sample_dependent = .false.
    phi%requires_grad = .false.
    phi%fix_pointer = .true.
    phi%is_temporary = .false.
    phi%val(:,1) = reshape(Q, [n * k])

    deallocate(B, Q)

  end function get_bases_ono_attn
!###############################################################################


!###############################################################################
  subroutine print_to_unit_ono_attn(this, unit)
    use coreutils, only: to_upper
    implicit none

    class(orthogonal_attention_layer_type), intent(in) :: this
    integer, intent(in) :: unit

    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs
    write(unit,'(3X,"NUM_BASIS = ",I0)') this%num_basis
    write(unit,'(3X,"KEY_DIM = ",I0)') this%key_dim
    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if

    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)   ! W_Q
    write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)   ! W_K
    write(unit,'(5(E16.8E2))') this%params(3)%val(:,1)   ! W_V
    write(unit,'(5(E16.8E2))') this%params(4)%val(:,1)   ! B
    write(unit,'(5(E16.8E2))') this%params(5)%val(:,1)   ! W
    if(this%use_bias)then
       write(unit,'(5(E16.8E2))') this%params(6)%val(:,1) ! b
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_ono_attn
!###############################################################################


!###############################################################################
  subroutine read_ono_attn(this, unit, verbose)
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    class(orthogonal_attention_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: stat, verbose_ = 0
    integer :: j, k, c, itmp1, iline
    integer :: num_inputs, num_outputs, num_basis, key_dim
    logical :: use_bias = .true.
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    class(base_actv_type), allocatable :: activation
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
    character(256) :: buffer, tag, err_msg
    real(real32), allocatable, dimension(:) :: data_list
    integer :: param_line, final_line, num_vals

    if(present(verbose)) verbose_ = verbose

    key_dim = 0
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
       case("KEY_DIM")
          call assign_val(buffer, key_dim, itmp1)
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

    if(key_dim .eq. 0) key_dim = num_basis

    call this%set_hyperparams( &
         num_outputs = num_outputs, &
         num_basis = num_basis, &
         key_dim = key_dim, &
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

       ! Read W_Q
       num_vals = key_dim * num_inputs
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

       ! Read W_K
       num_vals = key_dim * num_inputs
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

       ! Read W_V
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

       ! Read B
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
       this%params(4)%val(:,1) = data_list
       deallocate(data_list)

       ! Read W (bypass)
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
       this%params(5)%val(:,1) = data_list
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
          this%params(6)%val(:,1) = data_list(1:num_outputs)
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

  end subroutine read_ono_attn
!###############################################################################


!###############################################################################
  function read_orthogonal_attention_layer(unit, verbose) result(layer)
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer
    integer :: verbose_ = 0

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=orthogonal_attention_layer_type( &
         num_outputs=0, num_basis=1))
    call layer%read(unit, verbose=verbose_)

  end function read_orthogonal_attention_layer
!###############################################################################


!###############################################################################
  subroutine forward_ono_attn(this, input)
    !! Forward propagation for the Orthogonal Attention layer
    !!
    !! Computes:
    !!   Q = W_Q @ u                     [k, batch]
    !!   K = W_K @ u                     [k, batch]
    !!   attn = Q * K                    [k, batch]  (per-basis attention scores)
    !!
    !!   spectral = Q(B)^T @ u           [k, batch]  (project to spectral, B-tracked)
    !!   modulated = attn * spectral     [k, batch]  (modulate spectral coefficients)
    !!   decoded = Q(B) @ modulated      [n_in, batch]  (decode, B-tracked)
    !!
    !!   attn_out = W_V @ decoded        [n_out, batch]
    !!   bypass   = W @ u               [n_out, batch]
    !!
    !!   v = sigma( attn_out + bypass + b )
    implicit none

    class(orthogonal_attention_layer_type), intent(inout) :: this
    class(array_type), dimension(:,:), intent(in) :: input

    type(array_type), pointer :: ptr, ptr_attn, ptr_bypass
    type(array_type), pointer :: ptr_Q, ptr_K, ptr_coeff
    type(array_type), pointer :: ptr_spec, ptr_mod, ptr_decoded

    integer :: n, nb


    n = this%num_inputs
    nb = this%num_basis


    !---------------------------------------------------------------------------
    ! Attention scores from Q and K projections
    !---------------------------------------------------------------------------
    ptr_Q => matmul(this%params(1), input(1,1))    ! W_Q @ u: [k, batch]
    ptr_K => matmul(this%params(2), input(1,1))    ! W_K @ u: [k, batch]
    ptr_coeff => ptr_Q * ptr_K                     ! element-wise: [k, batch]

    !---------------------------------------------------------------------------
    ! Spectral pathway: modulate spectral coefficients by attention scores
    !---------------------------------------------------------------------------
    ptr_spec => ono_encode(input(1,1), this%params(4), n, nb)  ! [k, batch]
    ptr_mod  => ptr_coeff * ptr_spec                           ! [k, batch]
    ptr_decoded => ono_decode(ptr_mod, this%params(4), n, nb)  ! [n, batch]

    ! Value projection
    ptr_attn => matmul(this%params(3), ptr_decoded)  ! [n_out, batch]

    ! Bypass: W @ u
    ptr_bypass => matmul(this%params(5), input(1,1))   ! [n_out, batch]

    ! Combine: attn_out + bypass
    ptr => ptr_attn + ptr_bypass

    ! Add bias
    if(this%use_bias)then
       ptr => ptr + this%params(6)
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

  end subroutine forward_ono_attn
!###############################################################################

end module athena__orthogonal_attention_layer
