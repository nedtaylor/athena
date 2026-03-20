module athena__fastkan_layer
  !! Module containing implementation of a Kolmogorov-Arnold Network (KAN) layer
  !!
  !! This module implements a KAN layer using radial basis function (RBF)
  !! expansions in the style of FastKAN. Each input dimension is expanded
  !! into a set of RBF activations, which are then linearly combined to
  !! produce the output.
  !!
  !! Mathematical operation:
  !! \[ \phi_{i,k}(x_i) = \exp\left(-\frac{(x_i - c_{i,k})^2}
  !!    {2\sigma_{i,k}^2}\right) \]
  !! \[ y_j = \sum_{i,k} W_{j,i,k}\,\phi_{i,k}(x_i) + b_j \]
  !!
  !! where:
  !!   - \( x_i \) is the \(i\)-th input component
  !!   - \( c_{i,k} \), \( \sigma_{i,k} \) are trainable RBF centres and
  !!     bandwidths
  !!   - \( W_{j,i,k} \) are trainable output weights
  !!   - \( b_j \) is the bias
  !!
  !! Trainable parameters:
  !!   - centres: (input_dim * n_basis)
  !!   - bandwidths: (input_dim * n_basis)
  !!   - weights: (output_dim * input_dim * n_basis)
  !!   - bias: (output_dim)
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use diffstruc, only: array_type, matmul, &
       operator(+), operator(-), operator(*), operator(/), exp
  implicit none


  private

  public :: fastkan_layer_type
  public :: read_fastkan_layer


  type, extends(learnable_layer_type) :: fastkan_layer_type
     !! Type for KAN layer with RBF-based learnable activation functions
     integer :: num_inputs
     !! Number of inputs
     integer :: num_outputs
     !! Number of outputs
     integer :: n_basis
     !! Number of radial basis functions per input dimension
     type(array_type) :: expand_matrix
     !! Selection matrix to expand input [input_dim*n_basis, input_dim]
   contains
     procedure, pass(this) :: get_num_params => get_num_params_kan
     !! Get the number of parameters for KAN layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_kan
     !! Set the hyperparameters for KAN layer
     procedure, pass(this) :: init => init_kan
     !! Initialise KAN layer
     procedure, pass(this) :: print_to_unit => print_to_unit_kan
     !! Print the layer to a file
     procedure, pass(this) :: read => read_kan
     !! Read the layer from a file
     procedure, pass(this) :: forward => forward_kan
     !! Forward propagation

     final :: finalise_kan
     !! Finalise KAN layer
  end type fastkan_layer_type

  interface fastkan_layer_type
     !! Interface for setting up the KAN layer
     module function layer_setup( &
          num_outputs, n_basis, num_inputs, use_bias, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       !! Setup a KAN layer
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: n_basis
       !! Number of radial basis functions per input dimension
       integer, optional, intent(in) :: num_inputs
       !! Number of inputs
       logical, optional, intent(in) :: use_bias
       !! Whether to use bias
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       !! Kernel and bias initialisers
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(fastkan_layer_type) :: layer
       !! Instance of the KAN layer
     end function layer_setup
  end interface fastkan_layer_type



contains

!###############################################################################
  subroutine finalise_kan(this)
    !! Finalise KAN layer
    implicit none

    ! Arguments
    type(fastkan_layer_type), intent(inout) :: this
    !! Instance of the KAN layer

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(this%expand_matrix%allocated) call this%expand_matrix%deallocate()

  end subroutine finalise_kan
!###############################################################################


!###############################################################################
  pure function get_num_params_kan(this) result(num_params)
    !! Get the number of parameters for KAN layer
    implicit none

    ! Arguments
    class(fastkan_layer_type), intent(in) :: this
    !! Instance of the KAN layer
    integer :: num_params
    !! Number of parameters

    ! centres + bandwidths + weights + bias
    num_params = this%num_inputs * this%n_basis + &
         this%num_inputs * this%n_basis + &
         this%num_outputs * this%num_inputs * this%n_basis + &
         this%num_outputs

  end function get_num_params_kan
!###############################################################################


!###############################################################################
  module function layer_setup( &
       num_outputs, n_basis, num_inputs, &
       use_bias, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    !! Setup a KAN layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: n_basis
    !! Number of radial basis functions per input dimension
    integer, optional, intent(in) :: num_inputs
    !! Number of inputs
    logical, optional, intent(in) :: use_bias
    !! Whether to use bias
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(fastkan_layer_type) :: layer
    !! Instance of the KAN layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: n_basis_ = 8
    !! Default number of basis functions
    logical :: use_bias_ = .true.
    !! Whether to use bias
    class(base_init_type), allocatable :: kernel_initialiser_, bias_initialiser_
    !! Kernel and bias initialisers

    if(present(verbose)) verbose_ = verbose
    if(present(n_basis)) n_basis_ = n_basis
    if(present(use_bias)) use_bias_ = use_bias

    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if
    if(present(bias_initialiser))then
       bias_initialiser_ = initialiser_setup(bias_initialiser)
    end if

    call layer%set_hyperparams( &
         num_outputs = num_outputs, &
         n_basis = n_basis_, &
         use_bias = use_bias_, &
         kernel_initialiser = kernel_initialiser_, &
         bias_initialiser = bias_initialiser_, &
         verbose = verbose_ &
    )

    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_kan( &
       this, num_outputs, n_basis, &
       use_bias, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for KAN layer
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    ! Arguments
    class(fastkan_layer_type), intent(inout) :: this
    !! Instance of the KAN layer
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, intent(in) :: n_basis
    !! Number of radial basis functions per input dimension
    logical, intent(in) :: use_bias
    !! Whether to use bias
    class(base_init_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    character(len=256) :: buffer


    this%name = "fastkan"
    this%type = "fastkan"
    this%input_rank = 1
    this%output_rank = 1
    this%use_bias = use_bias
    this%num_outputs = num_outputs
    this%n_basis = n_basis

    ! KAN doesn't use a traditional activation function
    if(allocated(this%activation)) deallocate(this%activation)

    if(allocated(this%kernel_init)) deallocate(this%kernel_init)
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser("none")
       this%kernel_init = initialiser_setup(buffer)
    else
       allocate(this%kernel_init, source=kernel_initialiser)
    end if
    if(allocated(this%bias_init)) deallocate(this%bias_init)
    if(.not.allocated(bias_initialiser))then
       buffer = get_default_initialiser("none", is_bias=.true.)
       this%bias_init = initialiser_setup(buffer)
    else
       allocate(this%bias_init, source=bias_initialiser)
    end if

    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("KAN kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("KAN bias initialiser: ",A)') &
               trim(this%bias_init%name)
       end if
    end if

  end subroutine set_hyperparams_kan
!###############################################################################


!###############################################################################
  subroutine init_kan(this, input_shape, verbose)
    !! Initialise KAN layer
    implicit none

    ! Arguments
    class(fastkan_layer_type), intent(inout) :: this
    !! Instance of the KAN layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: d, K, m, dK
    !! Dimension shortcuts
    integer :: i, k_idx
    !! Loop variables
    real(real32) :: centre_min, centre_max


    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise dimensions
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_inputs = this%input_shape(1)
    this%output_shape = [this%num_outputs]
    this%num_params = this%get_num_params()

    d = this%num_inputs
    K = this%n_basis
    m = this%num_outputs
    dK = d * K


    !---------------------------------------------------------------------------
    ! Allocate parameter arrays
    ! params(1): centres      [d*K, 1]
    ! params(2): bandwidths   [d*K, 1]
    ! params(3): weights      [m*d*K, 1]  with shape [m, d*K]
    ! params(4): bias         [m, 1]      (if use_bias)
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(2,1))
    this%weight_shape(:,1) = [ m, dK ]

    if(this%use_bias)then
       this%bias_shape = [ m ]
       allocate(this%params(4))
    else
       allocate(this%params(3))
    end if

    ! Centres
    call this%params(1)%allocate([dK, 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.

    ! Bandwidths
    call this%params(2)%allocate([dK, 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%fix_pointer = .true.
    this%params(2)%is_sample_dependent = .false.
    this%params(2)%is_temporary = .false.

    ! Weights
    call this%params(3)%allocate([this%weight_shape(:,1), 1])
    call this%params(3)%set_requires_grad(.true.)
    this%params(3)%fix_pointer = .true.
    this%params(3)%is_sample_dependent = .false.
    this%params(3)%is_temporary = .false.

    ! Bias
    if(this%use_bias)then
       call this%params(4)%allocate([this%bias_shape, 1])
       call this%params(4)%set_requires_grad(.true.)
       this%params(4)%fix_pointer = .true.
       this%params(4)%is_sample_dependent = .false.
       this%params(4)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise centres: uniformly spaced over [-1, 1]
    !---------------------------------------------------------------------------
    centre_min = -1.0_real32
    centre_max =  1.0_real32
    do i = 1, d
       do k_idx = 1, K
          if(K > 1)then
             this%params(1)%val((i-1)*K + k_idx, 1) = &
                  centre_min + (centre_max - centre_min) * &
                  real(k_idx - 1, real32) / real(K - 1, real32)
          else
             this%params(1)%val((i-1)*K + k_idx, 1) = 0.0_real32
          end if
       end do
    end do


    !---------------------------------------------------------------------------
    ! Initialise bandwidths: set to spacing between centres
    !---------------------------------------------------------------------------
    if(K > 1)then
       this%params(2)%val(:, 1) = &
            (centre_max - centre_min) / real(K - 1, real32)
    else
       this%params(2)%val(:, 1) = 1.0_real32
    end if


    !---------------------------------------------------------------------------
    ! Initialise weights using kernel initialiser
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(3)%val(:,1), &
         fan_in = dK, fan_out = m, &
         spacing = [ m ] &
    )


    !---------------------------------------------------------------------------
    ! Initialise biases to zero
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       this%params(4)%val(:, 1) = 0.0_real32
    end if


    !---------------------------------------------------------------------------
    ! Build expansion matrix S: shape [d*K, d]
    ! S[(i-1)*K + k, i] = 1 for all i=1..d, k=1..K
    ! This repeats each x_i across K basis functions
    !---------------------------------------------------------------------------
    call this%expand_matrix%allocate([dK, d, 1])
    this%expand_matrix%val(:, 1) = 0.0_real32
    this%expand_matrix%is_sample_dependent = .false.
    this%expand_matrix%is_temporary = .false.
    this%expand_matrix%fix_pointer = .true.
    call this%expand_matrix%set_requires_grad(.false.)
    do i = 1, d
       do k_idx = 1, K
          ! Column-major: element (row, col) at index (col-1)*dK + row
          this%expand_matrix%val((i-1)*dK + (i-1)*K + k_idx, 1) = 1.0_real32
       end do
    end do


    !---------------------------------------------------------------------------
    ! Allocate output array
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))

  end subroutine init_kan
!###############################################################################


!###############################################################################
  subroutine print_to_unit_kan(this, unit)
    !! Print KAN layer to unit
    implicit none

    ! Arguments
    class(fastkan_layer_type), intent(in) :: this
    !! Instance of the KAN layer
    integer, intent(in) :: unit
    !! File unit


    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs
    write(unit,'(3X,"N_BASIS = ",I0)') this%n_basis
    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias

    ! Write parameters
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)  ! centres
    write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)  ! bandwidths
    write(unit,'(5(E16.8E2))') this%params(3)%val(:,1)  ! weights
    if(this%use_bias)then
       write(unit,'(5(E16.8E2))') this%params(4)%val(:,1)  ! bias
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_kan
!###############################################################################


!###############################################################################
  subroutine read_kan(this, unit, verbose)
    !! Read KAN layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(fastkan_layer_type), intent(inout) :: this
    !! Instance of the KAN layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    integer :: verbose_ = 0
    integer :: j, k, c, itmp1, iline, total_param_count
    integer :: num_inputs, num_outputs, n_basis
    logical :: use_bias = .true.
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
    character(256) :: buffer, tag, err_msg
    real(real32), allocatable, dimension(:) :: data_list
    integer :: param_line, final_line


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
       case("N_BASIS")
          call assign_val(buffer, n_basis, itmp1)
       case("USE_BIAS")
          call assign_val(buffer, use_bias, itmp1)
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
         n_basis = n_basis, &
         use_bias = use_bias, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[num_inputs])

    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in "// &
            to_upper(trim(this%name))//" not found"
    else
       call move(unit, param_line - iline, iostat=stat)

       ! Read all parameters: centres + bandwidths + weights [+ bias]
       total_param_count = num_inputs * n_basis + &          ! centres
            num_inputs * n_basis + &          ! bandwidths
            num_outputs * num_inputs * n_basis  ! weights
       if(use_bias) total_param_count = total_param_count + num_outputs

       allocate(data_list(total_param_count), source=0._real32)
       c = 1
       k = 1
       data_concat_loop: do while(c.le.total_param_count)
          read(unit,'(A)',iostat=stat) buffer
          if(stat.ne.0) exit data_concat_loop
          k = icount(buffer)
          read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
          c = c + k
       end do data_concat_loop

       ! Assign parameters from data_list
       c = 1
       ! centres
       this%params(1)%val(:,1) = data_list(c:c + num_inputs*n_basis - 1)
       c = c + num_inputs * n_basis
       ! bandwidths
       this%params(2)%val(:,1) = data_list(c:c + num_inputs*n_basis - 1)
       c = c + num_inputs * n_basis
       ! weights
       this%params(3)%val(:,1) = data_list(c:c + num_outputs*num_inputs*n_basis - 1)
       c = c + num_outputs * num_inputs * n_basis
       ! bias
       if(use_bias)then
          this%params(4)%val(:,1) = data_list(c:c + num_outputs - 1)
       end if
       deallocate(data_list)

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

  end subroutine read_kan
!###############################################################################


!###############################################################################
  function read_fastkan_layer(unit, verbose) result(layer)
    !! Read KAN layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the KAN layer

    ! Local variables
    integer :: verbose_ = 0

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=fastkan_layer_type(num_outputs=0, n_basis=1))
    call layer%read(unit, verbose=verbose_)

  end function read_fastkan_layer
!###############################################################################


!###############################################################################
  subroutine forward_kan(this, input)
    !! Forward propagation for KAN layer
    !!
    !! Computes RBF activations and linear output:
    !! phi_{i,k} = exp(-0.5 * (x_i - c_{i,k})^2 / sigma_{i,k}^2)
    !! y_j = sum_{i,k} W_{j,i,k} * phi_{i,k} + b_j
    implicit none

    ! Arguments
    class(fastkan_layer_type), intent(inout) :: this
    !! Instance of the KAN layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values

    type(array_type), pointer :: ptr_exp => null()
    type(array_type), pointer :: ptr_diff => null()
    type(array_type), pointer :: ptr_phi => null()
    type(array_type), pointer :: ptr => null()


    ! Expand input: x_expanded = S * x  [d*K, batch]
    !---------------------------------------------------------------------------
    ptr_exp => matmul(this%expand_matrix, input(1,1))

    ! Compute difference: (x_expanded - centres)  [d*K, batch]
    !---------------------------------------------------------------------------
    ptr_diff => ptr_exp - this%params(1)

    ! Compute RBF activations: exp(-0.5 * (diff/sigma)^2)  [d*K, batch]
    !---------------------------------------------------------------------------
    ptr_phi => exp( (-0.5_real32) * ((ptr_diff / this%params(2)) ** 2) )

    ! Compute output: W * phi [+ bias]  [m, batch]
    !---------------------------------------------------------------------------
    if(this%use_bias)then
       ptr => matmul(this%params(3), ptr_phi) + this%params(4)
    else
       ptr => matmul(this%params(3), ptr_phi)
    end if

    ! Store output
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_kan
!###############################################################################

end module athena__fastkan_layer
