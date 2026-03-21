module athena__kan_layer
  !! Module containing implementation of a B-spline KAN layer
  !!
  !! This module implements a KAN layer using B-spline basis functions in the
  !! style of the original Kolmogorov-Arnold Network (KAN) paper.
  !! Each input dimension is expanded into a set of B-spline basis activations,
  !! which are then linearly combined to produce the output.
  !!
  !! Mathematical operation:
  !! \[ \phi_{i,k}(x_i) = B_k(x_i) \]
  !! \[ y_j = \sum_{i,k} W_{j,i,k}\,\phi_{i,k}(x_i) + b_j \]
  !!
  !! where:
  !!   - \( x_i \) is the \(i\)-th input component
  !!   - \( B_k(x) \) are B-spline basis functions of degree \( p \)
  !!     on a fixed uniform knot grid
  !!   - \( W_{j,i,k} \) are trainable output weights
  !!   - \( b_j \) is the bias
  !!
  !! B-spline basis functions are evaluated using the Cox-de Boor recursion:
  !! \[ B_{i,0}(x) = \begin{cases} 1 & t_i \le x < t_{i+1} \\ 0 &
  !!    \text{otherwise} \end{cases} \]
  !! \[ B_{i,p}(x) = \frac{x - t_i}{t_{i+p} - t_i} B_{i,p-1}(x) +
  !!    \frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} B_{i+1,p-1}(x) \]
  !!
  !! Trainable parameters:
  !!   - weights: (output_dim * input_dim * n_basis)
  !!   - bias: (output_dim)
  !!
  !! Fixed storage:
  !!   - knots: uniform grid with augmented boundary knots
  use coreutils, only: real32, stop_program
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use diffstruc, only: array_type, matmul, &
       operator(+), operator(-), operator(*), operator(/)
  use athena__diffstruc_extd, only: swish
  implicit none


  private

  public :: kan_layer_type
  public :: read_kan_layer


  type, extends(learnable_layer_type) :: kan_layer_type
     !! Type for KAN layer with B-spline basis functions
     integer :: num_inputs
     !! Number of inputs
     integer :: num_outputs
     !! Number of outputs
     integer :: n_basis
     !! Number of B-spline basis functions per input dimension
     integer :: spline_degree
     !! Degree of the B-spline basis functions
     integer :: n_knots
     !! Total number of knots (n_basis + spline_degree + 1)
     logical :: use_base_activation = .false.
     !! Whether to use PyKAN-style base activation (silu + spline)
     real(real32), allocatable :: knots(:)
     !! Knot vector (n_knots)
     type(array_type) :: basis_matrix
     !! B-spline basis values [d*K, batch] (recomputed each forward)
   contains
     procedure, pass(this) :: get_num_params => get_num_params_bspline_kan
     !! Get the number of parameters
     procedure, pass(this) :: set_hyperparams => set_hyperparams_bspline_kan
     !! Set the hyperparameters
     procedure, pass(this) :: init => init_bspline_kan
     !! Initialise layer
     procedure, pass(this) :: print_to_unit => print_to_unit_bspline_kan
     !! Print the layer to a file
     procedure, pass(this) :: read => read_bspline_kan
     !! Read the layer from a file
     procedure, pass(this) :: forward => forward_bspline_kan
     !! Forward propagation
     procedure, pass(this) :: evaluate_bspline_basis
     !! Evaluate B-spline basis functions for a batch of inputs

     final :: finalise_bspline_kan
     !! Finalise layer
  end type kan_layer_type

  interface kan_layer_type
     !! Interface for setting up the B-spline KAN layer
     module function layer_setup( &
          num_outputs, n_basis, spline_degree, &
          num_inputs, use_bias, use_base_activation, &
          kernel_initialiser, bias_initialiser, verbose &
     ) result(layer)
       !! Setup a B-spline KAN layer
       integer, intent(in) :: num_outputs
       !! Number of outputs
       integer, optional, intent(in) :: n_basis
       !! Number of B-spline basis functions per input dimension
       integer, optional, intent(in) :: spline_degree
       !! Degree of B-spline (default: 3)
       integer, optional, intent(in) :: num_inputs
       !! Number of inputs
       logical, optional, intent(in) :: use_bias
       !! Whether to use bias
       logical, optional, intent(in) :: use_base_activation
       !! Whether to use PyKAN-style base activation (silu + spline)
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       !! Kernel and bias initialisers
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(kan_layer_type) :: layer
       !! Instance of the B-spline KAN layer
     end function layer_setup
  end interface kan_layer_type



contains

!###############################################################################
  subroutine finalise_bspline_kan(this)
    !! Finalise B-spline KAN layer
    implicit none

    ! Arguments
    type(kan_layer_type), intent(inout) :: this
    !! Instance of the B-spline KAN layer

    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%knots)) deallocate(this%knots)
    if(this%basis_matrix%allocated) call this%basis_matrix%deallocate()

  end subroutine finalise_bspline_kan
!###############################################################################


!###############################################################################
  pure function get_num_params_bspline_kan(this) result(num_params)
    !! Get the number of parameters for B-spline KAN layer
    implicit none

    ! Arguments
    class(kan_layer_type), intent(in) :: this
    !! Instance of the B-spline KAN layer
    integer :: num_params
    !! Number of parameters

    ! weights + bias
    num_params = this%num_outputs * this%num_inputs * this%n_basis + &
         this%num_outputs

    ! scale_base [m, d] + scale_sp [m, 1]
    if(this%use_base_activation)then
       num_params = num_params + &
            this%num_outputs * this%num_inputs + this%num_outputs
    end if

  end function get_num_params_bspline_kan
!###############################################################################


!###############################################################################
  module function layer_setup( &
       num_outputs, n_basis, spline_degree, &
       num_inputs, &
       use_bias, use_base_activation, &
       kernel_initialiser, bias_initialiser, verbose &
  ) result(layer)
    !! Setup a B-spline KAN layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, optional, intent(in) :: n_basis
    !! Number of B-spline basis functions per input dimension
    integer, optional, intent(in) :: spline_degree
    !! Degree of B-spline (default: 3)
    integer, optional, intent(in) :: num_inputs
    !! Number of inputs
    logical, optional, intent(in) :: use_bias
    !! Whether to use bias
    logical, optional, intent(in) :: use_base_activation
    !! Whether to use PyKAN-style base activation (silu + spline)
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    type(kan_layer_type) :: layer
    !! Instance of the B-spline KAN layer

    ! Local variables
    integer :: verbose_
    !! Verbosity level
    integer :: n_basis_
    !! Default number of basis functions
    integer :: spline_degree_
    !! Default spline degree
    logical :: use_bias_
    !! Whether to use bias
    logical :: use_base_activation_
    !! Whether to use base activation
    class(base_init_type), allocatable :: kernel_initialiser_, bias_initialiser_
    !! Kernel and bias initialisers

    verbose_ = 0
    n_basis_ = 8
    spline_degree_ = 3
    use_bias_ = .true.
    use_base_activation_ = .false.

    if(present(verbose)) verbose_ = verbose
    if(present(n_basis)) n_basis_ = n_basis
    if(present(spline_degree)) spline_degree_ = spline_degree
    if(present(use_bias)) use_bias_ = use_bias
    if(present(use_base_activation)) &
         use_base_activation_ = use_base_activation

    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if
    if(present(bias_initialiser))then
       bias_initialiser_ = initialiser_setup(bias_initialiser)
    end if

    call layer%set_hyperparams( &
         num_outputs = num_outputs, &
         n_basis = n_basis_, &
         spline_degree = spline_degree_, &
         use_bias = use_bias_, &
         use_base_activation = use_base_activation_, &
         kernel_initialiser = kernel_initialiser_, &
         bias_initialiser = bias_initialiser_, &
         verbose = verbose_ &
    )

    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_bspline_kan( &
       this, num_outputs, n_basis, spline_degree, &
       use_bias, use_base_activation, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for B-spline KAN layer
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    ! Arguments
    class(kan_layer_type), intent(inout) :: this
    !! Instance of the B-spline KAN layer
    integer, intent(in) :: num_outputs
    !! Number of outputs
    integer, intent(in) :: n_basis
    !! Number of B-spline basis functions per input dimension
    integer, intent(in) :: spline_degree
    !! Degree of B-spline
    logical, intent(in) :: use_bias
    !! Whether to use bias
    logical, intent(in) :: use_base_activation
    !! Whether to use PyKAN-style base activation
    class(base_init_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    !! Kernel and bias initialisers
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    character(len=256) :: buffer


    this%name = "kan"
    this%type = "kan "
    this%input_rank = 1
    this%output_rank = 1
    this%use_bias = use_bias
    this%use_base_activation = use_base_activation
    this%num_outputs = num_outputs
    this%n_basis = n_basis
    this%spline_degree = spline_degree

    ! B-spline KAN doesn't use a traditional activation function
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
          write(*,'("B-spline KAN kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
          write(*,'("B-spline KAN bias initialiser: ",A)') &
               trim(this%bias_init%name)
       end if
    end if

  end subroutine set_hyperparams_bspline_kan
!###############################################################################


!###############################################################################
  subroutine init_bspline_kan(this, input_shape, verbose)
    !! Initialise B-spline KAN layer
    implicit none

    ! Arguments
    class(kan_layer_type), intent(inout) :: this
    !! Instance of the B-spline KAN layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: verbose_
    !! Verbosity level
    integer :: d, K, m, dK, p
    !! Dimension shortcuts
    integer :: i
    !! Loop variable
    real(real32) :: grid_min, grid_max, h


    verbose_ = 0
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
    p = this%spline_degree
    dK = d * K


    !---------------------------------------------------------------------------
    ! Compute knot vector
    ! For K basis functions of degree p, we need K + p + 1 knots
    ! Use uniform spacing over [-1, 1] with augmented boundary knots
    !---------------------------------------------------------------------------
    this%n_knots = K + p + 1
    if(allocated(this%knots)) deallocate(this%knots)
    allocate(this%knots(this%n_knots))

    grid_min = -1.0_real32
    grid_max =  1.0_real32
    h = (grid_max - grid_min) / real(K - p, real32)

    do i = 1, this%n_knots
       this%knots(i) = grid_min + real(i - 1 - p, real32) * h
    end do

    ! Clamp boundary knots
    do i = 1, p + 1
       this%knots(i) = grid_min
    end do
    do i = this%n_knots - p, this%n_knots
       this%knots(i) = grid_max
    end do


    !---------------------------------------------------------------------------
    ! Allocate parameter arrays
    ! params(1): weights      [m, d*K]    (spline coefficients)
    ! params(2): bias         [m, 1]      (if use_bias)
    ! params(3): scale_base   [m, d]      (if use_base_activation)
    ! params(4): scale_sp     [m, 1]      (if use_base_activation)
    !---------------------------------------------------------------------------
    allocate(this%weight_shape(2,1))
    this%weight_shape(:,1) = [ m, dK ]

    if(this%use_base_activation)then
       if(this%use_bias)then
          allocate(this%params(4))
       else
          allocate(this%params(4))
       end if
    else
       if(this%use_bias)then
          allocate(this%params(2))
       else
          allocate(this%params(1))
       end if
    end if

    ! Weights
    call this%params(1)%allocate([this%weight_shape(:,1), 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.

    ! Bias
    if(this%use_bias .or. this%use_base_activation)then
       this%bias_shape = [ m ]
       call this%params(2)%allocate([m, 1])
       call this%params(2)%set_requires_grad(.true.)
       this%params(2)%fix_pointer = .true.
       this%params(2)%is_sample_dependent = .false.
       this%params(2)%is_temporary = .false.
    end if

    ! Scale base: weight matrix for base activation [m, d]
    if(this%use_base_activation)then
       call this%params(3)%allocate([m, d, 1])
       call this%params(3)%set_requires_grad(.true.)
       this%params(3)%fix_pointer = .true.
       this%params(3)%is_sample_dependent = .false.
       this%params(3)%is_temporary = .false.
    end if

    ! Scale sp: per-output spline scaling [m, 1]
    if(this%use_base_activation)then
       call this%params(4)%allocate([m, 1])
       call this%params(4)%set_requires_grad(.true.)
       this%params(4)%fix_pointer = .true.
       this%params(4)%is_sample_dependent = .false.
       this%params(4)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise weights using kernel initialiser
    !---------------------------------------------------------------------------
    call this%kernel_init%initialise( &
         this%params(1)%val(:,1), &
         fan_in = dK, fan_out = m, &
         spacing = [ m ] &
    )


    !---------------------------------------------------------------------------
    ! Initialise biases to zero
    !---------------------------------------------------------------------------
    if(this%use_bias .or. this%use_base_activation)then
       this%params(2)%val(:, 1) = 0.0_real32
    end if


    !---------------------------------------------------------------------------
    ! Initialise scale_base (zeros) and scale_sp (ones)
    !---------------------------------------------------------------------------
    if(this%use_base_activation)then
       this%params(3)%val(:, 1) = 0.0_real32
       this%params(4)%val(:, 1) = 1.0_real32
    end if


    !---------------------------------------------------------------------------
    ! Allocate output array
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)
    allocate(this%output(1,1))

  end subroutine init_bspline_kan
!###############################################################################


!###############################################################################
  subroutine evaluate_bspline_basis(this, x_flat, n_samples, basis_vals)
    !! Evaluate B-spline basis functions using Cox-de Boor recursion
    !!
    !! For each input sample and each input dimension, computes K basis
    !! function values. Output is stored column-major: basis_vals(d*K, n_samples)
    !!
    !! The Cox-de Boor recursion:
    !! B_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0
    !! B_{i,p}(x) = w1 * B_{i,p-1}(x) + w2 * B_{i+1,p-1}(x)
    !! where w1 = (x - t_i) / (t_{i+p} - t_i)
    !!       w2 = (t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1})
    implicit none

    ! Arguments
    class(kan_layer_type), intent(in) :: this
    !! Instance of the B-spline KAN layer
    real(real32), dimension(:,:), intent(in) :: x_flat
    !! Expanded input values [d*K, n_samples] (only d unique values per sample)
    integer, intent(in) :: n_samples
    !! Number of samples in batch
    real(real32), dimension(:,:), intent(out) :: basis_vals
    !! Output basis values [d*K, n_samples]

    ! Local variables
    integer :: d, K, p, n_knots_single
    integer :: i_dim, i_basis, i_order, s
    real(real32) :: x_val, denom1, denom2, w1, w2
    real(real32), allocatable :: B_prev(:), B_curr(:)
    !! Temporary storage for recursion over one input dimension


    d = this%num_inputs
    K = this%n_basis
    p = this%spline_degree
    n_knots_single = this%n_knots

    ! For each input dimension, there are K basis functions
    allocate(B_prev(K + p))
    allocate(B_curr(K + p))

    basis_vals = 0.0_real32

    do s = 1, n_samples
       do i_dim = 1, d
          ! Get the input value (same for all K basis functions of this dim)
          x_val = x_flat((i_dim - 1) * K + 1, s)

          ! Clamp to knot range
          x_val = max(this%knots(1), min(x_val, &
               this%knots(n_knots_single) - 1.0E-7_real32))

          !--------------------------------------------------------------------
          ! Degree 0: B_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0
          !--------------------------------------------------------------------
          B_prev = 0.0_real32
          do i_basis = 1, n_knots_single - 1
             if(x_val .ge. this%knots(i_basis) .and. &
                  x_val .lt. this%knots(i_basis + 1))then
                B_prev(i_basis) = 1.0_real32
             end if
          end do
          ! Handle right boundary: include x == t_{n_knots} in last interval
          if(x_val .ge. this%knots(n_knots_single - 1))then
             B_prev(n_knots_single - 1) = 1.0_real32
          end if

          !--------------------------------------------------------------------
          ! Recursive degrees 1..p
          !--------------------------------------------------------------------
          do i_order = 1, p
             B_curr = 0.0_real32
             do i_basis = 1, n_knots_single - 1 - i_order
                ! Left term: (x - t_i) / (t_{i+p} - t_i) * B_{i,p-1}
                denom1 = this%knots(i_basis + i_order) - this%knots(i_basis)
                if(abs(denom1) .gt. 1.0E-10_real32)then
                   w1 = (x_val - this%knots(i_basis)) / denom1
                else
                   w1 = 0.0_real32
                end if

                ! Right term: (t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1}) * B_{i+1,p-1}
                denom2 = this%knots(i_basis + i_order + 1) - &
                     this%knots(i_basis + 1)
                if(abs(denom2) .gt. 1.0E-10_real32)then
                   w2 = (this%knots(i_basis + i_order + 1) - x_val) / denom2
                else
                   w2 = 0.0_real32
                end if

                B_curr(i_basis) = w1 * B_prev(i_basis) + &
                     w2 * B_prev(i_basis + 1)
             end do
             B_prev = B_curr
          end do

          !--------------------------------------------------------------------
          ! Store K basis values for this dimension
          !--------------------------------------------------------------------
          do i_basis = 1, K
             basis_vals((i_dim - 1) * K + i_basis, s) = B_prev(i_basis)
          end do

       end do
    end do

    deallocate(B_prev, B_curr)

  end subroutine evaluate_bspline_basis
!###############################################################################


!###############################################################################
  subroutine print_to_unit_bspline_kan(this, unit)
    !! Print B-spline KAN layer to unit
    implicit none

    ! Arguments
    class(kan_layer_type), intent(in) :: this
    !! Instance of the B-spline KAN layer
    integer, intent(in) :: unit
    !! File unit


    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs
    write(unit,'(3X,"N_BASIS = ",I0)') this%n_basis
    write(unit,'(3X,"SPLINE_DEGREE = ",I0)') this%spline_degree
    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    write(unit,'(3X,"USE_BASE_ACTIVATION = ",L1)') this%use_base_activation

    ! Write parameters
    write(unit,'("WEIGHTS")')
    write(unit,'(5(E16.8E2))') this%params(1)%val(:,1)  ! weights
    if(this%use_bias .or. this%use_base_activation)then
       write(unit,'(5(E16.8E2))') this%params(2)%val(:,1)  ! bias
    end if
    if(this%use_base_activation)then
       write(unit,'(5(E16.8E2))') this%params(3)%val(:,1)  ! scale_base
       write(unit,'(5(E16.8E2))') this%params(4)%val(:,1)  ! scale_sp
    end if
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_bspline_kan
!###############################################################################


!###############################################################################
  subroutine read_bspline_kan(this, unit, verbose)
    !! Read B-spline KAN layer from file
    use athena__tools_infile, only: assign_val, assign_vec, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(kan_layer_type), intent(inout) :: this
    !! Instance of the B-spline KAN layer
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    integer :: verbose_
    integer :: j, k, c, itmp1, iline, total_param_count
    integer :: num_inputs, num_outputs, n_basis, spline_degree
    logical :: use_bias
    logical :: use_base_activation
    character(14) :: kernel_initialiser_name, bias_initialiser_name
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
    character(256) :: buffer, tag, err_msg
    real(real32), allocatable, dimension(:) :: data_list
    integer :: param_line, final_line


    if(present(verbose)) verbose_ = verbose

    verbose_ = 0
    spline_degree = 3
    use_bias = .true.
    use_base_activation = .false.
    kernel_initialiser_name = ''
    bias_initialiser_name = ''

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

       if(trim(adjustl(buffer)).eq."END "// &
            to_upper(trim(this%name)))then
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
       case("SPLINE_DEGREE")
          call assign_val(buffer, spline_degree, itmp1)
       case("USE_BIAS")
          call assign_val(buffer, use_bias, itmp1)
       case("USE_BASE_ACTIVATION")
          call assign_val(buffer, use_base_activation, itmp1)
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
         spline_degree = spline_degree, &
         use_bias = use_bias, &
         use_base_activation = use_base_activation, &
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

       ! Read all parameters: weights [+ bias] [+ scale_base + scale_sp]
       total_param_count = num_outputs * num_inputs * n_basis  ! weights
       if(use_bias .or. use_base_activation) &
            total_param_count = total_param_count + num_outputs  ! bias
       if(use_base_activation) &
            total_param_count = total_param_count + &
            num_outputs * num_inputs + num_outputs  ! scale_base + scale_sp

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
       ! weights
       this%params(1)%val(:,1) = data_list(c:c + &
            num_outputs*num_inputs*n_basis - 1)
       c = c + num_outputs * num_inputs * n_basis
       ! bias
       if(use_bias .or. use_base_activation)then
          this%params(2)%val(:,1) = data_list(c:c + num_outputs - 1)
          c = c + num_outputs
       end if
       ! scale_base
       if(use_base_activation)then
          this%params(3)%val(:,1) = data_list(c:c + &
               num_outputs*num_inputs - 1)
          c = c + num_outputs * num_inputs
       end if
       ! scale_sp
       if(use_base_activation)then
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
    if(trim(adjustl(buffer)).ne."END "// &
         to_upper(trim(this%name)))then
       write(err_msg,'("END ",A," not where expected")') &
            to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_bspline_kan
!###############################################################################


!###############################################################################
  function read_kan_layer(unit, verbose) result(layer)
    !! Read B-spline KAN layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the B-spline KAN layer

    ! Local variables
    integer :: verbose_

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=kan_layer_type( &
         num_outputs=0, n_basis=1, spline_degree=3))
    call layer%read(unit, verbose=verbose_)

  end function read_kan_layer
!###############################################################################


!###############################################################################
  subroutine forward_bspline_kan(this, input)
    !! Forward propagation for B-spline KAN layer
    !!
    !! When use_base_activation is false (default):
    !!   y_j = sum_{i,k} W_{j,i,k} * B_k(x_i) + b_j
    !!
    !! When use_base_activation is true (PyKAN-style):
    !!   y_j = scale_sp_j * sum_{i,k} W_{j,i,k} * B_k(x_i)
    !!       + sum_i scale_base_{j,i} * silu(x_i) + b_j
    implicit none

    ! Arguments
    class(kan_layer_type), intent(inout) :: this
    !! Instance of the B-spline KAN layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input values [d, batch]

    ! Local variables
    type(array_type), pointer :: ptr => null()
    type(array_type), pointer :: spline_ptr => null()
    type(array_type), pointer :: silu_ptr => null()
    type(array_type), pointer :: base_ptr => null()
    type(array_type), pointer :: scaled_spline_ptr => null()
    integer :: d, K, dK, n_samples, i_dim, i_basis, s
    real(real32) :: x_val
    real(real32), allocatable :: x_expanded(:,:), bvals(:,:)


    d = this%num_inputs
    K = this%n_basis
    dK = d * K

    ! Get batch size from input
    n_samples = size(input(1,1)%val, 2)


    !---------------------------------------------------------------------------
    ! Expand input: replicate each x_i K times → [d*K, batch]
    !---------------------------------------------------------------------------
    allocate(x_expanded(dK, n_samples))
    do s = 1, n_samples
       do i_dim = 1, d
          do i_basis = 1, K
             x_expanded((i_dim - 1) * K + i_basis, s) = &
                  input(1,1)%val(i_dim, s)
          end do
       end do
    end do


    !---------------------------------------------------------------------------
    ! Evaluate B-spline basis functions: [d*K, batch]
    !---------------------------------------------------------------------------
    allocate(bvals(dK, n_samples))
    call this%evaluate_bspline_basis(x_expanded, n_samples, bvals)
    deallocate(x_expanded)


    !---------------------------------------------------------------------------
    ! Store basis values in array_type for matmul with weights
    ! The basis matrix is sample-dependent and recomputed every forward pass
    !---------------------------------------------------------------------------
    if(this%basis_matrix%allocated) call this%basis_matrix%deallocate()
    call this%basis_matrix%allocate([dK, n_samples])
    this%basis_matrix%val(:,:) = bvals
    this%basis_matrix%is_sample_dependent = .true.
    this%basis_matrix%is_temporary = .false.
    this%basis_matrix%fix_pointer = .true.
    call this%basis_matrix%set_requires_grad(.false.)
    deallocate(bvals)


    !---------------------------------------------------------------------------
    ! Compute output
    !---------------------------------------------------------------------------
    if(this%use_base_activation)then

       ! Spline term: W * basis → [m, batch]
       spline_ptr => matmul(this%params(1), this%basis_matrix)

       ! Scaled spline: spline [m,batch] * scale_sp [m,1] → [m,batch]
       scaled_spline_ptr => spline_ptr * this%params(4)

       ! Base term: silu(input) → [d, batch], then scale_base[m,d] * silu → [m,batch]
       silu_ptr => swish(input(1,1), 1.0_real32)
       base_ptr => matmul(this%params(3), silu_ptr)

       ! Combine: scaled_spline + base + bias
       if(this%use_bias)then
          ptr => scaled_spline_ptr + base_ptr + this%params(2)
       else
          ptr => scaled_spline_ptr + base_ptr
       end if

    else

       ! Original behaviour: W * basis [+ bias]  [m, batch]
       if(this%use_bias)then
          ptr => matmul(this%params(1), this%basis_matrix) + this%params(2)
       else
          ptr => matmul(this%params(1), this%basis_matrix)
       end if

    end if

    ! Store output
    !---------------------------------------------------------------------------
    call this%output(1,1)%zero_grad()
    call this%output(1,1)%assign_and_deallocate_source(ptr)
    this%output(1,1)%is_temporary = .false.

  end subroutine forward_bspline_kan
!###############################################################################

end module athena__kan_layer
