module athena__misc_types
  !! Module containing custom derived types and interfaces for ATHENA
  !!
  !! This module contains interfaces and derived types for
  !! activation functions, initialisers, arrays, and facets.
  !! The activation and initialiser types are abstract types that are used
  !! to define the activation functions and initialisers for the
  !! weights and biases in the neural network. The array type is an
  !! abstract type that is used to define the operations that can be performed
  !! on the arrays used in the neural network. The facets type is used to store
  !! the faces, edges, and corners of the arrays for padding.
  use athena__constants, only: real32
  use athena__io_utils, only: stop_program
  implicit none


  private

  public :: activation_type
  public :: initialiser_type
  public :: array_type
  public :: array_container_type, array_ptr_type
!   public :: array1d_type, array2d_type, array3d_type, array4d_type, array5d_type
  public :: facets_type

  public :: operator(+), operator(-), operator(*), operator(/), &
       operator(**), operator(.mmul.), operator(.concat.), operator(.ltrim.), &
       operator(.rtrim.), operator(.index.), operator(.outer.)
  public :: operator(.lt.), operator(.gt.)

  public :: sign, merge, maxval, max, sum, mean, spread
  public :: pack, unpack
  public :: sin, cos, tan, exp, log, sqrt, tanh, sigmoid, transpose, add, concat


!-------------------------------------------------------------------------------
! Activation (transfer) function base type
!-------------------------------------------------------------------------------
  type, abstract :: activation_type
     !! Abstract type for activation functions
     character(10) :: name
     !! Name of the activation function
     real(real32) :: scale
     !! Scale of the activation function
     real(real32) :: threshold
     !! Threshold of the activation function
   contains
     procedure (activation_function_array), deferred, pass(this) :: activate_array
     procedure (activation_function_1d), deferred, pass(this) :: activate_1d
     !! Abstract procedure for 1D activation function
     procedure (derivative_function_1d), deferred, pass(this) :: &
          differentiate_1d
     !! Abstract procedure for 1D derivative of activation function
     procedure (activation_function_2d), deferred, pass(this) :: activate_2d
     !! Abstract procedure for 2D activation function
     procedure (derivative_function_2d), deferred, pass(this) :: &
          differentiate_2d
     !! Abstract procedure for 2D derivative of activation function
     procedure (activation_function_3d), deferred, pass(this) :: activate_3d
     !! Abstract procedure for 3D activation function
     procedure (derivative_function_3d), deferred, pass(this) :: &
          differentiate_3d
     !! Abstract procedure for 3D derivative of activation function
     procedure (activation_function_4d), deferred, pass(this) :: activate_4d
     !! Abstract procedure for 4D activation function
     procedure (derivative_function_4d), deferred, pass(this) :: &
          differentiate_4d
     !! Abstract procedure for 4D derivative of activation function
     procedure (activation_function_5d), deferred, pass(this) :: activate_5d
     !! Abstract procedure for 5D activation function
     procedure (derivative_function_5d), deferred, pass(this) :: &
          differentiate_5d
     !! Abstract procedure for 5D derivative of activation function
     generic :: activate => activate_array, activate_1d, activate_2d, &
          activate_3d , activate_4d, activate_5d
     !! Generic for activation function
     generic :: differentiate => differentiate_1d, differentiate_2d, &
          differentiate_3d, differentiate_4d, differentiate_5d
     !! Generic for derivative of activation function
  end type activation_type

  ! Interface for activation function
  !-----------------------------------------------------------------------------
  abstract interface

     function activation_function_1d(this, val) result(output)
       !! Interface for activation function
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:), intent(in) :: val
       real(real32), dimension(size(val,1)) :: output
     end function activation_function_1d

     function activation_function_2d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2)) :: output
     end function activation_function_2d

     function activation_function_3d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function activation_function_3d

     function activation_function_4d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function activation_function_4d

     function activation_function_5d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3), &
            size(val,4),size(val,5)) :: output
     end function activation_function_5d
  end interface

  ! Interface for derivative function
  !-----------------------------------------------------------------------------
  abstract interface
     function derivative_function_1d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:), intent(in) :: val
       real(real32), dimension(size(val,1)) :: output
     end function derivative_function_1d

     function derivative_function_2d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2)) :: output
     end function derivative_function_2d

     function derivative_function_3d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function derivative_function_3d

     function derivative_function_4d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function derivative_function_4d

     function derivative_function_5d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3), &
            size(val,4),size(val,5)) :: output
     end function derivative_function_5d
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Weights and biases initialiser base type
!-------------------------------------------------------------------------------
  type, abstract :: initialiser_type
     !! Abstract type for initialising weights and biases
     character(len=20) :: name
     !! Name of the initialiser
     real(real32) :: scale = 1._real32, mean = 1._real32, std = 0.01_real32
     !! Scale, mean, and standard deviation of the initialiser
   contains
     procedure (initialiser_subroutine), deferred, pass(this) :: initialise
     !! Abstract procedure for initialising weights and biases
  end type initialiser_type

  ! Interface for initialiser function
  !-----------------------------------------------------------------------------
  abstract interface
     !! Interface for initialiser function
     subroutine initialiser_subroutine(this, input, fan_in, fan_out, spacing)
       !! Interface for initialiser function
       import initialiser_type, real32
       class(initialiser_type), intent(inout) :: this
       !! Instance of the initialiser type
       real(real32), dimension(..), intent(out) :: input
       !! Array to initialise
       integer, optional, intent(in) :: fan_in, fan_out
       !! Number of input and output units
       integer, dimension(:), optional, intent(in) :: spacing
       !! Spacing of the array
     end subroutine initialiser_subroutine
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Base and extended array types
!-------------------------------------------------------------------------------
  type :: array_type
     !! Abstract type for array operations
     integer :: id = -1
     integer :: rank
     !! Rank of the array
     integer, dimension(:), allocatable :: shape
     !! Shape of the array
     integer :: size
     !! Size of the array
     logical :: is_sample_dependent = .true.
     !! Boolean whether array is sample-dependent
     logical :: is_scalar = .false.
     !! Boolean whether array is contains a scalar value
     logical :: is_forward = .false.
     !! Boolean whether operation is forward-mode
     logical :: allocated = .false.
     !! Logical flag for array allocation
     real(real32), dimension(:,:), allocatable :: val
     !! Array values in rank 2 (sample, batch)
     integer, dimension(:), allocatable :: indices ! store_1
     !! Indices for gradient accumulation
     integer, dimension(:,:), allocatable :: adj_ja ! store_2
     !! Sparse adjacency matrix for graph structure
     logical, dimension(:,:), allocatable :: mask
     !! Mask for operation
     logical :: requires_grad = .false.
     !! Flag indicating if gradients should be computed
     logical :: is_leaf = .true.
     !! Flag indicating if this is a leaf node (parameter)
     type(array_type), pointer :: grad => null()
     !! Gradient array (same type as value)
     type(array_type), pointer :: left_operand => null()
     !! Left operand for backward pass
     type(array_type), pointer :: right_operand => null()
     !! Right operand for backward pass
     character(len=32) :: operation = 'none'
     logical :: owns_gradient = .true.
     !! Flag indicating if this array owns its gradient memory
     logical :: fix_pointer = .false.

     real(real32), dimension(:), allocatable :: direction

     procedure(get_partial), pass(this), pointer :: get_partial_left => null()
     procedure(get_partial), pass(this), pointer :: get_partial_right => null()

   contains
     procedure, pass(this) :: allocate => allocate_array
     !! Abstract procedure for allocating array
     procedure, pass(this) :: deallocate => deallocate_array
     !! Abstract procedure for deallocating array
     !   procedure, pass(this) :: set_ptr => set_ptr_array
     !! Abstract procedure for setting pointers
     procedure, pass(this) :: flatten => flatten_array
     !! Procedure for flattening array
     procedure, pass(this) :: get => get_array
     !! Procedure for getting array
     procedure, pass(this) :: set => set_array
     !! Procedure for setting array
     !  generic, public :: operator(+) => add
     !  !! Generic for adding arrays
     !  generic, public :: operator(*) => multiply
     !  !! Generic for multiplying arrays
     procedure :: assign => assign_array
     generic, public :: assignment(=) => assign

     procedure, pass(this) :: set_direction
     procedure, pass(this) :: grad_reverse
     !! Reverse-mode: accumulate gradients wrt all inputs
     procedure, pass(this) :: grad_forward
     !! Forward-mode: return derivative wrt variable pointer

     !! Backward pass for gradient computation
     procedure, pass(this) :: zero_grad
     procedure, pass(this) :: zero_all_grads
     !! Zero the gradients
     procedure, pass(this) :: reset_graph
     procedure, pass(this) :: duplicate_graph
     !   procedure, pass(this) :: duplicate_graph_ptrs
     procedure, pass(this) :: get_ptr_from_id
     procedure, pass(this) :: detach
     !! Detach from computation graph
     procedure, private, pass(this) :: reverse_mode
     !   procedure, private, pass(this) :: forward_over_reverse
     !! Deferred procedure for operation-specific backward pass
     procedure, pass(this) :: set_requires_grad
     !! Set requires_grad flag
     procedure :: create_result => create_result_array
     !! Helper to safely create result arrays

     procedure, pass(this) :: print_graph

     ! final :: finalise_array
     ! !! Finaliser for array type
  end type array_type

  ! Interface for allocate, deallocate, and flattening array
  !-----------------------------------------------------------------------------
  interface
     module subroutine allocate_array(this, array_shape, source)
       class(array_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array

     module recursive subroutine deallocate_array(this, keep_shape)
       class(array_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array

     !   module subroutine set_ptr_array(this)
     !     class(array_type), intent(inout), target :: this
     !   end subroutine set_ptr_array

     module recursive subroutine finalise_array(this)
       type(array_type), intent(inout) :: this
     end subroutine finalise_array

     module function create_result_array(this, array_shape) result(result_ptr)
       class(array_type), intent(in) :: this
       integer, dimension(:), intent(in), optional :: array_shape
       type(array_type), pointer :: result_ptr
     end function create_result_array


     module subroutine print_graph(this)
       class(array_type), intent(in) :: this
     end subroutine print_graph


     module function grad_forward(this, variable) result(output)
       class(array_type), intent(inout) :: this
       type(array_type), intent(in) :: variable
       type(array_type), pointer :: output
     end function grad_forward

     module subroutine grad_reverse(this, record_graph, reset_graph)
       class(array_type), intent(inout) :: this
       logical, intent(in), optional :: record_graph
       logical, intent(in), optional :: reset_graph
     end subroutine grad_reverse

     !   module recursive function forward_over_reverse(this, variable, itmp) &
     !        result(output)
     !     class(array_type), intent(inout) :: this
     !     type(array_type), intent(inout) :: variable
     !     type(array_type) :: output
     !     integer :: itmp
     !   end function forward_over_reverse

     module recursive subroutine reverse_mode(this, upstream_grad, record_graph)
       class(array_type), intent(inout) :: this
       type(array_type), intent(in) :: upstream_grad
       logical, intent(in) :: record_graph
     end subroutine reverse_mode

     module subroutine set_requires_grad(this, requires_grad)
       class(array_type), intent(inout) :: this
       logical, intent(in) :: requires_grad
     end subroutine set_requires_grad

     module recursive subroutine reset_graph(this)
       !! Reset the gradients of this array
       class(array_type), intent(inout) :: this
     end subroutine reset_graph

     module subroutine zero_grad(this)
       !! Zero the gradients of this array
       class(array_type), intent(inout) :: this
     end subroutine zero_grad

     module recursive subroutine zero_all_grads(this)
       !! Zero the gradients of this array
       class(array_type), intent(inout) :: this
     end subroutine zero_all_grads

     module subroutine duplicate_graph(this)
       class(array_type), intent(inout) :: this
     end subroutine duplicate_graph

     !   module recursive subroutine duplicate_graph_ptrs(this, pointer_map)
     !     use iso_c_binding
     !     class(array_type), intent(inout) :: this
     !     type(c_ptr), dimension(:,:), allocatable, intent(inout) :: pointer_map
     !   end subroutine duplicate_graph_ptrs

     module recursive function get_ptr_from_id(this, id) result(ptr)
       use iso_c_binding
       class(array_type), intent(in), target :: this
       integer, intent(in) :: id
       type(array_type), pointer :: ptr
     end function get_ptr_from_id

     module subroutine detach(this)
       !! Detach this array from the computation graph
       class(array_type), intent(inout) :: this
     end subroutine detach
  end interface

  interface
     module function get_partial(this, upstream_grad) result(output)
       class(array_type), intent(inout) :: this
       type(array_type), intent(in) :: upstream_grad
       type(array_type) :: output
     end function get_partial
  end interface


  interface
     pure module function flatten_array(this) result(output)
       class(array_type), intent(in) :: this
       real(real32), dimension(this%size) :: output
     end function flatten_array

     pure module subroutine get_array(this, output)
       class(array_type), intent(in) :: this
       real(real32), dimension(..), allocatable, intent(out) :: output
     end subroutine get_array

     pure module subroutine set_array(this, input)
       class(array_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
     end subroutine set_array

     module subroutine assign_array(this, input)
       class(array_type), intent(out), target :: this
       type(array_type), intent(in) :: input
     end subroutine assign_array
  end interface

  !-----------------------------------------------------------------------------
  ! Operator interfaces
  !-----------------------------------------------------------------------------
  interface operator(+)
     module procedure add_arrays
     module procedure add_real2d
     module procedure real2d_add
     module procedure add_real1d
     module procedure real1d_add
     module procedure add_scalar
     module procedure scalar_add
  end interface

  interface operator(-)
     module procedure subtract_arrays
     module procedure subtract_real1d
     module procedure subtract_scalar
     module procedure scalar_subtract
     module procedure negate_array
  end interface

  interface operator(*)
     module procedure multiply_arrays
     module procedure multiply_scalar
     module procedure scalar_multiply
     module procedure multiply_logical
  end interface

  interface operator(/)
     module procedure divide_arrays
     module procedure divide_scalar
     module procedure scalar_divide
     module procedure divide_real1d
  end interface

  interface operator(**)
     module procedure power_arrays
     module procedure power_real_scalar
     module procedure power_int_scalar
     module procedure scalar_power
     module procedure int_scalar_power
  end interface

  interface operator(.mmul.)
     module procedure matmul_arrays
     module procedure real2d_matmul
     module procedure matmul_real2d
  end interface

  interface operator(.outer.)
     module procedure outer_product_arrays
  end interface

  interface operator(.concat.)
     module procedure concat_arrays
  end interface

  interface operator(.ltrim.)
     module procedure ltrim_array
  end interface

  interface operator(.rtrim.)
     module procedure rtrim_array
  end interface

  interface operator(.index.)
     module procedure index_array
  end interface

  interface operator(.lt.)
     module procedure lt_scalar
  end interface

  interface operator(.gt.)
     module procedure gt_scalar
  end interface

  interface sign
     module procedure sign_array
  end interface

  interface sum
     module procedure sum_array
     module procedure sum_array_output_array
  end interface

  interface mean
     module procedure mean_array
  end interface

  interface maxval
     module procedure maxval_array
  end interface

  interface max
     module procedure max_array
     module procedure max_scalar
  end interface

  interface merge
     module procedure merge_scalar
     module procedure merge_real2d
  end interface

  interface spread
     module procedure spread_array
  end interface

  interface unspread
     module procedure unspread_array
  end interface

  interface reverse_index
     module procedure reverse_index_array
  end interface

  interface add
     module procedure add_array_ptr
  end interface

  interface concat
     module procedure concat_array_ptr
  end interface

  interface pack
     module procedure pack_array
  end interface

  interface unpack
     module procedure unpack_array
  end interface

  !-----------------------------------------------------------------------------
  ! Mathematical function interfaces
  !-----------------------------------------------------------------------------
  interface sin
     module procedure sin_array
  end interface

  interface cos
     module procedure cos_array
  end interface

  interface tan
     module procedure tan_array
  end interface

  interface exp
     module procedure exp_array
  end interface

  interface log
     module procedure log_array
  end interface

  interface sqrt
     module procedure sqrt_array
  end interface

  interface tanh
     module procedure tanh_array
  end interface

  interface sigmoid
     module procedure sigmoid_array
  end interface

  interface transpose
     module procedure transpose_array
  end interface



  ! Array container types
  !-----------------------------------------------------------------------------
  type :: array_container_type
     class(array_type), allocatable :: array
  end type array_container_type

  type :: array_ptr_type
     type(array_type), pointer :: array(:,:)
  end type array_ptr_type


!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Facet type (for storing faces, edges, and corners for padding)
!-------------------------------------------------------------------------------
  type :: facets_type
     !! Type for storing faces, edges, and corners for padding
     integer :: num
     !! Number of facets
     integer :: rank
     !! Number of dimensions of the shape
     integer :: nfixed_dims
     !! Number of fixed dimensions
     character(6) :: type
     !! Type of facet, i.e. face, edge, corner
     integer, dimension(:), allocatable :: dim
     !! Dimension the facet is in, i.e.
     integer, dimension(:,:,:), allocatable :: orig_bound
     !! Original bounds of the facet (2, nfixed_dims, num)
     integer, dimension(:,:,:), allocatable :: dest_bound
     !! Destination bounds of the facet (2, nfixed_dims, num)
   contains
     procedure, pass(this) :: setup_bounds
     !! Procedure for setting up bounds
  end type facets_type

  interface
     !! Interface for setting up bounds
     module subroutine setup_bounds(this, length, pad, imethod)
       !! Procedure for setting up bounds
       class(facets_type), intent(inout) :: this
       !! Instance of the facets type
       integer, dimension(this%rank), intent(in) :: length, pad
       !! Length of the shape and padding
       integer, intent(in) :: imethod
       !! Method for setting up bounds
     end subroutine setup_bounds
  end interface
!-------------------------------------------------------------------------------

  abstract interface
     function activation_function_array(this, val) result(output)
       !! Interface for activation function
       import activation_type, real32, array_type
       class(activation_type), intent(in) :: this
       type(array_type), intent(in) :: val
       type(array_type), pointer :: output
     end function activation_function_array
  end interface
contains


  !-----------------------------------------------------------------------------
  ! Array pointer list operations
  !-----------------------------------------------------------------------------
  function add_array_ptr(a, idx1, idx2) result(c)
    !! Add two autodiff arrays
    type(array_ptr_type), dimension(:), intent(in) :: a
    integer, intent(in) :: idx1, idx2
    type(array_type), pointer :: c

    integer :: i

    c => a(1)%array(idx1, idx2) + a(2)%array(idx1, idx2)
    do i = 2, size(a)
       c => c + a(i)%array(idx1, idx2)
    end do
  end function add_array_ptr

  function concat_array_ptr(a, idx1, idx2, dim) result(c)
    !! Concatenate two autodiff arrays along a specified dimension
    type(array_ptr_type), dimension(:), intent(in) :: a
    integer, intent(in) :: idx1, idx2, dim
    type(array_type), pointer :: c

    integer :: i

    allocate(c)
    c => a(1)%array(idx1, idx2) .concat. a(2)%array(idx1, idx2)
    do i = 3, size(a)
       c => c .concat. a(i)%array(idx1, idx2)
    end do
  end function concat_array_ptr

  !-----------------------------------------------------------------------------
  ! Sign addition
  !-----------------------------------------------------------------------------
  function sign_array(scalar, array) result(c)
    !! Add a scalar sign to an autodiff array
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: array
    real(real32), dimension(:,:), allocatable :: c
    ! type(array_type), pointer :: c

    allocate(c(size(array%val,1), size(array%val,2)))
    c = sign(scalar, array%val)
    ! allocate(c)
    ! call c%allocate(array_shape=array%shape)
    ! c%val = sign(scalar, array%val)

    ! if(array%requires_grad) then
    !    c%requires_grad = .true.
    !    c%is_leaf = .false.
    !    c%operation = 'sign'
    !    c%left_operand => array
    ! end if
  end function sign_array


  !-----------------------------------------------------------------------------
  ! Partial derivative operations
  !-----------------------------------------------------------------------------
  subroutine set_direction(this, direction)
    !! Set the direction for the array (for higher-order derivatives)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(:), intent(in) :: direction

    if(allocated(this%direction)) deallocate(this%direction)
    if(size(this%val,1).ne.size(direction)) then
       call stop_program('Direction size does not match array size in set_direction')
    end if
    this%direction = direction

  end subroutine set_direction

  function get_partial_add(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad
  end function get_partial_add

  function get_partial_negate(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = -upstream_grad
  end function get_partial_negate

  function get_partial_exp(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad * this
  end function get_partial_exp

  function get_partial_multiply_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    if(this%right_operand%is_scalar)then
       output = upstream_grad * this%right_operand%val(1,1)
    else
       output = upstream_grad * this%right_operand
    end if
  end function get_partial_multiply_left

  function get_partial_multiply_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    if(this%left_operand%is_scalar)then
       output = upstream_grad * this%left_operand%val(1,1)
    else
       output = upstream_grad * this%left_operand
    end if
  end function get_partial_multiply_right

  function get_partial_divide_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    if(this%right_operand%is_scalar)then
       output = upstream_grad / this%right_operand%val(1,1)
    else
       output = upstream_grad / this%right_operand
    end if
  end function get_partial_divide_left

  function get_partial_divide_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: grad, div

    allocate(grad)
    allocate(div)
    if(this%left_operand%is_scalar)then
       grad = -upstream_grad * this%left_operand%val(1,1)
    else
       grad = -upstream_grad * this%left_operand
    end if
    div = this%right_operand * this%right_operand
    output = grad / div
  end function get_partial_divide_right

  function get_partial_power_base(this, upstream_grad) result(output)
    !! Get the partial gradient with respect to the base of the power operation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    if(all(abs(this%right_operand%val - 1._real32).lt.1.E-6_real32)) then
       output = upstream_grad
       return
    elseif(all(abs(this%right_operand%val - 2._real32).lt.1.E-6_real32)) then
       output = upstream_grad * 2._real32 * this%left_operand
       return
    end if
    if(this%right_operand%is_scalar)then
       output = upstream_grad * this%right_operand%val(1,1) * &
            this%left_operand ** ( this%right_operand%val(1,1) - 1.0_real32 )
    else
       output = upstream_grad * this%right_operand * &
            this%left_operand ** ( this%right_operand - 1.0_real32 )
    end if
  end function get_partial_power_base

  function get_partial_power_exponent(this, upstream_grad) result(output)
    !! Get the partial gradient with respect to the exponent of the power operation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    if(this%left_operand%is_scalar)then
       output = upstream_grad * log(this%left_operand%val(1,1)) * this
    else
       output = upstream_grad * log(this%left_operand) * this
    end if
  end function get_partial_power_exponent

  function get_partial_matmul_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand of matmul
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    if(size(this%right_operand%shape).eq.2)then
       if(this%is_forward)then
          output = upstream_grad .mmul. this%right_operand
       else
          output = upstream_grad .mmul. transpose(this%right_operand)
       end if
    elseif(size(upstream_grad%shape).eq.2)then
       if(this%is_forward)then
          output = upstream_grad .mmul. this%right_operand
       else
          output = transpose(upstream_grad) .mmul. this%right_operand
       end if
    else
       output = upstream_grad .outer. this%right_operand
    end if

  end function get_partial_matmul_left

  function get_partial_matmul_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand of matmul
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    if(size(this%left_operand%shape).eq.2)then
       if(this%is_forward)then
          output = this%left_operand .mmul. upstream_grad
       else
          output = transpose(this%left_operand) .mmul. upstream_grad
       end if
    elseif(size(upstream_grad%shape).eq.2)then
       if(this%is_forward)then
          output = this%left_operand .mmul. upstream_grad
       else
          output = this%left_operand .mmul. transpose(upstream_grad)
       end if
    else
       output = this%left_operand .outer. upstream_grad
    end if

  end function get_partial_matmul_right


  function get_partial_transpose_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = transpose(upstream_grad)

  end function get_partial_transpose_left

!   function get_partial_transpose_right(this, upstream_grad) result(output)
!     class(array_type), intent(inout) :: this
!     type(array_type), intent(in) :: upstream_grad
!     type(array_type) :: output

!     output = transpose(this%left_operand)

!   end function get_partial_transpose_right

  function get_partial_log(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad / this%left_operand

  end function get_partial_log

  function get_partial_sqrt(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad / ( 2._real32 * this )

  end function get_partial_sqrt

  function get_partial_sin(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad * cos( this%left_operand )

  end function get_partial_sin

  function get_partial_cos(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = -upstream_grad * sin( this%left_operand )

  end function get_partial_cos

  function get_partial_tan(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad / ( cos( this%left_operand ) ** 2._real32 )

  end function get_partial_tan

  function get_partial_tanh(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! as: this = tanh(this%left_operand)
    output = upstream_grad * tanh_reverse_array( this )
    ! output = upstream_grad * (1._real32 - this ** 2._real32)

  end function get_partial_tanh

  function get_partial_tanh_reverse(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: left

    allocate(left)
    left = -2._real32 * this%left_operand
    output = left * this

  end function get_partial_tanh_reverse

  function get_partial_index(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = reverse_index( &
         upstream_grad, indices=this%indices, from=.false., &
         new_index_size=size(this%left_operand%val, 2) &
    )

  end function get_partial_index

  function get_partial_merge(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = merge(upstream_grad, 0._real32, this%mask)

  end function get_partial_merge

  function get_partial_concat_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad .ltrim. this%left_operand%shape(1)

  end function get_partial_concat_left

  function get_partial_concat_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad .rtrim. this%right_operand%shape(1)

  end function get_partial_concat_right

  function get_partial_max_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad * (this%val .eq. this%left_operand%val)

  end function get_partial_max_left

  function get_partial_max_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad * (this%val .eq. this%right_operand%val)

  end function get_partial_max_right

  function get_partial_sum_reverse(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = spread( &
         upstream_grad, &
         dim=this%indices(1), &
         index=this%indices(2), &
         ncopies= size(this%left_operand%val, this%indices(1)) &
    )

  end function get_partial_sum_reverse

  function get_partial_sum_forward(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = sum( &
         upstream_grad, &
         dim = this%indices(1) &
    )

  end function get_partial_sum_forward

  function get_partial_mean(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    real(real32) :: rtmp1

    ! Calculate the number of elements that were averaged
    rtmp1 = real(size(this%left_operand%val, this%indices(1)), real32)

    if(this%is_forward)then
       output = sum( upstream_grad, dim = this%indices(1) ) / rtmp1
    else
       output = spread( &
            upstream_grad / rtmp1, &
            dim=this%indices(1), &
            index=this%indices(2), &
            ncopies= size(this%left_operand%val, this%indices(1)) &
       )
    end if

  end function get_partial_mean

  function get_partial_pack(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = unpack(upstream_grad, this%indices, this%adj_ja(1,1), this%adj_ja(2,1))

  end function get_partial_pack

  function get_partial_unpack(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = pack(upstream_grad, this%indices, this%adj_ja(1,1))

  end function get_partial_unpack

  function get_partial_spread(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    integer :: i, s

    output = unspread( &
         upstream_grad, &
         this%indices(1), &
         this%adj_ja(1,1), &
         this%adj_ja(2,1) &
    )

  end function get_partial_spread

  function get_partial_unspread(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    integer :: i, s

    output = spread( &
         upstream_grad, &
         this%indices(1), &
         this%adj_ja(1,1), &
         this%adj_ja(2,1) &
    )

  end function get_partial_unspread

  !-----------------------------------------------------------------------------
  ! Addition operation
  !-----------------------------------------------------------------------------
  function add_arrays(a, b) result(c)
    !! Add two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s

    ! Safely create result array
    c => a%create_result()
    if(b%is_sample_dependent)then
       c%val = a%val + b%val
    else
       do s = 1, size(a%val, 2)
          c%val(:,s) = a%val(:,s) + b%val(:,1)
       end do
    end if

    c%get_partial_left => get_partial_add
    c%get_partial_right => get_partial_add
    ! Set up computation graph
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'add'
       c%left_operand => a
       c%right_operand => b
    end if
  end function add_arrays

  function add_real2d(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val + b

    c%get_partial_left => get_partial_add
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'add'
       c%left_operand => a
    end if
  end function add_real2d

  function real2d_add(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_real2d(b, a)
  end function real2d_add

  function add_real1d(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s

    c => a%create_result()
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) + b(:)
    end do

    c%get_partial_left => get_partial_add
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'add'
       c%left_operand => a
    end if
  end function add_real1d

  function real1d_add(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    real(real32), dimension(:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_real1d(b, a)
  end function real1d_add

  function add_scalar(a, b) result(c)
    !! Add a scalar to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val + b

    c%get_partial_left => get_partial_add
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'add'
       c%left_operand => a
    end if
  end function add_scalar

  function scalar_add(a, b) result(c)
    !! Add a scalar to an autodiff array
    implicit none
    real(real32), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_scalar(b, a)
  end function scalar_add

  !-----------------------------------------------------------------------------
  ! Subtraction operation
  !-----------------------------------------------------------------------------
  function subtract_arrays(a, b) result(c)
    !! Subtract two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val - b%val

    c%get_partial_left => get_partial_add
    c%get_partial_right => get_partial_negate
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'subtract'
       c%left_operand => a
       c%right_operand => b
    end if
  end function subtract_arrays

  function subtract_real1d(a, b) result(c)
    !! Subtract a real array from an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s

    c => a%create_result()
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) - b(s)
    end do

    c%get_partial_left => get_partial_add
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'subtract_scalar'
       c%left_operand => a
    end if
  end function subtract_real1d

  function subtract_scalar(a, b) result(c)
    !! Subtract a scalar from an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val - b

    c%get_partial_left => get_partial_add
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'subtract_scalar'
       c%left_operand => a
    end if
  end function subtract_scalar

  function scalar_subtract(a, b) result(c)
    !! Subtract an autodiff array from a scalar
    implicit none
    real(real32), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => negate_array(b)
    c%val = a + c%val

    c%get_partial_left => get_partial_negate
    if(b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = b%is_forward
       c%is_leaf = .false.
       c%operation = 'subtract_scalar'
       c%left_operand => b
    end if
  end function scalar_subtract

  function negate_array(a) result(c)
    !! Negate an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = -a%val

    c%get_partial_left => get_partial_negate
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'negate'
       c%left_operand => a
    end if
  end function negate_array

  !-----------------------------------------------------------------------------
  ! Multiplication operations
  !-----------------------------------------------------------------------------
  function multiply_arrays(a, b) result(c)
    !! Multiply two autodiff arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s

    if(b%is_scalar)then
       c => a%create_result()
       c%val = a%val * b%val(1,1)
    elseif(.not.b%is_sample_dependent)then
       do s = 1, size(a%val,2)
          c%val(:,s) = a%val(:,s) * b%val(:,1)
       end do
    elseif(size(a%val,1).ne.size(b%val,1).and.size(a%val,2).eq.size(b%val,2))then
       if(size(a%val,1) .eq. 1)then
          c => b%create_result()
          do concurrent(s=1:size(a%val,2))
             c%val(:,s) = a%val(1,s) * b%val(:,s)
          end do
       elseif(size(b%val,1) .eq. 1)then
          c => a%create_result()
          do concurrent(s=1:size(a%val,2))
             c%val(:,s) = a%val(:,s) * b%val(1,s)
          end do
       end if
    else
       c => a%create_result()
       c%val = a%val * b%val
    end if

    c%get_partial_left => get_partial_multiply_left
    c%get_partial_right => get_partial_multiply_right
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'multiply'
       c%left_operand => a
       c%right_operand => b
    end if
  end function multiply_arrays

  function multiply_scalar(a, scalar) result(c)
    !! Multiply autodiff array by scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = a%val * scalar

    c%get_partial_left => get_partial_multiply_left
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'multiply_scalar'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array
  end function multiply_scalar

  function scalar_multiply(scalar, a) result(c)
    !! Multiply scalar by autodiff array
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => multiply_scalar(a, scalar)
  end function scalar_multiply

  function multiply_logical(a, b) result(c)
    !! Multiply two logical arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a
    logical, dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s, i

    c => a%create_result()
    do concurrent(s=1:size(a%val,2), i=1:size(a%val,1))
       if(b(i,s)) then
          c%val(i,s) = a%val(i,s)
       else
          c%val(i,s) = 0.0_real32
       end if
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'multiply_logical'
       c%left_operand => a
    end if

  end function multiply_logical

  function matmul_arrays(a, b) result(c)
    !! Matrix multiplication of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s
    real(real32), pointer :: temp(:,:)

    if(.not.a%is_sample_dependent)then
       if(size(b%shape).ne.1)then
          call stop_program( &
               'Matrix multiplication not implemented for these shapes yet' )
       end if
       c => a%create_result(array_shape=[a%shape(1), size(b%val,2)])
       temp(1:a%shape(1), 1:a%shape(2)) => a%val
       do concurrent(s=1:size(b%val,2))
          c%val(:,s) = matmul(temp, b%val(:,s))
       end do
    elseif(.not.b%is_sample_dependent)then
       if(size(a%shape).ne.1)then
          call stop_program( &
               'Matrix multiplication not implemented for these shapes yet' )
       end if
       c => b%create_result(array_shape=[b%shape(2), size(a%val,2)])
       temp(1:b%shape(1), 1:b%shape(2)) => b%val
       do concurrent(s=1:size(a%val,2))
          c%val(:,s) = matmul(a%val(:,s), temp)
       end do
    else
       write(0,*) "NOT SURE WHAT TO DO YET"
       stop 0
    end if

    c%is_sample_dependent = .true.
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'matmul'
       c%left_operand => a
       c%right_operand => b
    end if
  end function matmul_arrays

  function matmul_real2d(a, b) result(c)
    !! Matrix multiplication of a real array and an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    integer :: s, i

    c => a%create_result(array_shape = [size(b,2), size(a%val,2)])
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = matmul(a%val(:,s), b)
    end do

    c%is_sample_dependent = a%is_sample_dependent
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'matmul_scalar'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%shape = shape(b)
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[size(b,1), size(b,2), 1])
    do i = 1, size(b,2)
       b_array%val((i-1)*size(b,1)+1:i*size(b,1), 1) = b(:,i)
    end do
    c%right_operand => b_array
  end function matmul_real2d

  function real2d_matmul(a, b) result(c)
    !! Matrix multiplication of two autodiff arrays
    implicit none
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: a_array

    integer :: s, i

    c => b%create_result(array_shape = [size(a,1), size(b%val,2)])
    do concurrent(s=1:size(b%val,2))
       c%val(:,s) = matmul(a, b%val(:,s))
    end do

    c%is_sample_dependent = b%is_sample_dependent
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    if(b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = b%is_forward
       c%is_leaf = .false.
       c%operation = 'matmul_scalar'
       c%right_operand => b
    end if
    allocate(a_array)
    a_array%is_sample_dependent = .false.
    a_array%shape = shape(a)
    a_array%requires_grad = .false.
    a_array%is_leaf = .false.
    call a_array%allocate(array_shape=[size(a,1), size(a,2), 1])
    do i = 1, size(a,2)
       a_array%val((i-1)*size(a,1)+1:i*size(a,1), 1) = a(:,i)
    end do
    c%left_operand => a_array
  end function real2d_matmul

  function outer_product_arrays(a, b) result(c)
    !! Outer product of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, j, s

    c => a%create_result(array_shape = [size(a%val,1), size(b%val,1), size(a%val,2)])
    ! outer product 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:size(a%val,1), j=1:size(b%val,1))
          c%val(i + (j-1)*size(a%val,1),s) = a%val(i,s) * b%val(j,s)
       end do
    end do

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'outer_product'
       c%left_operand => a
       c%right_operand => b
    end if
  end function outer_product_arrays

  function concat_arrays(a, b) result(c)
    !! Concatenate two autodiff arrays along the first dimension
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, j, s

    c => a%create_result(array_shape = [size(a%val,1) + size(b%val,1), size(a%val,2)])
    ! concatenate 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:1, j=1:size(a%val,1))
          c%val( i, s) = a%val( i, s)
       end do
       do concurrent(i=1:1, j=1:size(b%val,1))
          c%val( size(a%val,1) + i, s) = b%val( i, s)
       end do
    end do

    c%get_partial_left => get_partial_concat_left
    c%get_partial_right => get_partial_concat_right
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'concat'
       c%left_operand => a
       c%right_operand => b
    end if
  end function concat_arrays

  function ltrim_array(a, b) result(c)
    !! Left trim an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    type(array_type), pointer :: c

    integer :: i, j, s

    c => a%create_result(array_shape = [b, size(a%val,2)])
    ! left trim 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       c%val( :, s) = a%val( 1:b, s)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'ltrim'
       c%left_operand => a
    end if
  end function ltrim_array

  function rtrim_array(a, b) result(c)
    !! Right trim an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    type(array_type), pointer :: c

    integer :: i, j, s

    c => a%create_result(array_shape = [b, size(a%val,2)])
    ! right trim 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       c%val( :, s) = a%val( size(a%val,1)-b+1:size(a%val,1), s)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'rtrim'
       c%left_operand => a
    end if
  end function rtrim_array

  function pack_array(a, indices, dim) result(c)
    !! Pack an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    integer, intent(in) :: dim
    type(array_type), pointer :: c

    integer :: i, s

    if(dim.eq.1)then
       c => a%create_result(array_shape=[size(indices), size(a%val,2)])
       do concurrent(s=1:size(a%val,2), i=1:size(indices))
          c%val(i, s) = a%val(indices(i), s)
       end do
    elseif(dim.eq.2)then
       c => a%create_result(array_shape=[size(a%val,1), size(indices)])
       do concurrent(s=1:size(indices), i=1:size(a%val,1))
          c%val(i, s) = a%val(i, indices(s))
       end do
    end if
    c%indices = indices
    allocate(c%adj_ja(2,1))
    c%adj_ja(:,1) = [ dim, size(a%val,dim) ]

    c%get_partial_left => get_partial_pack
    c%get_partial_right => get_partial_unpack
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'pack'
       c%left_operand => a
    end if
  end function pack_array

  function unpack_array(a, indices, dim, new_size) result(c)
    !! Unpack an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    integer, intent(in) :: new_size, dim
    type(array_type), pointer :: c

    integer :: i, s


    if(dim.eq.1)then
       c => a%create_result(array_shape = [ new_size, size(a%val,2) ])
       c%val = 0.0_real32
       do concurrent(i=1:size(indices,1), s=1:size(a%val,2))
          c%val(indices(i),s) = a%val(i,s)
       end do
    elseif(dim.eq.2)then
       c => a%create_result(array_shape = [ size(a%val,1), new_size ])
       c%val = 0.0_real32
       do concurrent(i=1:size(a%val,1), s=1:new_size)
          c%val(i,indices(s)) = a%val(i,s)
       end do
    end if
    c%indices = indices
    allocate(c%adj_ja(2,1))
    c%adj_ja(:,1) = [ dim, new_size ]

    c%get_partial_left => get_partial_unpack
    c%get_partial_right => get_partial_pack
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'unpack'
       c%left_operand => a
    end if
  end function unpack_array

  function index_array(a, indices) result(c)
    !! Index an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    type(array_type), pointer :: c

    integer :: i, s

    allocate(c)
    call c%allocate(array_shape=[size(a%val,1), size(indices)])
    do concurrent(s=1:size(indices), i=1:size(a%val,1))
       c%val(i, s) = a%val(i, indices(s))
    end do
    c%indices = indices

    c%get_partial_left => get_partial_index
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'index'
       c%left_operand => a
    end if
  end function index_array

  function reverse_index_array(a, indices, from, new_index_size) result(c)
    !! Index an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    logical, intent(in) :: from
    integer, intent(in) :: new_index_size
    type(array_type), pointer :: c

    integer :: i, s

    allocate(c)
    if(from) then
       call c%allocate(array_shape=[size(a%val,1), new_index_size])
       c%val = 0.0_real32
       do concurrent(s=1:size(indices), i=1:size(a%val,1))
          c%val(i, s) = a%val(i, indices(s))
       end do
    else
       call c%allocate(array_shape=[size(a%val,1), new_index_size])
       c%val = 0.0_real32
       do concurrent(s=1:size(indices), i=1:size(a%val,1))
          c%val(i, indices(s)) = a%val(i, s)
       end do
    end if
    c%indices = indices

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'index'
       c%left_operand => a
    end if
  end function reverse_index_array

  function transpose_array(a) result(c)
    !! Transpose an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    integer :: i, j, s

    if(size(a%shape) .ne. 2)then
       write(*,*) "ashape", a%shape
       call stop_program("transpose_array: only 2D arrays can be transposed")
    end if
    c => a%create_result(array_shape=[a%shape(2), a%shape(1), size(a%val,2)])
    ! transpose 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:a%shape(1), j=1:a%shape(2))
          c%val( (i-1)*a%shape(2) + j, s) = a%val( (j-1)*a%shape(1) + i, s)
       end do
    end do

    c%get_partial_left => get_partial_transpose_left
    ! c%get_partial_right => get_partial_transpose_right
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'transpose'
       c%left_operand => a
    end if
  end function transpose_array

  function lt_scalar(a, b) result(c)
    !! Less than comparison between autodiff array and scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    logical, dimension(size(a%val,1), size(a%val,2)) :: c

    c = a%val .lt. b

  end function lt_scalar

  function gt_scalar(a, b) result(c)
    !! Greater than comparison between autodiff array and scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    logical, dimension(size(a%val,1), size(a%val,2)) :: c

    c = a%val .gt. b

  end function gt_scalar

  function maxval_array(a, dim) result(c)
    !! Find maximum value along a dimension
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    real(real32), dimension(:), allocatable :: c

    integer :: i, s

    if(size(a%shape) .ne. 1)then
       call stop_program("maxval: only 1D arrays can be used")
    end if

    if(dim.eq.1)then
       allocate(c(size(a%val,2)))
       do concurrent(s=1:size(a%val,2))
          c(s) = maxval(a%val(:,s))
       end do
    else if(dim.eq.2)then
       allocate(c(size(a%val,1)))
       do concurrent(i=1:size(a%val,1))
          c(i) = maxval(a%val(i,:))
       end do
    else
       call stop_program("maxval: only 1 or 2 dimensions are supported")
    end if

  end function maxval_array

  function max_array(a, b) result(c)
    !! Find maximum value between two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = max(a%val, b%val)

    c%get_partial_left => get_partial_max_left
    c%get_partial_right => get_partial_max_right
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'max'
       c%left_operand => a
       c%right_operand => b
    end if
  end function max_array

  function max_scalar(a, scalar) result(c)
    !! Find maximum value between an autodiff array and a scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = max(a%val, scalar)

    c%get_partial_left => get_partial_max_left
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'max_scalar'
       c%left_operand => a
    end if
  end function max_scalar

  function sum_array(a, dim) result(c)
    !! Sum values along a dimension
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    type(array_type), pointer :: c

    integer :: i, s

    if(dim.eq.1)then
       c => a%create_result(array_shape=[1, size(a%val,2)])
       do concurrent(s=1:size(a%val,2))
          c%val(1,s) = sum(a%val(:,s))
       end do
    else if(dim.eq.2)then
       c => a%create_result(array_shape=[a%shape, 1])
       do concurrent(i=1:size(a%val,1))
          c%val(i,1) = sum(a%val(i,:))
       end do
       c%is_sample_dependent = .false.
    else
       call stop_program("sum_array: only 1 or 2 dimensions are supported")
    end if
    c%indices = [dim, 1]

    c%get_partial_left => get_partial_sum_reverse
    c%get_partial_right => get_partial_sum_forward
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'sum_array'
       c%left_operand => a
    end if

  end function sum_array

  function sum_array_output_real(a, dim) result(c)
    !! Sum values along a dimension
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    real(real32), dimension(:), allocatable :: c

    integer :: i, s

    if(size(a%shape) .ne. 1)then
       call stop_program("sum_array: only 1D arrays can be used")
    end if

    if(dim.eq.1)then
       allocate(c(size(a%val,2)))
       do concurrent(s=1:size(a%val,2))
          c(s) = sum(a%val(:,s))
       end do
    else if(dim.eq.2)then
       allocate(c(size(a%val,1)))
       do concurrent(i=1:size(a%val,1))
          c(i) = sum(a%val(i,:))
       end do
    else
       call stop_program("sum_array: only 1 or 2 dimensions are supported")
    end if

  end function sum_array_output_real

  function sum_array_output_array(a, dim, new_dim_index, new_dim_size) result(c)
    !! Sum values along a dimension and return an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    integer, intent(in) :: new_dim_index
    integer, intent(in) :: new_dim_size
    type(array_type), pointer :: c

    integer :: i, s

    if(size(a%shape) .ne. 1)then
       call stop_program("sum_array_output_array: only 1D arrays can be used")
    end if

    allocate(c)
    ! sum 1D array by using shape to swap dimensions
    if(dim.eq.1)then
       call c%allocate(array_shape=[new_dim_size, size(a%val,2)])
       c%val = 0.0_real32
       c%val(new_dim_index,:) = sum(a%val(:,:), dim=1)
    else if(dim.eq.2)then
       call c%allocate(array_shape=[size(a%val,1), new_dim_size])
       c%val = 0.0_real32
       c%val(:,new_dim_index) = sum(a%val(:,:), dim=2)
    end if

    c%get_partial_left => get_partial_sum_reverse
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'sum_array_output_array'
       c%left_operand => a
    end if
    c%indices = [dim, new_dim_index]
  end function sum_array_output_array

  function mean_array(a, dim) result(c)
    !! Compute mean values along a dimension
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    type(array_type), pointer :: c

    integer :: i, s
    real(real32) :: rtmp1

    ! if(size(a%shape) .ne. 1)then
    !    call stop_program("mean_array: only 1D arrays can be used")
    ! end if

    if(dim.eq.1)then
       c => a%create_result(array_shape = [1, size(a%val,2)])
       rtmp1 = real(size(a%val,1), real32)
       do concurrent(s=1:size(a%val,2))
          c%val(1,s) = sum(a%val(:,s)) / rtmp1
       end do
    else if(dim.eq.2)then
       c => a%create_result(array_shape = [a%shape, 1])
       rtmp1 = real(size(a%val,2), real32)
       do concurrent(i=1:size(a%val,1))
          c%val(i,1) = sum(a%val(i,:)) / rtmp1
       end do
       c%is_sample_dependent = .false.
    else
       call stop_program("mean_array: only 1 or 2 dimensions are supported")
    end if
    c%indices = [dim, 1]

    c%get_partial_left => get_partial_mean
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'mean_array'
       c%left_operand => a
    end if

  end function mean_array

  function merge_scalar(tsource, fsource, mask) result(c)
    !! Merge two autodiff arrays based on a mask
    implicit none
    class(array_type), intent(in), target :: tsource
    real(real32), intent(in) :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: c

    integer :: i, j, s

    if(size(tsource%shape) .ne. 1)then
       call stop_program("merge_array: only 1D arrays can be merged")
    end if

    allocate(c)
    call c%allocate(array_shape=[size(tsource%val,1), size(tsource%val,2)])
    ! merge 1D array by using shape to swap dimensions
    do concurrent(s=1:size(tsource%val,2))
       do concurrent(i=1:size(tsource%val,1), j=1:size(tsource%val,2))
          if(mask(i,j)) then
             c%val(i,j) = tsource%val(i,j)
          else
             c%val(i,j) = fsource
          end if
       end do
    end do
    c%mask = mask

    c%get_partial_left => get_partial_merge
    if(tsource%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = tsource%is_forward
       c%is_leaf = .false.
       c%operation = 'merge'
       c%left_operand => tsource
    end if
  end function merge_scalar

  function merge_real2d(tsource, fsource, mask) result(c)
    !! Merge two autodiff arrays based on a mask
    implicit none
    class(array_type), intent(in), target :: tsource
    real(real32), dimension(:,:), intent(in) :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: c

    integer :: i, j !, itmp1
    !  integer, dimension(:,:), allocatable :: adj_ja_tmp

    if(allocated(tsource%shape))then
       if(size(tsource%shape) .ne. 1)then
          call stop_program("merge_array: only 1D arrays can be merged")
       end if
    end if

    allocate(c)
    call c%allocate(array_shape=[size(tsource%val,1), size(tsource%val,2)])
    ! merge 1D array by using shape to swap dimensions
    !  allocate(adj_ja_tmp(1, size(mask)))
    !  itmp1 = 0
    do concurrent( i = 1: size(tsource%val,1), j = 1: size(tsource%val,2))
       if(mask(i,j)) then
          c%val(i,j) = tsource%val(i,j)
          !  if(.not.allocated(c%indices))then
          !    c%indices = [i]
          !  elseif(c%indices(size(c%indices)) .ne. i) then
          !    c%indices = [c%indices, i]
          !  end if
          !  itmp1 = itmp1 + 1
          !  adj_ja_tmp(1,itmp1) = j
       else
          c%val(i,j) = fsource(i,j)
       end if
    end do
    c%mask = mask
    !  allocate(c%adj_ja(1, itmp1))
    !  c%adj_ja(1,:) = adj_ja_tmp(1,1:itmp1)


    c%get_partial_left => get_partial_merge
    if(tsource%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = tsource%is_forward
       c%is_leaf = .false.
       c%operation = 'merge'
       c%left_operand => tsource
    end if
  end function merge_real2d

  function spread_array(source, dim, index, ncopies) result(c)
    !! Spread an autodiff array along a dimension
    implicit none
    class(array_type), intent(in), target :: source
    integer, intent(in) :: dim
    integer, intent(in) :: index
    integer, intent(in) :: ncopies
    type(array_type), pointer :: c

    integer :: i, s

    if(size(source%shape) .ne. 1)then
       call stop_program("spread: only 1D arrays can be used")
    end if

    if(dim.eq.1)then
       c => source%create_result(array_shape=[ncopies, size(source%val,2)])
       do concurrent(s=1:ncopies)
          c%val(s, :) = source%val(index, :)
       end do
    else if(dim.eq.2)then
       c => source%create_result(array_shape=[size(source%val,1), ncopies])
       do concurrent(s=1:ncopies)
          c%val(:, s) = source%val(:, index)
       end do
    else
       call stop_program("spread: only 1 or 2 dimensions are supported")
    end if
    c%indices = [index]
    allocate(c%adj_ja(2,1))
    c%adj_ja(:,1) = [ dim, size(source%val,dim) ]

    c%get_partial_left => get_partial_spread
    if(source%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = source%is_forward
       c%is_leaf = .false.
       c%operation = 'spread'
       c%left_operand => source
    end if
  end function spread_array

  function unspread_array(source, index, dim, new_size) result(c)
    !! Unpack an autodiff array
    implicit none
    class(array_type), intent(in), target :: source
    integer, intent(in) :: index
    integer, intent(in) :: new_size, dim
    type(array_type), pointer :: c

    integer :: i, s


    if(dim.eq.1)then
       c => source%create_result(array_shape = [ new_size, size(source%val,2) ])
       c%val = 0.0_real32
       do concurrent(i=1:size(source%val,1), s=1:size(source%val,2))
          c%val(index,s) = c%val(index,s) + source%val(i,s)
       end do
    elseif(dim.eq.2)then
       c => source%create_result( array_shape = [ size(source%val,1), new_size ] )
       c%val = 0.0_real32
       do concurrent(i=1:size(source%val,1), s=1:size(source%val,2))
          c%val(i,index) = c%val(i,index) + source%val(i,s)
       end do
    end if
    c%indices = [index]
    allocate(c%adj_ja(2,1))
    c%adj_ja(:,1) = [ dim, new_size ]

    c%get_partial_left => get_partial_unspread
    if(source%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = source%is_forward
       c%is_leaf = .false.
       c%operation = 'unspread'
       c%left_operand => source
    end if
  end function unspread_array

  !-----------------------------------------------------------------------------
  ! Division operations
  !-----------------------------------------------------------------------------
  function divide_arrays(a, b) result(c)
    !! Divide two autodiff arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s

    if(all(shape(a%val) .eq. shape(b%val))) then
       c => a%create_result()
       c%val = a%val / b%val
    elseif(size(a%val,1).ne.size(b%val,1).and.size(a%val,2).eq.size(b%val,2))then
       if(size(a%val,1) .eq. 1)then
          c => b%create_result()
          do concurrent(s=1:size(a%val,2))
             c%val(:,s) = a%val(1,s) / b%val(:,s)
          end do
       elseif(size(b%val,1) .eq. 1)then
          c => a%create_result()
          do concurrent(s=1:size(a%val,2))
             c%val(:,s) = a%val(:,s) / b%val(1,s)
          end do
       end if
    end if

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'divide'
       c%left_operand => a
       c%right_operand => b
    end if
  end function divide_arrays

  function divide_scalar(a, scalar) result(c)
    !! Divide autodiff array by scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = a%val / scalar

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'divide_scalar'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array
  end function divide_scalar

  function scalar_divide(scalar, a) result(c)
    !! Divide scalar by autodiff array
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = scalar / a%val

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'scalar_divide'
       c%right_operand => a
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%left_operand => b_array
  end function scalar_divide

  function divide_real1d(a, b) result(c)
    !! Divide autodiff array by a real array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    integer :: s

    c => a%create_result()
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) / b(s)
    end do

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'divide_real1d'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, size(b)])
    b_array%val(1,:) = b
    c%right_operand => b_array
  end function divide_real1d

  !-----------------------------------------------------------------------------
  ! Power operations
  !-----------------------------------------------------------------------------
  function power_arrays(a, b) result(c)
    !! Raise autodiff array to power of another array
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val ** b%val

    c%get_partial_left => get_partial_power_base
    c%get_partial_right => get_partial_power_exponent
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'power'
       c%left_operand => a
       c%right_operand => b
    end if
  end function power_arrays

  function power_real_scalar(a, scalar) result(c)
    !! Raise autodiff array to scalar power
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = a%val ** scalar

    c%get_partial_left => get_partial_power_base
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'power_scalar'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array

  end function power_real_scalar

  function power_int_scalar(a, scalar) result(c)
    !! Raise autodiff array to scalar power
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: scalar
    type(array_type), pointer :: c

    c => power_real_scalar(a, real(scalar, real32))
  end function power_int_scalar

  function scalar_power(scalar, a) result(c)
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = scalar ** a%val

    c%get_partial_left => get_partial_power_base
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'scalar_power'
       c%right_operand => a
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%left_operand => b_array

  end function scalar_power

  function int_scalar_power(scalar, a) result(c)
    implicit none
    integer, intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => scalar_power(real(scalar, real32), a)
  end function int_scalar_power

  !-----------------------------------------------------------------------------
  ! Mathematical functions
  !-----------------------------------------------------------------------------
  function sin_array(a) result(c)
    !! Sine function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    !  allocate(c)
    c => a%create_result()
    c%val = sin(a%val)

    c%get_partial_left => get_partial_sin
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'sin'
       c%left_operand => a
    end if
  end function sin_array

  function cos_array(a) result(c)
    !! Cosine function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    !  allocate(c)
    c => a%create_result()
    c%val = cos(a%val)

    c%get_partial_left => get_partial_cos
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'cos'
       c%left_operand => a
    end if
  end function cos_array

  function tan_array(a) result(c)
    !! Tangent function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = tan(a%val)

    c%get_partial_left => get_partial_tan
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'tan'
       c%left_operand => a
    end if
  end function tan_array

  function exp_array(a) result(c)
    !! Exponential function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = exp(a%val)

    c%get_partial_left => get_partial_exp
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'exp'
       c%left_operand => a
    end if
  end function exp_array

  function log_array(a) result(c)
    !! Natural logarithm function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = log(a%val)

    c%get_partial_left => get_partial_log
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'log'
       c%left_operand => a
    end if
  end function log_array

  function sqrt_array(a) result(c)
    !! Square root function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = sqrt(a%val)

    c%get_partial_left => get_partial_sqrt
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'sqrt'
       c%left_operand => a
    end if
  end function sqrt_array

  function tanh_array(a) result(c)
    !! Hyperbolic tangent function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = tanh(a%val)

    c%get_partial_left => get_partial_tanh
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'tanh'
       c%left_operand => a
    end if
  end function tanh_array

  function tanh_reverse_array(a) result(c)
    !! Reverse mode for tanh function
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    !  allocate(output)
    c => a%create_result()
    c%val = (1._real32 - a%val ** 2._real32)

    c%get_partial_left => get_partial_tanh_reverse
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'tanh_reverse'
       c%left_operand => a
    end if

  end function tanh_reverse_array

  function sigmoid_array(a) result(c)
    !! Sigmoid function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = 1.0_real32 / (1.0_real32 + exp(-a%val))

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'sigmoid'
       c%left_operand => a
    end if
  end function sigmoid_array


end module athena__misc_types
