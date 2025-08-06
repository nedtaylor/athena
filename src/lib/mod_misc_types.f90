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
  public :: array_container_type
  public :: array1d_type, array2d_type, array3d_type, array4d_type, array5d_type
  public :: facets_type

  public :: operator(+), operator(-), operator(*), operator(/), &
       operator(**), operator(.mmul.), operator(.concat.), operator(.ltrim.), &
       operator(.rtrim.), operator(.index.)
  public :: operator(.lt.)
  public :: merge, maxval, max, sum, spread, reverse_index
  public :: sin, cos, tan, exp, log, sqrt, tanh, sigmoid, transpose


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
     integer :: rank
     !! Rank of the array
     integer, dimension(:), allocatable :: shape
     !! Shape of the array
     integer :: size
     !! Size of the array
     logical :: is_constant = .false.
     !! Logical flag for constant array
     logical :: allocated = .false.
     !! Logical flag for array allocation
     real(real32), dimension(:,:), allocatable :: val
     !! Array values in rank 2 (sample, batch)
     integer, dimension(:), allocatable :: indices
     !! Indices for gradient accumulation
     logical :: requires_grad = .false.
     !! Flag indicating if gradients should be computed
     logical :: is_leaf = .true.
     !! Flag indicating if this is a leaf node (parameter)
     class(array_type), pointer :: grad => null()
     !! Gradient array (same type as value)
     class(array_type), pointer :: left_operand => null()
     !! Left operand for backward pass
     class(array_type), pointer :: right_operand => null()
     !! Right operand for backward pass
     character(len=32) :: operation = 'none'
     logical :: owns_gradient = .false.
     !! Flag indicating if this array owns its gradient memory
   contains
     procedure, pass(this) :: allocate => allocate_array
     !! Abstract procedure for allocating array
     procedure, pass(this) :: deallocate => deallocate_array
     !! Abstract procedure for deallocating array
     procedure, pass(this) :: set_ptr => set_ptr_array
     !! Abstract procedure for setting pointers
     procedure, pass(this) :: flatten => flatten_array
     !! Procedure for flattening array
     procedure, pass(this) :: get => get_array
     !! Procedure for getting array
     procedure, pass(this) :: set => set_array
     !! Procedure for setting array
     procedure :: add => add_array
     !! Procedure for adding arrays
     procedure :: multiply => multiply_array
     !! Procedure for multiplying arrays
     !  generic, public :: operator(+) => add
     !  !! Generic for adding arrays
     !  generic, public :: operator(*) => multiply
     !  !! Generic for multiplying arrays
     procedure :: assign => assign_array
     generic, public :: assignment(=) => assign

     procedure :: backward => backward_autodiff
     !! Backward pass for gradient computation
     procedure :: zero_grad => zero_grad_autodiff
     !! Zero the gradients
     procedure :: detach => detach_autodiff
     !! Detach from computation graph
     procedure :: backward_op => backward_op_array
     !! Deferred procedure for operation-specific backward pass
     procedure :: set_requires_grad => set_requires_grad_autodiff
     !! Set requires_grad flag
     procedure :: create_result => create_result_array
     !! Helper to safely create result arrays
     final :: finalise_array
     !! Finaliser for array type
  end type array_type

  ! Interface for allocate, deallocate, and flattening array
  !-----------------------------------------------------------------------------
  interface
     module subroutine allocate_array(this, array_shape, source)
       class(array_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array

     module subroutine deallocate_array(this, keep_shape)
       class(array_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array

     module subroutine set_ptr_array(this)
       class(array_type), intent(inout), target :: this
     end subroutine set_ptr_array

     module subroutine finalise_array(this)
       type(array_type), intent(inout) :: this
     end subroutine finalise_array

     module function create_result_array(this, shape_arr) result(result_ptr)
       class(array_type), intent(in) :: this
       integer, dimension(:), intent(in), optional :: shape_arr
       type(array_type), pointer :: result_ptr
     end function create_result_array



     module subroutine backward_op_array(this, upstream_grad)
       class(array_type), intent(inout) :: this
       class(array_type), intent(in) :: upstream_grad
     end subroutine backward_op_array

     module subroutine set_requires_grad_autodiff(this, requires_grad)
       class(array_type), intent(inout) :: this
       logical, intent(in) :: requires_grad
     end subroutine set_requires_grad_autodiff

     module subroutine backward_autodiff(this)
       !! Perform backward pass starting from this array
       class(array_type), intent(inout) :: this
     end subroutine backward_autodiff

     module subroutine zero_grad_autodiff(this)
       !! Zero the gradients of this array
       class(array_type), intent(inout) :: this
     end subroutine zero_grad_autodiff

     module subroutine detach_autodiff(this)
       !! Detach this array from the computation graph
       class(array_type), intent(inout) :: this
     end subroutine detach_autodiff
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

     module function add_array(a, b) result(output)
       class(array_type), intent(in) :: a, b
       type(array_type) :: output
     end function add_array

     module function multiply_array(a, b) result(output)
       class(array_type), intent(in) :: a, b
       type(array_type) :: output
     end function multiply_array

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
     module procedure power_scalar
  end interface

  interface operator(.mmul.)
     module procedure matmul_arrays
     module procedure real2d_matmul
     module procedure matmul_real2d
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

  interface sum
     module procedure sum_array
     module procedure sum_array_output_array
  end interface

  interface maxval
     module procedure maxval_array
  end interface

  interface max
     module procedure max_array
  end interface

  interface merge
     module procedure merge_scalar
  end interface

  interface spread
     module procedure spread_array
  end interface

  interface reverse_index
     module procedure reverse_index_array
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



  ! Extend the array type to 1d, 2d, 3d, 4d, and 5d arrays
  !-----------------------------------------------------------------------------
  type, extends(array_type) :: array1d_type
     !! Type for 1D array
     real(real32), pointer :: val_ptr(:) => null()
   contains
     procedure :: allocate => allocate_array1d
     procedure :: deallocate => deallocate_array1d
     procedure :: set_ptr => set_ptr_array1d
     final :: finalise_array1d
  end type array1d_type

  type, extends(array_type) :: array2d_type
     !! Type for 2D array
     real(real32), pointer :: val_ptr(:,:) => null()
     !! Pointer with rank 2 to the value of the array
   contains
     procedure :: allocate => allocate_array2d
     procedure :: deallocate => deallocate_array2d
     procedure :: set_ptr => set_ptr_array2d
     final :: finalise_array2d
  end type array2d_type

  type, extends(array_type) :: array3d_type
     !! Type for 3D array
     real(real32), pointer :: val_ptr(:,:,:) => null()
     !! Pointer with rank 3 to the value of the array
   contains
     procedure :: allocate => allocate_array3d
     procedure :: deallocate => deallocate_array3d
     procedure :: set_ptr => set_ptr_array3d
     procedure, pass(this) :: set => set_array3d
     final :: finalise_array3d
  end type array3d_type

  type, extends(array_type) :: array4d_type
     !! Type for 4D array
     real(real32), pointer :: val_ptr(:,:,:,:) => null()
     !! Pointer with rank 4 to the value of the array
   contains
     procedure :: allocate => allocate_array4d
     procedure :: deallocate => deallocate_array4d
     procedure :: set_ptr => set_ptr_array4d
     procedure, pass(this) :: set => set_array4d
     final :: finalise_array4d
  end type array4d_type

  type, extends(array_type) :: array5d_type
     !! Type for 5D array
     real(real32), pointer :: val_ptr(:,:,:,:,:) => null()
     !! Pointer with rank 5 to the value of the array
   contains
     procedure :: allocate => allocate_array5d
     procedure :: deallocate => deallocate_array5d
     procedure :: set_ptr => set_ptr_array5d
     procedure, pass(this) :: set => set_array5d
     final :: finalise_array5d
  end type array5d_type

  ! Interface for allocating array
  !-----------------------------------------------------------------------------
  interface
     module subroutine allocate_array1d(this, array_shape, source)
       class(array1d_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array1d

     module subroutine allocate_array2d(this, array_shape, source)
       class(array2d_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array2d

     module subroutine allocate_array3d(this, array_shape, source)
       class(array3d_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array3d

     module subroutine allocate_array4d(this, array_shape, source)
       class(array4d_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array4d

     module subroutine allocate_array5d(this, array_shape, source)
       class(array5d_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array5d
  end interface

  ! Interface for deallocating array
  !-----------------------------------------------------------------------------
  interface
     module subroutine deallocate_array1d(this, keep_shape)
       class(array1d_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array1d

     module subroutine deallocate_array2d(this, keep_shape)
       class(array2d_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array2d

     module subroutine deallocate_array3d(this, keep_shape)
       class(array3d_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array3d

     module subroutine deallocate_array4d(this, keep_shape)
       class(array4d_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array4d

     module subroutine deallocate_array5d(this, keep_shape)
       class(array5d_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array5d
  end interface

  ! Interface for finalising array
  !-----------------------------------------------------------------------------
  interface
     module subroutine finalise_array1d(this)
       type(array1d_type), intent(inout) :: this
     end subroutine finalise_array1d

     module subroutine finalise_array2d(this)
       type(array2d_type), intent(inout) :: this
     end subroutine finalise_array2d

     module subroutine finalise_array3d(this)
       type(array3d_type), intent(inout) :: this
     end subroutine finalise_array3d

     module subroutine finalise_array4d(this)
       type(array4d_type), intent(inout) :: this
     end subroutine finalise_array4d

     module subroutine finalise_array5d(this)
       type(array5d_type), intent(inout) :: this
     end subroutine finalise_array5d
  end interface

  ! Interface for setting pointers
  !-----------------------------------------------------------------------------
  interface
     module subroutine set_ptr_array1d(this)
       class(array1d_type), intent(inout), target :: this
     end subroutine set_ptr_array1d

     module subroutine set_ptr_array2d(this)
       class(array2d_type), intent(inout), target :: this
     end subroutine set_ptr_array2d

     module subroutine set_ptr_array3d(this)
       class(array3d_type), intent(inout), target :: this
     end subroutine set_ptr_array3d

     module subroutine set_ptr_array4d(this)
       class(array4d_type), intent(inout), target :: this
     end subroutine set_ptr_array4d

     module subroutine set_ptr_array5d(this)
       class(array5d_type), intent(inout), target :: this
     end subroutine set_ptr_array5d
  end interface

  ! Interface for setting array
  !-----------------------------------------------------------------------------
  interface
     pure module subroutine set_array3d(this, input)
       class(array3d_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
     end subroutine set_array3d

     pure module subroutine set_array4d(this, input)
       class(array4d_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
     end subroutine set_array4d

     pure module subroutine set_array5d(this, input)
       class(array5d_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
     end subroutine set_array5d
  end interface

  ! Interface for initialising array
  !-----------------------------------------------------------------------------
  interface array1d_type
     module function init_array1d(array_shape) result(output)
       integer, dimension(:), intent(in), optional :: array_shape
       type(array1d_type) :: output
     end function init_array1d
  end interface array1d_type

  interface array2d_type
     module function init_array2d(array_shape) result(output)
       integer, dimension(:), intent(in), optional :: array_shape
       type(array2d_type) :: output
     end function init_array2d
  end interface array2d_type

  interface array3d_type
     module function init_array3d(array_shape) result(output)
       integer, dimension(:), intent(in), optional :: array_shape
       type(array3d_type) :: output
     end function init_array3d
  end interface array3d_type

  interface array4d_type
     module function init_array4d(array_shape) result(output)
       integer, dimension(:), intent(in), optional :: array_shape
       type(array4d_type) :: output
     end function init_array4d
  end interface array4d_type

  interface array5d_type
     module function init_array5d(array_shape) result(output)
       integer, dimension(:), intent(in), optional :: array_shape
       type(array5d_type) :: output
     end function init_array5d
  end interface array5d_type

  ! Interface for assigning array
  !-----------------------------------------------------------------------------
  interface
     module subroutine assign_array1d(this, input)
       type(array1d_type), intent(out), target :: this
       type(array1d_type), intent(in) :: input
     end subroutine assign_array1d

     module subroutine assign_array2d(this, input)
       type(array2d_type), intent(out), target :: this
       type(array2d_type), intent(in) :: input
     end subroutine assign_array2d

     module subroutine assign_array3d(this, input)
       type(array3d_type), intent(out), target :: this
       type(array3d_type), intent(in) :: input
     end subroutine assign_array3d

     module subroutine assign_array4d(this, input)
       type(array4d_type), intent(out), target :: this
       type(array4d_type), intent(in) :: input
     end subroutine assign_array4d

     module subroutine assign_array5d(this, input)
       type(array5d_type), intent(out), target :: this
       type(array5d_type), intent(in) :: input
     end subroutine assign_array5d
  end interface

  type :: array_container_type
     class(array_type), allocatable :: array
  end type array_container_type


  interface assignment (=)
     !  module procedure assign_array
     module procedure assign_array1d
     module procedure assign_array2d
     module procedure assign_array3d
     module procedure assign_array4d
     module procedure assign_array5d
  end interface
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
  ! Addition operation
  !-----------------------------------------------------------------------------
  function add_arrays(a, b) result(c)
    !! Add two autodiff arrays
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    ! Safely create result array
    c => a%create_result()
    c%val = a%val + b%val

    ! Set up computation graph
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'add'
       c%left_operand => a
       c%right_operand => b
    end if
  end function add_arrays

  function add_real2d(a, b) result(c)
    !! Add a real array to an autodiff array
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val + b

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'add'
       c%left_operand => a
    end if
  end function add_real2d

  function real2d_add(a, b) result(c)
    !! Add a real array to an autodiff array
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_real2d(b, a)
  end function real2d_add

  function add_real1d(a, b) result(c)
    !! Add a real array to an autodiff array
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s

    c => a%create_result()
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) + b(:)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'add'
       c%left_operand => a
    end if
  end function add_real1d

  function real1d_add(a, b) result(c)
    !! Add a real array to an autodiff array
    real(real32), dimension(:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_real1d(b, a)
  end function real1d_add

  function add_scalar(a, b) result(c)
    !! Add a scalar to an autodiff array
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    type(array_type), pointer :: c

    allocate(c)
    call c%allocate(array_shape=[ a%shape, size(a%val,2) ])
    ! c => a%create_result()
    c%val = a%val + b

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'add'
       c%left_operand => a
    end if
  end function add_scalar

  function scalar_add(a, b) result(c)
    !! Add a scalar to an autodiff array
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
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val - b%val

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'subtract'
       c%left_operand => a
       c%right_operand => b
    end if
  end function subtract_arrays

  function subtract_real1d(a, b) result(c)
    !! Subtract a real array from an autodiff array
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s

    c => a%create_result()
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) - b(s)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'subtract_scalar'
       c%left_operand => a
    end if
  end function subtract_real1d

  function negate_array(a) result(c)
    !! Negate an autodiff array
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = -a%val

    if(a%requires_grad) then
       c%requires_grad = .true.
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
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val * b%val

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'multiply'
       c%left_operand => a
       c%right_operand => b
    end if
  end function multiply_arrays

  function multiply_scalar(a, scalar) result(c)
    !! Multiply autodiff array by scalar
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = a%val * scalar

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'multiply_scalar'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_constant = .true.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array
  end function multiply_scalar

  function scalar_multiply(scalar, a) result(c)
    !! Multiply scalar by autodiff array
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c = multiply_scalar(a, scalar)
  end function scalar_multiply

  function multiply_logical(a, b) result(c)
    !! Multiply two logical arrays (element-wise)
    class(array_type), intent(in), target :: a
    logical, dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s, i

    allocate(c)
    call c%allocate(array_shape=[size(a%val,1), size(a%val,2)])
    do concurrent(s=1:size(a%val,2), i=1:size(a%val,1))
       if(b(i,s)) then
          c%val(i,s) = a%val(i,s)
       else
          c%val(i,s) = 0.0_real32
       end if
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'multiply_logical'
       c%left_operand => a
    end if

  end function multiply_logical

  function matmul_arrays(a, b) result(c)
    !! Matrix multiplication of two autodiff arrays
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s
    real(real32), pointer :: temp(:,:)

    allocate(c)
    if(a%is_constant)then
       call c%allocate(array_shape=[a%shape(1), size(b%val,2)])
       temp(1:a%shape(1), 1:a%shape(2)) => a%val
       do concurrent(s=1:size(b%val,2))
          c%val(:,s) = matmul(temp, b%val(:,s))
       end do
    else
       call c%allocate(array_shape=[size(a%val,1), b%shape(2)])
       temp(1:b%shape(1), 1:b%shape(2)) => b%val
       do concurrent(s=1:size(a%val,2))
          c%val(:,s) = matmul(a%val(:,s), temp)
       end do
    end if

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'matmul'
       c%left_operand => a
       c%right_operand => b
    end if
  end function matmul_arrays

  function matmul_real2d(a, b) result(c)
    !! Matrix multiplication of a real array and an autodiff array
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    integer :: s, i

    allocate(c)
    call c%allocate(array_shape=[size(b,1), size(a%val,2)])
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = matmul(b, a%val(:,s))
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'matmul_scalar'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_constant = .true.
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
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: a_array

    integer :: s, i

    allocate(c)
    call c%allocate(array_shape=[size(a,1), size(b%val,2)])
    do concurrent(s=1:size(b%val,2))
       c%val(:,s) = matmul(a, b%val(:,s))
    end do

    if(b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'matmul_scalar'
       c%right_operand => b
    end if
    allocate(a_array)
    a_array%is_constant = .true.
    a_array%shape = shape(a)
    a_array%requires_grad = .false.
    a_array%is_leaf = .false.
    call a_array%allocate(array_shape=[size(a,1), size(a,2), 1])
    do i = 1, size(a,2)
       a_array%val((i-1)*size(a,1)+1:i*size(a,1), 1) = a(:,i)
    end do
    c%left_operand => a_array
  end function real2d_matmul

  function concat_arrays(a, b) result(c)
    !! Concatenate two autodiff arrays along the first dimension
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, j, s

    allocate(c)
    call c%allocate(array_shape=[size(a%val,1) + size(b%val,1), size(a%val,2)])
    ! concatenate 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:1, j=1:size(a%val,1))
          c%val( i, s) = a%val( i, s)
       end do
       do concurrent(i=1:1, j=1:size(b%val,1))
          c%val( size(a%val,1) + i, s) = b%val( i, s)
       end do
    end do

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'concat'
       c%left_operand => a
       c%right_operand => b
    end if
  end function concat_arrays

  function ltrim_array(a, b) result(c)
    !! Left trim an autodiff array
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    type(array_type), pointer :: c

    integer :: i, j, s

    allocate(c)
    call c%allocate(array_shape=[b, size(a%val,2)])
    ! left trim 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       c%val( :, s) = a%val( 1:b, s)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'ltrim'
       c%left_operand => a
    end if
  end function ltrim_array

  function rtrim_array(a, b) result(c)
    !! Right trim an autodiff array
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    type(array_type), pointer :: c

    integer :: i, j, s

    allocate(c)
    call c%allocate(array_shape=[b, size(a%val,2)])
    ! right trim 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       c%val( :, s) = a%val( size(a%val,1)-b+1:size(a%val,1), s)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'rtrim'
       c%left_operand => a
    end if
  end function rtrim_array

  function index_array(a, indices) result(c)
    !! Index an autodiff array
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

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'index'
       c%left_operand => a
    end if
  end function index_array

  function reverse_index_array(a, indices, from, new_index_size) result(c)
    !! Index an autodiff array
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    logical, intent(in) :: from
    integer, intent(in) :: new_index_size
    type(array_type), pointer :: c

    integer :: i, s

    allocate(c)
    if(from) then
       call c%allocate(array_shape=[size(a%val,1), new_index_size])
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
       c%is_leaf = .false.
       c%operation = 'index'
       c%left_operand => a
    end if
  end function reverse_index_array

  function transpose_array(a) result(c)
    !! Transpose an autodiff array
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    integer :: i, j, s

    if(size(a%shape) .ne. 2)then
       call stop_program("transpose_array: only 2D arrays can be transposed")
    end if
    allocate(c)
    call c%allocate(array_shape=[a%shape(2), a%shape(1), size(a%val,2)])
    ! transpose 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:a%shape(1), j=1:a%shape(2))
          c%val( (i-1)*a%shape(2) + j, s) = a%val( (j-1)*a%shape(1) + i, s)
       end do
    end do

    c%is_constant = a%is_constant
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'transpose'
       c%left_operand => a
    end if
  end function transpose_array

  function lt_scalar(a, b) result(c)
    !! Less than comparison between autodiff array and scalar
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    logical, dimension(size(a%val,1), size(a%val,2)) :: c

    c = a%val .lt. b

  end function lt_scalar

  function maxval_array(a, dim) result(c)
    !! Find maximum value along a dimension
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
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = max(a%val, b%val)

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'max'
       c%left_operand => a
       c%right_operand => b
    end if
  end function max_array

  function sum_array(a, dim) result(c)
    !! Sum values along a dimension
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

  end function sum_array

  function sum_array_output_array(a, dim, new_dim_index, new_dim_size) result(c)
    !! Sum values along a dimension and return an autodiff array
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

    c%is_constant = a%is_constant
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'sum_array_output_array'
       c%left_operand => a
    end if
    c%indices = [dim, new_dim_index]
  end function sum_array_output_array

  function merge_scalar(tsource, fsource, mask) result(c)
    !! Merge two autodiff arrays based on a mask
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

    if(tsource%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'merge'
       c%left_operand => tsource
    end if
  end function merge_scalar

  function spread_array(source, dim, index, ncopies) result(c)
    !! Spread an autodiff array along a dimension
    class(array_type), intent(in), target :: source
    integer, intent(in) :: dim
    integer, intent(in) :: index
    integer, intent(in) :: ncopies
    type(array_type), pointer :: c

    integer :: i, s

    if(size(source%shape) .ne. 1)then
       call stop_program("spread: only 1D arrays can be used")
    end if

    allocate(c)
    if(dim.eq.1)then
       call c%allocate(array_shape=[ncopies, size(source%val,2)])
       do concurrent(s=1:ncopies)
          c%val(s, :) = source%val(index, :)
       end do
    else if(dim.eq.2)then
       call c%allocate(array_shape=[size(source%val,1), ncopies])
       do concurrent(s=1:ncopies)
          c%val(:, s) = source%val(:, index)
       end do
    else
       call stop_program("spread: only 1 or 2 dimensions are supported")
    end if

    if(source%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'spread'
       c%left_operand => source
    end if
  end function spread_array

  !-----------------------------------------------------------------------------
  ! Division operations
  !-----------------------------------------------------------------------------
  function divide_arrays(a, b) result(c)
    !! Divide two autodiff arrays (element-wise)
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    allocate(c)
    c%val = a%val / b%val

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'divide'
       c%left_operand => a
       c%right_operand => b
    end if
  end function divide_arrays

  function divide_scalar(a, scalar) result(c)
    !! Divide autodiff array by scalar
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    allocate(c)
    call c%allocate(array_shape=[size(a%val,1), size(a%val,2)])
    c%val = a%val / scalar

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'divide_scalar'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_constant = .true.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array
  end function divide_scalar

  function scalar_divide(scalar, a) result(c)
    !! Divide scalar by autodiff array
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    allocate(c)
    call c%allocate(array_shape=[size(a%val,1), size(a%val,2)])
    c%val = scalar / a%val

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'scalar_divide'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_constant = .true.
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array
  end function scalar_divide

  function divide_real1d(a, b) result(c)
    !! Divide autodiff array by a real array
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    integer :: s

    allocate(c)
    call c%allocate(array_shape=[size(a%val,1), size(a%val,2)])
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) / b(s)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'divide_real1d'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_constant = .true.
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
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    allocate(c)
    c%val = a%val ** b%val

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'power'
       c%left_operand => a
       c%right_operand => b
    end if
  end function power_arrays

  function power_scalar(a, scalar) result(c)
    !! Raise autodiff array to scalar power
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c

    allocate(c)
    c%val = a%val ** scalar

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'power_scalar'
       c%left_operand => a
    end if
  end function power_scalar

  !-----------------------------------------------------------------------------
  ! Mathematical functions
  !-----------------------------------------------------------------------------
  function sin_array(a) result(c)
    !! Sine function for autodiff arrays
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    allocate(c)
    c%val = sin(a%val)

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'sin'
       c%left_operand => a
    end if
  end function sin_array

  function cos_array(a) result(c)
    !! Cosine function for autodiff arrays
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    allocate(c)
    c%val = cos(a%val)

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'cos'
       c%left_operand => a
    end if
  end function cos_array

  function tan_array(a) result(c)
    !! Tangent function for autodiff arrays
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    allocate(c)
    c%val = tan(a%val)

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'tan'
       c%left_operand => a
    end if
  end function tan_array

  function exp_array(a) result(c)
    !! Exponential function for autodiff arrays
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    allocate(c)
    call c%allocate(array_shape=[size(a%val,1), size(a%val,2)])
    c%val = exp(a%val)

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'exp'
       c%left_operand => a
    end if
  end function exp_array

  function log_array(a) result(c)
    !! Natural logarithm function for autodiff arrays
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    allocate(c)
    c%val = log(a%val)

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'log'
       c%left_operand => a
    end if
  end function log_array

  function sqrt_array(a) result(c)
    !! Square root function for autodiff arrays
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    allocate(c)
    c%val = sqrt(a%val)

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'sqrt'
       c%left_operand => a
    end if
  end function sqrt_array

  function tanh_array(a) result(c)
    !! Hyperbolic tangent function for autodiff arrays
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    allocate(c)
    c%val = tanh(a%val)

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'tanh'
       c%left_operand => a
    end if
  end function tanh_array

  function sigmoid_array(a) result(c)
    !! Sigmoid function for autodiff arrays
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    allocate(c)
    c%val = 1.0_real32 / (1.0_real32 + exp(-a%val))

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_leaf = .false.
       c%operation = 'sigmoid'
       c%left_operand => a
    end if
  end function sigmoid_array


end module athena__misc_types
