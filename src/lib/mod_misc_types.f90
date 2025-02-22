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
  implicit none


  private

  public :: activation_type
  public :: initialiser_type
  public :: array_type
  public :: array_container_type
  public :: array1d_type, array2d_type, array3d_type, array4d_type, array5d_type
  public :: facets_type


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
     generic :: activate => activate_1d, activate_2d, &
          activate_3d , activate_4d, activate_5d
     !! Generic for activation function
     generic :: differentiate => differentiate_1d, differentiate_2d, &
          differentiate_3d, differentiate_4d, differentiate_5d
     !! Generic for derivative of activation function
  end type activation_type

  ! Interface for activation function
  !-----------------------------------------------------------------------------
  abstract interface
     pure function activation_function_1d(this, val) result(output)
       !! Interface for activation function
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:), intent(in) :: val
       real(real32), dimension(size(val,1)) :: output
     end function activation_function_1d

     pure function activation_function_2d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2)) :: output
     end function activation_function_2d

     pure function activation_function_3d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function activation_function_3d

     pure function activation_function_4d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function activation_function_4d

     pure function activation_function_5d(this, val) result(output)
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
     pure function derivative_function_1d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:), intent(in) :: val
       real(real32), dimension(size(val,1)) :: output
     end function derivative_function_1d

     pure function derivative_function_2d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2)) :: output
     end function derivative_function_2d

     pure function derivative_function_3d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:), intent(in) :: val
       real(real32), dimension(size(val,1),size(val,2),size(val,3)) :: output
     end function derivative_function_3d

     pure function derivative_function_4d(this, val) result(output)
       import activation_type, real32
       class(activation_type), intent(in) :: this
       real(real32), dimension(:,:,:,:), intent(in) :: val
       real(real32), dimension(&
            size(val,1),size(val,2),size(val,3),size(val,4)) :: output
     end function derivative_function_4d

     pure function derivative_function_5d(this, val) result(output)
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


!!!-----------------------------------------------------------------------------
!!! base and extended array types
!!!-----------------------------------------------------------------------------
  type, abstract :: array_type
     !! Abstract type for array operations
     integer :: rank
     !! Rank of the array
     integer, dimension(:), allocatable :: shape
     !! Shape of the array
     integer :: size
     !! Size of the array
     logical :: allocated = .false.
     !! Logical flag for array allocation
     real(real32), dimension(:,:), allocatable :: val
     !! Array values in rank 2 (sample, batch)
   contains
     procedure (allocate_array), deferred, pass(this) :: allocate
     !! Abstract procedure for allocating array
     procedure (deallocate_array), deferred, pass(this) :: deallocate
     !! Abstract procedure for deallocating array
     procedure (set_ptr_array), deferred, pass(this) :: set_ptr
     !! Abstract procedure for setting pointers
     procedure, pass(this) :: flatten => flatten_array
     !! Procedure for flattening array
     procedure, pass(this) :: get => get_array
     !! Procedure for getting array
     procedure, pass(this) :: set => set_array
     !! Procedure for setting array
     procedure :: add => add_array
     !! Procedure for adding arrays
     generic, public :: operator(+) => add
     !! Generic for adding arrays
     !  procedure :: assign => assign_array
     !  generic, public :: assignment(=) => assign
  end type array_type

  ! Interface for allocate, deallocate, and flattening array
  !-----------------------------------------------------------------------------
  abstract interface
     module subroutine allocate_array(this, array_shape, source)
       class(array_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array

     pure module subroutine deallocate_array(this, keep_shape)
       class(array_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array

     module subroutine set_ptr_array(this)
       class(array_type), intent(inout), target :: this
     end subroutine set_ptr_array
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

     pure module function add_array(a, b) result(output)
       class(array_type), intent(in) :: a, b
       class(array_type), allocatable :: output
     end function add_array

     module subroutine assign_array(this, input)
      class(array_type), intent(out), target :: this
      class(array_type), intent(in) :: input
    end subroutine assign_array
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
    pure module subroutine deallocate_array1d(this, keep_shape)
      class(array1d_type), intent(inout) :: this
      logical, intent(in), optional :: keep_shape
    end subroutine deallocate_array1d

    pure module subroutine deallocate_array2d(this, keep_shape)
      class(array2d_type), intent(inout) :: this
      logical, intent(in), optional :: keep_shape
    end subroutine deallocate_array2d

    pure module subroutine deallocate_array3d(this, keep_shape)
      class(array3d_type), intent(inout) :: this
      logical, intent(in), optional :: keep_shape
    end subroutine deallocate_array3d

    pure module subroutine deallocate_array4d(this, keep_shape)
      class(array4d_type), intent(inout) :: this
      logical, intent(in), optional :: keep_shape
    end subroutine deallocate_array4d

    pure module subroutine deallocate_array5d(this, keep_shape)
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


!!!-----------------------------------------------------------------------------
!!! facet type (for storing faces, edges, and corners for padding)
!!!-----------------------------------------------------------------------------
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
    integer, dimension(:,:), allocatable :: orig_bound
    !! Original bounds of the facet (nfixed_dims, num)
    integer, dimension(:,:,:), allocatable :: dest_bound
    !! Destination bounds of the facet (2, nfixed_dims, num)
   contains
    procedure, pass(this) :: setup_replication_bounds
    !! Procedure for setting up replication bounds
  end type facets_type

  interface
    !! Interface for setting up replication bounds
    module subroutine setup_replication_bounds(this, length, pad)
      !! Procedure for setting up replication bounds
      class(facets_type), intent(inout) :: this
      !! Instance of the facets type
      integer, dimension(this%rank), intent(in) :: length, pad
      !! Length of the shape and padding
    end subroutine setup_replication_bounds
  end interface
!-------------------------------------------------------------------------------


end module athena__misc_types
