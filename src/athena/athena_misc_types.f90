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
  use coreutils, only: real32
  use diffstruc, only: array_type
  implicit none


  private

  public :: base_actv_type
  public :: base_init_type
  public :: facets_type
  public :: onnx_attribute_type, onnx_node_type, onnx_initialiser_type, &
       onnx_tensor_type






!-------------------------------------------------------------------------------
! Attributes type (for ONNX export)
!-------------------------------------------------------------------------------
  type :: onnx_attribute_type
     !! Type for storing attributes for ONNX export
     character(64), allocatable :: name
     !! Name of the attribute
     character(10), allocatable :: type
     !! Type of the attribute (e.g. 'int', 'float', 'string')
     character(len=:), allocatable :: val
     !! Value of the attribute as a string
     !! This allows for flexible storage of different types
     !! of attributes without needing to define a specific type
  end type onnx_attribute_type

  interface onnx_attribute_type
     pure module function create_attribute(name, type, val) result(attribute)
       !! Function to create an ONNX attribute
       character(*), intent(in) :: name
       !! Name of the attribute
       character(*), intent(in) :: type
       !! Type of the attribute
       character(len=*), intent(in) :: val
       !! Value of the attribute as a string
       type(onnx_attribute_type) :: attribute
       !! Resulting ONNX attribute
     end function create_attribute
  end interface
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! ONNX node type
!-------------------------------------------------------------------------------
  type :: onnx_node_type
     character(128) :: op_type = ''
     character(128) :: name = ''
     character(128), allocatable, dimension(:) :: inputs
     character(128), allocatable, dimension(:) :: outputs
     type(onnx_attribute_type), allocatable, dimension(:) :: attributes
     character(4096) :: attributes_json = ''
     !! Pre-formatted JSON for attributes block (empty = no attributes)
     integer :: num_inputs = 0
     integer :: num_outputs = 0
  end type onnx_node_type
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! ONNX initialiser type
!-------------------------------------------------------------------------------
  type :: onnx_initialiser_type
     character(128) :: name = ''
     integer :: data_type = 1
     !! 1=float32, 7=int64
     integer, allocatable, dimension(:) :: dims
     real(real32), allocatable, dimension(:) :: data
     integer, allocatable, dimension(:) :: int_data
  end type onnx_initialiser_type
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! ONNX tensor type
!-------------------------------------------------------------------------------
  type :: onnx_tensor_type
     character(128) :: name = ''
     integer :: elem_type = 1
     integer, allocatable, dimension(:) :: dims
     character(64), allocatable, dimension(:) :: dim_params
     !! If dim_params(i) /= '', it is a symbolic dimension name
  end type onnx_tensor_type
!-------------------------------------------------------------------------------



!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!



!-------------------------------------------------------------------------------
! Activation (aka transfer) function base type
!-------------------------------------------------------------------------------
  type, abstract :: base_actv_type
     !! Abstract type for activation functions
     character(10) :: name
     !! Name of the activation function
     real(real32) :: scale = 1._real32
     !! Scale of the activation function
     real(real32) :: threshold
     !! Threshold of the activation function
     logical :: apply_scaling = .false.
     !! Boolean to apply scaling or not
   contains
     procedure (apply_actv), deferred, pass(this) :: apply
     !! Abstract procedure for 5D derivative of activation function
     procedure(reset_actv), deferred, pass(this) :: reset
     !! Reset activation function attributes and variables
     procedure(apply_attributes_actv), deferred, pass(this) :: apply_attributes
     !! Set up ONNX attributes
     procedure(export_attributes_actv), deferred, pass(this) :: export_attributes
     !! Export ONNX attributes
     procedure, pass(this) :: print_to_unit => print_to_unit_actv
  end type base_actv_type

  ! Interface for activation function
  !-----------------------------------------------------------------------------
  abstract interface
     subroutine reset_actv(this)
       !! Interface for resetting activation function attributes and variables
       import base_actv_type
       class(base_actv_type), intent(inout) :: this
       !! Instance of the activation type
     end subroutine reset_actv

     function apply_actv(this, val) result(output)
       !! Interface for activation function
       import base_actv_type, real32, array_type
       class(base_actv_type), intent(in) :: this
       type(array_type), intent(in) :: val
       type(array_type), pointer :: output
     end function apply_actv

     subroutine apply_attributes_actv(this, attributes)
       !! Interface for loading ONNX attributes
       import base_actv_type, onnx_attribute_type
       class(base_actv_type), intent(inout) :: this
       !! Instance of the activation type
       type(onnx_attribute_type), dimension(:), intent(in) :: attributes
       !! ONNX attributes
     end subroutine apply_attributes_actv

     pure function export_attributes_actv(this) result(attributes)
       !! Interface for exporting ONNX attributes
       import base_actv_type, onnx_attribute_type
       class(base_actv_type), intent(in) :: this
       !! Instance of the activation type
       type(onnx_attribute_type), allocatable, dimension(:) :: attributes
     end function export_attributes_actv
  end interface

  interface
     module subroutine print_to_unit_actv(this, unit, identifier)
       !! Interface for printing activation function details
       class(base_actv_type), intent(in) :: this
       !! Instance of the activation type
       integer, intent(in) :: unit
       !! Unit number for output
       character(len=*), intent(in), optional :: identifier
       !! Optional identifier for the activation function
     end subroutine print_to_unit_actv
  end interface
!-------------------------------------------------------------------------------


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Weights and biases initialiser base type
!-------------------------------------------------------------------------------
  type, abstract :: base_init_type
     !! Abstract type for initialising weights and biases
     character(len=20) :: name
     !! Name of the initialiser
     real(real32) :: scale = 1._real32, mean = 1._real32, std = 0.01_real32
     !! Scale, mean, and standard deviation of the initialiser
   contains
     procedure (initialiser_subroutine), deferred, pass(this) :: initialise
     !! Abstract procedure for initialising weights and biases
  end type base_init_type

  ! Interface for initialiser function
  !-----------------------------------------------------------------------------
  abstract interface
     !! Interface for initialiser function
     subroutine initialiser_subroutine(this, input, fan_in, fan_out, spacing)
       !! Interface for initialiser function
       import base_init_type, real32
       class(base_init_type), intent(inout) :: this
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

end module athena__misc_types
