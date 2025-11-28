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

  public :: activation_type
  public :: initialiser_type
  public :: facets_type
  public :: onnx_attribute_type, onnx_node_type, onnx_initialiser_type



!-------------------------------------------------------------------------------
! Activation (transfer) function base type
!-------------------------------------------------------------------------------
  type, abstract :: activation_type
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
     procedure (activation_function), deferred, pass(this) :: activate
     !! Abstract procedure for 5D derivative of activation function
  end type activation_type

  ! Interface for activation function
  !-----------------------------------------------------------------------------
  abstract interface
     function activation_function(this, val) result(output)
       !! Interface for activation function
       import activation_type, real32, array_type
       class(activation_type), intent(in) :: this
       type(array_type), intent(in) :: val
       type(array_type), pointer :: output
     end function activation_function
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


!------------------------------------------------------------------------------!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!------------------------------------------------------------------------------!


!-------------------------------------------------------------------------------
! Attributes type (for ONNX export)
!-------------------------------------------------------------------------------
  type :: onnx_attribute_type
     !! Type for storing attributes for ONNX export
     character(20) :: name
     !! Name of the attribute
     character(20) :: type
     !! Type of the attribute (e.g. 'int', 'float', 'string')
     character(len=:), allocatable :: value
     !! Value of the attribute as a string
     !! This allows for flexible storage of different types
     !! of attributes without needing to define a specific type
  end type onnx_attribute_type
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! ONNX node type
!-------------------------------------------------------------------------------
  type :: onnx_node_type
     character(256) :: op_type
     character(20) :: name
     character(20), allocatable, dimension(:) :: inputs
     character(20), allocatable, dimension(:) :: outputs
     type(onnx_attribute_type), allocatable, dimension(:) :: attributes
     integer :: num_inputs, num_outputs
  end type onnx_node_type
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! ONNX initialiser type
!-------------------------------------------------------------------------------
  type :: onnx_initialiser_type
     character(20) :: name
     integer, allocatable, dimension(:) :: dims
     real(real32), allocatable, dimension(:) :: data
  end type onnx_initialiser_type
!-------------------------------------------------------------------------------

end module athena__misc_types
