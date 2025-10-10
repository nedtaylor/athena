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
  use diffstruc
  implicit none


  private

  public :: activation_type
  public :: initialiser_type
  public :: facets_type

  public :: array_container_type, array_ptr_type
  public :: add, concat


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
     function activation_function_array(this, val) result(output)
       !! Interface for activation function
       import activation_type, real32, array_type
       class(activation_type), intent(in) :: this
       type(array_type), intent(in) :: val
       type(array_type), pointer :: output
     end function activation_function_array

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
! Array container types
!-------------------------------------------------------------------------------
  type :: array_container_type
     class(array_type), allocatable :: array
  end type array_container_type

  type :: array_ptr_type
     type(array_type), pointer :: array(:,:)
  end type array_ptr_type

  ! Operator interfaces
  !-----------------------------------------------------------------------------
  interface add
     module procedure add_array_ptr
  end interface

  interface concat
     module procedure concat_array_ptr
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



contains

!###############################################################################
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
!###############################################################################


!###############################################################################
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
!###############################################################################

end module athena__misc_types
