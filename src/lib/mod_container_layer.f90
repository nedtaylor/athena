module athena__container_layer
  !! Module containing types and interfaces for the container type
  !!
  !! This module contains the container layer type which is a container for an
  !! individual layer.
  use coreutils, only: real32
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: onnx_node_type, onnx_initialiser_type
  implicit none


  private

  public :: container_layer_type
  public :: read_layer_container
  public :: list_of_layer_types
  public :: allocate_list_of_layer_types
  public :: onnx_create_layer_container
  public :: list_of_onnx_layer_creators
  public :: allocate_list_of_onnx_layer_creators
#if defined(GFORTRAN)
  public :: container_reduction
#endif


  type :: container_layer_type
     !! Container for a layer
     class(base_layer_type), allocatable :: layer
     !! Layer
   contains
#if defined(GFORTRAN)
     procedure, pass(this) :: reduce => container_reduction
     !! Reduce two layers via summation
     final :: finalise_container_layer
     !! Finalise the container layer
#endif
  end type container_layer_type


#if defined(GFORTRAN)
  interface
     module subroutine container_reduction(this, rhs)
       !! Reduce two layers via summation
       class(container_layer_type), intent(inout) :: this
       !! Present layer container
       class(container_layer_type), intent(in) :: rhs
       !! Input layer container
     end subroutine
  end interface
#endif


  type :: read_layer_container
     !! Type containing information needed to read a layer
     character(20) :: name
     !! Name of the layer
     procedure(read_layer), nopass, pointer :: read_ptr => null()
     !! Pointer to the specific layer read function
  end type read_layer_container
  type(read_layer_container), dimension(:), allocatable :: &
       list_of_layer_types
  !! List of layer names and their associated read functions

  type :: onnx_create_layer_container
     !! Type containing information needed to create a layer from ONNX
     character(20) :: op_type
     !! Name of the layer
     procedure(create_from_onnx_layer), nopass, pointer :: create_ptr => null()
     !! Pointer to the specific layer creation function from ONNX
  end type onnx_create_layer_container
  type(onnx_create_layer_container), dimension(:), allocatable :: &
       list_of_onnx_layer_creators
  !! List of layer names and their associated ONNX creation functions

  interface
     module function read_layer(unit, verbose) result(layer)
       !! Read a layer from a file
       integer, intent(in) :: unit
       !! Unit number
       integer, intent(in), optional :: verbose
       !! Verbosity level
       class(base_layer_type), allocatable :: layer
       !! Instance of a layer
     end function read_layer

     module function create_from_onnx_layer(nodes, initialisers, verbose) result(layer)
       !! Create a layer from ONNX nodes and initialisers
       type(onnx_node_type), intent(in) :: nodes
       !! ONNX nodes
       type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
       !! ONNX initialisers
       integer, intent(in), optional :: verbose
       !! Verbosity level
       class(base_layer_type), allocatable :: layer
       !! Instance of a layer
     end function create_from_onnx_layer
  end interface

  interface
     module subroutine allocate_list_of_layer_types(addit_list)
       !! Allocate the list of layer types
       type(read_layer_container), dimension(:), intent(in), optional :: &
            addit_list
       !! Additional list of layer types
     end subroutine allocate_list_of_layer_types

     module subroutine allocate_list_of_onnx_layer_creators(addit_list)
       !! Allocate the list of ONNX layer creation procedures
       type(onnx_create_layer_container), dimension(:), intent(in), optional :: &
            addit_list
       !! Additional list of ONNX layer creation procedures
     end subroutine allocate_list_of_onnx_layer_creators
  end interface

  interface
     module subroutine finalise_container_layer(this)
       !! Finalise the container layer
       class(container_layer_type), intent(inout) :: this
       !! Present layer container
     end subroutine finalise_container_layer
  end interface

end module athena__container_layer
