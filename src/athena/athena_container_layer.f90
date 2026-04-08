module athena__container_layer
  !! Module containing types and interfaces for the container type
  !!
  !! This module contains the container layer type which is a container for an
  !! individual layer.
  use coreutils, only: real32
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  implicit none


  private

  public :: container_layer_type
  public :: read_layer_container
  public :: list_of_layer_types
  public :: allocate_list_of_layer_types
  public :: onnx_create_layer_container
  public :: list_of_onnx_layer_creators
  public :: allocate_list_of_onnx_layer_creators
  public :: onnx_gnn_create_layer_container
  public :: list_of_onnx_gnn_layer_creators
  public :: allocate_list_of_onnx_gnn_layer_creators
  public :: onnx_nop_create_layer_container
  public :: list_of_onnx_nop_layer_creators
  public :: allocate_list_of_onnx_nop_layer_creators
  public :: onnx_expanded_nop_create_layer_container
  public :: list_of_onnx_expanded_nop_layer_creators
  public :: allocate_list_of_onnx_expanded_nop_layer_creators
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

  abstract interface
     function create_gnn_from_onnx_layer(meta_key, meta_value, inits, verbose) &
          result(layer)
       !! Create a GNN layer from ONNX metadata and initialisers
       import :: base_layer_type, onnx_initialiser_type
       character(*), intent(in) :: meta_key
       !! GNN metadata key (e.g. "athena_gnn_node_1")
       character(*), intent(in) :: meta_value
       !! Semicolon-separated GNN metadata value string
       type(onnx_initialiser_type), dimension(:), intent(in) :: inits
       !! ONNX initialisers (valid slice only)
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       class(base_layer_type), allocatable :: layer
       !! Constructed GNN layer
     end function create_gnn_from_onnx_layer
  end interface

  type :: onnx_gnn_create_layer_container
     !! Type containing information needed to create a GNN layer from ONNX
     character(20) :: gnn_subtype
     !! GNN subtype name (e.g. "duvenaud", "kipf")
     procedure(create_gnn_from_onnx_layer), nopass, pointer :: &
          create_ptr => null()
     !! Pointer to the GNN layer creation function
  end type onnx_gnn_create_layer_container
  type(onnx_gnn_create_layer_container), dimension(:), allocatable :: &
       list_of_onnx_gnn_layer_creators
  !! List of GNN subtype names and their associated ONNX creation functions

  type :: onnx_nop_create_layer_container
     !! Type containing information needed to create a NOP layer from ONNX
     character(30) :: nop_subtype
     !! NOP subtype name (e.g. "dynamic_lno", "fixed_lno")
     procedure(create_gnn_from_onnx_layer), nopass, pointer :: &
          create_ptr => null()
     !! Pointer to the NOP layer creation function
  end type onnx_nop_create_layer_container
  type(onnx_nop_create_layer_container), dimension(:), allocatable :: &
       list_of_onnx_nop_layer_creators
  !! List of NOP subtype names and their associated ONNX creation functions

  abstract interface
     logical function classify_onnx_expanded_nop_layer(prefix, nodes, &
          num_nodes)
       !! Return true when this creator handles the given
       !! expanded-ONNX NOP prefix.
       import :: onnx_node_type
       character(*), intent(in) :: prefix
       !! Expanded-ONNX layer prefix (e.g. "layer1")
       type(onnx_node_type), intent(in) :: nodes(:)
       !! Parsed ONNX nodes
       integer, intent(in) :: num_nodes
       !! Number of valid node entries
     end function classify_onnx_expanded_nop_layer

     function build_onnx_expanded_nop_layer( &
          prefix, nodes, num_nodes, inits, &
          num_inits) result(layer)
       !! Build one expanded-ONNX NOP layer from a node cluster.
       import :: base_layer_type, onnx_node_type, onnx_initialiser_type
       character(*), intent(in) :: prefix
       !! Expanded-ONNX layer prefix (e.g. "layer1")
       type(onnx_node_type), intent(in) :: nodes(:)
       !! Parsed ONNX nodes
       integer, intent(in) :: num_nodes
       !! Number of valid node entries
       type(onnx_initialiser_type), intent(in) :: inits(:)
       !! Parsed ONNX initialisers
       integer, intent(in) :: num_inits
       !! Number of valid initialiser entries
       class(base_layer_type), allocatable :: layer
       !! Constructed layer
     end function build_onnx_expanded_nop_layer
  end interface

  type :: onnx_expanded_nop_create_layer_container
     !! Registration entry for one expanded-ONNX NOP layer type
     character(30) :: nop_subtype
     !! Subtype name used for diagnostics (e.g. "dynamic_lno")
     procedure(classify_onnx_expanded_nop_layer), nopass, pointer :: &
          classify_ptr => null()
     !! Pointer to the classifier that recognises this layer type
     procedure(build_onnx_expanded_nop_layer), nopass, pointer :: &
          build_ptr => null()
     !! Pointer to the builder that constructs the layer
  end type onnx_expanded_nop_create_layer_container
  type(onnx_expanded_nop_create_layer_container), dimension(:), allocatable :: &
       list_of_onnx_expanded_nop_layer_creators
  !! List of expanded-ONNX NOP creators registered for pattern-matched ONNX
  !! import

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

     module function create_from_onnx_layer( &
          nodes, initialisers, value_info, verbose &
     ) result(layer)
       !! Create a layer from ONNX nodes and initialisers
       type(onnx_node_type), intent(in) :: nodes
       !! ONNX nodes
       type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
       !! ONNX initialisers
       type(onnx_tensor_type), dimension(:), intent(in) :: value_info
       !! ONNX value info
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

     module subroutine allocate_list_of_onnx_gnn_layer_creators(addit_list)
       !! Allocate the list of GNN ONNX layer creation procedures
       type(onnx_gnn_create_layer_container), &
            dimension(:), intent(in), optional :: addit_list
       !! Additional list of GNN ONNX layer creation procedures
     end subroutine allocate_list_of_onnx_gnn_layer_creators

     module subroutine allocate_list_of_onnx_nop_layer_creators(addit_list)
       !! Allocate the list of NOP ONNX layer creation procedures
       type(onnx_nop_create_layer_container), &
            dimension(:), intent(in), optional :: addit_list
       !! Additional list of NOP ONNX layer creation procedures
     end subroutine allocate_list_of_onnx_nop_layer_creators

     module subroutine allocate_list_of_onnx_expanded_nop_layer_creators( &
          addit_list)
       !! Allocate the list of expanded-ONNX NOP layer creation procedures
       type(onnx_expanded_nop_create_layer_container), &
            dimension(:), intent(in), optional :: addit_list
       !! Additional list of expanded-ONNX NOP layer creation procedures
     end subroutine allocate_list_of_onnx_expanded_nop_layer_creators
  end interface

  interface
     module subroutine finalise_container_layer(this)
       !! Finalise the container layer
       class(container_layer_type), intent(inout) :: this
       !! Present layer container
     end subroutine finalise_container_layer
  end interface

end module athena__container_layer
