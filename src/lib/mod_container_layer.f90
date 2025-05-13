module athena__container_layer
  !! Module containing types and interfaces for the container type
  !!
  !! This module contains the container layer type which is a container for an
  !! individual layer.
  use athena__constants, only: real32
  use athena__base_layer, only: base_layer_type
  implicit none


  private

  public :: container_layer_type
  public :: read_procedure_container
  public :: list_of_layer_types
  public :: allocate_list_of_layer_types
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


  type :: read_procedure_container
     !! Type containing information needed to read a layer
     character(20) :: name
     !! Name of the layer
     procedure(read_layer), nopass, pointer :: read_ptr => null()
     !! Pointer to the specific layer read function
  end type read_procedure_container
  type(read_procedure_container), dimension(:), allocatable :: &
       list_of_layer_types
  !! List of layer names and their associated read functions

  abstract interface
     module function read_layer(unit, verbose) result(layer)
       !! Read a layer from a file
       integer, intent(in) :: unit
       !! Unit number
       integer, intent(in), optional :: verbose
       !! Verbosity level
       class(base_layer_type), allocatable :: layer
       !! Instance of a layer
     end function read_layer
  end interface

  interface
     module subroutine allocate_list_of_layer_types(addit_list)
       !! Allocate the list of layer types
       type(read_procedure_container), dimension(:), intent(in), optional :: &
            addit_list
       !! Additional list of layer types
     end subroutine allocate_list_of_layer_types
  end interface

  interface
     module subroutine finalise_container_layer(this)
       !! Finalise the container layer
       class(container_layer_type), intent(inout) :: this
       !! Present layer container
     end subroutine finalise_container_layer
  end interface

end module athena__container_layer
