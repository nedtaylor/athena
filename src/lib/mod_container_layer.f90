!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains the container layer type for handling interactions ...
!!! ... between individual layers
!!!##################
!!! module contains the following derived types:
!!! - container_layer_type - type for handling interactions between layers
!!!##################
!!! module contains the following procedures:
!!! - forward             - forward pass
!!! - backward            - backward pass
!!! - container_reduction - reduction of container layers
!!!#############################################################################
module athena__container_layer
  use athena__constants, only: real32
  use athena__base_layer, only: base_layer_type
  implicit none


!!!------------------------------------------------------------------------
!!! layer container type
!!!------------------------------------------------------------------------
  type :: container_layer_type
     !! inpt, batc, conv, drop, full, pool, flat
     character(4) :: name
     class(base_layer_type), allocatable :: layer
   contains
#if defined(GFORTRAN)
     procedure, pass(this) :: reduce => container_reduction
#endif
  end type container_layer_type


#if defined(GFORTRAN)
  interface
    !!-----------------------------------------------------
    !! forward pass
    !!-----------------------------------------------------
    !! this = (T, io) present layer container
    !! rhs  = (T, in) input layer container
    module subroutine container_reduction(this, rhs)
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: rhs
    end subroutine 
  end interface
#endif


!!!-----------------------------------------------------------------------------
!!! list of layer types
!!!-----------------------------------------------------------------------------
  type :: read_procedure_container
     character(20) :: name
     procedure(read_layer), nopass, pointer :: read_ptr => null()
  end type read_procedure_container
  type(read_procedure_container), dimension(:), allocatable :: &
       list_of_layer_types
  !!! ACTUALLY ALLOCATE THIS SOMEWHERE IN THE CODE (i.e. whenever you initialise a network?)
  !!! Then, anyone who wants to add layers can just append to this list using:
  !!!    list_of_layer_types = [list_of_layer_types, new_layer]
  !!! Actually, populate it in the read procedure. If not yet allocated, allocate it and add the layers

  abstract interface
     !!-----------------------------------------------------
     !! read layer type
     !!-----------------------------------------------------
     !! this = (T, io) present layer container
     !! unit = (I, in) unit number
     !! verbose = (I, in) verbosity level
     module function read_layer(unit, verbose) result(layer)
       class(base_layer_type), allocatable :: layer
       integer, intent(in) :: unit
       integer, intent(in), optional :: verbose 
     end function read_layer
  end interface

  interface
     module subroutine allocate_list_of_layer_types(addit_list)
       type(read_procedure_container), dimension(:), intent(in), optional :: &
            addit_list
     end subroutine allocate_list_of_layer_types
  end interface


  private
  public :: container_layer_type
  public :: read_procedure_container
  public :: list_of_layer_types
  public :: allocate_list_of_layer_types
#if defined(GFORTRAN)
  public :: container_reduction
#endif


end module athena__container_layer
!!!#############################################################################
