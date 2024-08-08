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
module container_layer
  use constants, only: real32
  use base_layer, only: base_layer_type
  implicit none


!!!------------------------------------------------------------------------
!!! layer container type
!!!------------------------------------------------------------------------
  type :: container_layer_type
     !! inpt, batc, conv, drop, full, pool, flat
     character(4) :: name
     class(base_layer_type), allocatable :: layer
   contains
     procedure, pass(this) :: forward
     procedure, pass(this) :: backward

#if defined(GFORTRAN)
     procedure, pass(this) :: reduce => container_reduction
#endif
  end type container_layer_type


  interface
     !!-----------------------------------------------------
     !! forward pass
     !!-----------------------------------------------------
     !! this  = (T, io) present layer container
     !! input = (T, in) input layer container
     pure module subroutine forward(this, input)
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: input
     end subroutine forward
  end interface

  interface
     !!-----------------------------------------------------
     !! forward pass
     !!-----------------------------------------------------
     !! this     = (T, in) present layer container
     !! input    = (T, in) input layer container
     !! gradient = (R, in) backpropagated gradient
     pure module subroutine backward(this, input, gradient)
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: input
       real(real32), dimension(..), intent(in) :: gradient
     end subroutine backward
  end interface

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
     character(4) :: name
     procedure(read_layer), nopass, pointer :: read_ptr => null()
  end type read_procedure_container
  type(read_procedure_container), dimension(:), allocatable :: &
       list_of_layer_types

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


  private
  public :: container_layer_type
  public :: list_of_layer_types
#if defined(GFORTRAN)
  public :: container_reduction
#endif


end module container_layer
!!!#############################################################################
