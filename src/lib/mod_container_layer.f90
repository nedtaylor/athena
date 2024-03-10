!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
!!! module contains the container layer type for handling interactions ...
!!! ... between individual layers
!!!##################
!!! module contains the following types:
!!! container_layer_type - type for handling interactions between layers
!!!##################
!!! module contains the following procedures:
!!! forward             - forward pass
!!! backward            - backward pass
!!! container_reduction - reduction of container layers
!!!#############################################################################
module container_layer
  use constants, only: real12
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
       !import container_layer_type
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
       !import container_layer_type, real12
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: input
       real(real12), dimension(..), intent(in) :: gradient
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
       !import container_layer_type, real12
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: rhs
    end subroutine 
  end interface
#endif


  private
  public :: container_layer_type
#if defined(GFORTRAN)
  public :: container_reduction
#endif


end module container_layer
!!!#############################################################################
