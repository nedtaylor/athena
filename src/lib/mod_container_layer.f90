!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module container_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  implicit none


!!!------------------------------------------------------------------------
!!! layer container type
!!!------------------------------------------------------------------------
  type :: container_layer_type
     !! inpt, conv, full, pool, flat?
     character(4) :: name
     class(base_layer_type), allocatable :: layer
   contains
     procedure, pass(this) :: forward
     procedure, pass(this) :: backward
  end type container_layer_type



  interface
     pure module subroutine forward(this, input)
       !import container_layer_type
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: input
     end subroutine forward
  end interface

  interface
     pure module subroutine backward(this, input, gradient)
       !import container_layer_type, real12
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: input
       real(real12), dimension(..), intent(in) :: gradient
     end subroutine backward
  end interface


  private
  public :: container_layer_type


end module container_layer
!!!#############################################################################
