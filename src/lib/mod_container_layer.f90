!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module container_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  use conv2d_layer, only: conv2d_layer_type
  use maxpool2d_layer, only: maxpool2d_layer_type
  implicit none


!!!------------------------------------------------------------------------
!!! layer container type
!!!------------------------------------------------------------------------
  type :: container_layer_type
     !! conv, full, pool, flat, soft?
     character(4) :: name
     class(base_layer_type), allocatable :: layer !! probably needs to be allocatable
     !! KEEP THE ACTIVATION IN HERE
   contains
     procedure, pass(this) :: forward
     procedure, private, pass(this) :: backward
     
     !generic :: backward => backward_1d, backward_3d !, backward_4D
  end type container_layer_type



  interface
     pure module subroutine forward(this, input) !module?
       !import container_layer_type
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: input
     end subroutine forward
  end interface

  interface backward
     pure module subroutine backward (this, input, gradient)
       !import container_layer_type, real12
       class(container_layer_type), intent(inout) :: this
       class(container_layer_type), intent(in) :: input !! the input to this layer
       real(real12), dimension(..), intent(in) :: gradient
     end subroutine backward
  end interface backward


  private
  public :: container_layer_type


end module container_layer
!!!#############################################################################
