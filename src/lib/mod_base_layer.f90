!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module base_layer
  use constants, only: real12
  implicit none

!!!------------------------------------------------------------------------
!!! layer abstract type
!!!------------------------------------------------------------------------
  type, abstract :: base_layer_type !! give it parameterised values?

   contains
     procedure(initialise), deferred, pass(this) :: init
     procedure(forward), deferred, pass(this) :: forward
     procedure(backward), deferred, pass(this) :: backward
     !! NO NEED FOR DEFERRED PRODECURES
     !! instead, make this a generic type that just has a set of interfaces for (module) procedures that call 1D, 3D, and 4D forms
     !! Use subroutines because output data is trickier for function tricker to handle
     !! Use a general train subroutine that is called by the main model, which internally goes through forward and backward passes
     !! Input sizes have to be 1D, 3D, or 4D (any 2D data is simply 3D with num_channels=1)
     !! Output sizes defined by user
     !! For every forward, just pass in the whole previous layer container
     !! ... reverse for backward
     !! In each layer container, you know what size you are expecting for the input, so just take that based on a select type (or of a previous?)
     !! only some extended types have update
  end type base_layer_type
  !! is it a foward subroutine, or a function whose result is the output?

  abstract interface
     subroutine initialise(this, input_shape)
       import :: base_layer_type
       class(base_layer_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: input_shape
     end subroutine initialise
  end interface

  abstract interface
     pure subroutine forward(this, input)
       import :: base_layer_type, real12
       class(base_layer_type), intent(inout) :: this
       real(real12), dimension(..), intent(in) :: input
     end subroutine forward

     pure subroutine backward(this, input, gradient)
       import :: base_layer_type, real12
       class(base_layer_type), intent(inout) :: this
       real(real12), dimension(..), intent(in) :: input
       real(real12), dimension(..), intent(in) :: gradient
     end subroutine backward
  end interface



  public :: base_layer_type


end module base_layer
!!!#############################################################################
