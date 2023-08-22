!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module base_layer
  use constants, only: real12
  use optimiser, only: optimiser_type
  implicit none

!!!------------------------------------------------------------------------
!!! layer abstract type
!!!------------------------------------------------------------------------
  type, abstract :: base_layer_type !! give it parameterised values?
     integer, allocatable, dimension(:) :: input_shape, output_shape
   contains
     procedure, pass(this) :: print => print_base
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


!!!-----------------------------------------------------------------------------
!!! input derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: input_layer_type
     integer :: num_outputs
   contains
     procedure(set), deferred, pass(this) :: set
  end type input_layer_type

  abstract interface
     pure subroutine set(this, input)
       import :: input_layer_type, real12
       class(input_layer_type), intent(inout) :: this
       real(real12), dimension(this%num_outputs), intent(in) :: input
     end subroutine set
  end interface


!!!-----------------------------------------------------------------------------
!!! learnable derived extended type
!!!-----------------------------------------------------------------------------
  type, abstract, extends(base_layer_type) :: learnable_layer_type
     character(len=14) :: kernel_initialiser, bias_initialiser
   contains
     procedure(update), deferred, pass(this) :: update
  end type learnable_layer_type

  abstract interface
     pure subroutine update(this, optimiser, batch_size)
       import :: learnable_layer_type, optimiser_type
       class(learnable_layer_type), intent(inout) :: this
       type(optimiser_type), intent(in) :: optimiser
       integer, optional, intent(in) :: batch_size
     end subroutine update
  end interface


  private

  public :: base_layer_type
  public :: input_layer_type
  public :: learnable_layer_type


contains

!!!#############################################################################
!!! print layer to file (do nothing for a base layer)
!!!#############################################################################
  subroutine print_base(this, file)
    implicit none
    class(base_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    !! NO NEED TO WRITE ANYTHING FOR A DEFAULT LAYER
    return
  end subroutine print_base
!!!#############################################################################

end module base_layer
!!!#############################################################################
