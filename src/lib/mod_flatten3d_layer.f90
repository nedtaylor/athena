!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module flatten3d_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  implicit none
  
  
  type, extends(base_layer_type) :: flatten3d_layer_type
     integer :: num_outputs
     real(real12), allocatable, dimension(:) :: output
     real(real12), allocatable, dimension(:,:,:,:) :: di
   contains
     procedure :: forward  => forward_rank
     procedure :: backward => backward_rank
  end type flatten3d_layer_type

  interface flatten3d_layer_type
     pure module function layer_setup(input_shape) result(layer)
       integer, dimension(:), intent(in) :: input_shape
       type(flatten3d_layer_type) :: layer
     end function layer_setup
  end interface flatten3d_layer_type

  
  private
  public :: flatten3d_layer_type


contains

!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(flatten3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(4)
       this%output = reshape(input, [this%num_outputs])
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(flatten3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(gradient); rank(1)
       this%di = reshape(gradient, shape(this%di))
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure module function layer_setup(input_shape) result(layer)
    implicit none
    integer, dimension(:), intent(in) :: input_shape

    type(flatten3d_layer_type) :: layer

    allocate(layer%input_shape, source=input_shape)
    layer%num_outputs = size(input_shape)

    allocate(layer%output(layer%num_outputs), source=0._real12)
    allocate(layer%di(&
         input_shape(1), input_shape(2), &
         input_shape(3), input_shape(4)), &
         source=0._real12)

  end function layer_setup
!!!#############################################################################

end module flatten3d_layer
!!!#############################################################################
