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
     integer, allocatable, dimension(:) :: input_shape
     real(real12), allocatable, dimension(:) :: output
     real(real12), allocatable, dimension(:,:,:,:) :: di ! gradient of input (i.e. delta)

   contains
     procedure :: forward  => forward_rank
     procedure :: backward => backward_rank
     procedure :: init
  end type flatten3d_layer_type



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
  subroutine init(this, input_shape)
    implicit none
    class(flatten3d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape

    allocate(this%input_shape, source=input_shape)
    this%num_outputs = size(input_shape)

    allocate(this%output(this%num_outputs))
    allocate(this%di(&
         input_shape(1), input_shape(2), &
         input_shape(3), input_shape(4)))

  end subroutine init
!!!#############################################################################


end module flatten3d_layer
!!!#############################################################################
