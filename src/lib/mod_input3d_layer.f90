!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module input3d_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  implicit none
  
  
  type, extends(base_layer_type) :: input3d_layer_type
     integer :: num_outputs
     real(real12), allocatable, dimension(:,:,:) :: output

   contains
     procedure :: forward  => forward_rank
     procedure :: backward => backward_rank
     procedure :: init
  end type input3d_layer_type

  interface input3d_layer_type
     pure module function layer_setup(input_shape) result(layer)
       integer, dimension(:), intent(in) :: input_shape
       type(input3d_layer_type) :: layer
     end function layer_setup
  end interface input3d_layer_type

  
  private
  public :: input3d_layer_type


contains

!!!#############################################################################
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(input3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    return
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(input3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient
    return
  end subroutine backward_rank
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure module function layer_setup(input_shape) result(layer)
    implicit none
    integer, dimension(:), intent(in) :: input_shape

    type(input3d_layer_type) :: layer
    
    allocate(layer%output(input_shape(1),input_shape(2),input_shape(3)))
  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  subroutine init(this, input)
    implicit none
    class(input3d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:), intent(in) :: input

    this%output = input

  end subroutine init
!!!#############################################################################


end module input3d_layer
!!!#############################################################################
