!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module input3d_layer
  use constants, only: real12
  use base_layer, only: input_layer_type
  implicit none
  
  
  type, extends(input_layer_type) :: input3d_layer_type
     real(real12), allocatable, dimension(:,:,:) :: output

   contains
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, pass(this) :: init => init_3d
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
    
    layer%input_shape = input_shape
    layer%output_shape = input_shape
    allocate(layer%output(input_shape(1),input_shape(2),input_shape(3)))
    layer%num_outputs = product(input_shape)
  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!!#############################################################################
  pure subroutine init_3d(this, input)
    implicit none
    class(input3d_layer_type), intent(inout) :: this
    real(real12), dimension(this%num_outputs), intent(in) :: input

    this%output = reshape(input, shape=shape(this%output))
  end subroutine init_3d
!!!#############################################################################


end module input3d_layer
!!!#############################################################################
