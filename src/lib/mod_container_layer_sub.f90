!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
submodule(container_layer) container_layer_submodule
  use conv2d_layer, only: conv2d_layer_type
  use maxpool2d_layer, only: maxpool2d_layer_type
  use full_layer, only: full_layer_type

contains
  
  pure module subroutine forward(this, input) !module?
    implicit none
    class(container_layer_type), intent(inout) :: this
    class(container_layer_type), intent(in) :: input

    select type(previous => input%layer)
    type is(conv2d_layer_type)
       call this%layer%forward(previous%output)
    type is(maxpool2d_layer_type)
       call this%layer%forward(previous%output)
    type is(full_layer_type)
       call this%layer%forward(previous%output)
    end select

  end subroutine forward

  
  pure module subroutine backward(this, input, gradient)
    implicit none
    class(container_layer_type), intent(inout) :: this
    class(container_layer_type), intent(in) :: input !! the input to this layer
    real(real12), dimension(..), intent(in) :: gradient

    select type(previous => input%layer)
    type is(conv2d_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(maxpool2d_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(full_layer_type)
       call this%layer%backward(previous%output, gradient)
    end select

  end subroutine backward


end submodule container_layer_submodule
!!!#############################################################################
