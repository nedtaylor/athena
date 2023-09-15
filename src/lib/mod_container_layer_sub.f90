!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
submodule(container_layer) container_layer_submodule
  use input1d_layer, only: input1d_layer_type
  use input3d_layer, only: input3d_layer_type
  use input4d_layer, only: input4d_layer_type
  use conv2d_layer, only: conv2d_layer_type
  use conv3d_layer, only: conv3d_layer_type
  use dropout_layer, only: dropout_layer_type
  use dropblock2d_layer, only: dropblock2d_layer_type
  use dropblock3d_layer, only: dropblock3d_layer_type
  use maxpool2d_layer, only: maxpool2d_layer_type
  use maxpool3d_layer, only: maxpool3d_layer_type
  use flatten2d_layer, only: flatten2d_layer_type
  use flatten3d_layer, only: flatten3d_layer_type
  use full_layer, only: full_layer_type

contains
  
  pure module subroutine forward(this, input)
    implicit none
    class(container_layer_type), intent(inout) :: this
    class(container_layer_type), intent(in) :: input

    select type(previous => input%layer)
    type is(input1d_layer_type)
       call this%layer%forward(previous%output)
    type is(input3d_layer_type)
       call this%layer%forward(previous%output)
    type is(input4d_layer_type)
       call this%layer%forward(previous%output)

    type is(conv2d_layer_type)
       call this%layer%forward(previous%output)
    type is(conv3d_layer_type)
       call this%layer%forward(previous%output)

    type is(dropout_layer_type)
       call this%layer%forward(previous%output)
    type is(dropblock2d_layer_type)
       call this%layer%forward(previous%output)
    type is(dropblock3d_layer_type)
       call this%layer%forward(previous%output)

    type is(maxpool2d_layer_type)
       call this%layer%forward(previous%output)
    type is(maxpool3d_layer_type)
       call this%layer%forward(previous%output)

    type is(flatten2d_layer_type)
       call this%layer%forward(previous%output)
    type is(flatten3d_layer_type)
       call this%layer%forward(previous%output)

    type is(full_layer_type)
       call this%layer%forward(previous%output)
    end select

  end subroutine forward

  
  pure module subroutine backward(this, input, gradient)
    implicit none
    class(container_layer_type), intent(inout) :: this
    class(container_layer_type), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select type(previous => input%layer)
    type is(input1d_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(input3d_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(input4d_layer_type)
       call this%layer%backward(previous%output, gradient)

    type is(conv2d_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(conv3d_layer_type)
       call this%layer%backward(previous%output, gradient)

    type is(dropout_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(dropblock2d_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(dropblock3d_layer_type)
       call this%layer%backward(previous%output, gradient)

    type is(maxpool2d_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(maxpool3d_layer_type)
       call this%layer%backward(previous%output, gradient)

    type is(flatten2d_layer_type)
       call this%layer%backward(previous%output, gradient)
    type is(flatten3d_layer_type)
       call this%layer%backward(previous%output, gradient)

    type is(full_layer_type)
       call this%layer%backward(previous%output, gradient)
    end select

  end subroutine backward

  
  subroutine container_reduction(this, rhs)
    implicit none
    class(container_layer_type), intent(inout) :: this
    class(container_layer_type), intent(in) :: rhs

    write(*,*) "GOT HERE"

  end subroutine container_reduction


end submodule container_layer_submodule
!!!#############################################################################
