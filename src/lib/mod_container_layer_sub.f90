!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! submodule of the container_layer module
!!! submodule contains the associated methods from the container_layer module
!!!#############################################################################
submodule(container_layer) container_layer_submodule
  use base_layer, only: learnable_layer_type
  use custom_types, only: &
       array1d_type, array2d_type, array3d_type, array4d_type, array5d_type

contains
  
  pure module subroutine forward(this, input)
    implicit none
    class(container_layer_type), intent(inout) :: this
    class(container_layer_type), intent(in) :: input

    select type(output => input%layer%output)
    type is (array1d_type)
       call this%layer%forward(output%val_ptr)
    type is (array2d_type)
       call this%layer%forward(output%val_ptr)
    type is (array3d_type)
       call this%layer%forward(output%val_ptr)
    type is (array4d_type)
       call this%layer%forward(output%val_ptr)
    type is (array5d_type)
       call this%layer%forward(output%val_ptr)
    class default
       return
    end select

  end subroutine forward

  
  pure module subroutine backward(this, input, gradient)
    implicit none
    class(container_layer_type), intent(inout) :: this
    class(container_layer_type), intent(in) :: input
    real(real32), dimension(..), intent(in) :: gradient


    select type(output => input%layer%output)
    type is (array1d_type)
       call this%layer%backward(output%val_ptr, gradient)
    type is (array2d_type)
       call this%layer%backward(output%val_ptr, gradient)
    type is (array3d_type)
       call this%layer%backward(output%val_ptr, gradient)
    type is (array4d_type)
       call this%layer%backward(output%val_ptr, gradient)
    type is (array5d_type)
       call this%layer%backward(output%val_ptr, gradient)
    class default
       return
    end select

  end subroutine backward

  
#if defined(GFORTRAN)
  subroutine container_reduction(this, rhs)
    implicit none
    class(container_layer_type), intent(inout) :: this
    class(container_layer_type), intent(in) :: rhs

    select type(layer_this => this%layer)
    class is(learnable_layer_type)
       select type(layer_rhs => rhs%layer)
       class is(learnable_layer_type)
          call layer_this%reduce(layer_rhs)
       end select
    end select

  end subroutine container_reduction
#endif


end submodule container_layer_submodule
!!!#############################################################################
