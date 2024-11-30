!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! submodule of the container_layer module
!!! submodule contains the associated methods from the container_layer module
!!!#############################################################################
submodule(athena__container_layer) athena__container_layer_submodule
  use athena__base_layer, only: learnable_layer_type

contains

  
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


end submodule athena__container_layer_submodule
!!!#############################################################################
