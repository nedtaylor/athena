submodule(athena__container_layer) athena__container_layer_submodule
  !! Submodule containing the implementation for the container layer
  !!
  !! This submodule contains the implementation of the container layer
  !! which is a container for an individual layer.
  !! This also provides the initialisation of the list of layer types
  !! that can be used for reading layers into a network model from a file.
  use athena__base_layer, only: learnable_layer_type
  use athena__actv_layer, only: read_actv_layer
  use athena__avgpool1d_layer, only: read_avgpool1d_layer
  use athena__avgpool2d_layer, only: read_avgpool2d_layer
  use athena__avgpool3d_layer, only: read_avgpool3d_layer
  use athena__batchnorm1d_layer, only: read_batchnorm1d_layer
  use athena__batchnorm2d_layer, only: read_batchnorm2d_layer
  use athena__batchnorm3d_layer, only: read_batchnorm3d_layer
  use athena__conv1d_layer, only: read_conv1d_layer
  use athena__conv2d_layer, only: read_conv2d_layer
  use athena__conv3d_layer, only: read_conv3d_layer
  use athena__dropblock2d_layer, only: read_dropblock2d_layer
  use athena__dropblock3d_layer, only: read_dropblock3d_layer
  use athena__dropout_layer, only: read_dropout_layer
  use athena__duvenaud_msgpass_layer, only: read_duvenaud_msgpass_layer
  use athena__flatten_layer, only: read_flatten_layer
  use athena__full_layer, only: read_full_layer
  use athena__input_layer, only: read_input_layer
  use athena__kipf_msgpass_layer, only: read_kipf_msgpass_layer
  use athena__maxpool1d_layer, only: read_maxpool1d_layer
  use athena__maxpool2d_layer, only: read_maxpool2d_layer
  use athena__maxpool3d_layer, only: read_maxpool3d_layer
  use athena__pad1d_layer, only: read_pad1d_layer
  use athena__pad2d_layer, only: read_pad2d_layer
  use athena__pad3d_layer, only: read_pad3d_layer

contains

  module subroutine finalise_container_layer(this)
    !! Finalise the container layer
    implicit none
    class(container_layer_type), intent(inout) :: this

    if (allocated(this%layer)) deallocate(this%layer)

  end subroutine finalise_container_layer

!###############################################################################

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


  module subroutine allocate_list_of_layer_types(addit_list)
    implicit none
    type(read_procedure_container), dimension(:), intent(in), optional :: &
         addit_list


    allocate(list_of_layer_types(0))
    list_of_layer_types = [ &
         list_of_layer_types, &
         read_procedure_container('actv', read_actv_layer), &
         read_procedure_container('avgpool1d', read_avgpool1d_layer), &
         read_procedure_container('avgpool2d', read_avgpool2d_layer), &
         read_procedure_container('avgpool3d', read_avgpool3d_layer), &
         read_procedure_container('batchnorm1d', read_batchnorm1d_layer), &
         read_procedure_container('batchnorm2d', read_batchnorm2d_layer), &
         read_procedure_container('batchnorm3d', read_batchnorm3d_layer), &
         read_procedure_container('conv1d', read_conv1d_layer), &
         read_procedure_container('conv2d', read_conv2d_layer), &
         read_procedure_container('conv3d', read_conv3d_layer), &
         read_procedure_container('dropblock2d', read_dropblock2d_layer), &
         read_procedure_container('dropblock3d', read_dropblock3d_layer), &
         read_procedure_container('dropout', read_dropout_layer), &
         read_procedure_container('duvenaud', read_duvenaud_msgpass_layer), &
         read_procedure_container('flatten', read_flatten_layer), &
         read_procedure_container('full', read_full_layer), &
         read_procedure_container('input', read_input_layer), &
         read_procedure_container('kipf', read_kipf_msgpass_layer), &
         read_procedure_container('maxpool1d', read_maxpool1d_layer), &
         read_procedure_container('maxpool2d', read_maxpool2d_layer), &
         read_procedure_container('maxpool3d', read_maxpool3d_layer), &
         read_procedure_container('pad1d', read_pad1d_layer), &
         read_procedure_container('pad2d', read_pad2d_layer), &
         read_procedure_container('pad3d', read_pad3d_layer) &
    ]
    if (present(addit_list)) then
       list_of_layer_types = [list_of_layer_types, addit_list]
    end if

  end subroutine allocate_list_of_layer_types


end submodule athena__container_layer_submodule
