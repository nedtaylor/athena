submodule(athena__container_layer) athena__container_layer_submodule
  !! Submodule containing the implementation for the container layer
  !!
  !! This submodule contains the implementation of the container layer
  !! which is a container for an individual layer.
  !! This also provides the initialisation of the list of layer types
  !! that can be used for reading layers into a network model from a file.
  use athena__base_layer, only: learnable_layer_type
  use athena__actv_layer, only: read_actv_layer, create_from_onnx_actv_layer
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
  use athena__dropout_layer, only: read_dropout_layer, create_from_onnx_dropout_layer
  use athena__duvenaud_msgpass_layer, only: read_duvenaud_msgpass_layer
  use athena__flatten_layer, only: read_flatten_layer, create_from_onnx_flatten_layer
  use athena__full_layer, only: read_full_layer, create_from_onnx_full_layer
  use athena__input_layer, only: read_input_layer, create_from_onnx_input_layer
  use athena__kipf_msgpass_layer, only: read_kipf_msgpass_layer
  use athena__maxpool1d_layer, only: read_maxpool1d_layer
  use athena__maxpool2d_layer, only: read_maxpool2d_layer
  use athena__maxpool3d_layer, only: read_maxpool3d_layer
  use athena__pad1d_layer, only: read_pad1d_layer
  use athena__pad2d_layer, only: read_pad2d_layer
  use athena__pad3d_layer, only: read_pad3d_layer
  use athena__neural_operator_layer, only: read_neural_operator_layer
  use athena__dynamic_lno_layer, only: read_dynamic_lno_layer
  use athena__fixed_lno_layer, only: read_fixed_lno_layer
  use athena__graph_nop_layer, only: read_graph_nop_layer
  use athena__spectral_filter_layer, only: read_spectral_filter_layer
  use athena__orthogonal_attention_layer, only: &
       read_orthogonal_attention_layer
  use athena__orthogonal_nop_block, only: read_orthogonal_nop_block
  use athena__recurrent_layer, only: read_recurrent_layer
  use athena__reshape_layer, only: read_reshape_layer, create_from_onnx_reshape_layer

  use athena__onnx_creators, only: &
       create_from_onnx_avgpool_layer, &
       create_from_onnx_batchnorm_layer, &
       create_from_onnx_conv_layer, &
       create_from_onnx_maxpool_layer, &
       create_from_onnx_pad_layer, &
       create_from_onnx_duvenaud_layer, &
       create_from_onnx_kipf_layer

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
    type(read_layer_container), dimension(:), intent(in), optional :: &
         addit_list


    if(.not.allocated(list_of_layer_types)) allocate(list_of_layer_types(0))
    list_of_layer_types = [ &
         list_of_layer_types, &
         read_layer_container('actv', read_actv_layer), &
         read_layer_container('avgpool1d', read_avgpool1d_layer), &
         read_layer_container('avgpool2d', read_avgpool2d_layer), &
         read_layer_container('avgpool3d', read_avgpool3d_layer), &
         read_layer_container('batchnorm1d', read_batchnorm1d_layer), &
         read_layer_container('batchnorm2d', read_batchnorm2d_layer), &
         read_layer_container('batchnorm3d', read_batchnorm3d_layer), &
         read_layer_container('conv1d', read_conv1d_layer), &
         read_layer_container('conv2d', read_conv2d_layer), &
         read_layer_container('conv3d', read_conv3d_layer), &
         read_layer_container('dropblock2d', read_dropblock2d_layer), &
         read_layer_container('dropblock3d', read_dropblock3d_layer), &
         read_layer_container('dropout', read_dropout_layer), &
         read_layer_container('duvenaud', read_duvenaud_msgpass_layer), &
         read_layer_container('flatten', read_flatten_layer), &
         read_layer_container('full', read_full_layer), &
         read_layer_container('input', read_input_layer), &
         read_layer_container('kipf', read_kipf_msgpass_layer), &
         read_layer_container('maxpool1d', read_maxpool1d_layer), &
         read_layer_container('maxpool2d', read_maxpool2d_layer), &
         read_layer_container('maxpool3d', read_maxpool3d_layer), &
         read_layer_container('neural_operator', read_neural_operator_layer), &
         read_layer_container('fixed_lno', read_fixed_lno_layer), &
         read_layer_container('dynamic_lno', read_dynamic_lno_layer), &
         read_layer_container('graph_nop', read_graph_nop_layer), &
         read_layer_container('spectral_filter', read_spectral_filter_layer), &
         read_layer_container('orthogonal_attention', &
              read_orthogonal_attention_layer), &
         read_layer_container('orthogonal_nop', read_orthogonal_nop_block), &
         read_layer_container('pad1d', read_pad1d_layer), &
         read_layer_container('pad2d', read_pad2d_layer), &
         read_layer_container('pad3d', read_pad3d_layer), &
         read_layer_container('recurrent', read_recurrent_layer), &
         read_layer_container('reshape', read_reshape_layer) &
    ]
    if(present(addit_list))then
       list_of_layer_types = [list_of_layer_types, addit_list]
    end if

  end subroutine allocate_list_of_layer_types

  module subroutine allocate_list_of_onnx_layer_creators(addit_list)
    implicit none
    type(onnx_create_layer_container), dimension(:), intent(in), optional :: &
         addit_list

    ! make a global create_from_onnx_conv_layer that allocates depending on the attributes
    if(.not.allocated(list_of_onnx_layer_creators)) &
         allocate(list_of_onnx_layer_creators(0))
    list_of_onnx_layer_creators = [ &
         list_of_onnx_layer_creators, &
         onnx_create_layer_container('AvgPool', create_from_onnx_avgpool_layer), &
         onnx_create_layer_container('BatchNormalization', &
              create_from_onnx_batchnorm_layer &
         ), &
         onnx_create_layer_container('Conv', create_from_onnx_conv_layer), &
         onnx_create_layer_container('Dropout', create_from_onnx_dropout_layer), &
         onnx_create_layer_container('Flatten', create_from_onnx_flatten_layer), &
         onnx_create_layer_container('Gemm', create_from_onnx_full_layer), &
         onnx_create_layer_container('LeakyRelu', create_from_onnx_actv_layer), &
         onnx_create_layer_container('MatMul', create_from_onnx_full_layer), &
         onnx_create_layer_container('MaxPool', create_from_onnx_maxpool_layer), &
         onnx_create_layer_container('Pad', create_from_onnx_pad_layer), &
         onnx_create_layer_container('Relu', create_from_onnx_actv_layer), &
         onnx_create_layer_container('Reshape', create_from_onnx_reshape_layer), &
         onnx_create_layer_container('Selu', create_from_onnx_actv_layer), &
         onnx_create_layer_container('Sigmoid', create_from_onnx_actv_layer), &
         onnx_create_layer_container('Softmax', create_from_onnx_actv_layer), &
         onnx_create_layer_container('Swish', create_from_onnx_actv_layer), &
         onnx_create_layer_container('Tanh', create_from_onnx_actv_layer) &
    ]
    if(present(addit_list))then
       list_of_onnx_layer_creators = [list_of_onnx_layer_creators, addit_list]
    end if

  end subroutine allocate_list_of_onnx_layer_creators

  module subroutine allocate_list_of_onnx_gnn_layer_creators(addit_list)
    implicit none
    type(onnx_gnn_create_layer_container), &
         dimension(:), intent(in), optional :: addit_list

    if(.not.allocated(list_of_onnx_gnn_layer_creators)) &
         allocate(list_of_onnx_gnn_layer_creators(0))
    list_of_onnx_gnn_layer_creators = [ &
         list_of_onnx_gnn_layer_creators, &
         onnx_gnn_create_layer_container('duvenaud', &
              create_from_onnx_duvenaud_layer), &
         onnx_gnn_create_layer_container('kipf', &
              create_from_onnx_kipf_layer) &
    ]
    if(present(addit_list))then
       list_of_onnx_gnn_layer_creators = &
            [list_of_onnx_gnn_layer_creators, addit_list]
    end if

  end subroutine allocate_list_of_onnx_gnn_layer_creators

end submodule athena__container_layer_submodule
