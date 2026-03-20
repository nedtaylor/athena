module athena
  !! Module containing all publicly available types and procedures
  !!
  !! This module contains all publicly available types and procedures within the
  !! ATHENA library.
  !! No other types or procedures should be needed to be accessed by the user

  !! Libraries
  !!----------------------------------------------------------------------------
  use athena__io_utils, only: print_version, print_build_info, athena__version__
  use athena__misc_ml, only: shuffle, split, pad_data
  use athena__random, only: random_setup
  use athena__network, only: network_type
  use athena__metrics, only: metric_dict_type
  use graphstruc, only: graph_type, edge_type, vertex_type
  use athena__onnx, only: write_onnx, read_onnx


  ! Accuracy methods
  !-----------------------------------------------------------------------------
  use athena__accuracy, only: categorical_score, mae_score, mse_score, r2_score


  ! Optimisation, loss, and regularisation types
  !-----------------------------------------------------------------------------
  use athena__clipper, only: clip_type
  use athena__regulariser, only: &
       base_regulariser_type, &
       l1_regulariser_type, &
       l2_regulariser_type, &
       l1l2_regulariser_type
  use athena__learning_rate_decay, only: &
       base_lr_decay_type, &
       exp_lr_decay_type, &
       step_lr_decay_type, &
       inv_lr_decay_type
  use athena__optimiser, only: &
       base_optimiser_type, &
       sgd_optimiser_type, &
       rmsprop_optimiser_type, &
       adagrad_optimiser_type, &
       adam_optimiser_type
  use athena__loss, only: &
       base_loss_type, &
       bce_loss_type, &
       cce_loss_type, &
       mae_loss_type, &
       mse_loss_type, &
       nll_loss_type, &
       huber_loss_type


  ! Normalisation methods
  !-----------------------------------------------------------------------------
  use athena__normalisation, only: &
       linear_renormalise, &
       renormalise_norm, &
       renormalise_sum


  ! Activation functions (aka transfer functions)
  !-----------------------------------------------------------------------------
  use athena__activation_gaussian, only: gaussian_actv_type
  use athena__activation_leaky_relu, only: leaky_relu_actv_type
  use athena__activation_linear, only: linear_actv_type
  use athena__activation_none,   only: none_actv_type
  use athena__activation_piecewise, only: piecewise_actv_type
  use athena__activation_relu,   only: relu_actv_type
  use athena__activation_selu,   only: selu_actv_type
  use athena__activation_sigmoid,only: sigmoid_actv_type
  use athena__activation_softmax, only: softmax_actv_type
  use athena__activation_swish,  only: swish_actv_type
  use athena__activation_tanh,   only: tanh_actv_type

  ! Initialisation methods
  !-----------------------------------------------------------------------------
  use athena__misc_types, only: base_init_type
  use athena__initialiser_data, only: data_init_type
  use athena__initialiser_gaussian, only: gaussian_init_type
  use athena__initialiser_he, only: he_uniform_init_type, he_normal_init_type
  use athena__initialiser_glorot, only: &
       glorot_uniform_init_type, glorot_normal_init_type
  use athena__initialiser_ident, only: ident_init_type
  use athena__initialiser_lecun, only: &
       lecun_uniform_init_type, lecun_normal_init_type
  use athena__initialiser_ones, only: ones_init_type
  use athena__initialiser_zeros, only: zeros_init_type


  ! Abstract layer types
  !-----------------------------------------------------------------------------
  use athena__base_layer, only: &
       base_layer_type, &
       learnable_layer_type, &
       batch_layer_type, &
       conv_layer_type, &
       pool_layer_type, &
       merge_layer_type


  ! Input layer types
  !-----------------------------------------------------------------------------
  use athena__input_layer,   only: input_layer_type


  ! Activation layer types
  !-----------------------------------------------------------------------------
  use athena__actv_layer, only: actv_layer_type, read_actv_layer


  ! Batch normalisation layer types
  !-----------------------------------------------------------------------------
  use athena__batchnorm1d_layer, only: &
       batchnorm1d_layer_type, read_batchnorm1d_layer
  use athena__batchnorm2d_layer, only: &
       batchnorm2d_layer_type, read_batchnorm2d_layer
  use athena__batchnorm3d_layer, only: &
       batchnorm3d_layer_type, read_batchnorm3d_layer


  ! Convolution layer types
  !-----------------------------------------------------------------------------
  use athena__conv1d_layer,    only: conv1d_layer_type, read_conv1d_layer
  use athena__conv2d_layer,    only: conv2d_layer_type, read_conv2d_layer
  use athena__conv3d_layer,    only: conv3d_layer_type, read_conv3d_layer


! ! !   ! Deep set layer types
! ! !   !-----------------------------------------------------------------------------
! ! !   use athena__deepset_layer,   only: deepset_layer_type, read_deepset_layer


  ! Dropout layer types
  !-----------------------------------------------------------------------------
  use athena__dropout_layer, only: dropout_layer_type, read_dropout_layer
  use athena__dropblock2d_layer, only: &
       dropblock2d_layer_type, read_dropblock2d_layer
  use athena__dropblock3d_layer, only: &
       dropblock3d_layer_type, read_dropblock3d_layer


  ! Padding layer types
  !-----------------------------------------------------------------------------
  use athena__pad1d_layer, only: pad1d_layer_type, read_pad1d_layer
  use athena__pad2d_layer, only: pad2d_layer_type, read_pad2d_layer
  use athena__pad3d_layer, only: pad3d_layer_type, read_pad3d_layer


  ! Pooling layer types
  !-----------------------------------------------------------------------------
  use athena__avgpool1d_layer, only: avgpool1d_layer_type, read_avgpool1d_layer
  use athena__avgpool2d_layer, only: avgpool2d_layer_type, read_avgpool2d_layer
  use athena__avgpool3d_layer, only: avgpool3d_layer_type, read_avgpool3d_layer
  use athena__maxpool1d_layer, only: maxpool1d_layer_type, read_maxpool1d_layer
  use athena__maxpool2d_layer, only: maxpool2d_layer_type, read_maxpool2d_layer
  use athena__maxpool3d_layer, only: maxpool3d_layer_type, read_maxpool3d_layer


  ! Reshape layer types
  !-----------------------------------------------------------------------------
  use athena__flatten_layer, only: flatten_layer_type, read_flatten_layer
  use athena__reshape_layer, only: reshape_layer_type, read_reshape_layer


  ! Fully connected (dense) layer types
  !-----------------------------------------------------------------------------
  use athena__full_layer,      only: full_layer_type, read_full_layer


  ! Neural operator layer types
  !-----------------------------------------------------------------------------
  use athena__neural_operator_layer, only: &
       neural_operator_layer_type, read_neural_operator_layer
  use athena__dynamic_lno_layer, only: dynamic_lno_layer_type, read_dynamic_lno_layer
  use athena__fixed_lno_layer, only: fixed_lno_layer_type, read_fixed_lno_layer
  use athena__graph_nop_layer, only: &
       graph_nop_layer_type, read_graph_nop_layer
  use athena__spectral_filter_layer, only: &
       spectral_filter_layer_type, read_spectral_filter_layer
  use athena__orthogonal_attention_layer, only: &
       orthogonal_attention_layer_type, read_orthogonal_attention_layer
  use athena__orthogonal_nop_block, only: &
       orthogonal_nop_block_type, read_orthogonal_nop_block


  ! KAN layer types
  !-----------------------------------------------------------------------------
  use athena__kan_layer,       only: kan_layer_type, read_kan_layer


  ! Merge layer types
  !-----------------------------------------------------------------------------
  use athena__add_layer, only: add_layer_type, read_add_layer
  use athena__concat_layer, only: concat_layer_type, read_concat_layer

  ! Message passing layer types
  !-----------------------------------------------------------------------------
  use athena__msgpass_layer,      only: msgpass_layer_type
  use athena__kipf_msgpass_layer, only: kipf_msgpass_layer_type
  use athena__duvenaud_msgpass_layer, only: duvenaud_msgpass_layer_type

  ! Recurrent layer types
  !-----------------------------------------------------------------------------
  use athena__recurrent_layer, only: recurrent_layer_type

  ! Array types
  !-----------------------------------------------------------------------------
  use diffstruc
  use athena__diffstruc_extd, only: array_ptr_type


  ! List of layer types
  !-----------------------------------------------------------------------------
  use athena__container_layer, only: &
       list_of_layer_types, &
       allocate_list_of_layer_types, &
       list_of_onnx_layer_creators, &
       allocate_list_of_onnx_layer_creators, &
       read_layer_container, onnx_create_layer_container


  ! wandb logging (optional)
  !-----------------------------------------------------------------------------
#ifdef _WANDB
  use athena_wandb, only: wandb_network_type
#endif


  implicit none


  public


end module athena
