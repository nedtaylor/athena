!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains all publicly available types and procedures within the ...
!!! ... ATHENA library
!!! No other types or procedures should be needed to be accessed by the user
!!!#############################################################################
module athena
  use athena__io_utils, only: print_version, print_build_info
  use athena__misc_ml, only: shuffle, split, pad_data
  use athena__random, only: random_setup
  use athena__network, only: network_type
  use athena__metrics, only: metric_dict_type
  use graphstruc, only: graph_type, edge_type, vertex_type

  !! accuracy methods
  use athena__accuracy, only: categorical_score, mae_score, mse_score, r2_score

  !! optimisation and regularisation types
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

  !! normalisation methods
  use athena__normalisation, only: linear_renormalise, renormalise_norm, renormalise_sum

  !! abstract layer types
  use athena__base_layer, only: &
       base_layer_type, &
       learnable_layer_type, &
       batch_layer_type, &
       conv_layer_type, &
       pool_layer_type

  !! input layer types
  use athena__input_layer,   only: input_layer_type

  !! batch normalisation layer types
  use athena__batchnorm1d_layer, only: batchnorm1d_layer_type, read_batchnorm1d_layer
  use athena__batchnorm2d_layer, only: batchnorm2d_layer_type, read_batchnorm2d_layer
  use athena__batchnorm3d_layer, only: batchnorm3d_layer_type, read_batchnorm3d_layer

  !! convolution layer types
  use athena__conv1d_layer,    only: conv1d_layer_type, read_conv1d_layer
  use athena__conv2d_layer,    only: conv2d_layer_type, read_conv2d_layer
  use athena__conv3d_layer,    only: conv3d_layer_type, read_conv3d_layer

!   !! deep set layer types
!   use athena__deepset_layer,   only: deepset_layer_type, read_deepset_layer

  !! message passing layer types
  use athena__mpnn_layer,      only: mpnn_layer_type
  use athena__conv_mpnn_layer, only: conv_mpnn_layer_type

  !! dropout layer types
  use athena__dropout_layer, only: dropout_layer_type, read_dropout_layer
  use athena__dropblock2d_layer, only: dropblock2d_layer_type, read_dropblock2d_layer
  use athena__dropblock3d_layer, only: dropblock3d_layer_type, read_dropblock3d_layer

  !! pooling layer types
  use athena__avgpool2d_layer, only: avgpool2d_layer_type, read_avgpool2d_layer
  use athena__avgpool3d_layer, only: avgpool3d_layer_type, read_avgpool3d_layer
  use athena__maxpool2d_layer, only: maxpool2d_layer_type, read_maxpool2d_layer
  use athena__maxpool3d_layer, only: maxpool3d_layer_type, read_maxpool3d_layer

  !! flatten layer types
  use athena__flatten_layer, only: flatten_layer_type

  !! fully connected (dense) layer types
  use athena__full_layer,      only: full_layer_type, read_full_layer

  use athena__misc_types, only: &
       array_container_type, &
       array_type, &
       array1d_type, &
       array2d_type, &
       array3d_type, &
       array4d_type, &
       array5d_type

  implicit none


  public


end module athena
!!!#############################################################################
