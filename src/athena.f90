!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module athena
  use misc_ml, only: shuffle, split, pad_data
  use random, only: random_setup
  use network, only: network_type
  use metrics, only: metric_dict_type

  !! accuracy methods
  use accuracy, only: categorical_score, mae_score, mse_score, r2_score

  !! optimisation and regularisation types
  use clipper, only: clip_type
  use regulariser, only: &
       base_regulariser_type, &
       l1_regulariser_type, &
       l2_regulariser_type, &
       l1l2_regulariser_type
  use learning_rate_decay, only: &
       base_lr_decay_type, &
       exp_lr_decay_type, &
       step_lr_decay_type, &
       inv_lr_decay_type
  use optimiser, only: &
       base_optimiser_type, &
       sgd_optimiser_type, &
       rmsprop_optimiser_type, &
       adagrad_optimiser_type, &
       adam_optimiser_type

  !! normalisation methods
  use normalisation, only: linear_renormalise, renormalise_norm, renormalise_sum

  !! abstract layer types
  use base_layer, only: &
       base_layer_type, &
       learnable_layer_type, &
       input_layer_type, &
       batch_layer_type, &
       conv_layer_type, &
       pool_layer_type, &
       flatten_layer_type

  !! input layer types
  use input1d_layer,   only: input1d_layer_type
  use input3d_layer,   only: input3d_layer_type
  use input4d_layer,   only: input4d_layer_type

  !! batch normalisation layer types
  use batchnorm1d_layer, only: batchnorm1d_layer_type, read_batchnorm1d_layer
  use batchnorm2d_layer, only: batchnorm2d_layer_type, read_batchnorm2d_layer
  use batchnorm3d_layer, only: batchnorm3d_layer_type, read_batchnorm3d_layer

  !! convolution layer types
  use conv2d_layer,    only: conv2d_layer_type, read_conv2d_layer
  use conv3d_layer,    only: conv3d_layer_type, read_conv3d_layer

  !! dropout layer types
  use dropout_layer, only: dropout_layer_type, read_dropout_layer
  use dropblock2d_layer, only: dropblock2d_layer_type, read_dropblock2d_layer
  use dropblock3d_layer, only: dropblock3d_layer_type, read_dropblock3d_layer

  !! pooling layer types
  use avgpool2d_layer, only: avgpool2d_layer_type, read_avgpool2d_layer
  use avgpool3d_layer, only: avgpool3d_layer_type, read_avgpool3d_layer
  use maxpool2d_layer, only: maxpool2d_layer_type, read_maxpool2d_layer
  use maxpool3d_layer, only: maxpool3d_layer_type, read_maxpool3d_layer

  !! flatten layer types
  use flatten1d_layer, only: flatten1d_layer_type
  use flatten2d_layer, only: flatten2d_layer_type
  use flatten3d_layer, only: flatten3d_layer_type
  use flatten4d_layer, only: flatten4d_layer_type

  !! fully connected (dense) layer types
  use full_layer,      only: full_layer_type, read_full_layer

  implicit none


  public


end module athena
!!!#############################################################################
