!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module athena
  use network, only: network_type

  !! input layer types
  use input1d_layer,   only: input1d_layer_type
  use input3d_layer,   only: input3d_layer_type
  use input4d_layer,   only: input4d_layer_type

  !! convolution layer types
  use conv2d_layer,    only: conv2d_layer_type, read_conv2d_layer
  use conv3d_layer,    only: conv3d_layer_type, read_conv3d_layer

  !! pooling layer types
  use maxpool2d_layer, only: maxpool2d_layer_type, read_maxpool2d_layer
  use maxpool3d_layer, only: maxpool3d_layer_type, read_maxpool3d_layer

  !! flatten layer types
  use flatten2d_layer, only: flatten2d_layer_type
  use flatten3d_layer, only: flatten3d_layer_type

  !! fully connected (dense) layer types
  use full_layer,      only: full_layer_type, read_full_layer

  implicit none


  public


end module athena
!!!#############################################################################