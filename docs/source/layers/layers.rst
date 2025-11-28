Layers
======

The athena library provides a variety of layers commonly used in building neural networks.
These layers can be stacked and combined to create complex architectures for various machine learning tasks.

Available Layers
----------------

The following layers are currently implemented in the athena library:

* **Core layers**
* **Convolutional layers**
* **Input layer**
* **Normalisation layers**
* **Padding layers**
* **Pooling layers**
* **Regularisation layers**
* **Reshaping layer**

List of Layers
--------------

Core Layer
~~~~~~~~~~

* `activation_layer_type` - Applies an activation function element-wise to the input data
* `full_layer_type` - Standard fully connected (dense) layer

Convolutional Layers
~~~~~~~~~~~~~~~~~~~~

* `conv1d_layer_type` - 1D convolutional layer
* `conv2d_layer_type` - 2D convolutional layer
* `conv3d_layer_type` - 3D convolutional layer


Input Layer
~~~~~~~~~~~

* `input_layer_type` - Defines the input shape for the network (can be automated)

Normalisation Layers
~~~~~~~~~~~~~~~~~~~~

* `batchnorm1d_layer_type` - 1D batch normalization layer
* `batchnorm2d_layer_type` - 2D batch normalization layer
* `batchnorm3d_layer_type` - 3D batch normalization layer

Padding Layers
~~~~~~~~~~~~~~

* `pad1d_layer_type` - 1D padding layer
* `pad2d_layer_type` - 2D padding layer
* `pad3d_layer_type` - 3D padding layer

Pooling Layers
~~~~~~~~~~~~~~

* `avgpool1d_layer_type` - 1D average pooling layer
* `avgpool2d_layer_type` - 2D average pooling layer
* `avgpool3d_layer_type` - 3D average pooling layer
* `maxpool1d_layer_type` - 1D max pooling layer
* `maxpool2d_layer_type` - 2D max pooling layer
* `maxpool3d_layer_type` - 3D max pooling layer

Regularisation Layers
~~~~~~~~~~~~~~~~~~~~~

* `dropout_layer_type` - Dropout layer for regularisation
* `dropblock2d_layer_type` - 2D DropBlock layer for regularisation
* `dropblock3d_layer_type` - 3D DropBlock layer for regularisation

Reshaping Layer
~~~~~~~~~~~~~~~

* `flatten_layer_type` - Flattens the input data to a 1D array (can be automated)
