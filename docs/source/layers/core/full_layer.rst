Fully Connected Layer
=====================

`full_layer_type`
.. code-block:: fortran

  full_layer_type(
    num_outputs,
    num_inputs,
    num_addit_inputs,
    batch_size,
    activation_function="none",
    activation_scale=1.0,
    kernel_initialiser=empty,
    bias_initialiser=empty
  )


The `full_layer_type` derived type provides a fully-connected (aka dense) layer.
The layer contains `num_inputs` number of input features.

This layer creates a fully- (densely-) connected layer, a standard neural network layer.

Arguments
---------

* **num_outputs**: Positive integer. Number of output neurons (dimensionality of output space).
* **num_inputs**: Integer. Number of input neurons. Defaults to number of outputs of previous layer.
* **num_addit_inputs**: Positive integer. Number of additional inputs to this layer that have been exempt from entering previous layers.
* **batch_size**: Integer. The number samples in a batch. This is optional (the enclosing network structure can handle it instead).
* **activation_function**: Activation function for the layer (see [Activation Functions](activation-functions)).
* **activation_scale**: Real scalar. Defaults to `1.0`.
* **kernel_initialiser**: Initialiser for the kernel weights (see [Initialisers](initialisers)).
* **bias_initialiser**: Initialiser for the biases (see [Initialisers](initialisers)).
