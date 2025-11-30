.. _conv3d-layer:

3D Convolutional Layer
======================

``conv3d_layer_type``

.. code-block:: fortran

  conv3d_layer_type(
    num_filters,
    kernel_size,
    input_shape=...,
    stride=1,
    dilation=1,
    padding="valid",
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...,
    batch_size=...
  )


The ``conv3d_layer_type`` derived type provides a 3D convolutional layer.
The operation performed by this layer applies a 3D convolution over an input signal composed of several input planes.

Arguments
---------

* **num_filters** (`integer`): Number of output filters/channels in the convolution.
* **kernel_size** (`integer` or `integer, dimension(3)`): Size of the convolving kernel. If a single integer is provided, the same value is used for depth, height, and width.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (widht, height, depth, channels).
* **stride** (`integer` or `integer, dimension(3)`): Stride of the convolution. Default: ``1``.
* **dilation** (`integer` or `integer, dimension(3)`): Spacing between kernel elements. Default: ``1``.
* **padding** (`character(*)`): Padding method.

  * ``"valid"``: No padding (default).
  * ``"same"``: Padding to maintain spatial dimensions.

* **use_bias** (`logical`): If ``.false.``, the layer will not use a bias term. Default: ``.true.``.
* **activation** (`class(*)`): Activation function for the layer.

  * Accepts `character(*)` or `class(base_actv_type)`.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Default: ``none_actv_type``.

* **kernel_initialiser** (`class(*)`): Initialiser for the kernel weights (see :ref:`Initialisers <initialisers>`).

  * If ``activation`` is ``selu_actv_type``, default: ``lecun_normal_init_type``.
  * If ``activation`` is a version of ``relu_actv_type``, default: ``he_normal_init_type``.
  * For all other activations, default: ``glorot_uniform_init_type``.

* **bias_initialiser** (`class(*)`): Initialiser for the biases (see :ref:`Initialisers <initialisers>`). Default: ``zeros_init_type``.
* **batch_size** (`integer`): **SOON TO BE DEPRECATED**. Batch size for the layer. Handled automatically during training and inference.

Shape:
------

* Input: ``(width, height, depth, in_channels, batch_size)``.
* Output: ``(width_out, height_out, depth_out, num_filters, batch_size)``.

where:

.. math::

   \text{width_out} &= \left\lfloor \frac{\text{width} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1 \right\rfloor
   \text{height_out} &= \left\lfloor \frac{\text{height} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1 \right\rfloor \\
   \text{depth_out} &= \left\lfloor \frac{\text{depth} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1 \right\rfloor \\
