.. _conv1d-layer:

1D Convolutional Layer
======================

``conv1d_layer_type``

.. code-block:: fortran

  conv1d_layer_type(
    num_filters,
    kernel_size,
    input_shape=...,
    stride=1,
    dilation=1,
    padding="valid",
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...
  )


The ``conv1d_layer_type`` derived type provides a 1D convolutional layer.
The operation performed by this layer applies a 1D convolution over an input signal composed of several input planes.

Arguments
---------

* **num_filters** (`integer`): Number of output filters/channels in the convolution.
* **kernel_size** (`integer` or `integer, dimension(1)`): Size of the convolving kernel.
* **input_shape** (`integer, dimension(:)`): Shape of the input data (channels, width).
* **stride** (`integer` or `integer, dimension(1)`): Stride of the convolution. Default: ``1``.
* **dilation** (`integer` or `integer, dimension(1)`): Spacing between kernel elements. Default: ``1``.
* **padding** (`character(*)`): Padding method, if any, to be applied to the input data prior to convolution. Refer to :ref:`1D padding layer <pad1d-layer>` for options. Default: ``"valid"``, i.e. no padding.
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

Shape:
------

* Input: ``(width, in_channels, batch_size)``.
* Output: ``(width_out, num_filters, batch_size)``.

where:

.. math::

   \text{width_out} = \left\lfloor \frac{\text{width} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1 \right\rfloor
