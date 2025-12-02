.. _full-layer:

Fully-Connected Layer
=====================

``full_layer_type``

.. code-block:: fortran

  full_layer_type(
    num_outputs,
    num_inputs=...,
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...
  )


The ``full_layer_type`` derived type provides a fully-connected (aka dense) layer.
The operation performed by this layer is given by:

.. math::

   \text{output} = \text{activation}(\text{input} \cdot W + b)

where :math:`W` is the weight matrix, :math:`b` is the bias vector (if used), and :math:`\text{activation}` is the activation function applied element-wise to the output.

Arguments
---------

* **num_outputs** (`integer`): Size of each output sample
* **num_inputs** (`integer`): Size of each input sample
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

* Input: ``(num_inputs, batch_size)``.
* Output: ``(num_outputs, batch_size)``.
