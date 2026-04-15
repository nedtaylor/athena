.. _neural-operator-layer:

Neural Operator Layer
=====================

``neural_operator_layer_type``

.. code-block:: fortran

  neural_operator_layer_type(
    num_outputs,
    num_inputs=...,
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...
  )


The ``neural_operator_layer_type`` derived type provides a simple neural operator layer for discretised functions.
It combines a local affine transform with a global mean-field correction term:

.. math::

   \mathbf{v} = \sigma\left(\mathbf{W}\mathbf{u} + \mathbf{w}_k\langle\mathbf{u}\rangle + \mathbf{b}\right)

where:

* :math:`\mathbf{u} \in \mathbb{R}^{n_{in}}` is the input function sampled on a grid
* :math:`\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}` is the local weight matrix
* :math:`\mathbf{w}_k \in \mathbb{R}^{n_{out}}` is the global integral-kernel coupling vector
* :math:`\langle\mathbf{u}\rangle = \frac{1}{n_{in}}\sum_j u_j` is the mean of the input sample
* :math:`\mathbf{b} \in \mathbb{R}^{n_{out}}` is the bias vector when ``use_bias=.true.``
* :math:`\sigma` is the activation function

This gives a rank-1 approximation to a non-local integral operator while retaining a standard dense mapping for local interactions.

Arguments
---------

* **num_outputs** (``integer``): Number of output discretisation points.
* **num_inputs** (``integer``): Number of input discretisation points. If not provided, it is inferred when the layer is initialised.
* **use_bias** (``logical``): If ``.false.``, the layer will not use a bias term. Default: ``.true.``.
* **activation** (``class(*)``): Activation function for the layer.

  * Accepts ``character(*)`` or ``class(base_actv_type)``.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Default: ``none_actv_type``.

* **kernel_initialiser** (``class(*)``): Initialiser for the local weights :math:`\mathbf{W}` and kernel weights :math:`\mathbf{w}_k` (see :ref:`Initialisers <initialisers>`).

  * If ``activation`` is ``selu_actv_type``, default: ``lecun_normal_init_type``.
  * If ``activation`` is a version of ``relu_actv_type``, default: ``he_normal_init_type``.
  * For all other activations, default: ``glorot_uniform_init_type``.

* **bias_initialiser** (``class(*)``): Initialiser for the biases (see :ref:`Initialisers <initialisers>`). Default: ``zeros_init_type``.

Shape
-----

* Input: ``(num_inputs, batch_size)``.
* Output: ``(num_outputs, batch_size)``.

Parameters
----------

The layer contains the following learnable parameters:

* **W**: Local weight matrix of shape ``(num_outputs, num_inputs)``.
* **W_k**: Global kernel coupling vector of shape ``(num_outputs, 1)``.
* **b**: Bias vector of shape ``(num_outputs)`` when ``use_bias=.true.``.

Total parameters:

* With bias: ``num_outputs * (num_inputs + 2)``
* Without bias: ``num_outputs * (num_inputs + 1)``

Examples
--------

**Basic neural operator block:**

.. code-block:: fortran

   use athena
   type(network_type) :: network

   call network%add(neural_operator_layer_type( &
        num_inputs=64, &
        num_outputs=64, &
        activation="relu" &
   ))

**Stacked neural operator network:**

.. code-block:: fortran

   call network%add(neural_operator_layer_type( &
        num_inputs=128, &
        num_outputs=128, &
        activation="swish" &
   ))
   call network%add(neural_operator_layer_type( &
        num_outputs=128, &
        activation="swish" &
   ))
   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="none" &
   ))

Notes
-----

* The non-local term uses the mean of each input sample, so it captures only global average information.
* This layer is most useful as a lightweight neural-operator building block or baseline.
* Because the global correction is rank-1, it is cheaper than spectral operator layers but less expressive.

See Also
--------

* :ref:`fixed_lno_layer_type <fixed-lno-layer>` - Laplace neural operator layer with fixed encoder/decoder bases and spectral mixing
* :ref:`dynamic_lno_layer_type <dynamic-lno-layer>` - More flexible Laplace neural operator layer with learnable bases
* :ref:`full_layer_type <full-layer>` - Standard dense layer
