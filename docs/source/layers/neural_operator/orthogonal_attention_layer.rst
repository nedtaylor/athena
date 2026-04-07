.. _orthogonal-attention-layer:

Orthogonal Attention Layer
==========================

``orthogonal_attention_layer_type``

.. code-block:: fortran

  orthogonal_attention_layer_type(
    num_outputs,
    num_basis,
    key_dim=...,
    num_inputs=...,
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...
  )


The ``orthogonal_attention_layer_type`` derived type provides an orthogonal attention layer.
It uses a learned low-rank orthonormal basis to build an efficient global projection, then combines that with a local bypass:

.. math::

   \mathbf{v} = \sigma\left(\text{Attn}(\mathbf{u}) + \mathbf{W}\,\mathbf{u} + \mathbf{b}\right)

where the attention operation is defined as:

.. math::

   \mathrm{Attn}(\mathbf{u})
   =
   \mathbf{\Phi}
   \left(\mathbf{\Phi}^T \mathbf{W}_Q \mathbf{u}\right)
   \left(\mathbf{\Phi}^T \mathbf{W}_K \mathbf{u}\right)^T
   \mathbf{W}_V \mathbf{u}.

and:

* :math:`\mathbf{u} \in \mathbb{R}^{n_{in}}` is the input sampled on a grid
* :math:`\mathbf{\Phi} \in \mathbb{R}^{n_{in} \times k}` is the learned orthonormal basis obtained from basis weights :math:`\mathbf{B}`
* :math:`\mathbf{W}_Q \in \mathbb{R}^{d_k \times n_{in}}` and :math:`\mathbf{W}_K \in \mathbb{R}^{d_k \times n_{in}}` are the query and key projection weights
* :math:`\mathbf{W}_V \in \mathbb{R}^{n_{out} \times n_{in}}` is the value projection matrix
* :math:`\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}` is the local bypass matrix
* :math:`\mathbf{b} \in \mathbb{R}^{n_{out}}` is the bias vector when ``use_bias=.true.``
* :math:`k` is ``num_basis`` and :math:`d_k` is ``key_dim``
* :math:`\sigma` is the activation function

The current implementation exposes the full orthogonal-attention parameter set, but evaluates the attention path as an orthogonal basis projection :math:`\mathbf{\Phi}\mathbf{\Phi}^T\mathbf{u}` followed by a value projection and local bypass.

Arguments
---------

* **num_outputs** (``integer``): Number of output discretisation points.
* **num_basis** (``integer``): Number of orthogonal basis functions.
* **key_dim** (``integer``): Dimension of the query and key projections. If not provided, it defaults to ``num_basis``.
* **num_inputs** (``integer``): Number of input discretisation points. If not provided, it is inferred when the layer is initialised.
* **use_bias** (``logical``): If ``.false.``, the layer will not use a bias term. Default: ``.true.``.
* **activation** (``class(*)``): Activation function for the layer.

  * Accepts ``character(*)`` or ``class(base_actv_type)``.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Default: ``none_actv_type``.

* **kernel_initialiser** (``class(*)``): Initialiser for :math:`\mathbf{W}_Q`, :math:`\mathbf{W}_K`, :math:`\mathbf{W}_V`, :math:`\mathbf{B}`, and :math:`\mathbf{W}` (see :ref:`Initialisers <initialisers>`).

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

* **W_Q**: Query projection matrix of shape ``(key_dim, num_inputs)``.
* **W_K**: Key projection matrix of shape ``(key_dim, num_inputs)``.
* **W_V**: Value projection matrix of shape ``(num_outputs, num_inputs)``.
* **B**: Basis weight matrix of shape ``(num_inputs, num_basis)``.
* **W**: Local bypass matrix of shape ``(num_outputs, num_inputs)``.
* **b**: Bias vector of shape ``(num_outputs)`` when ``use_bias=.true.``.

The following tensor is derived from the basis weights and rebuilt during forward propagation:

* **Phi**: Orthogonal basis of shape ``(num_inputs, num_basis)``.

Total learnable parameters:

* With bias: ``2 * key_dim * num_inputs + 2 * num_outputs * num_inputs + num_inputs * num_basis + num_outputs``
* Without bias: ``2 * key_dim * num_inputs + 2 * num_outputs * num_inputs + num_inputs * num_basis``

Examples
--------

**Basic orthogonal attention block:**

.. code-block:: fortran

   use athena
   type(network_type) :: network

   call network%add(orthogonal_attention_layer_type( &
        num_inputs=128, &
        num_outputs=128, &
        num_basis=16, &
        key_dim=16, &
        activation="relu" &
   ))

**Orthogonal attention stack with dense readout:**

.. code-block:: fortran

   call network%add(orthogonal_attention_layer_type( &
        num_inputs=256, &
        num_outputs=256, &
        num_basis=32, &
        key_dim=32, &
        activation="swish" &
   ))
   call network%add(orthogonal_attention_layer_type( &
        num_outputs=128, &
        num_basis=16, &
        key_dim=16, &
        activation="swish" &
   ))
   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="none" &
   ))

Notes
-----

* ``num_basis`` controls the rank of the orthogonal projection used to approximate the global interaction.
* ``key_dim`` controls the size of the exposed query and key parameterisation, even though the present forward path uses the orthogonal projection form.
* This layer is useful when you want an operator-style global coupling block without fixing a spectral basis analytically.

See Also
--------

* :ref:`orthogonal_nop_block_type <orthogonal-nop-block>` - Orthogonal neural operator block with spectral mixing on the same learned basis
* :ref:`neural_operator_layer_type <neural-operator-layer>` - Simpler mean-field neural operator layer
* :ref:`laplace_nop_layer_type <laplace-nop-layer>` - Spectral neural operator layer with fixed bases
* :ref:`full_layer_type <full-layer>` - Standard dense layer
