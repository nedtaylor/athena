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


The ``orthogonal_attention_layer_type`` derived type provides a stabilised orthogonal attention layer.
It uses a learned low-rank orthonormal basis to construct a global spectral representation, applies
normalised per-mode attention in that basis, and combines this with a local bypass:

.. math::

   \mathbf{v} = \sigma\left(\text{Attn}(\mathbf{u}) + \mathbf{W}\,\mathbf{u} + \mathbf{b}\right)

The attention operation is defined in three stages: projection to an orthogonal basis, stable
attention weighting, and reconstruction:

.. math::

   \mathbf{c} = \mathbf{\Phi}^T \mathbf{u}
   \quad \in \mathbb{R}^{k}

.. math::

   \mathbf{q} = \mathbf{W}_Q \mathbf{u}, \qquad
   \mathbf{k} = \mathbf{W}_K \mathbf{u}

.. math::

   \mathbf{a} = \mathrm{softmax}\!\left(
   \tanh\!\left(
   \frac{\mathbf{q} \odot \mathbf{k}}{\sqrt{d_k}}
   \right)
   \right)
   \quad \in \mathbb{R}^{k}

.. math::

   \tilde{\mathbf{c}} = \mathbf{c} + \mathbf{a} \odot \mathbf{c}

.. math::

   \text{Attn}(\mathbf{u}) =
   \mathbf{W}_V \left( \mathbf{\Phi} \tilde{\mathbf{c}} \right)

where:

* :math:`\mathbf{u} \in \mathbb{R}^{n_{in}}` is the input sampled on a grid
* :math:`\mathbf{\Phi} \in \mathbb{R}^{n_{in} \times k}` is the learned orthonormal basis obtained from basis weights :math:`\mathbf{B}`
* :math:`\mathbf{c}` are spectral coefficients in the orthogonal basis
* :math:`\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_k \times n_{in}}` are the query and key projection weights
* :math:`\mathbf{a} \in \mathbb{R}^{k}` are normalised per-basis attention weights
* :math:`\mathbf{W}_V \in \mathbb{R}^{n_{out} \times n_{in}}` is the value projection matrix
* :math:`\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}` is the local bypass matrix
* :math:`\mathbf{b} \in \mathbb{R}^{n_{out}}` is the bias vector when ``use_bias=.true.``
* :math:`k` is ``num_basis`` and :math:`d_k` is ``key_dim``
* :math:`\odot` denotes element-wise multiplication
* :math:`\sigma` is the activation function

This formulation differs from a standard dot-product attention mechanism in that attention is applied directly to orthogonal spectral coefficients rather than pairwise token interactions.
The use of bounded interactions (:math:`\tanh`) and softmax normalisation ensures numerical stability, while the residual spectral update preserves information across basis modes.

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
* :ref:`fixed_lno_layer_type <fixed-lno-layer>` - Laplace neural operator layer with fixed encoder/decoder bases and spectral mixing
* :ref:`full_layer_type <full-layer>` - Standard dense layer
