.. _orthogonal-nop-block:

Orthogonal Neural Operator Block
================================

``orthogonal_nop_block_type``

.. code-block:: fortran

  orthogonal_nop_block_type(
    num_outputs,
    num_basis,
    num_inputs=...,
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...
  )


The ``orthogonal_nop_block_type`` derived type provides an orthogonal neural operator block.
It combines a learned orthonormal basis, a spectral mixing path, and a local bypass:

.. math::

   \mathbf{v} = \sigma\left(\mathbf{W}\,\mathbf{\Phi}\,\mathbf{R}\,\mathbf{\Phi}^T\,\mathbf{u} + \mathbf{W}\,\mathbf{u} + \mathbf{b}\right)

where:

* :math:`\mathbf{u} \in \mathbb{R}^{n_{in}}` is the input sampled on a grid
* :math:`\mathbf{\Phi} \in \mathbb{R}^{n_{in} \times k}` is the learned orthonormal basis obtained from basis weights :math:`\mathbf{B}`
* :math:`\mathbf{R} \in \mathbb{R}^{k \times k}` is the learnable spectral mixing matrix
* :math:`\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}` is the learnable bypass and output projection matrix
* :math:`\mathbf{b} \in \mathbb{R}^{n_{out}}` is the bias vector when ``use_bias=.true.``
* :math:`k` is ``num_basis``
* :math:`\sigma` is the activation function

The basis matrix :math:`\mathbf{\Phi}` is formed by orthogonalising the learnable basis weights with modified Gram-Schmidt. This gives a low-rank operator block whose non-local interaction scales with the chosen basis size rather than the full grid resolution.

Arguments
---------

* **num_outputs** (``integer``): Number of output discretisation points.
* **num_basis** (``integer``): Number of orthogonal basis functions.
* **num_inputs** (``integer``): Number of input discretisation points. If not provided, it is inferred when the block is initialised.
* **use_bias** (``logical``): If ``.false.``, the block will not use a bias term. Default: ``.true.``.
* **activation** (``class(*)``): Activation function for the block.

  * Accepts ``character(*)`` or ``class(base_actv_type)``.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Default: ``none_actv_type``.

* **kernel_initialiser** (``class(*)``): Initialiser for the spectral matrix :math:`\mathbf{R}`, basis weights :math:`\mathbf{B}`, and bypass weights :math:`\mathbf{W}` (see :ref:`Initialisers <initialisers>`).

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

The block contains the following learnable parameters:

* **R**: Spectral mixing matrix of shape ``(num_basis, num_basis)``.
* **B**: Basis weight matrix of shape ``(num_inputs, num_basis)``.
* **W**: Local bypass and output projection matrix of shape ``(num_outputs, num_inputs)``.
* **b**: Bias vector of shape ``(num_outputs)`` when ``use_bias=.true.``.

The following tensors are derived from the basis weights and rebuilt during forward propagation:

* **Phi**: Orthogonal basis of shape ``(num_inputs, num_basis)``.
* **Phi_T**: Transposed orthogonal basis of shape ``(num_basis, num_inputs)``.

Total learnable parameters:

* With bias: ``num_basis * num_basis + num_inputs * num_basis + num_outputs * num_inputs + num_outputs``
* Without bias: ``num_basis * num_basis + num_inputs * num_basis + num_outputs * num_inputs``

Examples
--------

**Basic orthogonal neural operator block:**

.. code-block:: fortran

   use athena
   type(network_type) :: network

   call network%add(orthogonal_nop_block_type( &
        num_inputs=128, &
        num_outputs=128, &
        num_basis=16, &
        activation="relu" &
   ))

**Stacked orthogonal operator block:**

.. code-block:: fortran

   call network%add(orthogonal_nop_block_type( &
        num_inputs=256, &
        num_outputs=256, &
        num_basis=32, &
        activation="swish" &
   ))
   call network%add(orthogonal_nop_block_type( &
        num_outputs=128, &
        num_basis=16, &
        activation="swish" &
   ))
   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="none" &
   ))

**Heat-equation operator example:**

.. code-block:: fortran

   call network%add(orthogonal_nop_block_type( &
        num_inputs=N_grid, num_outputs=N_hidden, &
        num_basis=k_basis, activation="relu"))
   call network%add(orthogonal_nop_block_type( &
        num_outputs=N_grid, &
        num_basis=k_basis))

Notes
-----

* Smaller ``num_basis`` gives a cheaper low-rank operator but can limit spectral expressiveness.
* The basis is learned from data rather than fixed analytically, unlike :ref:`fixed_lno_layer_type <fixed-lno-layer>`.
* The orthogonality quality can be monitored through the block's ``get_orthogonality_metric()`` method, which reports :math:`\max |\mathbf{\Phi}^T\mathbf{\Phi} - \mathbf{I}|`.
* In the current implementation, the same matrix :math:`\mathbf{W}` is used for both the local bypass and the decoded spectral projection.

See Also
--------

* :ref:`orthogonal_attention_layer_type <orthogonal-attention-layer>` - Orthogonal attention layer built from the same learned low-rank basis idea
* :ref:`fixed_lno_layer_type <fixed-lno-layer>` - Laplace neural operator layer with fixed encoder/decoder bases and spectral mixing
* :ref:`neural_operator_layer_type <neural-operator-layer>` - Simpler mean-field neural operator layer
* :ref:`full_layer_type <full-layer>` - Standard dense layer
