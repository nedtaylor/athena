.. _dynamic-lno-layer:

Dynamic Laplace Neural Operator Layer
=====================================

``dynamic_lno_layer_type``

.. code-block:: fortran

  dynamic_lno_layer_type(
    num_outputs,
    num_modes,
    num_inputs=...,
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...
  )


The ``dynamic_lno_layer_type`` derived type provides a Laplace neural operator layer.
It combines a local bypass with a spectral pathway defined via a learnable pole–residue decomposition:

.. math::

   \mathbf{v} = \sigma\left(\mathbf{D}(\boldsymbol{\mu})\,\mathrm{diag}(\boldsymbol{\beta})\,\mathbf{E}(\boldsymbol{\mu})\,\mathbf{u} + \mathbf{W}\mathbf{u} + \mathbf{b}\right)

where:

* :math:`\mathbf{u} \in \mathbb{R}^{n_{in}}` is the input sampled on a grid
* :math:`\boldsymbol{\mu} \in \mathbb{R}^{M}` are learnable Laplace-domain poles
* :math:`\mathbf{E}(\boldsymbol{\mu}) \in \mathbb{R}^{M \times n_{in}}` is the pole-residue encoder: :math:`E_{k,j}=\exp(-\mu_k\,t_j)`, :math:`t_j = (j{-}1)/(n_{in}{-}1)`
* :math:`\boldsymbol{\beta} \in \mathbb{R}^{M}` are learnable residues forming :math:`\mathrm{diag}(\boldsymbol{\beta})`
* :math:`\mathbf{D}(\boldsymbol{\mu}) \in \mathbb{R}^{n_{out} \times M}` is the pole-residue decoder: :math:`D_{i,k}=\exp(-\mu_k\,\tau_i)`, :math:`\tau_i = (i{-}1)/(n_{out}{-}1)`
* :math:`\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}` is the learnable local bypass matrix
* :math:`\mathbf{b} \in \mathbb{R}^{n_{out}}` is the bias vector when ``use_bias=.true.``
* :math:`M` is ``num_modes``
* :math:`\sigma` is the activation function

Arguments
---------

* **num_outputs** (``integer``): Number of output discretisation points.
* **num_modes** (``integer``): Number of spectral poles.
* **num_inputs** (``integer``): Number of input discretisation points. If not provided, it is inferred when the layer is initialised.
* **use_bias** (``logical``): If ``.false.``, the layer will not use a bias term. Default: ``.true.``.
* **activation** (``class(*)``): Activation function for the layer.

  * Accepts ``character(*)`` or ``class(base_actv_type)``.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Default: ``none_actv_type``.

* **kernel_initialiser** (``class(*)``): Initialiser for the learnable residues :math:`\boldsymbol{\beta}` and local bypass weights :math:`\mathbf{W}` (see :ref:`Initialisers <initialisers>`).

* **bias_initialiser** (``class(*)``): Initialiser for the biases (see :ref:`Initialisers <initialisers>`). Default: ``zeros_init_type``.

Shape
-----

* Input: ``(num_inputs, batch_size)``.
* Output: ``(num_outputs, batch_size)``.

Parameters
----------

The layer contains the following learnable parameters:

* **μ**: Spectral poles of shape ``(num_modes)``.
* **β**: Residues of shape ``(num_modes)``.
* **W**: Local bypass matrix of shape ``(num_outputs, num_inputs)``.
* **b**: Bias vector of shape ``(num_outputs)`` when ``use_bias=.true.``.

The following tensors are fixed after initialisation and are not learnable:

* **E(μ)**: Encoder basis of shape ``(num_modes, num_inputs)``.
* **D(μ)**: Decoder basis of shape ``(num_outputs, num_modes)``.

Total learnable parameters:

* With bias: ``2*num_modes + num_outputs*num_inputs + num_outputs``
* Without bias: ``2*num_modes + num_outputs*num_inputs``

Examples
--------

**Basic Laplace neural operator block:**

.. code-block:: fortran

   use athena
   type(network_type) :: network

   call network%add(dynamic_lno_layer_type( &
        num_inputs=128, &
        num_outputs=128, &
        num_modes=16, &
        activation="relu" &
   ))

**Stacked operator network with dense readout:**

.. code-block:: fortran

   call network%add(dynamic_lno_layer_type( &
        num_inputs=256, &
        num_outputs=256, &
        num_modes=32, &
        activation="swish" &
   ))
   call network%add(dynamic_lno_layer_type( &
        num_outputs=128, &
        num_modes=16, &
        activation="swish" &
   ))
   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="none" &
   ))

Notes
-----

* Larger ``num_modes`` increases spectral expressiveness but also increases the quadratic parameter cost of :math:`\mathbf{R}`.
* The fixed encoder and decoder bases make this layer resolution-aware through the chosen input and output grid sizes.
* This layer is more expressive than the mean-field neural operator layer, but more expensive to evaluate.

See Also
--------

* :ref:`neural_operator_layer_type <neural-operator-layer>` - Simpler mean-field neural operator layer
* :ref:`fixed_lno_layer_type <fixed-lno-layer>` - Laplace neural operator layer with fixed encoder/decoder bases and spectral mixing
* :ref:`full_layer_type <full-layer>` - Standard dense layer
