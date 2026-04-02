.. _spectral-filter-layer:

Spectral Filter Layer
=====================

``spectral_filter_layer_type``

.. code-block:: fortran

  spectral_filter_layer_type(
    num_outputs,
    num_modes,
    num_inputs=...,
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...
  )


The ``spectral_filter_layer_type`` derived type provides a spectral neural operator layer.
It combines a local bypass with a spectral pathway defined in a fixed DCT-II basis:

.. math::

   \mathbf{v} = \sigma\left(\boldsymbol{\Phi}^{-1}\,\mathrm{diag}(\mathbf{w}_s)\,\boldsymbol{\Phi}\,\mathbf{u} + \mathbf{W}\mathbf{u} + \mathbf{b}\right)

where:

* :math:`\mathbf{u} \in \mathbb{R}^{n_{in}}` is the input sampled on a grid
* :math:`\boldsymbol{\Phi} \in \mathbb{R}^{M \times n_{in}}` is the fixed forward DCT basis with :math:`\Phi_{k,j}=\cos\left(\pi (k{-}1)(j{-}\tfrac{1}{2})/n_{in}\right)`
* :math:`\mathbf{w}_s \in \mathbb{R}^{M}` are learnable spectral filter weights
* :math:`\boldsymbol{\Phi}^{-1} \in \mathbb{R}^{n_{out} \times M}` is the fixed inverse DCT basis with :math:`\Phi^{-1}_{i,k}=\cos\left(\pi (k{-}1)(i{-}\tfrac{1}{2})/n_{out}\right)`
* :math:`\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}` is the learnable local bypass matrix
* :math:`\mathbf{b} \in \mathbb{R}^{n_{out}}` is the bias vector when ``use_bias=.true.``
* :math:`M` is ``num_modes``
* :math:`\sigma` is the activation function

The forward and inverse DCT bases are fixed after layer initialisation and are not learnable.

Arguments
---------

* **num_outputs** (``integer``): Number of output discretisation points.
* **num_modes** (``integer``): Number of retained spectral cosine modes.
* **num_inputs** (``integer``): Number of input discretisation points. If not provided, it is inferred when the layer is initialised.
* **use_bias** (``logical``): If ``.false.``, the layer will not use a bias term. Default: ``.true.``.
* **activation** (``class(*)``): Activation function for the layer.

  * Accepts ``character(*)`` or ``class(base_actv_type)``.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Default: ``none_actv_type``.

* **kernel_initialiser** (``class(*)``): Initialiser for local bypass weights :math:`\mathbf{W}` (see :ref:`Initialisers <initialisers>`).
* **bias_initialiser** (``class(*)``): Initialiser for the bias (see :ref:`Initialisers <initialisers>`). Default: ``zeros_init_type``.

Shape
-----

* Input: ``(num_inputs, batch_size)``.
* Output: ``(num_outputs, batch_size)``.

Parameters
----------

The layer contains the following learnable parameters:

* **w_s**: Spectral filter weights of shape ``(num_modes)``.
* **W**: Local bypass matrix of shape ``(num_outputs, num_inputs)``.
* **b**: Bias vector of shape ``(num_outputs)`` when ``use_bias=.true.``.

The following tensors are fixed after initialisation and are not learnable:

* **Phi**: Forward DCT basis of shape ``(num_modes, num_inputs)``.
* **Phi_inv**: Inverse DCT basis of shape ``(num_outputs, num_modes)``.

Total learnable parameters:

* With bias: ``num_modes + num_outputs*num_inputs + num_outputs``
* Without bias: ``num_modes + num_outputs*num_inputs``

Examples
--------

**Basic spectral filtering block:**

.. code-block:: fortran

   use athena
   type(network_type) :: network

   call network%add(spectral_filter_layer_type( &
        num_inputs=128, &
        num_outputs=128, &
        num_modes=32, &
        activation="relu" &
   ))

**Stacked spectral-filter network with dense readout:**

.. code-block:: fortran

   call network%add(spectral_filter_layer_type( &
        num_inputs=256, &
        num_outputs=256, &
        num_modes=64, &
        activation="swish" &
   ))
   call network%add(spectral_filter_layer_type( &
        num_outputs=128, &
        num_modes=32, &
        activation="swish" &
   ))
   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="none" &
   ))

Notes
-----

* ``num_modes`` controls spectral resolution. Larger values can improve expressiveness but increase compute and parameter count.
* Initialising ``w_s`` near one yields a near-identity spectral path, while the local bypass captures fine local corrections.
* This layer uses fixed cosine bases, so it is lightweight and stable compared with learnable-basis operator layers.

See Also
--------

* :ref:`neural_operator_layer_type <neural-operator-layer>` - Mean-field neural operator layer
* :ref:`fixed_lno_layer_type <fixed-lno-layer>` - Laplace neural operator with pole-residue spectral mixing
* :ref:`dynamic_lno_layer_type <dynamic-lno-layer>` - Laplace neural operator with learnable encoder/decoder bases
* :ref:`full_layer_type <full-layer>` - Standard dense layer
