.. _recurrent-layer:

Recurrent Layer
===============

``recurrent_layer_type``

.. code-block:: fortran

  recurrent_layer_type(
    hidden_size,
    input_size=...,
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...
  )


The ``recurrent_layer_type`` derived type provides a simple recurrent neural network (RNN) layer.
The operation performed by this layer is given by:

.. math::

   \mathbf{h}_t = \text{activation}(\mathbf{W}_{ih} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)

where:

* :math:`\mathbf{x}_t` is the input at time step :math:`t`
* :math:`\mathbf{h}_t` is the hidden state at time step :math:`t`
* :math:`\mathbf{W}_{ih}` is the input-to-hidden weight matrix
* :math:`\mathbf{W}_{hh}` is the hidden-to-hidden weight matrix
* :math:`\mathbf{b}_h` is the bias vector (if used)
* :math:`\text{activation}` is the activation function applied element-wise

The layer maintains a hidden state across time steps, allowing it to capture temporal dependencies in sequential data.

Arguments
---------

* **hidden_size** (`integer`): Number of features in the hidden state
* **input_size** (`integer`): Number of features in the input. If not provided, it will be inferred when the layer is initialised.
* **use_bias** (`logical`): If ``.false.``, the layer will not use bias terms. Default: ``.true.``.
* **activation** (`class(*)`): Activation function for the hidden state.

  * Accepts `character(*)` or `class(base_actv_type)`.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Default: ``tanh_actv_type``.
  * Common choices: ``tanh``, ``relu``, ``sigmoid``

* **kernel_initialiser** (`class(*)`): Initialiser for the weight matrices :math:`\mathbf{W}_{ih}` and :math:`\mathbf{W}_{hh}` (see :ref:`Initialisers <initialisers>`).

  * If ``activation`` is ``selu_actv_type``, default: ``lecun_normal_init_type``.
  * If ``activation`` is a version of ``relu_actv_type``, default: ``he_normal_init_type``.
  * For all other activations, default: ``glorot_uniform_init_type``.

* **bias_initialiser** (`class(*)`): Initialiser for the biases (see :ref:`Initialisers <initialisers>`). Default: ``zeros_init_type``.

Shape
-----

* Input: ``(input_size, batch_size)``
* Output: ``(hidden_size, batch_size)``

The layer maintains an internal hidden state of shape ``(hidden_size, batch_size)`` that persists across forward passes.

Parameters
----------

The layer contains the following learnable parameters:

* **W_ih**: Input-to-hidden weight matrix of shape ``(hidden_size, input_size)``
* **W_hh**: Hidden-to-hidden weight matrix of shape ``(hidden_size, hidden_size)``
* **b_h**: Hidden bias vector of shape ``(hidden_size)`` (if ``use_bias=.true.``)
* **b_o**: Output bias vector of shape ``(hidden_size)`` (if ``use_bias=.true.``)

Total parameters: ``hidden_size * input_size + hidden_size * hidden_size + 2 * hidden_size`` (with bias)

Examples
--------

**Basic RNN layer for sequence processing:**

.. code-block:: fortran

   use athena
   type(network_type) :: network

   ! Create RNN layer with 10 hidden units processing 5-dimensional input
   call network%add(recurrent_layer_type( &
        input_size=5, &
        hidden_size=10, &
        activation="tanh" &
   ))

**RNN layer with custom initialisation:**

.. code-block:: fortran

   call network%add(recurrent_layer_type( &
        input_size=3, &
        hidden_size=20, &
        activation="relu", &
        kernel_initialiser="he_uniform", &
        bias_initialiser="zeros" &
   ))

**Multi-layer RNN network:**

.. code-block:: fortran

   ! Stack multiple RNN layers
   call network%add(recurrent_layer_type( &
        input_size=8, &
        hidden_size=32, &
        activation="tanh" &
   ))
   call network%add(recurrent_layer_type( &
        hidden_size=16, &
        activation="tanh" &
   ))
   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="sigmoid" &
   ))

Notes
-----

* The hidden state is initialised to zero at the start of training
* The hidden state persists across forward passes within an epoch
* For proper sequence processing, consider resetting the hidden state between different sequences
* The ``time_step`` counter tracks the number of forward passes

See Also
--------

* :ref:`full_layer_type <full-layer>` - Fully-connected layer
* :ref:`lstm_layer_type <lstm-layer>` - Long Short-Term Memory layer (future)
* :ref:`gru_layer_type <gru-layer>` - Gated Recurrent Unit layer (future)
