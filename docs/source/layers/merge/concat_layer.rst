.. _concat-layer:

Concatenate Layer
=================

``concat_layer_type``

.. code-block:: fortran

  concat_layer_type(
    input_layer_ids,
    input_rank=...,
    verbose=0
  )


The ``concat_layer_type`` concatenates outputs from multiple layers along a specified dimension.
This is commonly used to combine features from different processing paths or to implement skip connections that preserve information.

The operation concatenates tensors along the feature dimension:

.. math::

   \text{output} = [\text{input}_1 \parallel \text{input}_2 \parallel \cdots \parallel \text{input}_N]

where :math:`[\cdot \parallel \cdot]` denotes concatenation and :math:`N` is the number of input layers.

Arguments
---------

* **input_layer_ids** (`integer, dimension(:)`): Array of layer indices to concatenate.

  * Layer IDs refer to the position in the network (0-indexed)
  * Special value ``-1`` refers to the immediately previous layer
  * Example: ``[0, -1]`` concatenates layer 0 with the previous layer
  * Inputs must have the same shape except along the concatenation dimension

* **input_rank** (`integer`, optional): Rank (number of dimensions) of input tensors.

  * Used to determine concatenation dimension
  * For rank 2 (graphs): concatenates along first dimension (features)
  * For rank 3 (images): concatenates along third dimension (channels)
  * Typically inferred automatically from input layers

* **verbose** (`integer`, optional): Verbosity level for initialisation. Default: ``0``.

Shape
-----

* **Input**: Multiple tensors of shape ``(input_shape, batch_size)``
* **Output**: ``(input_shape(:-1), sum(input_shape(end)), batch_size)``
* **Concatenation dimension**: last dimension before batch size


Key Features
------------

* **Feature combination**: Combines features from multiple sources
* **Information preservation**: Retains all information from inputs
* **Shape expansion**: Output size grows with number of inputs
* **Gradient splitting**: Distributes gradients to corresponding input portions
* **No learnable parameters**: Pure operation layer

Usage Example
-------------

Basic Skip Connection
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   use athena

   type(network_type) :: network

   ! Layer 0: Input layer
   call network%add(input_layer_type(input_shape=[100]))

   ! Layer 1: First dense layer (100 → 64)
   call network%add(full_layer_type( &
        num_inputs=100, num_outputs=64, activation="relu"))

   ! Layer 2: Second dense layer (64 → 32)
   call network%add(full_layer_type( &
        num_outputs=32, activation="relu"))

   ! Layer 3: Concatenate with original input
   ! Output will be (32 + 100, batch_size)
   call network%add(concat_layer_type(input_layer_ids=[0, -1]))

   ! Layer 4: Process combined features (132 → 10)
   call network%add(full_layer_type( &
        num_inputs=132, num_outputs=10, activation="softmax"))

Multi-Input Concatenation
~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple layers can be concatenated together using this layer.

.. code-block:: fortran

    ! Concatenate three different layers together
    call network%add(concat_layer_type(input_layer_ids=[2, 5, 7]))

Implicit inclusion
~~~~~~~~~~~~~~~~~~

The ``concat_layer_type`` can be implicitly included when specifying skip connections in other layers.
This can be achieved by providing multiple input layer IDs and the ``operator="||"`` argument when using the ``add()`` method of the ``network_type``.

.. code-block:: fortran

    ! Implicit add layer by specifying multiple inputs and operator
    call network%add(conv2d_layer_type( &
        num_filters=64, kernel_size=[3,3], activation="relu"), &
        input_layer_ids=[0, 3], operator="||")

Notes
-----

**Concatenation Dimension**
  The dimension along which concatenation occurs depends on input rank:

  .. code-block:: fortran

     ! Rank 2 (graphs, dense): concatenate along dimension 1 (features)
     ! Input: [(f1, batch), (f2, batch), ...]
     ! Output: [(f1+f2+..., batch)]

     ! Rank 3 (images): concatenate along dimension 3 (channels)
     ! Input: [(w, h, c1), (w, h, c2), ...]
     ! Output: [(w, h, c1+c2+...)]

**Shape Compatibility**
  Inputs must have matching shapes except along concatenation dimension:

  .. code-block:: fortran

     ! Valid: same width, height, different channels
     ! Input 1: (28, 28, 32)
     ! Input 2: (28, 28, 64)
     ! Output:  (28, 28, 96) ✓

     ! Invalid: different width
     ! Input 1: (28, 28, 32)
     ! Input 2: (14, 14, 32)  ✗
     ! Need to resize first

**Gradient Distribution**
  During backpropagation, gradients are split among inputs:

  .. math::

     \frac{\partial L}{\partial \text{input}_i} = \frac{\partial L}{\partial \text{output}_{[\text{slice}_i]}}

  Each input receives gradients only for its corresponding slice of the output.

**Comparison with Addition**

  .. list-table::
     :widths: 30 35 35
     :header-rows: 1

     * - Aspect
       - ``concat_layer_type``
       - ``add_layer_type``
     * - Output shape
       - Larger along concat dimension
       - Same as inputs
     * - Shape requirement
       - Can differ along concat dimension
       - All inputs identical
     * - Information
       - Preserves all information
       - Combines information
     * - Use case
       - Feature combination
       - Residual connections
     * - Memory
       - Increases
       - Same as inputs

**When to Use**
  * **Skip connections**: Preserve early features in deep networks
  * **Multi-scale features**: Combine information at different scales
  * **Feature fusion**: Merge outputs from parallel processing paths
  * **U-Net architectures**: Connect encoder and decoder paths
  * **Dense connections**: DenseNet-style architectures
  * **Graph neural networks**: Concatenate input features at each layer

**When Not to Use**
  * Memory is constrained (consider :ref:`add_layer_type <add-layer>` instead)
  * Features should be combined additively (use add or learned attention)
  * Output dimension growth becomes problematic for subsequent layers
  * Simple residual connections suffice (add is more efficient)

**Benefits**
  * **Information preservation**: No information loss from inputs
  * **Gradient flow**: Provides direct path for gradients
  * **Feature reuse**: Allows later layers to access early features
  * **Flexibility**: Combines features without learning parameters

Typical Hyperparameters
-----------------------

Architectural considerations:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Consideration
     - Guidelines
   * - Number of inputs
     - 2-4 typical; >5 may cause dimension explosion
   * - Skip distance
     - Connect complementary feature levels
   * - Channel growth
     - Monitor total channels after concatenation
   * - Subsequent layers
     - May need more parameters to process larger inputs

See Also
--------

* :ref:`add_layer_type <add-layer>` - Add instead of concatenate
* :ref:`kipf_msgpass_layer_type <kipf-msgpass-layer>` - Use with GNN skip connections
* :ref:`conv2d_layer_type <conv2d-layer>` - Common in U-Net architectures
* :ref:`Message Passing Example <msgpass-example>` - Skip connection examples
