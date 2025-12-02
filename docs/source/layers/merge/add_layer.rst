.. _add-layer:

Add Layer
=========

``add_layer_type``

.. code-block:: fortran

  add_layer_type(
    input_layer_ids,
    batch_size=...,
    input_rank=...,
    verbose=0
  )


The ``add_layer_type`` performs element-wise addition of outputs from multiple layers.
This is commonly used to implement skip connections (residual connections) in deep networks.

The operation is defined as:

.. math::

   \text{output} = \sum_{i=1}^{N} \text{input}_i

where :math:`N` is the number of input layers. All inputs must have the same shape.

Arguments
---------

* **input_layer_ids** (`integer, dimension(:)`): Array of layer indices to add together.

  * Layer IDs refer to the position in the network (0-indexed)
  * Special value ``-1`` refers to the immediately previous layer
  * Example: ``[0, -1]`` adds layer 0 with the previous layer
  * All specified layers must have identical output shapes

* **batch_size** (`integer`, optional): **SOON TO BE DEPRECATED**. Batch size for the layer.

* **input_rank** (`integer`, optional): Rank (number of dimensions) of input tensors.

  * Used to determine proper shape handling
  * Typically inferred automatically from input layers
  * Manual specification useful for graph data (rank 2) vs image data (rank 3)

* **verbose** (`integer`, optional): Verbosity level for initialisation. Default: ``0``.

Shape
-----

* **Input**: Multiple tensors of shape ``(d1, d2, ..., dn, batch_size)``

  * All inputs must have identical shapes
  * Number of inputs determined by length of ``input_layer_ids``

* **Output**: Same shape as inputs ``(d1, d2, ..., dn, batch_size)``

Key Features
------------

* **Element-wise**: Adds corresponding elements from each input
* **Shape preservation**: Output has same shape as inputs
* **Gradient flow**: Distributes gradients equally to all inputs during backpropagation
* **No learnable parameters**: Pure operation layer

Usage Example
-------------

Basic Residual Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   use athena

   type(network_type) :: network

   ! Layer 0: Input layer
   call network%add(input_layer_type(input_shape=[28, 28, 1]))

   ! Layer 1: Convolutional layer
   call network%add(conv2d_layer_type( &
        num_filters=32, kernel_size=[3,3], activation="relu"))

   ! Layer 2: Another convolutional layer
   call network%add(conv2d_layer_type( &
        num_filters=32, kernel_size=[3,3], activation="relu"))

   ! Layer 3: Add layer - creates residual connection
   ! Adds output of layer 1 with output of layer 2
   call network%add(add_layer_type(input_layer_ids=[1, -1]))

   ! Continue network...
   call network%add(conv2d_layer_type( &
        num_filters=64, kernel_size=[3,3], activation="relu"))

Multi-Input Addition
~~~~~~~~~~~~~~~~~~~~

Multiple layers can be added together using this layer.

.. code-block:: fortran

   ! Add three different layers together
   call network%add(add_layer_type(input_layer_ids=[2, 5, 7]))

Implicit inclusion
~~~~~~~~~~~~~~~~~~

The ``add_layer_type`` can be implicitly included when specifying skip connections in other layers.
This can be achieved by providing multiple input layer IDs and the ``operator="+"`` argument when using the ``add()`` method of the ``network_type``.

.. code-block:: fortran

   ! Implicit add layer by specifying multiple inputs and operator
   call network%add(conv2d_layer_type( &
        num_filters=64, kernel_size=[3,3], activation="relu"), &
        input_layer_ids=[0, 3], operator="+")

Notes
-----

**Shape Requirements**
  All inputs must have exactly the same shape. If shapes differ, consider:

  * Using projection layers to match dimensions
  * Using :ref:`concat_layer_type <concat-layer>` instead
  * Applying reshape/pooling operations first

**Layer ID Indexing**
  * Layer IDs are 0-indexed (first layer after input is 0)
  * Use ``-1`` to refer to the immediately previous layer
  * Use ``-2`` for two layers back, etc.
  * Forward references (future layers) are not allowed

**Gradient Distribution**
  During backpropagation, gradients are copied (not split) to all inputs:

  .. math::

     \frac{\partial L}{\partial \text{input}_i} = \frac{\partial L}{\partial \text{output}}

  This allows gradients to flow equally through all paths.

**Comparison with Concatenation**

  .. list-table::
     :widths: 30 35 35
     :header-rows: 1

     * - Aspect
       - ``add_layer_type``
       - ``concat_layer_type``
     * - Output shape
       - Same as inputs
       - Larger along concatenation dimension
     * - Shape requirement
       - All inputs identical
       - Can differ along concat dimension
     * - Use case
       - Residual connections
       - Feature combination
     * - Parameters
       - None
       - None
     * - Gradient flow
       - Copied to all inputs
       - Split along concat dimension

**When to Use**
  * **Residual connections**: Add input to output of processing block
  * **Deep networks**: Help gradient flow in very deep architectures
  * **Skip connections**: Connect distant layers in encoder-decoder architectures
  * **Ensemble-like behavior**: Combine predictions from parallel paths
  * **Identity mappings**: Allow network to learn when to bypass transformations

**When Not to Use**
  * Inputs have different shapes (use concat or reshape first)
  * Want to combine features additively (consider learned weighted sum)
  * Need to concatenate features (use :ref:`concat_layer_type <concat-layer>`)

**Benefits for Training**
  * **Gradient flow**: Provides direct path for gradients to flow to early layers
  * **Training stability**: Helps prevent vanishing gradients in deep networks
  * **Faster convergence**: Often leads to faster training
  * **Better optimisation**: Creates implicit ensemble of shallow networks

Typical Hyperparameters
-----------------------

There are no hyperparameters to tune for this layer, only architectural choices:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Consideration
     - Guidelines
   * - Number of inputs
     - Usually 2 (main path + skip connection); rarely >3
   * - Skip distance
     - 2-4 layers for CNNs; 1-2 layers for GNNs
   * - Activation placement
     - Often apply activation *after* add layer
   * - Batch norm placement
     - Apply before add layer in both paths

See Also
--------

* :ref:`concat_layer_type <concat-layer>` - Concatenate instead of add
* :ref:`kipf_msgpass_layer_type <kipf-msgpass-layer>` - Use with GNN skip connections
* :ref:`conv2d_layer_type <conv2d-layer>` - Commonly used with residual connections
