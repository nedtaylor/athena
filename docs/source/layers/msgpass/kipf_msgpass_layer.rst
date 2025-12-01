.. _kipf-msgpass-layer:

Kipf Message Passing Layer
===========================

``kipf_msgpass_layer_type``

.. code-block:: fortran

  kipf_msgpass_layer_type(
    num_vertex_features,
    num_time_steps,
    batch_size=...,
    activation="none",
    kernel_initialiser=...,
    verbose=0
  )


The ``kipf_msgpass_layer_type`` implements the Graph Convolutional Network (GCN) from Kipf & Welling (2017).
This layer performs message passing with degree normalisation, making it effective for semi-supervised learning on graphs.

The operation is defined as:

.. math::

   H^{(t+1)} = \text{activation}\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(t)} W^{(t)}\right)

where:

* :math:`H^{(t)}` is the node feature matrix at layer :math:`t`
* :math:`\tilde{A} = A + I` is the adjacency matrix with added self-loops
* :math:`\tilde{D}` is the degree matrix of :math:`\tilde{A}`
* :math:`W^{(t)}` is a learnable weight matrix
* The normalisation :math:`\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}` ensures features are properly scaled by node degree

This layer preserves the graph structure, producing node-level outputs rather than a single graph-level representation.

Arguments
---------

* **num_vertex_features** (`integer, dimension(:)`): Dimensionality of vertex features.

  * Specifies ``[input_dim, output_dim]`` for the layer
  * Example: ``[16, 32]`` transforms 16-dimensional features to 32 dimensions
  * All nodes share the same transformation
  * For single time step, use 2-element array

* **num_time_steps** (`integer`): Number of message passing iterations.

  * Typically set to ``1`` when stacking multiple Kipf layers
  * Larger values apply the same transformation multiple times
  * For deep architectures, prefer stacking layers over increasing time steps

* **batch_size** (`integer`, optional): **SOON TO BE DEPRECATED**. Batch size for the layer.

* **activation** (`class(*)`): Activation function applied after aggregation.

  * Accepts ``character(*)`` or ``class(base_actv_type)``
  * See :ref:`Activation Functions <activation-functions>` for available options
  * Default: ``"none"`` (linear)
  * Common choices: ``"relu"``, ``"softmax"``, ``"tanh"``, ``"swish"``

* **kernel_initialiser** (`character(*)`): Initialiser for the weight matrix (see :ref:`Initialisers <initialisers>`).

  * Default depends on activation function
  * If ``activation`` is ReLU-based: ``"he_normal"``
  * Otherwise: ``"glorot_uniform"``

* **verbose** (`integer`, optional): Verbosity level for initialisation. Default: ``0``.

Shape
-----

* **Input**: Graph structure represented by ``graph_type``:

  * ``vertex_features``: ``(num_vertex_features(1), num_vertices)`` - Node features
  * ``adjacency_matrix``: Sparse or dense connectivity matrix
  * Batch dimension handled internally

* **Output**: Graph structure with updated features:

  * ``vertex_features``: ``(num_vertex_features(2), num_vertices)`` - Transformed node features
  * Graph structure (adjacency) preserved

Key Features
------------

* **Degree normalisation**: Symmetric normalisation prevents features from exploding/vanishing
* **Node-level outputs**: Preserves graph structure for further processing
* **Efficient**: Works well with sparse adjacency matrices
* **Stackable**: Can be chained with skip connections for deep architectures
* **Permutation equivariant**: Node ordering doesn't affect relative node features

Usage Example
-------------

Stacked Layers with Skip Connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   ! Layer 0: First message passing
   call network%add(kipf_msgpass_layer_type( &
        num_vertex_features=[3, 6], &
        num_time_steps=1, &
        activation="softmax"))

   ! Layer 1: Concatenate with input (skip connection)
   call network%add(kipf_msgpass_layer_type( &
        num_vertex_features=[9, 14], &  ! 6 + 3 = 9 from concatenation
        num_time_steps=1, &
        activation="softmax"), &
        input_list=[0, -1], &           ! Concatenate layer 0 and previous
        operator="concatenate")

   ! Layer 2: Continue pattern
   call network%add(kipf_msgpass_layer_type( &
        num_vertex_features=[17, 32], & ! 14 + 3 = 17
        num_time_steps=1, &
        activation="softmax"), &
        input_list=[0, -1], &
        operator="concatenate")

Notes
-----

**Graph Preprocessing**
  Always add self-loops and use sparse format:

  .. code-block:: fortran

     call graph%add_self_loops()
     call graph%convert_to_sparse()

**Computational Complexity**
  * Time: :math:`O(|E| \cdot F_{\text{in}} \cdot F_{\text{out}})` where :math:`|E|` is number of edges
  * Space: :math:`O(|V| \cdot F_{\text{out}})` where :math:`|V|` is number of vertices
  * Sparse matrices significantly reduce computation for large sparse graphs

**When to Use**
  * Node classification tasks
  * Graph-to-graph prediction (e.g., flow fields, deformation)
  * Semi-supervised learning on graphs
  * When graph structure should be preserved
  * Deep graph networks (stack multiple layers)
  * Point cloud processing

See Also
--------

* :ref:`duvenaud_msgpass_layer_type <duvenaud-msgpass-layer>` - For graph-level predictions
* :ref:`Message Passing Example <msgpass-example>` - Complete tutorial with examples
* :ref:`concat_layer_type <concat-layer>` - For skip connection concatenation
* :ref:`add_layer_type <add-layer>` - For skip connection addition

References
----------

Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks."
*International Conference on Learning Representations (ICLR)*, DOI: `10.48550/arXiv.1609.02907 <https://arxiv.org/abs/1609.02907>`_.
