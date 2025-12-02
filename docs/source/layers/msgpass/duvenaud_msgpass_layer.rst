.. _duvenaud-msgpass-layer:

Duvenaud Message Passing Layer
===============================

``duvenaud_msgpass_layer_type``

.. code-block:: fortran

  duvenaud_msgpass_layer_type(
    num_vertex_features,
    num_edge_features,
    num_time_steps,
    max_vertex_degree,
    num_outputs,
    min_vertex_degree=1,
    batch_size=...,
    message_activation="none",
    readout_activation="none",
    kernel_initialiser=...,
    verbose=0
  )


The ``duvenaud_msgpass_layer_type`` implements the message passing architecture from Duvenaud et al. (2015) for learning on molecular graphs.
This layer aggregates information from neighboring nodes through message passing iterations and performs graph-level readout to produce a fixed-size vector representation.

The operation is performed in two stages:

1. **Message Passing**: For each time step :math:`t`, update node features by aggregating neighbor information:

.. math::

   h_v^{(t+1)} = \text{activation}(h_v^{(t)} + \sum_{u \in N(v)} M(h_v^{(t)}, h_u^{(t)}, e_{vu}))

where :math:`h_v` is the feature vector for node :math:`v`, :math:`N(v)` is the set of neighbors, :math:`e_{vu}` are edge features, and :math:`M` is a learned message function.

2. **Graph Readout**: Aggregate all node features into a graph-level representation:

.. math::

   h_{\text{graph}} = \text{readout\_activation}\left(\sum_{d=1}^{D} \sum_{v: \deg(v)=d} W_d h_v^{(T)}\right)

where :math:`D` is the maximum vertex degree, :math:`T` is the number of time steps, and :math:`W_d` are degree-specific weight matrices.

Arguments
---------

* **num_vertex_features** (`integer, dimension(:)`): Dimensionality of vertex features at each layer.

  * If single value: all time steps use same feature dimension
  * If array: specifies input and output dimensions (e.g., ``[16, 32]`` means 16→32)
  * Defines the evolution of feature dimensions through message passing

* **num_edge_features** (`integer, dimension(:)`): Dimensionality of edge features.

  * Similar to ``num_vertex_features``
  * Edge features are incorporated into message function
  * Essential for molecular graphs (bond types, bond orders)

* **num_time_steps** (`integer`): Number of message passing iterations.

  * Controls how far information propagates across the graph
  * Typical values: 2-6 for molecular graphs
  * More steps = larger receptive field but slower computation

* **max_vertex_degree** (`integer`): Maximum degree of any vertex in the graph.

  * Used to allocate degree-specific weight matrices for readout
  * Should be set to maximum degree in your dataset
  * Example: for molecules, often 4-6 (typical valence)

* **num_outputs** (`integer`): Dimensionality of graph-level output after readout.

  * Output size of the graph representation vector
  * This output typically feeds into dense layers for prediction
  * Does not affect node-level features during message passing

* **min_vertex_degree** (`integer`, optional): Minimum degree of vertices to consider in readout. Default: ``1``.

  * Vertices with degree less than this are excluded from readout
  * Useful for ignoring isolated nodes
  * Typically set to 1 (include all connected nodes)

* **batch_size** (`integer`, optional): **SOON TO BE DEPRECATED**. Batch size for the layer.

* **message_activation** (`class(*)`): Activation function for message updates.

  * Accepts ``character(*)`` or ``class(base_actv_type)``
  * See :ref:`Activation Functions <activation-functions>` for available options
  * Default: ``"none"`` (linear)
  * Common choices: ``"relu"``, ``"leaky_relu"``, ``"tanh"``

* **readout_activation** (`class(*)`): Activation function for graph readout aggregation.

  * Applied to the aggregated graph representation
  * Default: ``"none"`` (linear)
  * Common choices: ``"softmax"`` (weighted sum), ``"sigmoid"``
  * ``"softmax"`` creates an attention-like mechanism over node degrees

* **kernel_initialiser** (`character(*)`): Initialiser for the weight matrices (see :ref:`Initialisers <initialisers>`).

  * Default depends on activation function (Glorot/He initialisation)
  * All weight matrices (message and readout) use the same initialiser

* **verbose** (`integer`, optional): Verbosity level for initialisation. Default: ``0``.

Shape
-----

* **Input**: Graph structure represented by ``graph_type``:

  * ``vertex_features``: ``(num_vertex_features(1), num_vertices)`` - Node features
  * ``edge_features``: ``(num_edge_features(1), num_edges)`` - Edge features
  * ``adjacency_matrix``: Sparse or dense connectivity matrix
  * Batch dimension handled internally

* **Output**: ``(num_outputs, batch_size)`` - Graph-level representation vector

Key Features
------------

* **Degree-aware readout**: Uses separate weights for nodes of different degrees
* **Edge feature incorporation**: Considers both node and edge attributes in message passing
* **Permutation invariant**: Graph-level output is invariant to node ordering
* **Differentiable**: Supports backpropagation through the graph structure

Usage Example
-------------

Molecular Property Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   use athena

   type(network_type) :: network
   type(graph_type), allocatable :: molecules(:,:)

   ! Add Duvenaud message passing layer
   call network%add(duvenaud_msgpass_layer_type( &
        num_vertex_features=[6, 16, 16, 16], &  ! 4 time steps
        num_edge_features=[4, 8, 8, 8], &
        num_time_steps=4, &
        max_vertex_degree=6, &
        num_outputs=32, &
        min_vertex_degree=1, &
        message_activation="relu", &
        readout_activation="softmax", &
        kernel_initialiser="glorot_normal", &
        batch_size=8))

   ! Add dense layers for prediction
   call network%add(full_layer_type( &
        num_inputs=32, num_outputs=128, activation="relu"))
   call network%add(full_layer_type( &
        num_outputs=64, activation="relu"))
   call network%add(full_layer_type( &
        num_outputs=1, activation="none"))

   ! Compile and train
   call network%compile( &
        optimiser=adam_optimiser_type(), &
        loss_method="mse", &
        batch_size=8)

   call network%train(molecules, target_energies, num_epochs=100)

Notes
-----

**Graph Data Requirements**
  Graphs should have ``add_self_loops()`` called and be converted to sparse format for efficiency:

  .. code-block:: fortran

     call molecule%add_self_loops()
     call molecule%convert_to_sparse()

**Computational Complexity**
  * Time complexity: :math:`O(T \cdot |E| \cdot F)` where :math:`T` is time steps, :math:`|E|` is edges, :math:`F` is feature dimension
  * Space complexity: :math:`O(D \cdot F \cdot F_{\text{out}})` for readout weights

**When to Use**
  * Molecular property prediction (energy, solubility, toxicity)
  * Chemical reaction prediction
  * Graph classification tasks
  * When edge features are important
  * When graph-level (not node-level) predictions are needed

See Also
--------

* :ref:`kipf_msgpass_layer_type <kipf-msgpass-layer>` - Alternative message passing (GCN)
* :ref:`Message Passing Example <msgpass-example>` - Complete tutorial with examples
* :ref:`concat_layer_type <concat-layer>` - For combining with skip connections

References
----------

Duvenaud, D. K., et al. (2015). "Convolutional Networks on Graphs for Learning Molecular Fingerprints."
*Advances in Neural Information Processing Systems*, 28, DOI: `10.48550/arXiv.1509.09292 <https://arxiv.org/abs/1509.09292>`_.
