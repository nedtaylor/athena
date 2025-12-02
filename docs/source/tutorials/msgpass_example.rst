.. _msgpass-example:

Message Passing Neural Networks
================================

Complete examples demonstrating graph neural networks (GNNs) using message passing layers for learning on graph-structured data.

This tutorial covers the ``example/msgpass_chemical`` and ``example/msgpass_euler`` examples from the athena repository.

Overview
--------

Message passing neural networks operate on graph data by:

* **Aggregating information** from neighboring nodes
* **Updating node features** based on local graph structure
* **Learning graph-level** or node-level representations
* **Preserving graph topology** throughout the network

These examples demonstrate two different applications:

1. **Chemical graphs** (``msgpass_chemical``): Predicting molecular energy from atomic structure
2. **Euler flow** (``msgpass_euler``): Predicting steady-state fluid flow over geometry

Chemical Graph Example
----------------------

Predicting Molecular Energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``example/msgpass_chemical`` uses Duvenaud message passing [#f1]_ to predict total energy from molecular graphs.
Note that this is an illustrative example; for production use, consider more advanced architectures.

Network Architecture
~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   ! Message passing layer with graph readout
   call network%add(duvenaud_msgpass_layer_type( &
        num_time_steps=4, &
        num_vertex_features=[num_atom_features], &
        num_edge_features=[num_bond_features], &
        num_outputs=10, &
        kernel_initialiser="glorot_normal", &
        readout_activation="softmax", &
        min_vertex_degree=1, &
        max_vertex_degree=10))

   ! Dense layers for prediction
   call network%add(full_layer_type( &
        num_inputs=10, &
        num_outputs=128, &
        activation="leaky_relu", &
        kernel_initialiser="he_normal"))

   call network%add(full_layer_type( &
        num_outputs=64, &
        activation="leaky_relu"))

   call network%add(full_layer_type( &
        num_outputs=1, &
        activation="leaky_relu"))

**Architecture components:**

1. **Duvenaud message passing**: Aggregates neighbor features over multiple time steps
2. **Graph readout**: Reduces graph to fixed-size vector using softmax aggregation
3. **Dense layers**: Learn from aggregated graph representation to scalar energy prediction

Complete Program Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   program msgpass_chemical_example
     use athena
     use read_chemical_graphs, only: read_extxyz_db
     implicit none

     type(network_type) :: network
     type(graph_type), allocatable :: graphs_in(:,:)
     type(array_type) :: output(1,1)
     type(metric_dict_type) :: metric_dict(2)
     class(clip_type), allocatable :: clip

     integer, parameter :: num_epochs = 100
     integer, parameter :: batch_size = 8
     integer, parameter :: num_time_steps = 4
     integer :: seed = 42

     ! Load chemical graphs from XYZ format
     call read_extxyz_db("database.xyz", graphs_in, output)

     ! Add self-loops and convert to sparse format
     do i = 1, size(graphs_in)
       call graphs_in(1,i)%add_self_loops()
       if (.not. graphs_in(1,i)%is_sparse) then
         call graphs_in(1,i)%convert_to_sparse()
       end if
     end do

     ! Initialise random seed
     call random_setup(seed, restart=.false.)

     ! Build network (see architecture above)

     ! Compile with gradient clipping
     allocate(clip, source=clip_type(clip_norm=0.1_real32))

     metric_dict%active = .false.
     metric_dict(1)%key = "loss"
     metric_dict(2)%key = "accuracy"
     metric_dict%threshold = 0.1

     call network%compile( &
          optimiser=adam_optimiser_type( &
               clip_dict=clip, &
               learning_rate=0.01_real32), &
          loss_method="mse", &
          accuracy_method="mse", &
          metrics=metric_dict, &
          batch_size=batch_size, &
          verbose=1)

     ! Normalise outputs
     output_min = minval(output(1,1)%val)
     output_max = maxval(output(1,1)%val)
     output(1,1)%val = (output(1,1)%val - output_min) / &
                       (output_max - output_min)

     ! Train network
     call network%train( &
          graphs_in, &
          output, &
          num_epochs=num_epochs, &
          shuffle_batches=.true.)

     ! Test and save
     call network%test(graphs_in, output)
     call network%print(file="network.txt")

   end program msgpass_chemical_example

Graph Data Format
~~~~~~~~~~~~~~~~~

Chemical graphs contain:

.. code-block:: fortran

   type(graph_type) :: molecule

   ! Vertex (atom) features: [num_features, num_atoms]
   ! Example: atomic number, valence, hybridisation, etc.
   molecule%num_vertex_features = 6
   molecule%vertex_features(:, atom_id)

   ! Edge (bond) features: [num_features, num_edges]
   ! Example: bond type, bond order, ring membership
   molecule%num_edge_features = 4
   molecule%edge_features(:, bond_id)

   ! Sparse adjacency representation
   molecule%adjacency_matrix  ! Connectivity
   molecule%is_sparse = .true.

Euler Flow Example
------------------

Predicting Steady-State Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``example/msgpass_euler`` uses Kipf message passing (Graph Convolutional Network) [#f2]_ to predict steady-state fluid flow from initial conditions.
Unlike the chemical example, the Kipf layer outputs node-level features, preserving graph structure.
This example utilises skip connections via concatenation to improve information flow.

Network Architecture
~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   ! First message passing layer
   call network%add(kipf_msgpass_layer_type( &
        num_time_steps=1, &
        num_vertex_features=[3, 6], &  ! [input, output] dimensions
        activation="softmax", &
        kernel_initialiser="he_normal"))

   ! Second layer with concatenation
   call network%add(kipf_msgpass_layer_type( &
        num_time_steps=1, &
        num_vertex_features=[9, 14], &  ! 6 + 3 = 9 from concatenation
        activation="softmax"), &
        input_list=[0, -1], &           ! Concatenate layer 0 and previous
        operator="concatenate")

   ! Additional layers continuing pattern
   call network%add(kipf_msgpass_layer_type( &
        num_time_steps=1, &
        num_vertex_features=[17, 32], &
        activation="softmax"), &
        input_list=[0, -1], &
        operator="concatenate")

   ! ... more layers with increasing then decreasing dimensions ...

   ! Final layer
   call network%add(kipf_msgpass_layer_type( &
        num_time_steps=1, &
        num_vertex_features=[17, 7], &  ! Output: 7 flow features
        activation="swish"), &
        input_list=[0, -1], &
        operator="concatenate")

**Key architecture features:**

1. **U-Net style**: Features expand then contract (6 → 14 → 32 → 64 → 32 → 14 → 7)
2. **Skip connections**: Concatenate original input at each layer via ``input_list=[0, -1]``
3. **Kipf message passing**: Graph convolution normalises by node degree
4. **Multiple aggregations**: Information propagates through graph structure

Complete Program
~~~~~~~~~~~~~~~~

.. code-block:: fortran

   program msgpass_euler_example
     use athena
     use read_euler, only: read_graph
     implicit none

     type(network_type) :: network
     type(graph_type), allocatable :: graphs_in(:,:), graphs_out(:,:)
     type(graph_type), allocatable :: graphs_predicted(:,:)
     class(clip_type), allocatable :: clip

     integer, parameter :: num_epochs = 200
     integer, parameter :: batch_size = 2
     integer :: seed = 1

     ! Load graph data from files
     allocate(graphs_in(1, 2), graphs_out(1, 2))

     do i = 1, 2
       call read_graph( &
            vertex_file="bump_nodeData_in_"//trim(str(i))//".txt", &
            edge_file="bump_edgeData_1.txt", &
            graph=graphs_in(1,i))

       call read_graph( &
            vertex_file="bump_nodeData_out_"//trim(str(i))//".txt", &
            edge_file="bump_edgeData_1.txt", &
            graph=graphs_out(1,i))
     end do

     ! Initialise random seed
     call random_setup(seed, restart=.false.)

     ! Build network (see architecture above)

     ! Compile with gradient clipping and learning rate decay
     allocate(clip, source=clip_type(-1.0_real32, 1.0_real32))

     call network%compile( &
          optimiser=adam_optimiser_type( &
               clip_dict=clip, &
               learning_rate=0.02_real32, &
               lr_decay=exp_lr_decay_type(0.001_real32)), &
          loss_method="mse", &
          accuracy_method="mse", &
          batch_size=batch_size, &
          verbose=1)

     ! Train on graph pairs
     call network%train( &
          graphs_in, &
          graphs_out, &
          num_epochs=num_epochs)

     ! Test and predict
     call network%test(graphs_in, graphs_out)
     graphs_predicted = network%predict(graphs_in)

     ! Save results
     call network%print(file="network.txt")

   end program msgpass_euler_example

Message Passing Types
---------------------

Duvenaud Message Passing
~~~~~~~~~~~~~~~~~~~~~~~~~

Designed for molecular graphs:

.. code-block:: fortran

   duvenaud_msgpass_layer_type( &
        num_time_steps=4, &          ! Number of message passing iterations
        num_vertex_features=[...], & ! Node feature dimensions
        num_edge_features=[...], &   ! Edge feature dimensions
        num_outputs=10, &            ! Graph-level output dimension
        readout_activation="softmax" & ! Aggregation method
   )

**Characteristics:**

* Considers edge features in message passing
* Performs graph-level readout (reduces entire graph to vector)
* Suitable for graph-level prediction tasks

Kipf Message Passing (GCN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graph Convolutional Network:

.. code-block:: fortran

   kipf_msgpass_layer_type( &
        num_time_steps=1, &          ! Usually 1 per layer
        num_vertex_features=[in, out], & ! [input_dim, output_dim]
        activation="softmax" &
   )

**Characteristics:**

* Degree-normalised aggregation
* Node-level outputs (preserves graph structure)
* Can be stacked with skip connections
* Suitable for node-level prediction tasks

Training on Graphs
------------------

Key Differences from Standard Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Graph batching:**

.. code-block:: fortran

   ! Graphs are batched as array of graphs
   type(graph_type), allocatable :: graphs(:,:)
   ! Shape: [1, num_samples]

   ! Call train with graphs directly
   call network%train(graphs_in, graphs_out, num_epochs=100)

**Sparse representation:**

.. code-block:: fortran

   ! Convert to sparse format for efficiency
   call graph%add_self_loops()        ! Add diagonal connections
   call graph%convert_to_sparse()     ! Use sparse matrix format

**Output formats:**

.. code-block:: fortran

   ! Graph-level output (scalar per graph)
   type(array_type) :: output(1,1)
   output(1,1)%val  ! [1, num_samples]

   ! Node-level output (features per node)
   type(graph_type) :: output_graphs(:,:)
   output_graphs(1,s)%vertex_features  ! [num_features, num_nodes]

Gradient Clipping
~~~~~~~~~~~~~~~~~

Essential for stable training on graphs:

.. code-block:: fortran

   ! Clip by norm (recommended)
   allocate(clip, source=clip_type(clip_norm=0.1_real32))

   ! Or clip by value
   allocate(clip, source=clip_type(-1.0_real32, 1.0_real32))

   call network%compile( &
        optimiser=adam_optimiser_type(clip_dict=clip, ...), &
        ...)

Learning Rate Decay
~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   ! Exponential decay
   lr_decay = exp_lr_decay_type(decay_rate=0.001_real32)

   ! Step decay
   lr_decay = step_lr_decay_type(decay_factor=0.5_real32, decay_steps=5)

   optimiser = adam_optimiser_type( &
        learning_rate=0.02_real32, &
        lr_decay=lr_decay)


Key Takeaways
-------------

From These Examples
~~~~~~~~~~~~~~~~~~~

1. **Graph structure matters**: Message passing leverages connectivity information
2. **Sparse is faster**: Use sparse representation for large graphs
3. **Gradient clipping essential**: Prevents exploding gradients in deep message passing
4. **Skip connections help**: Concatenating early features improves information flow
5. **Layer choice matters**: Duvenaud for graph-level, Kipf for node-level predictions

When to Use Message Passing NNs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Good for:**

* Molecular property prediction
* Social network analysis
* Traffic/flow prediction on networks
* Point cloud processing
* Physics simulations on meshes

**Not ideal for:**

* Regular grid data (use CNNs instead)
* Sequential data (use RNNs instead)
* When graph structure is unknown
* Very large graphs (>100k nodes)

See Also
--------

* :ref:`MNIST Example <mnist-example>` - Standard CNNs for comparison
* :ref:`PINN Example <pinn-example>` - Physics-informed extensions
* :ref:`Regression Examples <regression-example>` - Basic network training


.. rubric:: Footnotes

.. [#f1] Duvenaud, D. K., Maclaurin, D., Aguilera-Iparraguirre, J., Gómez-Bombarelli, R., Hirzel, T., Aspuru-Guzik, A., & Adams, R. P. (2015). Convolutional Networks on Graphs for Learning Molecular Fingerprints. In Advances in Neural Information Processing Systems (Vol. 28). `<https://doi.org/10.48550/arXiv.1509.09292>`_
.. [#f2] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In Proceedings of the 5th International Conference on Learning Representations (ICLR). DOI: `<https://doi.org/10.48550/arXiv.1609.02907>`_
