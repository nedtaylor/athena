.. _network-outputs:

Accessing Network Outputs
=========================

This tutorial covers both basic and advanced techniques for accessing and manipulating network outputs.

Overview
--------

The athena library provides several methods for accessing network outputs depending on your use case:

* ``predict()``: Inference-oriented forward pass that switches the network to inference mode
* ``forward_eval()``: Forward pass that returns output pointers for custom operations
* Direct access to layer outputs -- for advanced use cases

These enable access to network predictions for basic use.
For advanced use, they enable custom loss functions, multi-task learning, physics-informed neural networks (PINNs), and other advanced training scenarios.

.. _basic-methods:

Basic Methods
-------------

This section is focused on networks with a single output layer.
For multi-output networks, see the :ref:`Advanced Methods <advanced-patterns>` section.

For basic usage of an athena neural network, the convention is to train the network using ``train()``, then make predictions using ``predict()``.
The ``train()`` method performs a forward pass, computes the loss, backpropagates gradients automatically for ``num_epochs`` iterations (internally sets the network to training mode and reverts to its previous state).
The ``predict()`` method performs an inference-oriented forward pass, returns the output values (internally sets the network to inference mode and reverts to its previous state).

If you are writing a custom low-level loop around ``forward()`` or
``forward_eval()``, switch modes explicitly with
:ref:`set_training_mode() and set_inference_mode() <network-modes>`.

Inference Mode - ``predict()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depending on the input type provided to the ``predict()`` method, the output can be either an ``array_type``, a ``graph_type``, or a ``real`` type array.

The allowed input types and shapes are:

* ``real``: array of rank 1-5, with shape :math:`(D_1, D_2, ..., D_{r-1}, N)`, where :math:`r` is the rank (1-5) and :math:`D_i` are the dimensions, with the final dimension representing the number of samples :math:`N`,
* ``array_type``: scalar or rank 2 array either with shape :math:`(1, 1)` for single-input networks, or :math:`(I, N)` for multi-input networks with :math:`I` inputs and :math:`N` samples, or
* ``graph_type``: array of rank 1 (representing a list of graphs) or rank 2 with shape :math:`(I, N)` for multi-input networks with :math:`I` inputs and :math:`N` samples.

Beyond this, there is the more generic ``predict_generic()`` method that can handle any of the above input types in addition to the ``array_ptr_type``.
The allowed input types and shapes are:

* Any of the above input types, or
* ``array_ptr_type``: rank 1 array containing :math:`I` inputs, one for each input layer.

.. note::

  ``array_ptr_type`` is not yet fully implemented, so it only offers the same use of ``array_type``.
  The intended use is for it to handle both array and graph type data for multi-input networks.

Based on the input type, the output type will vary, though it will typically be the same type as the input data.
The values returned by ``predict()`` do not track gradients and are suitable for evaluation, testing, and deployment.

``real`` Array Input/Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the input is a ``real`` array of any shape (up to rank 5 currently supported), the output will be a ``real`` two-dimensional (2D) array with shape :math:`Y \in \mathbb{R}^{O \times N}`, where

- :math:`O` is the number of output features (e.g., classes), and
- :math:`N` is the number of samples.

The returned output is a flattened (or "unwrapped") array for easy use in standard Fortran code.
This means that the output is always a two-dimensional array :math:`Y \in \mathbb{R}^{O \times N}`.
If the expected output has shape :math:`(H, W, C, N)`, corresponding to height, width, channels, and samples,
then the flattened array has shape :math:`Y \in \mathbb{R}^{(H \cdot W \cdot C) \times N}`.
The indexing of :math:`Y` follows **column-major order** (Fortran-style), meaning that the first dimension
(features) varies fastest when iterating through the array.

An example of using ``predict()`` with a ``real`` array input is shown below:

.. code-block:: fortran

   real(real32), dimension(:,:), allocatable :: test_input
   real(real32), dimension(:,:), allocatable :: predictions

   ! Load or prepare test_input data
   ! ...

   ! Inference without gradient tracking
   predictions = network%predict(test_input)

``array_type`` Input/Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |diffstruc readthedocs| raw:: html

   <a href="https://diffstruc.readthedocs.io/en/latest/">diffstruc Read<em>the</em>Docs</a>

The ``array_type`` is a core data structure in the athena library that supports automatic differentiation and gradient tracking and is provided by the diffstruc library.
Simply, the ``array_type`` is a wrapper around standard Fortran arrays that adds functionality for machine learning tasks.
Its data is stored in a component called ``val``, which is a 2D array, with dimensions :math:`(E, N)`, where :math:`E` is the number of elements and :math:`N` is the number of samples.
As such, a single ``array_type`` instance always represents a batch of samples.
For more details on the ``array_type``, refer to the |diffstruc readthedocs| (the current branch being utilised is the ``development`` branch).

If the input is an ``array_type`` (scalar or rank 2), the output will be a 2D ``array_type`` array.
If the expected output for the each sample is a fixed size (_aka_ homogeneous) array, then the output shape will be :math:`(1, 1)`, with the data stored in the ``val`` component.
Representing a scalar output as a 2D array with shape :math:`(1, 1)` allows for consistent handling of different data formats.

An example of using ``predict()`` with an ``array_type`` input is shown below:

.. code-block:: fortran

   type(array_type) :: test_input
   type(array_type), dimension(:,:), allocatable :: predictions

   ! Load or prepare test_input data
   ! ...

   ! Inference without gradient tracking
   predictions = network%predict(test_input)

The data within the output ``array_type`` can be accessed via the ``val`` component or using the ``extract()`` type-bound procedure.

Below an example of accessing the output values of a 3D array stored in an ``array_type``:

.. code-block:: fortran

   real(real32), dimension(:, :, :), allocatable :: pred_vals

   ! Print the data shape
   write(*,*) "Prediction shape: ", predictions(1,1)%shape
   write(*,*) "Number of samples: ", size(predictions(1,1)%val, 2)

   ! Directly access the val component
   write(*,*) "Predictions:"
   do i = 1, size(predictions(1,1)%val, 2)  ! Loop over samples
      write(*,*) "Sample ", i, ": ", predictions(1,1)%val(:, i)
   end do

   ! Alternatively, use the extract() method
   call predictions(1,1)%extract(pred_vals)
   write(*,*) "Extracted Predictions:"
   do i = 1, size(pred_vals, 3)  ! Loop over samples
      write(*,*) "Sample ", i, ": ", pred_vals(:, :, i)
   end do


``graph_type`` Input/Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``graph_type`` is another core data structure in the athena library that represents computational graphs for graph neural networks (GNNs) and message-passing architectures.
It encapsulates nodes, edges, and their features, allowing for efficient representation and processing of graph-structured data.
Documentation for `graphstruc <https://graphstruc.readthedocs.io/en/latest/>`_ is available, but is currently limited.
Examples of its usage be found in the ``example/example_library``, ``example/msgpass_chemical`` and ``example/msgpass_euler`` directories.
The source code for the ``graph_type`` can be found in the `graphstruc GitHub repository <https://github.com/nedtaylor/graphstruc>`_ (the current branch being utilised is the ``main`` branch).

A brief outline of the data type and its components is provided here:

* The ``graph_type`` contains arrays for node features, edge features, and global graph features.
* It also includes connectivity information, such as adjacency lists or matrices, to define relationships between nodes and edges.
* ``vertex_features``: A 2D ``array_type`` array storing features for each node in the graph, shape :math:`(F_v, N_v)`, where :math:`F_v` is the number of vertex features and :math:`N_v` is the number of vertices.
* ``edge_features``: A 2D ``array_type`` array storing features for each edge in the graph, shape :math:`(F_e, N_e)`, where :math:`F_e` is the number of edge features and :math:`N_e` is the number of edges.
* ``graph_features``: A 1D ``array_type`` array storing global features for the entire graph, shape :math:`(F_g)`, where :math:`F_g` is the number of graph features (note: this is not yet utilised in athena).
* ``adj_ia`` and ``adj_ja``: Arrays representing the adjacency information for efficient message passing, look up Compressed Sparse Row (CSR) format for more details.
* ``num_vertices`` and ``num_edges``: Integers representing the number of vertices and edges in the graph.
* ``num_vertex_features``, ``num_edge_features``, and ``num_graph_features``: Integers representing the number of features for vertices, edges, and the graph.
* ``edge_weights``: Optional array for edge weights, shape :math:`(N_e)`.

The useful type-bound procedures provided in the ``graph_type`` include:

* ``add_vertex()``: Add a vertex with features to the graph.
* ``add_edge()``: Add an edge with features between two vertices.
* ``set_num_vertices()``: Set the total number of vertices in the graph.
* ``set_num_edges()``: Set the total number of edges in the graph.
* ``remove_vertices()``: Remove specified vertices from the graph.
* ``remove_edges()``: Remove specified edges from the graph.
* ``add_self_loops()``: Add self-loops to all vertices in the graph.
* ``remove_self_loops()``: Remove self-loops from the graph.

There also exists non-sparse formatting for the vertex features, edge features, and adjacency information, but these are less efficient for large graphs and are not recommended for general use beyond testing.

If the input is a ``graph_type`` (ranks 1 or 2), the output will be a 2D ``graph_type`` array (like with ``array_type``, rank 2 output is used for ):

.. code-block:: fortran

   type(graph_type), dimension(:), allocatable :: test_input
   type(graph_type), dimension(:,:), allocatable :: predictions

   ! Load or prepare test_input data
   ! ...

   ! Inference without gradient tracking
   predictions = network%predict(test_input)

These different input/output types provide flexibility for various use cases.


Polymorphic Input/Output - ``predict_generic()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, the ``predict_generic()`` method can handle any of the above input types, as well as the ``array_ptr_type``.
Its main use is to return a different type than the input type, controlled by the optional argument ``output_as_graph``.
The input can be any data type and, if the network is expected to output a graph, setting ``output_as_graph=.false.`` will return an ``array_type`` output instead.


Custom Operations - ``forward_eval()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``forward_eval()`` method returns a pointer to the network output, allowing custom operations while preserving the computation graph.
Internally, this method calls the standard ``forward()`` method but returns a pointer to the output array instead of a copy.
As such, this only works for networks with a single output layer.
This is useful for implementing custom training loops, loss functions, and physics-informed neural networks (PINNs).

.. code-block:: fortran

   type(array_type), pointer :: output(:,:)
   type(array_type), pointer :: output_ptr
   type(array_type), pointer :: loss

   ! Forward pass returns output pointer
   output => network%forward_eval(input_batch)

   ! Preserve computation graph
   output_ptr => output(1,1)%duplicate_graph()

   ! Custom loss computation
   loss => (output_ptr - target)**2

   ! Backpropagate custom loss
   call loss%grad_reverse()
   call network%update()

Note that this only works for a network with a single output layer, otherwise it does not know what to point to.
The fact that it returns a pointer means that the graphs are preserved, allowing for gradient computation through custom operations.


Example: Recurrent Network
--------------------------

This example demonstrates custom loss computation with a recurrent network for time series prediction.
The complete code is available in ``example/rnn_timeseries``.

The key aspect here is that we use ``forward_eval()`` to get the output pointer at each time step, allowing us to accumulate a custom loss over the entire sequence.
This makes it easier to extract the output of a single-output network without having to access the layers directly.

.. rubric:: :h3style:`Network Setup:`

.. code-block:: fortran

   use athena
   implicit none

   type(network_type) :: network

   ! Build RNN architecture
   call network%add(input_layer_type(input_shape=[1]))

   call network%add(recurrent_layer_type( &
        num_inputs=1, &
        num_outputs=32, &
        activation_type="tanh", &
        use_bias=.true.))

   call network%add(full_layer_type( &
        num_outputs=1, &
        activation_type="none"))

   ! Compile with optimiser (no loss function needed)
   call network%compile( &
        optimiser_type=optimiser_type( &
             name="adam", &
             learning_rate=0.001_real32), &
        loss_method="none")

Note that we don't specify a loss function because we'll compute it manually.

.. rubric:: :h3style:`Custom Training Loop:`

.. code-block:: fortran

   type(array_type), pointer :: output(:,:)
   type(array_type), pointer :: output_ptr
   type(array_type), pointer :: loss
   type(array_type), dimension(1,1) :: x_array, y_array
   integer :: epoch, t

   do epoch = 1, num_epochs
      ! Reset hidden state at start of sequence
      call network%reset_state()

      ! Initialise loss accumulator
      nullify(loss)

      ! Process sequence
      do t = 1, sequence_length - 1
         ! Prepare input and target
         x_array(1,1) = array_type(reshape([x(t)], [1,1]))
         y_array(1,1) = array_type(reshape([x(t+1)], [1,1]))

         ! Forward pass returns output pointer
         output => network%forward_eval(x_array)

         ! Preserve computation graph
         output_ptr => output(1,1)%duplicate_graph()

         ! Accumulate custom loss (MSE for each time step)
         if (.not. associated(loss)) then
            loss => (output_ptr - y_array(1,1))**2
         else
            loss => loss + (output_ptr - y_array(1,1))**2
         end if
      end do

      ! Average loss over sequence
      loss => loss / real(sequence_length - 1, real32)

      ! Backpropagate accumulated loss
      call loss%grad_reverse()

      ! Update network parameters
      call network%update()
   end do

.. rubric:: :h3style:`Prediction:`

After training, use ``predict()`` for inference:

.. code-block:: fortran

   real(real32), dimension(1,1) :: x_test
   real(real32), dimension(1,1) :: pred

   ! Reset state for new sequence
   call network%reset_state()

   ! Generate predictions
   do t = 1, test_length
      x_test(1,1) = test_data(t)
      pred = network%predict(x_test)

      print *, "Time:", t, "Prediction:", pred(1,1), "Actual:", test_data(t+1)
   end do

Remember that the outputs from ``predict()`` do not track gradients and are suitable only for evaluation.

Another example can be found in the ``example/pinn_burgers`` directory.
However, this example uses direct access to the layer outputs instead of ``forward_eval()``.

.. _advanced-patterns:

Advanced Methods
----------------

Multi-Output Operations - ``forward_eval_multi()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For networks with multiple output layers, the ``forward_eval_multi()`` method can be used to return an array of output pointers, one for each output layer.
This returns a pointer to a 1D ``array_ptr_type`` array, where each element points to the output of a different layer.

Each element of the ``array_ptr_type`` array contains a rank 2 ``array_type`` array for the corresponding output layer.
Hence, the componenet ``array`` of each element can be accessed to get the actual output data, just like a single-output network.

An example of using ``forward_eval_multi()`` with a multi-output network is shown below:

.. code-block:: fortran

   type(array_ptr_type), pointer :: outputs(:)
   type(array_type), pointer :: output1(:,:), output2(:,:)
   type(array_type), pointer :: loss

   ! Forward pass returns array of output pointers
   outputs => network%forward_eval_multi(input_batch)

   ! Access different outputs
   output1 => outputs(1)%array
   output2 => outputs(2)%array

   ! Multi-task loss
   loss => mse(output1, target1) + 0.5_real32*mse(output2, target2)

   call loss%grad_reverse()
   call network%update()

.. note::

  The ``forward_eval_multi()`` method is currently under development, so may not be fully stable yet.

Intermediate Layer Access
~~~~~~~~~~~~~~~~~~~~~~~~~~

Access intermediate layer outputs for constraints or visualisation:

.. code-block:: fortran

   type(array_type), pointer :: output(:,:)
   type(array_type), dimension(:,:), pointer :: hidden_layer

   ! Forward pass
   output => network%forward_eval(input_batch)

   ! Access specific layer output
   hidden_layer => network%model(3)%layer%output

   ! Add constraint on hidden layer
   ! e.g., sparsity penalty
   loss => task_loss + lambda*sum(abs(hidden_layer(1,1)))

Related Topics
--------------

See also:

* :ref:`custom-loss`: Creating custom loss functions
* :ref:`pinn-example`: Complete PINN examples
* :ref:`Recurrent Layer <recurrent-layer>`: RNN layer documentation

Examples
--------

Complete working examples demonstrating these techniques:

* ``example/rnn_timeseries``: Custom RNN training loop with forward_eval
* ``example/pinn_burgers``: Solving PDEs with physics-informed loss
* ``example/pinn_chemical``: Molecular force prediction with automatic differentiation
* ``example/msgpass_euler``: Graph neural networks with custom message passing

Summary
-------

Key takeaways:

1. **predict()** only for inference without gradients
2. **forward_eval()** enables custom operations for single-output networks while preserving gradients
3. **forward_eval_multi()** enables custom operations while preserving gradients
4. **duplicate_graph()** is essential when accumulating or storing outputs
5. Manual loss computation allows arbitrary physics and constraints
6. Combine methods for advanced architectures like sequence-to-sequence models
7. If you're comfortable with it, direct access of layer outputs is also an option
