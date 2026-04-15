.. _graph-nop-layer:

Graph Neural Operator Layer
===========================

``graph_nop_layer_type``

.. code-block:: fortran

  graph_nop_layer_type(
    num_outputs,
    coord_dim,
    kernel_hidden=...,
    num_inputs=...,
    use_bias=.true.,
    activation="none",
    kernel_initialiser=...,
    bias_initialiser=...,
    verbose=0
  )


The ``graph_nop_layer_type`` derived type provides a graph neural operator layer for irregular meshes and general sparse graphs.
It combines a local linear bypass with a learned kernel evaluated on per-edge geometric features, typically relative coordinates:

.. math::

   h_i^{(l+1)} = \sigma\left(
      \mathbf{W} h_i^{(l)} +
      \sum_{j \in \mathcal{N}(i)}
      \kappa_\theta(\Delta x_{ij}) h_j^{(l)} + \mathbf{b}
   \right)

where:

* :math:`h_i^{(l)} \in \mathbb{R}^{F_{in}}` is the input feature vector at node :math:`i`
* :math:`\Delta x_{ij} \in \mathbb{R}^{d}` is the edge geometry for the directed edge :math:`j \to i`
* :math:`\kappa_\theta : \mathbb{R}^{d} \to \mathbb{R}^{F_{out} \times F_{in}}` is a learnable kernel network
* :math:`\mathbf{W} \in \mathbb{R}^{F_{out} \times F_{in}}` is the learnable local bypass matrix
* :math:`\mathbf{b} \in \mathbb{R}^{F_{out}}` is the bias vector when ``use_bias=.true.``
* :math:`\sigma` is the activation function

The kernel network is a one-hidden-layer MLP applied independently to every directed edge:

.. math::

   \kappa_\theta(\Delta x) = \mathbf{V} \, \mathrm{relu}(\mathbf{U} \Delta x + \mathbf{b}_u) + \mathbf{b}_v

where:

* :math:`\Delta x` is supplied as an edge feature vector, for example :math:`x_i - x_j`
* :math:`\mathbf{U} \in \mathbb{R}^{H \times d}`
* :math:`\mathbf{V} \in \mathbb{R}^{(F_{out} F_{in}) \times H}`
* :math:`\mathbf{b}_u \in \mathbb{R}^{H}`
* :math:`\mathbf{b}_v \in \mathbb{R}^{F_{out} F_{in}}`
* :math:`H` is ``kernel_hidden``

This makes the layer suitable for operator learning on point clouds, unstructured meshes, and graph-structured discretisations where neighbor interactions depend on geometry.

Arguments
---------

* **num_outputs** (``integer``): Number of output node features :math:`F_{out}`.
* **coord_dim** (``integer``): Dimensionality of the edge geometric features :math:`d`.
* **kernel_hidden** (``integer``): Hidden width :math:`H` of the kernel MLP. If not provided, it defaults to ``num_outputs``.
* **num_inputs** (``integer``): Number of input node features :math:`F_{in}`. If not provided, it is inferred when the layer is initialised.
* **use_bias** (``logical``): If ``.false.``, the layer omits the output bias term. Default: ``.true.``.
* **activation** (``class(*)``): Activation function applied after aggregation and bypass combination.

  * Accepts ``character(*)`` or ``class(base_actv_type)``.
  * See :ref:`Activation Functions <activation-functions>` for available options.
  * Default: ``none_actv_type``.

* **kernel_initialiser** (``class(*)``): Initialiser for the kernel MLP and bypass weights (see :ref:`Initialisers <initialisers>`).

  * If ``activation`` is ``selu_actv_type``, default: ``lecun_normal_init_type``.
  * If ``activation`` is a version of ``relu_actv_type``, default: ``he_normal_init_type``.
  * For all other activations, default: ``glorot_uniform_init_type``.

* **bias_initialiser** (``class(*)``): Initialiser for the biases (see :ref:`Initialisers <initialisers>`). Default: ``zeros_init_type``.
* **verbose** (``integer``, optional): Verbosity level for initialisation. Default: ``0``.

Shape
-----

This layer consumes graph-structured data with two input channels per sample:

* ``input(1, s)``: Node features of shape ``(num_inputs, num_vertices)``
* ``input(2, s)``: Edge geometric features of shape ``(coord_dim, num_edges)``

The output is node-level for vertex features and preserves edge geometry:

* ``output(1, s)``: Updated node features of shape ``(num_outputs, num_vertices)``
* ``output(2, s)``: Propagated edge features of shape ``(coord_dim, num_edges)``

The graph connectivity is not inferred from the edge-feature tensor. It must be provided separately through ``set_graph(...)`` on the layer, or by passing graph-valued training data through ``network%train(...)`` so the network can propagate adjacency information to the message-passing layer.

When stacking GNO layers, ``output(2, s)`` is forwarded unchanged so later GNO blocks continue to receive the same geometric edge descriptors.

Parameters
----------

The layer contains the following learnable parameters:

* **U**: First kernel MLP weight matrix of shape ``(kernel_hidden, coord_dim)``.
* **b_u**: First kernel MLP bias vector of shape ``(kernel_hidden)``.
* **V**: Second kernel MLP weight matrix of shape ``(num_outputs * num_inputs, kernel_hidden)``.
* **b_v**: Kernel output bias vector of shape ``(num_outputs * num_inputs)``.
* **W**: Local bypass matrix of shape ``(num_outputs, num_inputs)``.
* **b**: Output bias vector of shape ``(num_outputs)`` when ``use_bias=.true.``.

Let :math:`F = num_outputs \cdot num_inputs`. The total number of learnable parameters is:

* With bias: ``kernel_hidden * coord_dim + kernel_hidden + F * kernel_hidden + 2 * F + num_outputs``
* Without bias: ``kernel_hidden * coord_dim + kernel_hidden + F * kernel_hidden + 2 * F``

Examples
--------

**Single GNO layer on a graph:**

.. code-block:: fortran

   use athena
   type(graph_type), dimension(1) :: graph
   type(graph_nop_layer_type) :: layer
   type(array_type), allocatable :: input(:,:)

   layer = graph_nop_layer_type( &
        num_inputs=3, &
        num_outputs=8, &
        coord_dim=2, &
        kernel_hidden=16, &
        activation="relu")

   call layer%set_graph(graph)

   allocate(input(2, 1))
   call input(1,1)%allocate(array_shape=[3, graph(1)%num_vertices])
   call input(2,1)%allocate(array_shape=[2, graph(1)%num_edges])
   call layer%forward(input)

**Stacked GNO network:**

.. code-block:: fortran

   use athena
   type(network_type) :: network

   call network%add(graph_nop_layer_type( &
        num_inputs=1, &
        num_outputs=8, &
        coord_dim=1, &
        kernel_hidden=8, &
        activation="relu"))

   call network%add(graph_nop_layer_type( &
        num_outputs=2, &
        coord_dim=1, &
        kernel_hidden=8, &
        activation="none"))

   call network%compile( &
        optimiser=base_optimiser_type(learning_rate=0.01_real32), &
        loss_method="mse", &
        metrics=["loss"])

Notes
-----

* This layer extends ``msgpass_layer_type`` and preserves node-level outputs rather than performing graph-level pooling.
* The learnable kernel depends on edge geometry, so translation-invariant interactions can be represented by supplying relative coordinates such as :math:`x_i - x_j` as edge features.
* The forward pass is built entirely from ``array_type``-based differentiable operations, making the layer compatible with ATHENA's autodiff workflows, including physics-informed use cases.
* For stable performance on sparse graphs, ensure the graph adjacency is prepared before calling ``forward`` or training the enclosing network.

See Also
--------

* :ref:`neural_operator_layer_type <neural-operator-layer>` - Mean-field neural operator layer on regular discretisations
* :ref:`fixed_lno_layer_type <fixed-lno-layer>` - Laplace neural operator layer with fixed encoder/decoder bases and spectral mixing
* :ref:`kipf_msgpass_layer_type <kipf-msgpass-layer>` - Degree-normalised message passing layer
