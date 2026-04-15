.. _layers:

Layers
======

The athena library provides a variety of layers commonly used in building neural networks.
These layers can be stacked and combined to create complex architectures for various machine learning tasks.

.. toctree::
   :hidden:
   :maxdepth: 2

   core/core_layers
   convolutional/convolutional_layers
   input/input_layers
   merge/merge_layers
   msgpass/msgpass_layers
   neural_operator/neural_operator_layers
   normalisation/normalisation_layers
   padding/padding_layers
   pooling/pooling_layers
   recurrent/recurrent_layers
   regularisation/regularisation_layers
   reshaping/reshaping_layers

Available Layers
----------------

The following types of layers are available in the athena library:

* :ref:`core-layers`
* :ref:`convolutional-layers`
* :ref:`input-layers`
* :ref:`merge-layers`
* :ref:`msgpass-layers`
* :ref:`neural-operator-layers`
* :ref:`normalisation-layers`
* :ref:`padding-layers`
* :ref:`pooling-layers`
* :ref:`recurrent-layers`
* :ref:`regularisation-layers`
* :ref:`reshaping-layers`

Creating custom layers
----------------------

The athena library is designed with extensibility in mind, allowing users to create custom layers by extending the ``base_layer_type``.

.. rubric:: Blocks

Blocks are a convenient way to group multiple layers together into a single unit that can be added to a network.
An example of a custom block is the :ref:`orthogonal_nop_block_type <orthogonal-nop-block>`, which combines an orthogonal basis projection with a spectral mixing path and a local bypass.

See the tutorial: :ref:`Creating Custom Layers <custom-layers>`
