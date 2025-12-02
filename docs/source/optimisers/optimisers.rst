.. _optimisers:

Optimisers
==========

Optimisers (or optimisers) are algorithms that adjust neural network weights to minimise the loss function during training.
The athena library implements several popular optimisation algorithms, each with different characteristics and use cases.

.. toctree::
   :maxdepth: 1
   :caption: Available Optimisers

   sgd
   adam
   rmsprop
   adagrad

Creating custom optimisers
--------------------------

The athena library is designed with extensibility in mind, allowing users to create custom optimisers by extending the ``base_optimiser_type``.

See the tutorial: :ref:`Creating Custom Optimisers <custom-optimisers>`
