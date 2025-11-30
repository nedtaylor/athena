.. _optimisers:

Optimisers
==========

Optimisers (or optimizers) are algorithms that adjust neural network weights to minimize the loss function during training.
The athena library implements several popular optimization algorithms, each with different characteristics and use cases.

.. toctree::
   :maxdepth: 1
   :caption: Available Optimisers

   sgd
   adam
   rmsprop
   adagrad
   custom_optimisers

Creating custom optimisers
--------------------------

The athena library is designed with extensibility in mind, allowing users to create custom optimisers by extending the ``base_optimiser_type``.

See :ref:`Creating Custom Optimisers <custom-optimisers>` for a detailed guide.
