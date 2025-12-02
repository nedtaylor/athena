.. _initialisers:

Initialisers
============

Weight initialisation is crucial for training neural networks effectively. Proper initialisation helps prevent vanishing or exploding gradients and can significantly impact convergence speed.
The athena library provides various initialisation strategies suited for different activation functions and network architectures.

.. toctree::
   :maxdepth: 1
   :caption: Available Initialisers

   glorot_uniform
   glorot_normal
   he_uniform
   he_normal
   lecun_uniform
   lecun_normal
   zeros
   ones
   identity
   gaussian


Creating custom initialisers
----------------------------

The athena library is designed with extensibility in mind, allowing users to create custom initialisers by extending the ``base_initialiser_type``.

See the tutorial: :ref:`Creating Custom Initialisers <custom-initialisers>`
