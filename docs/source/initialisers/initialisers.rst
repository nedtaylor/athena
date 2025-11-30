.. _initialisers:

Initialisers
============

Weight initialization is crucial for training neural networks effectively. Proper initialization helps prevent vanishing or exploding gradients and can significantly impact convergence speed.
The athena library provides various initialization strategies suited for different activation functions and network architectures.

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
