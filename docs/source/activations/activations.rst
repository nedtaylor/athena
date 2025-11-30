.. _activation-functions:

Activation Functions
====================

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.
The athena library provides a variety of commonly used activation functions.

.. toctree::
   :maxdepth: 1
   :caption: Available Activation Functions

   gaussian
   linear
   relu
   leaky_relu
   sigmoid
   tanh
   softmax
   swish
   selu
   none
   custom_activations

Creating custom activation functions
------------------------------------

The athena library is designed with extensibility in mind, allowing users to create custom activation functions by extending the ``base_activation_type``.

See :ref:`Creating Custom Activations <custom-activations>` for a detailed guide.
