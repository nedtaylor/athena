=====
About
=====


**athena** is an open-source Fortran library enabling the building and training of neural networks in Fortran.
It does this by providing a suite of extendable derived types and procedures for constructing various types of neural network layers, loss functions, and optimisation algorithms.
The library focuses on convolutional layers, message passing layers, and physics informed neural networks, but also includes other layer types.
The key derived types are `network_type`, `base_layer_type`, `base_loss_type`, and `base_optimiser_type`, with the latter two being extendable to create custom layers, loss functions, and optimisers.

The library is designed with the open-closed principle [#f1]_ in mind, allowing users to easily extend the library without modifying the source code.
The library is also designed to be user-friendly (hopefully this has been achieved), allowing users to easily integrate neural networks into their Fortran projects with minimal setup.
A core philosophy is to make this as user-friendly as possible, with clear documentation and examples provided.

The library utilises modern Fortran features such as modules, derived types, and pointers to manage memory and data structures effectively.
Much of the data handling is managed by the `diffstruc <https://github.com/nedtaylor/diffstruc>`_ library, which provides automatic differentiation capabilities in Fortran.
If you have any interest in contributing to the athena library or have suggestions for improvements, we are happy to have you get involved; for more information on contributing, please refer to the (:git:`contributing guidelines<CONTRIBUTING.md>`).

The format of names and arguments are made to closely align with industry standards in popular Python libraries such as PyTorch and TensorFlow/Keras, to facilitate ease of understanding and accessibility for newcomers to the athena library.

The code is freely available under the `MIT License <https://opensource.org/licenses/MIT>`_.

Footnotes
---------

.. [#f1] The open-closed principle is a software design principle that states that software entities (modules, functions, derived types, etc.) should be open for extension but closed for modification.
   This means that the behaviour of a module can be extended without modifying its source code, typically through mechanisms such as inheritance, interfaces, and composition.
   This is a key principle of object-oriented design and is one of the SOLID principles of software development.
