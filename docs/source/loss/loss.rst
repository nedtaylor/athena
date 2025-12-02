.. _loss-functions:

Loss Functions
==============

Loss functions measure how well a neural network's predictions match the expected outputs. They guide the optimisation process by providing a scalar value to minimise during training.

The athena library provides several commonly used loss functions for different types of problems.

.. toctree::
   :maxdepth: 1
   :caption: Available Loss Functions

   bce
   cce
   mae
   mse
   nll
   huber

Overview
--------

Choosing the Right Loss Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Binary Classification**: Use :ref:`BCE <bce-loss>` (Binary Cross Entropy)
* **Multi-class Classification**: Use :ref:`CCE <cce-loss>` or :ref:`NLL <nll-loss>`
* **Regression**: Use :ref:`MSE <mse-loss>`, :ref:`MAE <mae-loss>`, or :ref:`Huber <huber-loss>`

Loss Function Properties
~~~~~~~~~~~~~~~~~~~~~~~~~

All loss functions in athena:

* Extend ``base_loss_type``
* Implement a ``compute`` method
* Work with differentiable ``array_type`` for automatic gradient computation
* Support batch processing

Usage Example
-------------

.. code-block:: fortran

   use athena__loss
   use athena__network

   ! Create a network
   type(network_type) :: net
   type(mse_loss_type) :: loss

   ! Initialise loss function
   loss = mse_loss_type()

   ! Use in training
   call net%train(train_data, train_labels, loss=loss)

Creating custom loss functions
-------------------------------

The athena library is designed with extensibility in mind, allowing users to create custom loss functions by extending the ``base_loss_type``.

See the tutorial: :ref:`Creating Custom Loss Functions <custom-loss>`
