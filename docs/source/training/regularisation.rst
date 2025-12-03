.. _regularisation:

Regularisation
==============

Adds penalty terms to the loss function to prevent overfitting by encouraging simpler models.

Available Types
---------------

* **l1_regulariser_type**: L1 regularisation (Lasso)
* **l2_regulariser_type**: L2 regularisation (Ridge)
* **l1l2_regulariser_type**: Combined L1 and L2 regularisation

L1 Regularisation
-----------------

``l1_regulariser_type``

Adds penalty based on absolute values of weights: :math:`L_{total} = L + \lambda_1 \sum_i |w_i|`

Encourages sparsity (many weights become exactly zero).

.. code-block:: fortran

  l1_regulariser_type()

Attributes
~~~~~~~~~~

* **l1** (`real`): L1 regularisation parameter. Default: ``0.01``.

Usage
~~~~~

.. code-block:: fortran

   use athena

   type(l1_regulariser_type) :: regulariser

   regulariser = l1_regulariser_type()
   regulariser%l1 = 0.001

   call network%compile( &
        optimiser_type=adam_optimiser_type( &
             learning_rate=0.001, &
             regulariser=regulariser), &
        loss_method="mse")

L2 Regularisation
-----------------

``l2_regulariser_type``

Adds penalty based on squared weights: :math:`L_{total} = L + \lambda_2 \sum_i w_i^2`

Encourages small weights distributed across all parameters.

.. code-block:: fortran

  l2_regulariser_type()

Attributes
~~~~~~~~~~

* **l2** (`real`): L2 regularisation parameter. Default: ``0.01``.
* **l2_decoupled** (`real`): Decoupled weight decay parameter (AdamW). Default: ``0.01``.
* **decoupled** (`logical`): Use decoupled weight decay. Default: ``.true.``.

Usage
~~~~~

.. code-block:: fortran

   type(l2_regulariser_type) :: regulariser

   regulariser = l2_regulariser_type()
   regulariser%l2 = 0.0001

   call network%compile( &
        optimiser_type=adam_optimiser_type( &
             learning_rate=0.001, &
             regulariser=regulariser), &
        loss_method="categorical_crossentropy")

Decoupled Weight Decay (AdamW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Adam optimiser, decoupled weight decay is recommended:

.. code-block:: fortran

   regulariser = l2_regulariser_type()
   regulariser%l2_decoupled = 0.01
   regulariser%decoupled = .true.  ! Default

   call network%compile( &
        optimiser_type=adam_optimiser_type( &
             learning_rate=0.001, &
             regulariser=regulariser), &
        loss_method="mse")

Combined L1/L2 Regularisation
------------------------------

``l1l2_regulariser_type``

Also known as Elastic Net regularisation. Combines both L1 and L2 penalties.

.. code-block:: fortran

  l1l2_regulariser_type()

Attributes
~~~~~~~~~~

* **l1** (`real`): L1 regularisation parameter. Default: ``0.01``.
* **l2** (`real`): L2 regularisation parameter. Default: ``0.01``.

Usage
~~~~~

.. code-block:: fortran

   type(l1l2_regulariser_type) :: regulariser

   regulariser = l1l2_regulariser_type()
   regulariser%l1 = 0.0001
   regulariser%l2 = 0.001

   call network%compile( &
        optimiser_type=sgd_optimiser_type( &
             learning_rate=0.01, &
             regulariser=regulariser), &
        loss_method="mse")

Typical Values
--------------

L1 Regularisation
~~~~~~~~~~~~~~~~~

* Weak: ``l1`` = 0.00001 to 0.0001
* Moderate: ``l1`` = 0.0001 to 0.001
* Strong: ``l1`` = 0.001 to 0.01

Use L1 when you want feature selection or sparse models.

L2 Regularisation
~~~~~~~~~~~~~~~~~

* Weak: ``l2`` = 0.00001 to 0.0001
* Moderate: ``l2`` = 0.0001 to 0.001
* Strong: ``l2`` = 0.001 to 0.01

Use L2 as the default choice for most problems.

L1/L2 Combined
~~~~~~~~~~~~~~

* Typically use ``l1`` << ``l2`` (e.g., ``l1`` = 0.0001, ``l2`` = 0.001)
* Start with L2 alone, add L1 if you need sparsity

When to Use
-----------

* **Small datasets**: Use stronger regularisation (higher λ values)
* **Large models**: Models with many parameters benefit from regularisation
* **Overfitting symptoms**: Large gap between training and validation performance
* **Less with dropout**: If using dropout layers, reduce regularisation strength
* **Less with batch norm**: Batch normalisation provides some regularisation effect

See Also
--------

* :ref:`Training Configuration <training-config>`: Overview
* :ref:`Gradient Clipping <gradient-clipping>`: Preventing exploding gradients
* :ref:`Learning Rate Decay <lr-decay>`: Gradually reducing learning rate
