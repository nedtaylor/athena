.. _lr-decay:

Learning Rate Decay
===================

Gradually reduces the learning rate during training.

Available Types
---------------

* **base_lr_decay_type**: No decay (constant learning rate)
* **exp_lr_decay_type**: Exponential decay
* **step_lr_decay_type**: Step-wise decay at fixed intervals
* **inv_lr_decay_type**: Inverse time decay

Exponential Decay
-----------------

``exp_lr_decay_type``

.. code-block:: fortran

  exp_lr_decay_type(decay_rate=...)

Smooth exponential decay: :math:`\eta_t = \eta_0 \cdot e^{-t \cdot r}`

Arguments
~~~~~~~~~

* **decay_rate** (`real`, optional): Decay rate. Default: ``0.9``.

Usage
~~~~~

.. code-block:: fortran

   use athena

   type(exp_lr_decay_type) :: lr_schedule

   lr_schedule = exp_lr_decay_type(decay_rate=0.01)

   call network%compile( &
        optimiser_type=adam_optimiser_type( &
             learning_rate=0.001, &
             lr_decay=lr_schedule), &
        loss_method="categorical_crossentropy")

Step Decay
----------

``step_lr_decay_type``

.. code-block:: fortran

  step_lr_decay_type(decay_rate=..., decay_steps=...)

Discrete drops every *n* epochs: :math:`\eta_t = \eta_0 \cdot r^{\lfloor t / s \rfloor}`

Arguments
~~~~~~~~~

* **decay_rate** (`real`, optional): Multiplicative decay factor. Default: ``0.1``.
* **decay_steps** (`integer`, optional): Number of epochs between decays. Default: ``100``.

Usage
~~~~~

.. code-block:: fortran

   type(step_lr_decay_type) :: lr_schedule

   ! Reduce by half every 10 epochs
   lr_schedule = step_lr_decay_type( &
        decay_rate=0.5, &
        decay_steps=10)

   call network%compile( &
        optimiser_type=sgd_optimiser_type( &
             learning_rate=0.1, &
             lr_decay=lr_schedule), &
        loss_method="mse")

Inverse Time Decay
------------------

``inv_lr_decay_type``

.. code-block:: fortran

  inv_lr_decay_type(decay_rate=..., decay_power=...)

Inverse time decay: :math:`\eta_t = \frac{\eta_0}{(1 + r \cdot t)^p}`

Arguments
~~~~~~~~~

* **decay_rate** (`real`, optional): Decay rate coefficient. Default: ``0.001``.
* **decay_power** (`real`, optional): Exponent for decay. Default: ``1.0``.

Usage
~~~~~

.. code-block:: fortran

   type(inv_lr_decay_type) :: lr_schedule

   lr_schedule = inv_lr_decay_type( &
        decay_rate=0.001, &
        decay_power=1.0)

   call network%compile( &
        optimiser_type=adam_optimiser_type( &
             learning_rate=0.01, &
             lr_decay=lr_schedule), &
        loss_method="binary_crossentropy")

Typical Values
--------------

Exponential Decay
~~~~~~~~~~~~~~~~~

* Small datasets (<1k samples): ``decay_rate`` = 0.001 to 0.01
* Medium datasets (1k-100k): ``decay_rate`` = 0.01 to 0.05
* Large datasets (>100k): ``decay_rate`` = 0.05 to 0.1

Step Decay
~~~~~~~~~~

* Conservative: ``decay_rate`` = 0.5, ``decay_steps`` = 20-50
* Aggressive: ``decay_rate`` = 0.1, ``decay_steps`` = 10-30
* Fine-tuning: ``decay_rate`` = 0.3, ``decay_steps`` = 5-10

Inverse Time Decay
~~~~~~~~~~~~~~~~~~

* Long training: ``decay_rate`` = 0.0001 to 0.001, ``decay_power`` = 0.5 to 1.0
* Medium training: ``decay_rate`` = 0.001 to 0.01, ``decay_power`` = 1.0

See Also
--------

* :ref:`Training Configuration <training-config>`: Overview
* :ref:`Gradient Clipping <gradient-clipping>`: Preventing exploding gradients
* :ref:`Regularisation <regularisation>`: Preventing overfitting
