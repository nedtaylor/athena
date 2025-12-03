.. _gradient-clipping:

Gradient Clipping
=================

``clip_type``

Prevents gradient explosion by limiting gradient magnitudes during backpropagation.

.. code-block:: fortran

  clip_type(
    clip_min=...,
    clip_max=...,
    clip_norm=...
  )

Arguments
---------

* **clip_min** (`real`, optional): Minimum allowed gradient value
* **clip_max** (`real`, optional): Maximum allowed gradient value
* **clip_norm** (`real`, optional): Maximum allowed L2-norm

Value clipping and norm clipping can be used independently or together.

Usage
-----

Norm Clipping
~~~~~~~~~~~~~

.. code-block:: fortran

   use athena

   type(clip_type) :: clipper

   clipper = clip_type(clip_norm=1.0)

   call network%compile( &
        optimiser_type=adam_optimiser_type( &
             learning_rate=0.001, &
             clip_dict=clipper), &
        loss_method="mse")

Value Clipping
~~~~~~~~~~~~~~

.. code-block:: fortran

   clipper = clip_type(clip_min=-0.5, clip_max=0.5)

   call network%compile( &
        optimiser_type=sgd_optimiser_type( &
             learning_rate=0.01, &
             clip_dict=clipper), &
        loss_method="categorical_crossentropy")

Combined Clipping
~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   clipper = clip_type( &
        clip_min=-1.0, &
        clip_max=1.0, &
        clip_norm=5.0)

Typical Values
--------------

* RNNs/LSTMs: ``clip_norm`` = 0.5 to 2.0
* GRUs: ``clip_norm`` = 1.0 to 5.0
* CNNs: ``clip_norm`` = 5.0 to 10.0
* GNNs: ``clip_norm`` = 0.5 to 2.0
* PINNs: ``clip_min`` = -0.1, ``clip_max`` = 0.1

See Also
--------

* :ref:`Training Configuration <training-config>`: Overview
* :ref:`Learning Rate Decay <lr-decay>`: Gradually reducing learning rate
* :ref:`Regularisation <regularisation>`: Preventing overfitting
