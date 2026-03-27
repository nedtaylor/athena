.. _train-subroutine:

train() Subroutine
==================

The high-level ``train()`` method runs minibatch training for a compiled
``network_type``.

Signature
---------

.. code-block:: fortran

   call network%train( &
        input, output, num_epochs, &
        batch_size, plateau_threshold, shuffle_batches, &
        batch_print_step, verbose, print_precision, scientific_print)

Arguments
---------

* ``input`` (required): Training input data.
* ``output`` (required): Training targets.
* ``num_epochs`` (required): Number of epochs.
* ``batch_size`` (optional): Batch size used for training.
* ``plateau_threshold`` (optional): Plateau check threshold.
* ``shuffle_batches`` (optional): Whether to shuffle batch order each epoch.
* ``batch_print_step`` (optional): Batch print interval for verbose batch logs.
* ``verbose`` (optional): Controls print cadence.
* ``print_precision`` (optional): Decimal precision for printed metrics.
* ``scientific_print`` (optional): Print metrics in scientific notation.

Print Options
-------------

``train()`` prints loss and (optionally) accuracy using one shared formatting
path:

* ``print_precision`` controls precision for both ``loss`` and
  ``accuracy``.
* ``scientific_print=.true.`` prints metrics using exponential notation.
* ``scientific_print=.false.`` prints metrics using fixed-point notation.

Print cadence is controlled by ``verbose``:

* ``verbose = 0``: prints one epoch summary line.
* ``abs(verbose) > 0``: prints batch progress, starting at batch 1 and then
  every ``batch_print_step`` batches.

Accuracy Printing Behavior
--------------------------

Accuracy is optional.

* If an accuracy method is configured (for example via
  ``compile(..., accuracy_method=...)``), ``accuracy`` is printed.
* If no accuracy method is configured, accuracy is not computed or printed, and
  output lines contain only loss.

Examples
--------

With accuracy and scientific metric printing:

.. code-block:: fortran

   call network%compile( &
        optimiser=adam_optimiser_type(learning_rate=1.0e-3), &
        loss_method="mse", &
        accuracy_method="mse")

   call network%train( &
        input=train_x, output=train_y, num_epochs=100, &
        batch_size=32, verbose=1, batch_print_step=20, &
        print_precision=6, scientific_print=.true.)

Without accuracy (loss-only logging):

.. code-block:: fortran

   call network%compile( &
        optimiser=adam_optimiser_type(learning_rate=1.0e-3), &
        loss_method="mse")

   call network%train( &
        input=train_x, output=train_y, num_epochs=100, &
        verbose=0, print_precision=4)
