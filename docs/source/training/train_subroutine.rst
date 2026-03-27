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
        batch_print_step, verbose, print_precision, scientific_print, &
        early_stopping, val_input, val_output)

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
* ``early_stopping`` (optional): Whether to check for early stopping conditions.
* ``val_input`` (optional): Validation input data. Must be provided together with ``val_output``.
* ``val_output`` (optional): Validation target data. Must be provided together with ``val_input``.

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

Convergence and Early Stopping
------------------------------

The ``train()`` method checks for convergence at the end of each epoch using the training loss metric.
If the optional argument ``early_stopping`` is set to ``.false.``, training continues for the full number of epochs regardless of convergence.
The default behavior is to check for convergence and stop training early if a plateau is detected (i.e. ``early_stopping`` is ``.true.``).

When validation data is provided and ``early_stopping`` is ``.true.``, the validation loss is compared against ``plateau_threshold`` directly. If the validation loss falls below ``plateau_threshold``, training stops early.


Validation
----------

If ``val_input`` and ``val_output`` are supplied, validation is performed at the
end of every epoch:

* The network is switched to inference mode (dropout/batchnorm behave as at
  test time).
* Each validation sample is evaluated individually (batch size 1).
* Validation loss (``val_loss``) and accuracy (``val_accuracy``, when an accuracy
  method is configured) are computed and printed alongside the training metrics.
* After evaluation, the network is restored to training mode with the original
  batch size and training data.

Both ``val_input`` and ``val_output`` must be provided together; supplying only
one raises an error.

Validation metrics are printed at the epoch level regardless of the ``verbose``
setting:

* ``verbose = 0``: validation appears in the epoch summary line.
* ``abs(verbose) > 0``: a separate validation line is printed after the last
  batch line of each epoch.

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


With validation data:

.. code-block:: fortran

   call network%compile( &
        optimiser=adam_optimiser_type(learning_rate=1.0e-3), &
        loss_method="mse", accuracy_method="mse")

   call network%train( &
        input=train_x, output=train_y, num_epochs=100, &
        batch_size=32, verbose=0, &
        val_input=val_x, val_output=val_y)
