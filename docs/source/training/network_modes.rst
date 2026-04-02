.. _network-modes:

Network Modes
=============

athena provides explicit network-level mode switching procedures similar in
intent to PyTorch's ``train()`` and ``eval()`` mode controls:

.. code-block:: fortran

   call model%set_training_mode()
   call model%set_inference_mode()

These procedures control layers whose behaviour differs between training and
inference, such as dropout, dropblock, and batch-normalisation layers.

Optionally, in both of these procedures, you can obtain a snapshot of the current mode state of all layers before switching modes using the ``mode_store`` argument.
You can also pass a list of layer indices to control which layers are switched to the desired mode using the ``layer_indices`` argument.

set_training_mode()
-------------------

``set_training_mode()`` puts all layers in training mode.

.. code-block:: fortran

   call model%set_training_mode()

Use this when you want stochastic or training-time layer behaviour during a
manual forward or optimisation loop.

In training mode:

* dropout-style layers apply masking
* batch-normalisation layers use training-time statistics
* the network is ready for low-level loops based on ``forward()``,
  ``loss_eval()``, and ``update()``

set_inference_mode()
--------------------

``set_inference_mode()`` puts all layers in inference mode.

.. code-block:: fortran

   call model%set_inference_mode()

Use this when you want deterministic evaluation behaviour.

In inference mode:

* dropout-style layers do not apply stochastic masking
* batch-normalisation layers use their inference path
* the network is ready for evaluation-oriented forward passes and predictions

Automatic Use by athena
-----------------------

The high-level network procedures use these mode setters automatically:

* ``train()`` calls ``set_training_mode()``
* ``test()`` calls ``set_inference_mode()``
* ``predict()`` calls ``set_inference_mode()``

For standard workflows, you usually do not need to call the mode procedures
yourself.
Note, the current mode of the network is reset to its previous state after each high-level call, so you can safely call
``train()``, ``test()``, or ``predict()`` from within a manual loop without worrying about the mode state.

They are most useful when writing custom low-level logic around ``forward()``
or ``forward_eval()``.

Example
-------

.. code-block:: fortran

   call model%set_training_mode()
   call model%forward(train_batch)
   loss => model%loss_eval(1, 1)
   call loss%grad_reverse()
   call model%update()

   call model%set_inference_mode()
   predictions = model%predict(test_batch)

See Also
--------

* :ref:`train() Subroutine <train-subroutine>`
* :ref:`Network Outputs <network-outputs>`
* :ref:`Training a Model <training-model>`
