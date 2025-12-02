.. _bce-loss:

Binary Cross Entropy Loss
==========================

``bce_loss_type``

.. code-block:: fortran

  bce_loss_type()


Binary Cross Entropy (BCE) loss measures the performance of a classification model whose output is a probability value between 0 and 1.

.. math::

   L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]

where:
- :math:`y_i` is the true label (0 or 1)
- :math:`\hat{y}_i` is the predicted probability
- :math:`N` is the number of samples

Use Cases
---------

* Binary classification problems
* Multi-label classification (independent binary decisions per label)
* Probability estimation tasks

Example
-------

.. code-block:: fortran

   use athena__loss

   type(bce_loss_type) :: loss
   type(array_type), dimension(:,:) :: predicted, expected
   type(array_type), pointer :: loss_value

   ! Initialise loss function
   loss = bce_loss_type()

   ! Compute loss
   loss_value => loss%compute(predicted, expected)

Notes
-----

* Assumes predicted values are in the range [0, 1] (typically from sigmoid activation)
* Uses small epsilon (1e-10) to prevent log(0) errors
* Suitable for networks with sigmoid output activation

See Also
--------

* :ref:`CCE Loss <cce-loss>` - For multi-class classification
* :ref:`NLL Loss <nll-loss>` - For classification with log probabilities
