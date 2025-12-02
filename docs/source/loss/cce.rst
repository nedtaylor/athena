.. _cce-loss:

Categorical Cross Entropy Loss
===============================

``cce_loss_type``

.. code-block:: fortran

  cce_loss_type()


Categorical Cross Entropy (CCE) loss measures the performance of a classification model with multiple classes where each sample belongs to exactly one class.

.. math::

   L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})

where:
- :math:`y_{i,c}` is 1 if sample :math:`i` belongs to class :math:`c`, 0 otherwise
- :math:`\hat{y}_{i,c}` is the predicted probability for class :math:`c`
- :math:`N` is the number of samples
- :math:`C` is the number of classes

Use Cases
---------

* Multi-class classification (mutually exclusive classes)
* Image classification
* Text categorisation
* Any problem where each sample has one true class

Example
-------

.. code-block:: fortran

   use athena__loss

   type(cce_loss_type) :: loss
   type(array_type), dimension(:,:) :: predicted, expected
   type(array_type), pointer :: loss_value

   ! Initialise loss function
   loss = cce_loss_type()

   ! Compute loss (expected should be one-hot encoded)
   loss_value => loss%compute(predicted, expected)

Notes
-----

* Expects one-hot encoded labels
* Typically used with softmax output activation
* Uses small epsilon (1e-10) to prevent log(0) errors
* Sum of predicted probabilities should equal 1 for each sample

See Also
--------

* :ref:`BCE Loss <bce-loss>` - For binary classification
* :ref:`NLL Loss <nll-loss>` - Alternative formulation with log probabilities
