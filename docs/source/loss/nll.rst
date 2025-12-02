.. _nll-loss:

Negative Log Likelihood Loss
=============================

``nll_loss_type``

.. code-block:: fortran

  nll_loss_type()


Negative Log Likelihood (NLL) loss is used for multi-class classification when the model outputs log probabilities.

.. math::

   L = -\frac{1}{N} \sum_{i=1}^{N} \log(\hat{y}_{i,c_i})

where:
- :math:`\hat{y}_{i,c_i}` is the predicted log probability for the correct class :math:`c_i`
- :math:`N` is the number of samples

Use Cases
---------

* Multi-class classification with log-softmax output
* When working with pre-computed log probabilities
* Maximum likelihood estimation
* Statistical modeling

Example
-------

.. code-block:: fortran

   use athena__loss

   type(nll_loss_type) :: loss
   type(array_type), dimension(:,:) :: predicted, expected
   type(array_type), pointer :: loss_value

   ! Initialise loss function
   loss = nll_loss_type()

   ! Compute loss (predicted should be log probabilities)
   loss_value => loss%compute(predicted, expected)

Relationship to Cross Entropy
------------------------------

NLL loss with log-softmax output is mathematically equivalent to categorical cross entropy with softmax output:

.. math::

   \text{CCE}(\text{softmax}(x), y) = \text{NLL}(\text{log\_softmax}(x), y)

However, the log-softmax + NLL combination is often more numerically stable.

Notes
-----

* Assumes input predictions are log probabilities (not raw logits or probabilities)
* Typically used with log-softmax activation
* More numerically stable than CCE with softmax for large magnitude logits
* Equivalent to minimizing negative log likelihood in maximum likelihood estimation

Numerical Stability
-------------------

Using log probabilities directly avoids numerical issues:

* No need to exponentiate (which can overflow)
* Logarithm of very small probabilities is still representable
* More stable gradient computation

See Also
--------

* :ref:`CCE Loss <cce-loss>` - Related loss with raw probabilities
* :ref:`BCE Loss <bce-loss>` - Binary version
