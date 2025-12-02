.. _mae-loss:

Mean Absolute Error Loss
=========================

``mae_loss_type``

.. code-block:: fortran

  mae_loss_type()


Mean Absolute Error (MAE) loss measures the average magnitude of errors between predicted and expected values.

.. math::

   L = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|

where:
- :math:`y_i` is the true value
- :math:`\hat{y}_i` is the predicted value
- :math:`N` is the number of samples

Use Cases
---------

* Regression problems
* When you want equal weighting of all errors
* When outliers should have less influence than with MSE
* Time series prediction

Example
-------

.. code-block:: fortran

   use athena__loss

   type(mae_loss_type) :: loss
   type(array_type), dimension(:,:) :: predicted, expected
   type(array_type), pointer :: loss_value

   ! Initialise loss function
   loss = mae_loss_type()

   ! Compute loss
   loss_value => loss%compute(predicted, expected)

Comparison with MSE
-------------------

**MAE advantages:**
- More robust to outliers
- Linear error scale makes interpretation easier
- Constant gradient magnitude

**MSE advantages:**
- Penalises large errors more heavily
- Smooth gradients everywhere
- Often faster convergence

Notes
-----

* Also known as L1 loss
* Gradient has constant magnitude (less sensitive to error size)
* More robust to outliers compared to MSE
* Non-differentiable at zero (in practice, automatic differentiation handles this)

See Also
--------

* :ref:`MSE Loss <mse-loss>` - For penalizing large errors more
* :ref:`Huber Loss <huber-loss>` - Combines benefits of MAE and MSE
