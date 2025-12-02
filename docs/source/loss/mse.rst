.. _mse-loss:

Mean Squared Error Loss
========================

``mse_loss_type``

.. code-block:: fortran

  mse_loss_type()


Mean Squared Error (MSE) loss measures the average squared difference between predicted and expected values.

.. math::

   L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2

where:
- :math:`y_i` is the true value
- :math:`\hat{y}_i` is the predicted value
- :math:`N` is the number of samples

Use Cases
---------

* Regression problems
* When large errors should be penalised more than small errors
* Forecasting continuous values
* Function approximation

Example
-------

.. code-block:: fortran

   use athena__loss

   type(mse_loss_type) :: loss
   type(array_type), dimension(:,:) :: predicted, expected
   type(array_type), pointer :: loss_value

   ! Initialise loss function
   loss = mse_loss_type()

   ! Compute loss
   loss_value => loss%compute(predicted, expected)

Properties
----------

**Advantages:**
- Differentiable everywhere
- Convex for linear models
- Penalises large errors more heavily
- Standard in many regression tasks

**Disadvantages:**
- Sensitive to outliers
- Units are squared (less interpretable than MAE)
- Can be dominated by a few large errors

Mathematical Properties
-----------------------

* **Gradient**: :math:`\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)`
* **Convexity**: Convex in prediction space
* **Scale dependence**: Not invariant to target scaling

Notes
-----

* Also known as L2 loss
* Most commonly used loss for regression
* Gradient magnitude increases with error (faster correction of large errors)
* Consider normalizing targets to similar scales

See Also
--------

* :ref:`MAE Loss <mae-loss>` - More robust to outliers
* :ref:`Huber Loss <huber-loss>` - Combines MSE and MAE benefits
