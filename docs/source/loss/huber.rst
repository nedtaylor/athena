.. _huber-loss:

Huber Loss
==========

``huber_loss_type``

.. code-block:: fortran

  huber_loss_type(gamma=1.0)


Huber loss combines the best properties of MSE and MAE. It is quadratic for small errors and linear for large errors, making it robust to outliers while maintaining smooth gradients.

.. math::

   L_\gamma(y, \hat{y}) = \begin{cases}
   \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \gamma \\
   \gamma |y - \hat{y}| - \frac{1}{2}\gamma^2 & \text{otherwise}
   \end{cases}

where :math:`\gamma` is the threshold parameter that determines the transition point.

Arguments
---------

* **gamma** (`real(real32)`): Threshold parameter. Default: ``1.0``.

  * Smaller values make the loss more similar to MAE
  * Larger values make it more similar to MSE

Use Cases
---------

* Regression with outliers
* Robust regression problems
* Time series with anomalies
* When you want a balance between MSE and MAE properties

Example
-------

.. code-block:: fortran

   use athena__loss

   type(huber_loss_type) :: loss
   type(array_type), dimension(:,:) :: predicted, expected
   type(array_type), pointer :: loss_value

   ! Initialize loss function with custom gamma
   loss = huber_loss_type()
   loss%gamma = 0.5  ! Adjust sensitivity to outliers

   ! Compute loss
   loss_value => loss%compute(predicted, expected)

Properties
----------

**Advantages:**
- Robust to outliers (like MAE)
- Smooth gradients everywhere (like MSE)
- Adjustable sensitivity via gamma parameter
- Convex function

**Characteristics:**
- Differentiable everywhere
- Less sensitive to outliers than MSE
- Faster convergence than MAE for small errors
- Computationally efficient

Choosing Gamma
--------------

The gamma parameter controls the trade-off:

* **Small gamma (e.g., 0.1-0.5)**: More robust to outliers, behaves more like MAE
* **Medium gamma (e.g., 1.0-2.0)**: Balanced approach (default)
* **Large gamma (e.g., 5.0+)**: Less robust to outliers, behaves more like MSE

Rule of thumb: Set gamma to approximately the expected scale of typical residuals.

Mathematical Properties
-----------------------

**Gradient:**

.. math::

   \frac{\partial L_\gamma}{\partial \hat{y}} = \begin{cases}
   \hat{y} - y & \text{if } |y - \hat{y}| \leq \gamma \\
   \gamma \cdot \text{sign}(\hat{y} - y) & \text{otherwise}
   \end{cases}

Notes
-----

* Also known as smooth L1 loss
* Reduces to MSE when gamma → ∞
* Reduces to MAE when gamma → 0
* Commonly used in robust statistics and reinforcement learning
* Particularly effective for regression with heterogeneous noise

See Also
--------

* :ref:`MSE Loss <mse-loss>` - For problems without outliers
* :ref:`MAE Loss <mae-loss>` - For maximum outlier robustness
