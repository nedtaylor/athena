
API Reference
=============

The full Fortran API documentation is generated using FORD (Fortran Automatic Documentation).

You can browse the complete API documentation here:

.. raw:: html

   <div style="margin: 20px 0;">
      <a href="_static/ford/index.html" target="_blank" class="btn btn-primary">
         📚 Open FORD API Documentation
      </a>
   </div>

The API documentation includes:

* **Module Reference**: Detailed documentation of all modules
* **Type Definitions**: Documentation of derived types like ``base_layer_type``
* **Procedures**: All public procedures and their interfaces
* **Source Code**: Annotated source code with cross-references
* **Call Graphs**: Visual representation of procedure dependencies

Key Components
--------------

The following are some of the key derived types and their main methods documented in the API:


* ``base_layer_type``: The abstract base type for all layers

  * ``forward()``: Method to perform forward propagation

* ``base_actv_type``: The abstract base type for all activation functions

  * ``apply()``: Method to apply the activation function

* ``base_init_type``: The abstract base type for all initialisers

  * ``initialise()``: Method to initialise parameters

* ``base_optimiser_type``: The base type for all optimisers

  * ``minimise()``: Method to update parameters based on gradients

* ``base_loss_type``: The base type for all loss functions

  * ``compute()``: Method to compute the loss value

* ``network_type``: The type representing a neural network

  * ``add()``: Method to add layers to the network
  * ``compile()``: Method to compile the network for training
  * ``train()``: Method to train the network
  * ``test()``: Method to evaluate the network on test data
  * ``predict()``: Method to make predictions with the network
  * ``update()``: Method to update learnable parameters

For complete details and source code, please refer to the FORD documentation linked above.
