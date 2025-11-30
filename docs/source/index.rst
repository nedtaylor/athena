Welcome to athena's documentation!
===================================

**athena** is a Fortran library for implementing neural networks.

The code is provided freely available under the `MIT License <https://opensource.org/licenses/MIT>`_.

.. note::

    THE DOCUMENTATION IS CURRENTLY UNDER DEVELOPMENT. Please refer to the [wiki](https://github.com/nedtaylor/athena/wiki) for now.

The library is aimed at providing tools for building and training neural networks in Fortran;
the focus is on convolutional layers, message passing layers, and physics informed neural networks, but other layers are also provided.

The athena library, once installed, can be imported into a Fortran program with the statement:

.. code-block:: fortran

   use athena

The list of supported layers can be found `here <layers.html>`_.

An example of how to use the library is shown below:



Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
   install
   layers/layers
   optimisers/optimisers
   activations/activations
   initialisers/initialisers
   Fortran API <api>
