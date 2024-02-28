# Contributing guide

This document outlines the organisation of the athena codebase to help guide code contributors.

## Overall code organization

The source code organisation follows the usual `fpm` convention:
the library code is in [src/](src/), test programs are in [test/](test/),
and example programs are in [example/](example/).

The top-level module that suggests the public, user-facing API is in
[src/athena.f90](src/athena.f90).
All other library source files are in [src/lib/](src/lib/).

The code is separated into distinct modules for specific uses. There currently exists very little in the way of submodule usage for clean procedure interfaces.

Each library source file contains either one module or one submodule.
The few source files that define the submodule end with `_sub.f90`.

## Components

Athena defines several components, described here:

* Networks
* Containers
* Layers
* Optimisers
* Activation functions
* Initialisers

### Networks

A network is the main deferred type that users work with. It is a container for a collection of layers (i.e. the highest level container).

The network container is defined by the `network_type` derived type in the `nf_network` module, in the [mod_network.f90](src/lib/mod_network.f90) source file.

The `network_type` type contains multiple variables describing the form of the network, e.g. to batch size, number of layers, number of outputs.
The type also contains optimisation variables including `optimiser` and `metrics`. Finally, within the `network_type` is the `model` variable, which is an allocatable array of type `type(container_layer_type)`.

### Containers

Layer containers have multiple uses. First, they handle the passing of information between any general layer. Second, they allow for the allocation of an exact layer type to the enclosed `layer` variable.

The interfaces for `container_layer_type` type can be found in [mod_container_layer.f90](src/lib/mod_container_layer.f90); the key features are that it contains a variale `layer` of class `base_layer_type` (this variable is the hidden layer).
The container type also contains two procedures, `forward` and `backward` for performing training.

### Layers

Layers are hidden layers of a neural network. There is an abstract type from which all specific types are extended, `base_layer_type` found in [mod_base_layer.f90](src/lib/mod_base_layer.f90).
This contains interfaces for the shared procedures of all layers.

If developing a new layer, it is recommended to first study the [max-pooling](src/lib/mod_maxpool2d_layer.f90) [convolutional](src/lib/mod_conv2d_layer.f90), or [fully-connected](src/lib/mod_full_layer.f90) are implemented.
All new layer types must also be imported, and correctly handled in the [mod_container_layer](src/lib/mod_container_layer.f90) source file and its corresponding submodule, the [mod_network.f90](src/lib/mod_network.f90) file, and imported in the [athena](src/athena.f90) main library file.

### Optimisers

Optimisers are derived types that contain the corresponding algorithm for model parameter optimisation, i.e. the optimisers update model parameters during training.

Optimisers are curently implemented in the [mod_optimiser.f90](src/lib/mod_optimiser.f90) file.
An optimiser is either the base type `base_optimiser_type` or an extended type. The type is stored in the network variable `network_type%optimiser` and can either be set manually or passed to the network during the network compilation call, `network_type%compile()`.

### Activation functions

The setup of activation functions is handled in the [mod_activation.f90](src/lib/mod_activation.f90) source file.
The `activation_type` base activation type is defined in the (mod_types.f90)[src/lib/mod_types.f90] source file.
Extended derived types for implementations of specific activation functions are contained within files `mod_activation_<NAME>.f90`, where <NAME> corresponds to the activation function.

If developing a new activation function, the abstract activation type `activation_type` needs to be extended.
Implementations of the activation and its derivative must be provided for 1-5D.
All new activation functions must also be imported, and correctly handled in the [mod_activation.f90](src/lib/mod_activation.f90) and [mod_network.f90](src/lib/mod_network.f90) files.

### Initialisers

The setup of initialisers is handled in the [mod_initialiser.f90](src/lib/mod_initialiser.f90) source file.
The `initialiser_type` base initialiser type is defined in the (mod_types.f90)[src/lib/mod_types.f90] source file.
Extended derived types for implementations of specific initialisation methods are contained within files `mod_initialiser_<NAME>.f90`, where <NAME> corresponds to the initialisation method.

If developing a new initialiser, the abstract initialiser type `initialiser_type` needs to be extended.
All new initialisers functions must also be imported in the [mod_initialiser.f90](src/lib/mod_activation.f90) file.
