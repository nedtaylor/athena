# Contributing Guidelines

Thank you for considering contributing to the athena project!
We appreciate your time and effort.
It is our hope that this library is found to be useful (but not limited) to scientific research communities looking to take advantage of the capabilities of modern machine learning techniques in existing Fortran code bases.
To ensure a smooth collaboration, please follow the guidelines provided below.

Please first discuss potential changes you wish to make to the project via [issues](https://github.com/nedtaylor/athena/issues) (preferably), or [email](#contact).

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Mention it on social media platforms
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

This document has been copied from the neural-fortran repository and used as a template from which to design this version. The original document can be found here: https://github.com/modern-fortran/neural-fortran/blob/main/CONTRIBUTING.md

<!-- omit in toc -->

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want to Contribute](#i-want-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contact](#contact)
- [License](#license)


## Code of Conduct

This project and everyone participating in it is governed by the
[athena Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to .

## I Have a Question

> If you want to ask a question, we assume that you have read the available [wiki](https://github.com/nedtaylor/athena/wiki).

Before you ask a question, it is best to search for existing [Issues](https://github.com/nedtaylor/athena/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/nedtaylor/athena/issues/new/choose).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (nodejs, npm, etc), depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs
If you encounter any issues or have suggestions for improvement, please open an [Issue](https://github.com/nedtaylor/athena/issues/new?template=bug_report.yaml) on the repository's issue tracker.

When reporting, please provide as much context as possible and describe the reproduction steps that someone else can follow to recreate the issue on their own.
This usually includes your code.
For good bug reports you should isolate the problem and create a reduced test case.


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for graphstruc, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

<!-- omit in toc -->
#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [wiki](https://github.com/nedtaylor/athena/wiki) carefully and find out if the functionality is already covered.
- Perform a [search](https://github.com/nedtaylor/graphstruc/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.
- Open a [Feature Proposal](https://github.com/nedtaylor/athena/issues/new?template=feature_request.yaml) on the issue board (open a new issue and select the **Feature Proposal** template).


### Contributing Code

This guide provides the recommended route to contributing to graphstruc:

1. Fork the repository.
2. Clone the forked repository to your local machine.
3. Create a new branch for your changes.
4. Make your changes and commit them.
5. Push the changes to your forked repository.
6. Open a pull request to the main repository.

When submitting your contributions, please ensure the following:
- Provide a clear and descriptive title for your pull request.
- Include a detailed description of the changes made.
- Reference any related issues or pull requests, if applicable.
- Write unit tests for your contributions
- Ensure all existing tests pass before submitting your changes.
- Update the documentation to reflect your changes, if necessary (i.e. through FORD style commenting).
- Provide examples and usage instructions, if applicable.

Follow the [Code Style](#code-style) when contributing code to this project to ensure compatibility and a uniform format to the project.


# Code Style

Rules to adhere to for maintaining a coherent code base:
- Follow the existing code style and conventions.
- Use meaningful variable and function names.
- Write clear and concise comments. Follow [FORD Fortran Documenter](https://forddocs.readthedocs.io/en/stable/) (preferred) or existing file commenting to keep consistency.
    - **NOTE:** The intention is to move to using compilable comments and documentation using FORD. As such, it is preferred that this format is used to reduce the need to upgrade later.
    - Update the wiki with new features and example guides to make athena easier to use.


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

If developing a new layer, it is recommended to first study the [max-pooling](src/lib/mod_maxpool2d_layer.f90) [convolutional](src/lib/mod_conv2d_layer.f90), or [fully-connected](src/lib/mod_full_layer.f90) layers to understand how layers are implemented.
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



## Contact
If you have any questions or need further assistance, feel free to contact [Ned Taylor](mailto:n.t.taylor@exeter.ac.uk?subject=graphstruc%20-%query).

## License
This project is licensed under the [MIT License](LICENSE).

<!-- omit in toc -->
## Attribution
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!



