[![MIT workflow](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit/ "View MIT license")
[![Latest Release](https://img.shields.io/github/v/release/nedtaylor/athena?sort=semver)](https://github.com/nedtaylor/athena/releases "View on GitHub")
[![Downloads](https://img.shields.io/github/downloads/nedtaylor/athena/total)](https://github.com/nedtaylor/athena/releases "View on GitHub")
[![status](https://joss.theoj.org/papers/7806cc51a998f872034abfe0bb24bc24/status.svg)](https://joss.theoj.org/papers/7806cc51a998f872034abfe0bb24bc24)
[![FPM](https://img.shields.io/badge/fpm-0.9.0-purple)](https://github.com/fortran-lang/fpm "View Fortran Package Manager")
[![CMAKE](https://img.shields.io/badge/cmake-3.17.5-red)](https://github.com/Kitware/CMake/releases/tag/v3.17.5 "View cmake")
[![GCC compatibility](https://img.shields.io/badge/gcc-13.2.0-green)](https://gcc.gnu.org/gcc-13/ "View GCC")
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/nedtaylor/fd7c07c046ecc92d92eaf7bdcb78c4b5/raw/test.json)](https://nedtaylor.github.io/athena/ "View coverage report")

# athena

by Ned Thaddeus Taylor

ATHENA (Adaptive Training for High Efficiency Neural network Applications) is a Fortran library for developing and handling neural networks (with a focus on convolutional neural networks).

## New Repository Location

This repository has been migrated from the University of Exeter GitLab to GitHub to facilitate community interaction and support. The latest version, updates, and collaboration now take place on this GitHub repository.

**GitLab Repository (Archived):** https://git.exeter.ac.uk/hepplestone/athena

## Why the Migration?

It was decided that this project should be migrated to allow for better community support (i.e. allowing community users to raise issues). All information has been ported over where possible. Issues have not been migrated over, these can be found in the old repository. Releases prior to 1.2.0 have not been migrated over, but they can still be found as tags in this repository.

---

ATHENA is distributed with the following directories:

| Directory | Description |
|---|---|
|  example/  |    A set of example programs utilising the ATHENA library |
|  _src/_ |      Source code  |
|  _tools/_ |    Additional shell script tools for automating learning  |
|  test/  |    A set of test programs to check functionality of the library works after compilation |


Documentation
-----

For extended details on the functionality of this library, please check out the [wiki](https://github.com/nedtaylor/athena/wiki)

**NOTE: There currently exists no manual document. This will be included at a later date**


Setup
-----

The ATHENA library can be obtained from the git repository. Use the following commands to get started:
```
  git clone https://github.com/nedtaylor/athena.git
  cd athena
```

### Dependencies

The library has the following dependencies:
- A Fortran compiler (compatible with Fortran 2018 or later)
- [fpm](https://github.com/fortran-lang/fpm) or [CMake](https://cmake.org) for building the library

The library has been developed and tested using the following compilers:
- gfortran -- gcc 13.2.0
- ifort -- Intel 2021.10.0.20230609
- ifx -- IntelLLVM 2023.2.0

### Building with fpm

The library is set up to work with the Fortran Package Manager (fpm).

With gfortran, the following command in the repository main directory:
```
  fpm build --profile release
```

#### Testing with fpm

To check whether ATHENA has installed correctly and that the compilation works as expected, the following command can be run:
```
  fpm test
```

This runs a set of test programs (found within the test/ directory) to ensure the expected output occurs when layers and networks are set up.

### Building with cmake

Run the following commands in the directory containing _CMakeLists.txt_:
```
  mkdir build  
  cd build  
  cmake [-DCMAKE_BUILD_TYPE="optim;mp"] ..  
  make install  
```
This will build the library in the build/ directory. All library files will then be found in:
```
  ${HOME}/.local/athena
```
Inside this directory, the following files will be generated:
```
  include/athena.mod
  lib/libathena.a
```

#### Testing with cmake

To check whether ATHENA has installed correctly and that the compilation works as expected, the following command can be run:
```
  ctest
```

This runs a set of test programs (found within the test/ directory) to ensure the expected output occurs when layers and networks are set up.


Examples
-------
After the library has been installed, a set of example programs can be compiled and run to test the capabilities of ATHENA on the MNIST dataset. Some of the examples can be run as-is, and do not require external databases. For those that require the MNIST (a set of 60,000 hand-written numbers for training and 10,000 for testing, 0-9) dataset (i.e. 'example/mnist_' directories ), the dataset must first be downloaded. The example program has been developed to accept a text-based format of the MNIST dataset. The .txt database that these examples have been developed for can be found here:
https://github.com/halimb/MNIST-txt/tree/master

The link to the original MNIST database is: http://yann.lecun.com/exdb/mnist/

__NOTE:__ For the mnist examples, the MNIST dataset must be downloaded. By default, the database is expected to be found in the directory path ``../../DMNIST``. However, this can be chaned by editing the following line in the ``example/mnist[_VAR]/test_job.in`` file to point to the desired path:

```
  dataset_dir = "../../DMNIST"
```

#### Running examples using fpm

Using fpm, the examples are built alongside the library. To list all available examples, use:
```
  fpm run --example --list
```

To run a particular example, execute the following command:

```
  fpm run --example [NAME]
```

where [_NAME_] is the name of the example found in the list.


#### Running examples manually

To compile and run the examples, run the following commands in the directory containing _CMakeLists.txt_:
```
  cd example/mnist
  make build optim [FC=FORTRAN-COMPILER]
  ./bin/athena_test -f test_job.in
```
After the example program is compiled, the following directories will also exist:

| Directory | Description |
|---|---|
|  _example/mnist/bin/_  |     Contains binary executable | 
|  _example/mnist/obj/_  |     Contains module/object files (non-linked binary files)|

The example will perform a train over the MNIST dataset. Once complete, it will print its weights and biases to file, and test the trained network on the training set. The output from this can then be compared to the file _expected_output_COMPILER.txt_.

In the tools/ directory, there exist scripts that take utilise the wandb python package (Weights and Biases, a machine learning data tracker). Wandb is a Python module and, as such, a Python interface has been provided to call and run the Fortran example. The Python interface then reads the Fortran output files and logs the results to the wandb project.

Example wandb project link: https://wandb.ai/ntaylor/cnn_mnist_test/overview?workspace=user-ntaylor



How-to
-------
To call/reference the ATHENA library in a program, include the following use statement at the beginning of the necessary Fortran file:
  use athena

During compilation, include the following flags in the compilation (gfortran) command:
```
-I${HOME}/.local/athena/include -L${HOME}/.local/athena/lib -lathena
```


Developers
----------
- Ned Thaddeus Taylor

Contributing
------------

Please note that this project adheres to the [Contributing Guide](CONTRIBUTING.md). If you are interested in contributing to this project, please contact [Ned Taylor](mailto:n.t.taylor@exeter.ac.uk?subject=ATHENA%20-%20contribution%20request).


License
-------
This work is licensed under an [MIT license](https://opensource.org/license/mit/).

Code Coverage
-------------

Automated reporting on unit test code coverage in the README is achieved through utilising the [cmake-modules](https://github.com/rpavlik/cmake-modules) and [dynamic-badges-action](https://github.com/Schneegans/dynamic-badges-action?tab=readme-ov-file) projects.


Files
-----


|Source file | Description|
|-----------|------------|
|_src/athena.f90_                      | the module file that imports all necessary user-accessible procedures  |
|_src/lib/mod_accuracy.f90_            | accuracy calculation procedures |
|_src/lib/mod_activation.f90_          | generic node activation (transfer) setup  |
|_src/lib/mod_activation__[_NAME_]_.f90_   | [_NAME_] activation method  |
|_src/lib/mod_base_layer.f90_          | abstract layer construct type  |
|_src/lib/mod_clipper.f90_             | gradient clipping procedures |
|_src/lib/mod_constants.f90_           | a set of global constants used in this code  |
|_src/lib/mod_container.f90_           | layer container construct for handling multiple layers in a network  |
|_src/lib/mod_container_sub.f90_       | layer container submodule  |
|_src/lib/mod__[_NAME_]__layer.f90_        | [_NAME_] layer-type  |
|_src/lib/mod_initialiser.f90_         | generic kernel (and bias) initialiser setup  |
|_src/lib/mod_initialiser__[_NAME_]_.f90_  | [_NAME_] kernel initialisation method  |
|_src/lib/mod_loss.f90_                | loss and corresponding derivatives calculation procedures |
|_src/lib/mod_lr_decay.f90_            | learning rate decay procedures |
|_src/lib/mod_metrics.f90_             | training convergence metric derived type and procedures  |
|_src/lib/mod_misc.f90_                | miscellaneous procedures  |
|_src/lib/mod_misc_ml.f90_             | miscellaneous machine learning procedures  |
|_srcs/lib/mod_network.f90_            | neural network derived type and procedures  |
|_src/lib/mod_normalisation.f90_       | data normalisation procedures  |
|_src/lib/mod_optimiser.f90_           | learning optimisation derived type and procedures  |
|_src/lib/mod_random.f90_              | random number procedures  |
|_src/lib/mod_tools_infile.f90_        | tools to read input files  |
|_src/lib/mod_types.f90_               | neural network-associated derived types  |



| Additional file | Description |
|-----|------|
|_CHANGELOG_                        | human-readable athena codebase version history |
|_CMakeLists.txt_                   | the makefile used for compiling the library  |
|_CONTRIBUTING.md_                  | Guidelines for organisation of athena codebase |
|_fpm.toml_                         | [Fortran Package Manager (fpm)](https://github.com/fortran-lang/fpm) compilation file |
|_LICENSE_                          | licence of ATHENA code |
|_README.md_                        | a readme file with a brief description of the code and files  |
|_TODO_                             | todo-list in addition to useful machine learning and fortran references |
|_cmake/CodeCoverage.cmake_         | [cmake-modules](https://github.com/rpavlik/cmake-modules) file to automate unit test coverage reporting| 
|_example/example_library_          | Utility library shared between the examples |
|_example/__[_NAME_]__/expected_output.txt_   | expected output from executing [_NAME_] example program  |
|_example/__[_NAME_]__/test_job.in_           | input file for [_NAME_] example program  |
|_example/__[_NAME_]__/src_                   | source directory for [_NAME_] example program  |
|_test/test__[_NAME_]__.f90_           | [_NAME_] test program to check library expected functionality |
|_tools/coverage_badge.py_          | script to extract code coverage percentage from GitHub Action |
|_tools/sweep_init.py_              | script to initialise wandb sweep  |
|_tools/sweep_train.py_             | script to perform training and log learning to wandb  |
|_tools/template.in_                | input file for program in test/bin/ (once compiled)  |
|_tools/wandb-metadata.json_        | metadata defining default plots on wandb website  |
