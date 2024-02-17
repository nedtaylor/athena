ATHENA (Adaptive Training for High Efficiency Neural Network Applications)
=========================================================================
by Ned Thaddeus Taylor

ATHENA is a Fortran library for developing and handling neural networks (with a focus on convolutional neural networks).

ATHENA is distributed with the following directories:

| Directory | Description |
|---|---|
|  _doc/_ |      Documentation  |
|  example/  |    A set of example programs utilising the ATHENA library |
|  _src/_ |      Source code  |
|  _tools/_ |    Additional shell script tools for automating learning  |
|  test/  |    A set of test programs to check functionality of the library works after compilation |


Documentation
-----

For extended details on the functinoality of this library, please check out the [wiki](https://github.com/nedtaylor/athena/wiki)

**NOTE: There currently exists no manual document. This will be included at a later date**


Setup
-----

The ATHENA library can be obtained from the git repository. Use the following commands to get started:
```
  git clone https://git.exeter.ac.uk/hepplestone/athena.git
  cd athena
```

### Dependencies

The library has the following dependencies:
- A Fortran compiler (compatible with Fortran 2018 or later)
- [CMake](https://cmake.org) for building the library

The library has been developed and tested using the following compilers:
- gfortran -- gcc 13.2.0
- ifort -- Intel 2021.10.0.20230609
- ifx -- IntelLLVM 2023.2.0


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


Acknowledgements
---------------

The author thanks the Leverhulme for funding via Grant No. RPG-2021-086. The development of this code has benefitted through discussions with and contributions from many members of the Hepplestone research group, including Steven Paul Hepplestone, Francis Huw Davies, Harry McClean, Shane Davies, Ed Baker, Joe Pitfield, and Conor Price. Of particular note, Francis has provided contributions towards the development of code in some procedures focused on handling variables and files.

License
-------
This work is licensed under an [MIT license](https://opensource.org/license/mit/).



|Source file | Description|
|-----------|------------|
|_src/athena.f90_                      | the module file that imports all necessary user-accessible procedures  |
|_src/lib/mod_activation.f90_          | generic node activation (transfer) setup  |
|_src/lib/mod_activation__[_NAME_]_.f90_   | [_NAME_] activation method  |
|_src/lib/mod_base_layer.f90_          | abstract layer construct type  |
|_src/lib/mod_container.f90_           | layer container construct for handling multiple layers in a network  |
|_src/lib/mod_container_sub.f90_       | layer container submodule  |
|_src/lib/mod_constants.f90_           | a set of global constants used in this code  |
|_src/lib/mod__[_NAME_]__layer.f90_        | [_NAME_] layer-type  |
|_src/lib/mod_initialiser.f90_         | generic kernel (and bias) initialiser setup  |
|_src/lib/mod_initialiser__[_NAME_]_.f90_  | [_NAME_] kernel initialisation method  |
|_src/lib/mod_loss_categorical.f90_    | categorical loss methods and their respective derivatives | 
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
|_README.md_                        | a readme file with a brief description of the code and files  |
|_CMakeLists.txt_                   | the makefile used for compiling the library  |
|_LICENCE_                          | licence of ATHENA code  |
|_example/mnist/expected_output.txt_   | expected output from executing test program  |
|_example/mnist/test_job.in_           | input file for test program  |
|_test/test_...f90_                 | test program to check that the library works as expected  |
|_tools/sweep_init.py_              | script to initialise wandb sweep  |
|_tools/sweep_train.py_             | script to perform training and log learning to wandb  |
|_tools/template.in_                | input file for program in test/bin/ (once compiled)  |
|_tools/wandb-metadata.json_        | metadata defining default plots on wandb website  |
