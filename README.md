ATHENA (Adaptive Training for High Efficiency Neural Network Applications/Algorithm/Architecture)
=========================================================================
by Ned Thaddeus Taylor

ATHENA is a Fortran library for developing and handling neural networks (with a focus on convolutional neural networks).

ATHENA is distributed with the following directories:

| Directory | Description |
|---|---|
|  _doc/_ |      Documentation  |
|  _src/_ |      Source code  |
|  _tools/_ |    Additional shell script tools for automating learning  |
|  _test/_  |    Example input and output file for the test makefile  |

**There currently does not exist a wiki or a manual. One will be included at a later date**

The library has been developed using the gcc 13.2.0 fortran compiler.

The library has not been tested on ifort (due to outdated ifort compilers on our current setup).



Setup
-----
Run the following commands in the directory containing _CMakeLists.txt_:
```
  mkdir build  
  cd build  
  cmake -DCMAKE_BUILD_TYPE="optim;mp" ..  
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


Testing
-------
After the library has been installed, a test program can be compiled and run to test the capabilities of ATHENA on the MNIST dataset.

To do so, the MNIST dataset (a set of 60,000 hand-written numbers for training and 10,000 for testing, 0-9) must first be downloaded. The first the format the test program has been developed for follows a text-based format. The .txt database developed for can be found here:
https://github.com/halimb/MNIST-txt/tree/master

The link to the original MNIST database is: http://yann.lecun.com/exdb/mnist/

To compile and run the test, run the following commands in the directory containing _CMakeLists.txt_:
```
  cd test
  make build optim
  ./bin/athena_test -f test_job.in
```
After the test program is compiled, the following directories will also exist:

| Directory | Description |
|---|---|
|  _test/bin/_  |     Contains binary executable | 
|  _test/obj/_  |     Contains module/object files (non-linked binary files)|

The test will perform a train over 200 mini-batch steps. It will then exit prematurely, print its weights and biases to file, and test the partially-trained network on the training set. The output from this cna then be compared to the file _expected_output.txt_.

In the tools/ directory, there exist scripts that take utilise the wandb python package (Weights and Biases, a machine learning data tracker). Wandb is a Python module and, as such, a Python interface has been provided to call and run the Fortran test. The Python interface then reads the Fortran output files and logs the results to the wandb project.

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


License
-------
This work is licensed under a Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0) License.
https://creativecommons.org/licenses/by-nc/3.0/



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
|_README.md_                  | a readme file with a brief description of the code and files  |
|_CMakeLists.txt_             | the makefile used for compiling the library  |
|_LICENCE_                    | licence of ATHENA code  |
|_test/expected_output.txt_   | expected output from executing test program  |
|_test/test_job.in_           | input file for test program  |
|_tools/sweep_init.py_        | script to initialise wandb sweep  |
|_tools/sweep_train.py_       | script to perform training and log learning to wandb  |
|_tools/template.in_          | input file for program in test/bin/ (once compiled)  |
|_tools/wandb-metadata.json_  | metadata defining default plots on wandb website  |
