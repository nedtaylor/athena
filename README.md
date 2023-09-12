ATHENA (Adaptive Training for High Efficiency Neural Network Applications/Algorithm/Architecture)

This is a program to highlight the uses of convolutional neural networks in Fortran.

It is currently being tested on the MNIST dataset (a set of 60,000 hand-written numbers for training and 10,000 for testing, 0-9). The version used is a text-form one.

The link to the original database is: http://yann.lecun.com/exdb/mnist/

The link to the .txt dabase is: https://github.com/halimb/MNIST-txt/tree/master

**NOTE:** Whilst the code is now a library (which is built using cmake), there still exists a Makefile.old that uses the src/inputs.f90 and src/main.f90 files to test using the MNIST dataset. The dataset must be downloaded in a text file from above.

The software is intended to be developed into an independent Fortran package which people can include/use in their own codes to add neural network capabilities (likely with a focus on convolutional neural networks, but not exclusively).

The software currently uses a namelist input file system. There is currently no help function. However, an example input file has been provided in example/param.in.

This code has been developed using the gcc 13.2.0 fortran compiler. This has not been tested on ifort (due to outdated ifort compilers on our current setup).

The code has a makefile. You SHOULD be able to build with it. Current methods are:

-- basic make --

make -f Makefile.old

-- build with intentions --

make build <opt> [opt] -f Makefile.old

make build mp -f Makefile.old

make build optim -f Makefile.old

make build dev -f Makefile.old

make build mp optim -f Makefile.old

make build mp dev -f Makefile.old

make build mp fast -f Makefile.old


To define the compiler, use make FC=<compiler_name>


Training/learning data is currently being logged using wandb (Weights and Biases, a machine learning data tracker). Wandb is a Python module and, as such, a Python interface has been made to call and run the Fortran CNN. The Python interface then reads the Fortran output files and logs the results to the wandb project.

wandb project link: https://wandb.ai/ntaylor/cnn_mnist_test/overview?workspace=user-ntaylor


For cmake, run the following commands:

mkdir build

cd build

cmake -DCMAKE_BUILD_TYPE="optim;mp" ..

make install


This will install athena into ${HOME}/.local/athena

Inside there, you will find

include/athena.mod

lib/libathena.a
