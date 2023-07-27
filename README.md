This is a program to highlight the uses of convolutional neural networks in Fortran.

It is currently being tested on the MNIST dataset (a set of arund 600,000 hand-written numbers, 0-9). The version used is a text-form one.
The link to the original database is: http://yann.lecun.com/exdb/mnist/
The link to the .txt dabase is: https://github.com/halimb/MNIST-txt/tree/master
**NOTE:** The code is currently hardcoded in main.tex to point to the MNIST files, they are not yet set up as an input file tag. This is simply laziness and because it is a testing version and not meant to train on the MNIST database when complete.

Once it works for 2D systems, which it seems to currently, it will be extended to 3D systems.

The software is intended to be developed into an independent Fortran package which people can include/use in their own codes to add neural network capabilities (likely with a focus on convolutional neural networks, but not exclusively).

The software currently uses a namelist input file system. There is currently no help function. However, an example input file has been provided in example/param.in.

This code has been developed using the gcc 12.2.0 fortran compiler. This has not been tested on ifort (due to outdated ifort compilers on our current setup).
