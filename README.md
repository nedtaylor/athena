This is a program to highlight the uses of convolutional neural networks in Fortran.

It is currently being tested on the MNIST dataset (a set of arund 600,000 hand-written numbers, 0-9).

Once it works for 2D systems, which it seems to currently, it will be extended to 3D systems.

The software is intended to be developed into an independent Fortran package which people can include/use in their own codes to add neural network capabilities (likely with a focus on convolutional neural networks, but not exclusively).

This code has been developed using the gcc 12.2.0 fortran compiler. This has not been tested on ifort (due to outdated ifort compilers on our current setup).
