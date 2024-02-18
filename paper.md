---
title: 'ATHENA: A Fortran package for neural networks'
tags:
  - fortran
  - neural network
  - machine leanring
  - convolution
  - 3D convolution
authors:
  - name: Ned Thaddeus Taylor
    orcid: 0000-0002-9134-9712
    corresponding: true
    affiliation: "1"
affiliations:
 - name: Department of Physics and Astronomy, University of Exeter, United Kingdom, EX4 4QL
   index: 1
date: 18 February 2024
bibliography: paper.bib


---

# Summary

In the landscape of modern Fortran programming, there exists a compelling need for neural network libraries tailored to the language. Given the extensive set of legacy codes built with Fortran, there is an ever-growing necessity to provide new libraries implementing on modern data science tools and methodologies. Fortran's inherent compatibility with high-performance computing resources, particularly CPUs, positions it as a language of choice for many machine learning problems.

The vast amount of computing capabilities available within current supercomputers worldwide would be an invaluable asset to the growing demand for machine learning and artificial intelligence. The `ATHENA` library is developed as a resource to bridge this gap; It provides a  robust suite of tools designed for building, training, and testing fully-connected and convolutional feed-forward neural networks. 

# Statement of need

`ATHENA` (Adaptive Training for High Efficiency Neural Network Applications) is a Fortran-based library aimed at providing users with the ability to build, train, and test feed-forward neural networks. The library leverages Fortran's strong support of array arithmatics, and its compatibility with parallel and high-performance computing resources. Additionally, there exist many improvements made available since Fortran 95, specifically in Fortran 2018 [@reid2018new] (and upcoming ones in Fortran 2023), as well as continued development by the Fortran Standards committee. All of this provides a clear incentive to develop further libraries and frameworks focused on providing machine learning capabilities to the Fortran community.

While existing Fortran-based libraries, such as neural-fortran (@curcic2019parallel), address many aspects of neural networks, the focus on convolutional neural networks is drastically reduced. `ATHENA` is developed to handle both fully-connected and convolutional layers, including the ability to handle 3D data for convolutional layers (a domain sometimes underappreciated in comparison to its 3D counterpart). The `ATHENA` library is developed to handle diverse layer types, including fully-connected, Dropout, pooling, and convolution.

Notably, discussions with a spectrum of stakeholders have significantly influenced the development of `ATHENA`, placing paramount importance on accessibility and usability. This user-centric approach ensures that `ATHENA` is not just a library but a tool that seamlessly integrates with the evolving needs of the neural network community.

# Features

A full list of features available within the `ATHENA` library, including available layer types, optimisers, activation functions, and initialisers, can be found on the repository's wiki.

`ATHENA` is developed to handle the following network layer types: batch normalisation (2D and 3D; @ioffe2015batch), convolution (2D and 3D), Dropout [@srivastava2014dropout], DropBlock (2D and 3D; @ghiasi2018dropblock), flatten, fully-connected (dense), pooling (2D and 3D; average and maximum).

The library can handle feed-forward networks with an arbirtray number of hidden layers and neurons (or filter sizes). There exist several activation functions, including Gaussian, linear, sigmoid, ReLU, leaky ReLU, tangent hyberbolic functions, and more. Optimiser functions include stochastic gradient decent (SGD), RMSprop, Adam, and AdaGrad. Network models can be saved to and loaded from files.

# Ongoing research projects

The `ATHENA` library is being used in ongoing materials science research, with a focus on structural and materials property prediction.

# Acknowledgements

The author thanks the Leverhulme for funding via Grant No. RPG-2021-086. The development of this code has benefitted through discussions with and contributions from many members of the Hepplestone research group, including Steven Paul Hepplestone, Francis Huw Davies, Harry McClean, Shane Davies, Ed Baker, Joe Pitfield, and Conor Price. Of particular note, Francis has provided contributions towards the development of code in some procedures focused on handling variables and files.

# References
