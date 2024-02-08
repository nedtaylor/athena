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
date: 8 February 2024
bibliography: paper.bib


---

# Summary

There exists a need for neural network libraries made available within the modern Fortran programming language. With the extensive set of legacy codes built using Fortran, there is an ever impressing need to provide new libraries within the language. Additionally, the Fortran language is well suited to use on high performance computing resources, particularly CPUs. With the nearly exhaustive amount of computing power available on supercomputers across the world, their use in the growing fields of machine learning and AI would be invaluable. The `ATHENA` library offers tools for designing fully connected and convolutional neural networks, training, and testing them.

# Statement of need

`ATHENA` is a Fortran library for building, training, and testing neural networks. Fortran is well suited for machine learning problems due to its high efficiency and speed when it comes to handling matrix maths, as well as its operability with parallel and high performance computing resources. With the additions to the language since FORTRAN95, the language has become more flexible than ever before. The `ATHENA` library is designed to take advantage of the latest developments in the Fortran language for the purposes of neural networks. Whilst other Fortran-based neural network libraries exist (such as `neural-fortran` [@curcic2019parallel], there appears limited focus on convolutional neural networks. This library has a focus on 3D convolutional neural networks, which is often less well suported than its 2D counterpart.

The `ATHENA` library is developed to handle the following network layer types: batch normalisation (2D and 3D; [@ioffe2015batch]), convolution (2D and 3D), Dropout ([@srivastava2014dropout]), DropBlock (2D and 3D; [@ghiasi2018dropblock]), flatten, fully-connected (dense), pooling (2D and 3D; average and maximum).

Through discussions with various stakeholders, accessibility and usability have been at the forefront of its development.

# Acknowledgements

The author thanks the Leverhulme for funding via Grant No. RPG-2021-086. The author also thanks and acknowledges the contribution of Francis Huw Davies to various miscellaneous Fortran procedures focused on variable and file handling.

The development of this code has benefitted through discussions with and contributions from many members of the Hepplestone research group, including Steven Paul Hepplestone, Francis Huw Davies, Harry McClean, Shane Graham Davies, and Conor Jason Price.

# References
