# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install athena
#
# You can edit this file again by typing:
#
#     spack edit athena
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class Athena(CMakePackage):
    """A Fortran library for building, training and testing feed-forward neural networks."""

    homepage = "https://github.com/nedtaylor/athena"
    url = "https://github.com/nedtaylor/athena/archive/refs/tags/1.3.3.tar.gz"
    git = "https://github.com/nedtaylor/athena.git"

    maintainers("nedtaylor")

    license("MIT", checked_by="nedtaylor")

    version("1.3.3", sha256="c73a908bc2c5b006f8acb316976416cd1661b458a0c95f5819f24bf7bf281f24")

    variant("mpi", default=False, description="Enable parallel execution")

    depends_on("fortran", type="build")

    # FIXME: Add dependencies if required.
    depends_on("fpm@0.9:", type="build")
    depends_on("mpi", when="+mpi")

