# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import subprocess
from spack.package import *
from spack.util.executable import Executable


class Athena(GitHubArchivePackage):
    """A Fortran library for building, training and testing neural networks."""

    homepage = "https://github.com/nedtaylor/athena"
    git = "https://github.com/nedtaylor/athena.git"

    # GitHub repository
    github = "nedtaylor/athena"

    maintainers("nedtaylor")

    license("MIT", checked_by="nedtaylor")

    version("1.3.3", sha256="c73a908bc2c5b006f8acb316976416cd1661b458a0c95f5819f24bf7bf281f24")

    variant("mpi", default=False, description="Enable parallel execution")

    depends_on("fortran", type="build")
    depends_on("fpm@0.13:", type="build")
    depends_on("mpi", when="+mpi")
    depends_on("blas")
    depends_on("lapack")

    def install(self, spec, prefix):
        fpm = Executable("fpm")
        fpm("install", "--prefix", prefix)
        install_tree(".", prefix)
