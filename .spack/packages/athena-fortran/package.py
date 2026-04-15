# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
# NOTE: the package name has been set to "athena_fortran" to avoid conflicts with the existing "athena" package. Please rename as appropriate.

from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class AthenaFortran(CMakePackage):
    """ATHENA is a Fortran library for building, training and testing neural networks."""

    homepage = "https://athena-fortran.readthedocs.io/en/latest/"
    url = "https://github.com/nedtaylor/athena/archive/refs/tags/v2.1.0.tar.gz"
    git = "https://github.com/nedtaylor/athena.git"

    maintainers("nedtaylor")

    license("MIT", checked_by="nedtaylor")

    # --- Versions ---
    version("development", branch="development")
    version("main", branch="main")
    version("2.1.0", sha256="f468d077378e61ec7f4b74ee75cabf6c244d34afc4bd6930c2990727056cc7c1")
    version("2.0.0", sha256="db15a25b73d4f94dea3fb119df2be1974cdfe52209b0437eb9dcdf8fd34fa834")
    version("1.3.3", sha256="c73a908bc2c5b006f8acb316976416cd1661b458a0c95f5819f24bf7bf281f24")

    # --- Variants ---
    variant("mpi", default=False, description="Enable MPI support")

    # --- Dependencies ---
    depends_on("mpi", when="+mpi")
    depends_on("openblas")
#    depends_on("cmake@3.18:", type="build")
#    depends_on("gcc@15:", type="build")  # for Fortran compiler
    depends_on("fortran", type="build")  # generated

    # --- Compiler constraints ---
    conflicts("%gcc@:14.2", when="@main", msg="Requires GCC 14.3 or later for Fortran support")
    conflicts("%gcc@:14.2", when="@development", msg="Requires GCC 14.3 or later for Fortran support")
    conflicts("%gcc@:14.2", when="@2.0.0:", msg="Requires GCC 14.3 or later for Fortran support")
    conflicts("%gcc@:11.4", when="@1.3.3", msg="Requires GCC 12.1 or later for Fortran support")
    conflicts("%apple-clang", msg="Requires a Fortran-capable compiler (e.g. GCC)")

    # --- Version URL handling ---
    def url_for_version(self, version):
        version_str = str(version)
        major = int(version_str.split(".", 1)[0])
        tag = f"v{version_str}" if major >= 2 else version_str
        return f"https://github.com/nedtaylor/athena/archive/refs/tags/{tag}.tar.gz"

    # --- macOS fix for SDK issues ---
    def setup_build_environment(self, env):
        if self.spec.satisfies("platform=darwin"):
            # Avoid macOS SDK / deployment target mismatches
            env.set("MACOSX_DEPLOYMENT_TARGET", "14.0")

    # --- Optional: ensure OpenBLAS is preferred ---
    def setup_run_environment(self, env):
        # Helps avoid weird provider resolution in some environments
        env.set("SPACK_BLAS_VENDOR", "openblas")
