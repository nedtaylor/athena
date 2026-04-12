# Copyright Spack Project Developers. See COPYRIGHT file for details.
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
#     spack install athena_fortran
#
# You can edit this file again by typing:
#
#     spack edit athena_fortran
#
# See the Spack documentation for more information on packaging.
# NOTE: the package name has been set to "athena_fortran" to avoid conflicts with the existing "athena" package. Please rename as appropriate.
# ----------------------------------------------------------------------------

from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class AthenaFortran(CMakePackage):
    """A Fortran library for building, training and testing neural networks."""

    homepage = "https://athena-fortran.readthedocs.io/en/latest/"
    url = "https://github.com/nedtaylor/athena/archive/refs/tags/v2.1.0.tar.gz"
    git = "https://github.com/nedtaylor/athena.git"

    maintainers("nedtaylor")

    license("MIT", checked_by="nedtaylor")

    version("development", branch="development")
    version("main", branch="main")
    version("2.1.0", sha256="f468d077378e61ec7f4b74ee75cabf6c244d34afc4bd6930c2990727056cc7c1")
    version("2.0.0", sha256="db15a25b73d4f94dea3fb119df2be1974cdfe52209b0437eb9dcdf8fd34fa834")


    variant("mpi", default=False, description="Enable MPI support")


    # --- Dependencies ---
    depends_on("cmake@3.18:", type="build")

    depends_on("mpi", when="+mpi")

    # Use virtual deps but constrain providers safely
    depends_on("blas")
    depends_on("lapack")

    # --- Compiler constraints ---
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
