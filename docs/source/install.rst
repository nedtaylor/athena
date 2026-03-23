.. _install:

Installation
============

This guide will help you install and set up the **athena** library for automatic differentiation in Fortran.

.. contents::
   :local:
   :depth: 2

Using athena in your Fortran Projects
-------------------------------------

The easiest way to include athena in your Fortran projects is by using the Fortran Package Manager (fpm).
To add athena as a dependency, include the following in your ``fpm.toml`` file:

.. code-block:: toml

   [dependencies]
   athena = { git = "https://github.com/nedtaylor/athena" }

For specific development versions, you can specify a branch, tag, or commit hash:

.. code-block:: toml

   [dependencies]
   athena = { git = "https://github.com/nedtaylor/athena", branch = "development" }

This is the recommended way of using athena in your projects.

Alternatively, you can clone the repository directly and build it manually.

Getting the Source Code
-----------------------

The athena library can be obtained from the GitHub repository:

.. code-block:: bash

   git clone https://github.com/nedtaylor/athena.git
   cd athena

Prerequisites
-------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

To build and use athena, you need:

1. **A Fortran Compiler** (compatible with Fortran 2018 or later)
2. A Fortran build system; we recommend using:

   a. **Fortran Package Manager (fpm >= 0.13.0)** - https://github.com/fortran-lang/fpm
   b. Alt. **CMake** - https://cmake.org
   c. Alt. **Spack** - https://spack.io

3. **coreutils** - (dependency handled automatically by fpm) https://github.com/nedtaylor/coreutils
4. **diffstruc** - (dependency handled automatically by fpm) https://github.com/nedtaylor/diffstruc
5. **graphstruc** - (dependency handled automatically by fpm) https://github.com/nedtaylor/graphstruc

.. important::
   athena and diffstruc are known to be **incompatible** with all versions of the gfortran compiler below ``14.3.0`` due to issues with the calling of the ``final`` procedure of ``array_type``.

coreutils is a lightweight Fortran library that provides essential precision types, mathematical constants, and utility functions.
diffstruc is a Fortran library providing automatic differentiation capabilities.
graphstruc is a Fortran library for constructing and managing computational graphs.
The installation of coreutils, diffstruc, and graphstruc are managed automatically by fpm when building athena.

For some of the examples, other libraries may be required, such as:

* **atomstruc** - (dependency handled automatically by fpm for the required examples)

atomstruc is a Fortran library for handling atomic structures and is used in some of the chemistry-related examples.

Supported Compilers
~~~~~~~~~~~~~~~~~~~

The library has been developed and tested with:

* **gfortran** -- gcc 14.3.0, 15.2.0
* **ifx** -- Intel Fortran Compiler 2025.2.0
* **flang** -- Flang 22.1.1

Installing Dependencies
-----------------------

Installing fpm
~~~~~~~~~~~~~~

**Linux/macOS:**

You can install fpm using one of the following methods:

.. code-block:: bash

   # Using conda
   conda install -c conda-forge fpm

   # Or download pre-built binary from GitHub releases
   # https://github.com/fortran-lang/fpm/releases

**Manual Installation:**

.. code-block:: bash

   git clone https://github.com/fortran-lang/fpm
   cd fpm
   ./install.sh

See the `fpm documentation <https://fpm.fortran-lang.org/install/index.html>`_ for detailed installation instructions.

Installing a Fortran Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ubuntu/Debian:**

.. code-block:: bash

   # For GCC/gfortran (ensure version >= 14.3.0)
   sudo apt-get update
   sudo apt-get install gfortran

**macOS:**

.. code-block:: bash

   # Using Homebrew
   brew install gcc

   # This typically installs as gfortran-<version>
   # Check your version
   gfortran --version

**Intel Fortran (ifx):**

Download from the `Intel oneAPI website <https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html>`_.

Installing Fortran dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

diffstruc, coreutils, and graphstruc will be installed automatically by fpm when you build diffstruc, so no manual installation is necessary.


Building athena
---------------

With fpm
~~~~~~~~

Once you have installed the prerequisites, building athena is straightforward using fpm.

In the repository main directory, run:

.. code-block:: bash

   fpm build --profile release

This will compile the library with optimisation flags for production use.

For development and debugging, you can build without the release profile:

.. code-block:: bash

   fpm build --profile debug

This compiles faster but without optimisations.
If you do not specify a profile, fpm defaults to the debug profile.

With built-in W&B support
~~~~~~~~~~~~~~~~~~~~~~~~~

The W&B integration in athena is provided through an ``fpm`` feature (available in ``fpm >= 0.13.0``).

.. code-block:: bash

   source tools/setup_wf_env.sh
   fpm build --features wandb

To run the built-in W&B examples:

.. code-block:: bash

   source tools/setup_wf_env.sh
   fpm run --example wandb_sine --features wandb

Available W&B examples are:

* ``wandb_sine``
* ``wandb_network_sine``
* ``wandb_sweep``
* ``wandb_pinn_burgers``


With CMake
~~~~~~~~~~

Run the following commands in the root directory of the repository.

.. code-block:: bash

  mkdir build
  cd build
  cmake [-DCMAKE_BUILD_TYPE="optim;mp"] ..
  make install


This will build the library in the `build/` directory. All library files will then be found in:

.. code-block:: bash

  ${HOME}/.local/athena

Inside this directory, the following files will be generated:

.. code-block:: bash

  include/athena.mod
  lib/libathena.a


With Spack
~~~~~~~~~~

The library can also be installed using the Spack package manager.
This can be achieved by running the following commands in the main directory:

.. code-block:: bash

  spack repo add .spack
  spack install athena

Currently, Spack compilation requires manual download of ATHENA.
NOTE: There already exists an athena package directly on Spack, be aware that these are not related.


Testing the Installation
------------------------

To verify that athena has been installed correctly and works as expected, run the test suite:

.. code-block:: bash

   fpm test

This runs a set of test programs (found in the ``test/`` directory) to ensure:

* Core functionality works correctly
* Automatic differentiation is functioning as intended

If all tests pass, your installation is successful!


Using athena in Your Project
----------------------------

With fpm
~~~~~~~~

The easiest way to use athena in your own fpm project is to add it as a dependency in your ``fpm.toml``:

.. code-block:: toml

   [dependencies]
   athena = { git = "https://github.com/nedtaylor/athena.git" }

If you are using a specific branch, tag, or commit, specify it as follows:

.. code-block:: toml

   [dependencies]
   athena = { git = "https://github.com/nedtaylor/athena.git", branch = "BRANCH_NAME" }

where ``BRANCH_NAME`` is the name of the branch you wish to use.
Alternatively, if you are modifying the athena source code directly, you can use a local path:

.. code-block:: toml

   [dependencies]
   athena = { path = "../path/to/athena" }


Then in your Fortran code:

.. code-block:: fortran

   program my_program
     use athena
     implicit none

     type(network_type) :: network

     ! Your code here...
   end program my_program

With CMake
~~~~~~~~~~

Once athena has been installed using CMake, it can be pointed to during compilation of a Fortran program, which allows its associated procedures and variables to be used within said program.
To include it during compilation, the following flags must be used.

.. code-block:: bash

  <COMPILER> <OBJECTS> -I${ATHENA_PATH}/include -L${ATHENA_PATH}/lib -o a.out

Here, `<COMPILER>` refers to the Fortran compiler command in use, `<OBJECTS>` refers to the Fortran program files, `${ATHENA_PATH}` refers to the ATHENA library directory, and `a.out` is a placeholder name for the output name of the executable. If the Setup steps above were followed, then

.. code-block:: bash

  ${ATHENA_PATH} = ${HOME}/.local/athena

As an example, for a Fortran program that contains only a `main.f90`,  with the intended executable name of `a.out`, using the `gfortran` compiler, the command line would be:

.. code-block:: bash

  gfortran main.f90 -I${HOME}/.local/athena/include -L${HOME}/.local/athena/lib -o a.out




Troubleshooting
---------------

Compiler Version Issues
~~~~~~~~~~~~~~~~~~~~~~~

athena and diffstruc require a Fortran compiler that fully supports Fortran 2018 features.
If you encounter errors related to ``final`` or ``finalise_array``, this will likely be due to using an outdated Fortran compiler.

**Solution:** Ensure your gfortran version is at least 14.3.0:

.. code-block:: bash

   gfortran --version

If your version is older, upgrade your compiler.


Flang compiler issues
~~~~~~~~~~~~~~~~~~~~~

diffstruc and athena have been tested with Flang 22.1.1 and work correctly with this version.
However, diffstruc is known to have issues with Flang when chaining overloaded operators on ``array_type`` objects, which may lead to compilation errors or incorrect results..
See the `diffstruc documentation <https:https://diffstruc.readthedocs.io/en/development/tutorials/operations.html#chaining-operations-issues>`_ for more details and workarounds.
This is only important for users intending on writing their own loss functions, custom layers, or ``array_type`` operations, and does not affect the use of pre-defined layers and loss functions provided by athena.

fpm Not Found
~~~~~~~~~~~~~

If you get ``fpm: command not found``:

**Solution:** Ensure fpm is installed and in your PATH:

.. code-block:: bash

   which fpm
   # If not found, install fpm or add its location to PATH
   export PATH="$HOME/.local/bin:$PATH"

Module File Errors
~~~~~~~~~~~~~~~~~~

If you see errors about missing ``.mod`` files:

**Solution:** Clean the build directory and rebuild:

.. code-block:: bash

   fpm clean --all
   fpm build --profile release


Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/nedtaylor/athena/issues>`_ page
2. Review the :doc:`api` documentation
3. Open an issue on the GitHub issue tracker, making sure to follow the (:git:`contributing guidelines<CONTRIBUTING.md>`).
