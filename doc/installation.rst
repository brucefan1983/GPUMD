.. _installation:
.. index:: Installation

Installation
************

Download
========

Download the source from the `GPUMD repository <https://github.com/brucefan1983/GPUMD>`_, or clone it with Git::

  git clone https://github.com/brucefan1983/GPUMD.git
  cd GPUMD

The commands below refer to this repository directory.

Prerequisites
=============

For NVIDIA GPUs, use a CUDA toolkit and driver that support the target GPU, with a compatible host compiler.

* **Make:** the Linux makefile uses C++14. On Windows, use ``cl.exe`` from Microsoft Visual Studio and a `64-bit make.exe <http://www.equation.com/servlet/equation.cmd?fa=make>`_.
* **CMake (pre-release):** use CMake 3.24 or newer and a host compiler with C++17 support. On Windows, use ``cl.exe`` from Microsoft Visual Studio.

For AMD GPUs, use ROCm/HIP and follow :ref:`Building for AMD GPUs <build_amd_hip>`.

.. _compilation:

Compilation
===========

Choose one build method.
Both produce ``gpumd``, ``nep``, and ``tb`` by default.

.. tabs::

   .. tab:: Make

      Run ``make`` in ``src``. Set ``CUDA_ARCH`` for your GPU and CUDA toolkit, for example::

        cd src
        make CUDA_ARCH=-arch=sm_89

      This example targets an RTX 4090. The executables are written to ``src``.
      To build only one executable, add its target name, for example ``make gpumd CUDA_ARCH=-arch=sm_89``.

      The makefile defaults to ``sm_60``. CUDA 13.0 removed offline compilation support for Pascal GPUs, including ``sm_60``; see the `CUDA release notes <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html>`_.
      Check the comments at the beginning of ``src/makefile`` for other options.
      After changing compiler flags, GPU architecture, or optional libraries, run ``make clean`` before rebuilding.

   .. tab:: CMake (pre-release)

      Configure and build from the repository root::

        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
        cmake --build build --config Release --parallel

      The executables are placed in ``build`` with a single-configuration generator, or normally in ``build/Release`` with Visual Studio.
      ``CMAKE_BUILD_TYPE`` selects the configuration for generators such as Unix Makefiles and Ninja; ``--config`` selects it for generators such as Visual Studio.
      Omit ``--parallel`` for serial compilation.

      The GPU architecture defaults to ``native``.
      To choose it explicitly, add ``-DCMAKE_CUDA_ARCHITECTURES`` to the configure command, for example for an RTX 4090::

        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89

      To build only one executable, add its target name::

        cmake --build build --config Release --target gpumd --parallel

.. _build_amd_hip:

Building for AMD GPUs (ROCm/HIP)
================================

Use ``src/makefile.hip`` with a ROCm installation that provides ``hipcc``, rocThrust/rocPRIM, hipBLAS, hipSOLVER, and hipFFT::

  cd src
  make -f makefile.hip

The target architecture defaults to ``gfx90a``. Select another AMD GPU with ``HIP_ARCH``, for example::

  make -f makefile.hip HIP_ARCH=gfx1100

The executables are written to ``src``.
After changing build options, clean with ``make -f makefile.hip clean`` before rebuilding.
CMake builds for AMD GPUs are not yet supported.

Running a program
=================

Run the executable from the directory containing the input files, using its full path or adding its directory to ``PATH``.
For example, in a Linux shell after a Make build::

  export PATH="/path/to/GPUMD/src:$PATH"

Replace ``/path/to/GPUMD/src`` with your executable directory; for a CMake build, use ``/path/to/GPUMD/build`` instead.
Then run ``gpumd`` from the simulation directory or ``nep`` from the training directory.
Follow the :ref:`MD workflow <gpumd_executable>` or :ref:`NEP training and prediction workflow <nep_executable>` for the required inputs and outputs.
The repository's `examples directory <https://github.com/brucefan1983/GPUMD/tree/master/examples>`_ provides examples for both programs.

GPU selection and multi-GPU execution
=====================================

The ``nep`` executable uses all GPUs visible to the process. On CUDA systems,
use ``CUDA_VISIBLE_DEVICES`` to select a subset, for example::

  CUDA_VISIBLE_DEVICES=0 nep
  CUDA_VISIBLE_DEVICES=0,1 nep

The ``gpumd`` executable also uses all visible GPUs when running standard NEP
or temperature-dependent NEP models, and ``nep_spin`` spin models.
For example::

  CUDA_VISIBLE_DEVICES=0,1 gpumd

For ordinary MD, charge NEP models currently use the single-GPU implementation in ``gpumd``.
For replica dynamics, :ref:`multi_replica <kw_multi_replica>` assigns complete replicas of an energy NEP or charge NEP system to the visible GPUs.
On managed clusters, request GPUs through the scheduler and use the device
visibility configured by the scheduler.

For multi-GPU runs, GPUs with similar performance are recommended because
synchronization can cause faster devices to wait for slower ones. For NEP
molecular dynamics, GPUMD chooses the spatial partition direction automatically.
The optional manual ``x``, ``y``, or ``z`` partition direction is documented
with the :ref:`potential keyword <kw_potential>`.

GNEP setup
==========

GNEP stands for a method of training NEP models using analytical Gradients (G stands for Gradients).
See the `implementation paper <https://doi.org/10.1016/j.cpc.2025.109994/>`_ for details.

.. tabs::

   .. tab:: Make

      To compile the ``gnep`` executable, run ``make gnep`` in the ``src`` directory.

   .. tab:: CMake (pre-release)

      To compile the ``gnep`` executable, configure with CMake as described in the :ref:`compilation` section above, then run:

      .. code-block:: bash

        cmake --build build --config Release --target gnep

The usage of the ``gnep`` executable is similar to that of the ``nep`` executable.
The major difference is that training hyperparameters are written in ``gnep.in`` instead of ``nep.in``
Below we use an explicit example with default parameters (except for the ``type`` keyword) to illustrate the inputs in ``gnep.in``::

  type         2 Ge Se      # same usage as in nep.in
  prediction   0            # same usage as in nep.in
  cutoff       8 4          # same usage as in nep.in
  n_max        4 4          # same usage as in nep.in
  basis_size   8 8          # same usage as in nep.in
  l_max        4            # same usage as in nep.in but does not support 4-body and 5-body descriptors
  neuron       30           # same usage as in nep.in
  lambda_e     1.0          # same usage as in nep.in
  lambda_f     2.0          # same usage as in nep.in but defaults to 2
  lambda_v     0.1          # same usage as in nep.in
  start_lr     1e-3         # new keyword to set the starting learning rate, which should be a non-negative floating-point number
  stop_lr      1e-7         # new keyword to set the stopping learning rate, which should be a non-negative floating-point number
  weight_decay 0.0          # new keyword to set the weight decay parameter, which should be a non-negative floating-point number
  batch        2            # global batch size; it does not grow with the number of visible GPUs
  epoch        50           # one epoch equals #structures/#batchsize training steps
  seed         20260831     # optional non-negative seed for parameter initialization and batch shuffling

GNEP uses every GPU made visible through ``CUDA_VISIBLE_DEVICES`` on one node.
Each complete configuration belongs to exactly one GPU shard; configurations
are never split across devices. A training step still performs one global Adam
update, so ``batch`` has the same meaning for one, two, or more GPUs. For
example, the following commands run an otherwise identical deterministic job
on one, two, and four GPUs::

  CUDA_VISIBLE_DEVICES=0 ./gnep
  CUDA_VISIBLE_DEVICES=0,1 ./gnep
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./gnep

Global batch size and multi-GPU efficiency
------------------------------------------

The ``batch`` keyword is the number of complete configurations used by one
global Adam update across all visible GPUs. It is a global batch size, not a
per-GPU batch size, and it must be kept unchanged when comparing otherwise
identical one-, two-, and four-GPU runs. Changing ``batch`` changes the
optimization problem seen by each Adam step and can therefore change the
training trajectory.

GNEP distributes complete configurations and never splits the atoms or
neighbor graph of one configuration across GPUs. Consequently, the number of
active GPUs in a step is::

  min(number of visible GPUs, number of configurations in the current batch)

``batch 1`` is valid and follows the same training algorithm, but only one GPU
can perform useful work in each step. To allow every visible GPU to participate,
use a global batch of at least 2 for two GPUs or at least 4 for four GPUs. The
last incomplete batch of an epoch can use fewer GPUs when it contains fewer
configurations. For example, a global batch of 4 is analogous to assigning one
configuration to each of four GPUs before one synchronized Adam update; it is
not equivalent to four independent ``batch 1`` updates.

Multi-GPU efficiency depends on the amount of work in each global batch. Very
small configurations or a small global batch can be slower on multiple GPUs
because host worker startup, gradient reduction, and synchronization dominate
the GPU computation. Increase ``batch`` only as permitted by GPU memory and the
desired optimization behavior. Benchmark with the same global batch, dataset,
seed, and number of epochs on every GPU count, and exclude the first warm-up
step when reporting step time.

As a reference rather than a hardware-independent guarantee, a validation run
with 3810 training configurations, 100 test configurations, global ``batch 80``,
and RTX 4090 GPUs measured median step times of 0.462856, 0.263450, and 0.151553
seconds on one, two, and four GPUs, respectively. These correspond to 1.76x and
3.05x speedups, while the serialized loss and RMSE trajectories were identical
for all three runs.

When ``seed`` is omitted, GNEP retains the historical behavior: parameter
initialization uses its existing default seed and epoch batch shuffling obtains
entropy from ``std::random_device``. Model, restart, prediction, and training
output formats are unchanged.

.. _netcdf_setup:
.. index::
   single: NetCDF setup

NetCDF setup
============

The :ref:`dump_netcdf command <kw_dump_netcdf>` requires netCDF-C 4.6.3 or newer.
The steps below use Make; append the flags after the existing variable definitions in ``src/makefile``.

Install NetCDF
--------------

Install `netCDF-C 4.6.3 <https://github.com/Unidata/netcdf-c/releases/tag/v4.6.3>`_ or a `newer release <https://github.com/Unidata/netcdf-c/releases>`_, following its build instructions.
For uncompressed output, a minimal configuration is::

  ./configure --prefix=<path> --disable-netcdf-4 --disable-dap
  make -j
  make install

Here, ``<path>`` is the installation directory.
This configuration avoids HDF5 and zlib dependencies.
For ``compression deflate`` output, omit ``--disable-netcdf-4`` and build with NetCDF4/HDF5 and zlib support.
If a newer release reports a missing ``xml2-config``, add ``--disable-libxml2`` to use its bundled XML parser.

Compile GPUMD
-------------

Append the following to ``src/makefile``, replacing ``<path>`` with the NetCDF installation directory:

.. code:: make

  CFLAGS += -DUSE_NETCDF
  INC += -I<path>/include
  LDFLAGS += -L<path>/lib -Xlinker=-rpath -Xlinker=<path>/lib
  LIBS += -lnetcdf

The library path is used both when linking and when running GPUMD.
Run ``make clean`` in ``src``, then rebuild as described under :ref:`Compilation <compilation>`.


.. _plumed_setup:
.. index::
   single: PLUMED setup

PLUMED setup
============

The :ref:`plumed command <kw_plumed>` requires a PLUMED installation.
The steps below use Make; append the flags after the existing variable definitions in ``src/makefile``.

Install PLUMED
--------------

Download a `PLUMED release <https://github.com/plumed/plumed2/releases/>`_ and follow its installation instructions.
For example, the following configuration can be used with `PLUMED 2.8.2 <https://www.plumed.org/doc-v2.8/user-doc/html/_installation.html>`_::

  ./configure --prefix=<path> --disable-mpi --enable-openmp --enable-modules=all
  make -j6
  make install

Here, ``<path>`` is the installation directory.
Set the runtime environment, for example in your shell startup file:

.. code:: bash

  export PLUMED_KERNEL=<path>/lib/libplumedKernel.so
  export LD_LIBRARY_PATH=<path>/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export PATH=<path>/bin:$PATH

Replace ``<path>`` with the same installation directory and open a new shell if you edited its startup file.

Compile GPUMD
-------------

Append the following to ``src/makefile``:

.. code:: make

  CFLAGS += -DUSE_PLUMED
  INC += -I<path>/include
  LDFLAGS += -L<path>/lib
  LIBS += -lplumed -lplumedKernel

Run ``make clean`` in ``src``, then rebuild as described under :ref:`Compilation <compilation>`.

.. _use_dp_in_gpumd:
.. index::
   single: Deep Potential

DP potential support
====================

The DP interface uses the DeePMD-kit C++ library and is enabled at compile time with ``USE_DEEPMD``.

Supported model formats
-----------------------

* ``.pb``: TensorFlow backend.
* ``.pth``: PyTorch TorchScript backend.
* ``.pt2``: PyTorch AOTInductor backend.

The installed DeePMD-kit version and backend must support the model being used.
Follow the `model-freezing instructions <https://docs.deepmodeling.com/projects/deepmd/en/stable/freeze/freeze.html>`_ for that version to produce the appropriate file.
The DeePMD-kit C++ API selects the backend from the model file.

Install the DeePMD-kit C++ interface
------------------------------------

Install DeePMD-kit with the backend and GPU support required by your model, following its `installation documentation <https://docs.deepmodeling.com/projects/deepmd/en/stable/install/install-from-source.html#install-the-c-interface>`_.
The GPUMD interface includes ``DeepPot.h`` and links to ``libdeepmd_cc``.
Check that both the headers and the C++ library are installed; a working ``dp`` command alone does not verify their availability.
Use a C++ compiler and ABI compatible with the installed DeePMD-kit library.

Compile GPUMD
-------------

Use the ``src`` directory of the current `GPUMD repository <https://github.com/brucefan1983/GPUMD>`_.
For a Linux installation with headers in ``<path>/include/deepmd`` and libraries in ``<path>/lib``, append the following after the existing flag definitions in ``src/makefile``:

.. code:: make

   CFLAGS += -DUSE_DEEPMD
   INC += -I<path>/include/deepmd
   LDFLAGS += -L<path>/lib -Xlinker=-rpath -Xlinker=<path>/lib
   LIBS += -ldeepmd_cc

Replace ``<path>`` with the DeePMD-kit installation prefix.
If using a Python wheel, set the include and library paths to the actual locations supplied by that package.
Keep the existing CUDA libraries in ``LIBS``. Adding ``-ldeepmd_cc`` to ``LIBS`` places it after the object files in the link command.
The `DeePMD-kit C++ interface documentation <https://docs.deepmodeling.com/projects/deepmd/en/stable/inference/cxx.html>`_ describes backend and runtime library requirements.

From ``src``, rebuild with the selected GPU architecture, for example:

.. code:: bash

   make clean
   make gpumd -j CUDA_ARCH=-arch=sm_89

This example targets an RTX 4090. If backend dependencies are installed in another directory, add that directory to the runtime library search path before running GPUMD, for example:

.. code:: bash

   export LD_LIBRARY_PATH=/path/to/backend/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

Run GPUMD
---------

The interface requires a setting file and a frozen DP model file.
The setting file lists the number of atom types and their chemical symbols in exactly the same order as the model type map.
For example, ``dp_settings.txt`` for a model with type map ``O H`` contains::

   dp 2 O H

Select one model in ``run.in``, for example::

   potential dp_settings.txt frozen_model.pb

For another backend, replace the model filename with the corresponding ``.pth`` or ``.pt2`` file.
The model type map can be inspected with DeePMD-kit, for example:

.. code:: bash

   python -c "from deepmd.infer import DeepPot; print(DeepPot('frozen_model.pb').get_type_map())"

Run the compiled ``gpumd`` executable from the directory containing ``run.in`` and ``model.xyz``.

.. _use_nnap_in_gpumd:
.. index::
   single: NNAP Potential

NNAP support
============

GPUMD supports NNAP (Neural-Network Atomic Potentials) through `jse <https://github.com/liqa1024/jse>`_.
This interface is a development version.
The instructions below describe a Linux build with CUDA.

Prerequisites
-------------

In addition to the GPUMD build requirements, install jse 4.1.0 or newer with a working JNI build (``jse --jnibuild``).
Prepare an NNAP model (``.json`` or ``.jnn``) and a corresponding GPUMD setting file as shown below.
For the complete jse installation and usage guides, see `jse <https://github.com/liqa1024/jse>`_ and `jse-skill <https://github.com/liqa1024/jse-skill>`_.
For support, contact Qing'an Li (liqa1024@vip.qq.com) or Ke Xu (twtdq@qq.com).

Installation details
--------------------

Prepare the environment
~~~~~~~~~~~~~~~~~~~~~~~

Check the system environment::

  nvcc --version

Set the installation directory::

  export install_dir="$HOME/software/GPUMD-NNAP"
  mkdir -p ${install_dir}
  cd ${install_dir}

Install jse
~~~~~~~~~~~

Install jse from the dev channel to obtain the latest development version (for linux):

.. code:: bash

  bash <(curl -fsSL https://raw.githubusercontent.com/liqa1024/jse/dev/scripts/get.sh)

Optionally, manually run the JNI build to detect any potential environment issues in advance:

.. code:: bash

  jse --jnibuild
  jse -t 'jse.gpu.CudaCore.InitHelper.init()'


Configure and compile GPUMD-NNAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Make

      Detect the required paths using ``jse`` and export them as environment variables:

      .. code:: bash

        export JSE_JAR_PATH=$(jse -t 'println(jse.code.OS.JAR_PATH)')
        export JVM_INCLUDE=$(jse -t 'println(jse.clib.JVM.INCLUDE_DIR)')
        export JVM_LLIB_DIR=$(jse -t 'println(jse.clib.JVM.LLIB_DIR)')

      Enable NNAP support and add the jse class path by modifying ``CFLAGS``:

      .. code:: make

        CFLAGS += -DUSE_NNAP -DJVM_CLASS_PATH=\"-Djava.class.path=$(JSE_JAR_PATH)\"

      Add the JVM header paths by modifying ``INC``:

      .. code:: make

        INC += -I$(JVM_INCLUDE) -I$(JVM_INCLUDE)/linux

      Here ``$(JVM_INCLUDE)/linux`` corresponds to Linux; for other systems, replace it with the corresponding platform directory.

      Add the JVM library search paths and library:

      .. code:: make

        LDFLAGS += -L$(JVM_LLIB_DIR) -Xlinker -rpath -Xlinker $(JVM_LLIB_DIR)
        LIBS += -ljvm

      Here, ``JSE_JAR_PATH``, ``JVM_INCLUDE``, and ``JVM_LLIB_DIR`` should be
      replaced by the actual paths obtained from the jse commands above, or exported
      as environment variables before running ``make``.

      Rebuild the MD executable from the ``src`` directory:

      .. code:: bash

        make clean
        make gpumd -j

      If compilation is successful, the ``gpumd`` executable is generated.

   .. tab:: CMake (pre-release)

      CMake uses ``PKG_NNAP`` to control NNAP support:

      .. code-block:: bash

        -D PKG_NNAP=yes

      Add environment variable to import jse jar path:

      .. code:: bash

        export JSE_JAR_PATH=$(jse -t 'println(jse.code.OS.JAR_PATH)')

      In Windows, you need to additionally specify the JVM library path (PowerShell):

      .. code:: bash

        $env:JSE_JAR_PATH = "$(jse -t 'println(jse.code.OS.JAR_PATH)')"
        $env:JVM_LIB_PATH = "$(jse -t 'println(jse.clib.JVM.LIB_PATH)')"


Run the NNAP test
-----------------

In the GPUMD input file, use the NNAP potential as follows:

.. code::

  potential nnap.txt CuZr-sphs.json

An example GPUMD setting file ``nnap.txt`` file is:

.. code::

  nnap 2 Cu Zr

The element order in ``nnap.txt`` must be consistent with that in the NNAP potential
file ``CuZr-sphs.json``.

For example, if the NNAP potential file uses the element order ``Zr Cu``, then
``nnap.txt`` should be written as:

.. code::

  nnap 2 Zr Cu

Run the test with::

  gpumd

References
~~~~~~~~~~
* jse: https://github.com/liqa1024/jse
* jse-skill: https://github.com/liqa1024/jse-skill
* jsex-NNAP: https://github.com/liqa1024/jsex-NNAP
