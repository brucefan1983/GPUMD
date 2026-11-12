.. _installation:
.. index:: Installation

Installation
************

Download
========

The source code is hosted on `github <https://github.com/brucefan1983/GPUMD>`_.


Prerequisites
=============

.. tabs::

   .. tab:: Make

      To compile and run :program:`GPUMD` on NVIDIA GPUs, use a CUDA toolkit and driver that support the target GPU.
      On Linux systems, the default makefile uses C++14; the host compiler must support this standard and be compatible with the selected CUDA toolkit.
      On Windows systems, one also needs the ``cl.exe`` compiler from Microsoft Visual Studio and a `64-bit version of make.exe <http://www.equation.com/servlet/equation.cmd?fa=make>`_.

      Alternatively, :program:`GPUMD` can be built for AMD GPUs through ROCm/HIP (see :ref:`Building for AMD GPUs <build_amd_hip>` below); this requires a ROCm installation.

   .. tab:: CMake (pre-release)

      CMake 3.24 or newer is required on all platforms.
      Use an NVIDIA GPU, a compatible CUDA toolkit and driver, and a host compiler that supports C++17.
      On Windows systems, the host compiler is ``cl.exe`` from Microsoft Visual Studio.

.. _compilation:

Compilation
===========

.. tabs::

   .. tab:: Make

      In the ``src`` directory run ``make``, which generates three executables: ``nep``, ``gpumd``, and ``tb``.
      To build only the MD executable, run ``make gpumd``.
      Set ``CUDA_ARCH`` to an architecture supported by both your GPU and CUDA toolkit, for example::

        make CUDA_ARCH=-arch=sm_89

      This example targets an RTX 4090. The makefile defaults to ``sm_60``; CUDA 13.0 removed offline compilation support for Pascal GPUs, including ``sm_60``.
      See the `CUDA release notes <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html>`_ for architecture compatibility.
      Check the comments at the beginning of the makefile for additional options.

   .. tab:: CMake (pre-release)

      Configure and compile from the project root:

      .. code-block:: bash

        cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
        cmake --build build --config Release --parallel

      The default targets are ``gpumd``, ``nep``, and ``tb``.
      With a single-configuration generator, the executables are placed in ``build``; with Visual Studio, they are normally in ``build/Release``.
      ``CMAKE_BUILD_TYPE=Release`` selects optimization for single-configuration generators such as Unix Makefiles and Ninja; ``--config Release`` selects the configuration for generators such as Visual Studio.
      Omit ``--parallel`` to fall back to serial compilation.

      To target a specific GPU architecture, add ``-D`` at configure time:

      .. code-block:: bash

        -D CMAKE_CUDA_ARCHITECTURES=value   # 80, 90, ... (default native)


When changing compiler flags, GPU architecture, or optional library support in a Make build, run ``make clean`` before rebuilding.
The optional-library examples below append flags after the existing variable definitions in ``src/makefile``.

.. _build_amd_hip:

Building for AMD GPUs (ROCm/HIP)
================================

.. tabs::

   .. tab:: Make

      :program:`GPUMD` also runs on AMD GPUs through ROCm/HIP. In the ``src`` directory, build with the HIP makefile instead of the default one::

        make -f makefile.hip

      This uses ``hipcc`` and links rocThrust/rocPRIM, hipBLAS, hipSOLVER, and hipFFT, so it requires a ROCm installation. The target GPU architecture defaults to ``gfx90a``; build for another AMD GPU by setting ``HIP_ARCH``::

        make -f makefile.hip HIP_ARCH=gfx1100

   .. tab:: CMake (pre-release)

      CMake build for AMD GPUs is not yet supported.


Examples
========

You can find several examples for how to use both the ``gpumd`` and ``nep`` executables in `the examples directory <https://github.com/brucefan1983/GPUMD/tree/master/examples>`_ of the :program:`GPUMD` repository.
   
GPU selection and multi-GPU execution
=====================================

The ``nep`` executable uses all GPUs visible to the process. On CUDA systems,
use ``CUDA_VISIBLE_DEVICES`` to select a subset, for example::

  CUDA_VISIBLE_DEVICES=0 ./nep
  CUDA_VISIBLE_DEVICES=0,1 ./nep

The ``gpumd`` executable also uses all visible GPUs when running standard NEP
or temperature-dependent NEP models, and ``nep_spin`` spin models.
For example::

  CUDA_VISIBLE_DEVICES=0,1 ./gpumd

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

To use `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ (see :ref:`dump_netcdf keyword <kw_dump_netcdf>`) with :program:`GPUMD`, a few extra steps must be taken before building :program:`GPUMD`.
First, you must download and install a compatible version of NetCDF.
:program:`GPUMD` requires netCDF-C 4.6.3 or later; version 4.6.3 remains a known-compatible baseline.

The setup instructions are below:

* Download `netCDF-C 4.6.3 <https://github.com/Unidata/netcdf-c/releases/tag/v4.6.3>`_ or a `newer release <https://github.com/Unidata/netcdf-c/releases>`_.
* Configure and build NetCDF.
  It is best to follow the instructions included with the software but, for the configuration, please use the following flags seen in our example line

  .. code:: bash

     ./configure --prefix=<path> --disable-netcdf-4 --disable-dap

  This configuration supports the default uncompressed NetCDF output and avoids the
  additional HDF5 and zlib dependencies. To use the optional ``compression deflate``
  mode, install NetCDF-C with NetCDF4/HDF5 and zlib support by omitting
  ``--disable-netcdf-4``. For newer NetCDF-C releases, if ``configure`` reports that
  ``xml2-config`` cannot be found, add ``--disable-libxml2`` to use the bundled XML
  parser. Here, the :attr:`--prefix` determines the output directory of the build.
  Then make and install NetCDF:

  .. code:: bash

     make -j && make install

* Enable the NetCDF functionality.
  To do this, one must enable the :attr:`USE_NETCDF` flag.
  In the makefile, this will look as follows:

  .. code:: make

     CFLAGS += -DUSE_NETCDF

  In addition to that line the makefile must also be updated to the following:

  .. code:: make

     INC += -I<path>/include
     LDFLAGS += -L<path>/lib -Xlinker=-rpath -Xlinker=<path>/lib
     LIBS += -lnetcdf

  where :attr:`<path>` should be replaced with the installation path for NetCDF
  (defined in :attr:`--prefix` of the ``./configure`` command). The ``-L`` option
  specifies where to find NetCDF while linking, and the ``-rpath`` linker option
  records this location so that the shared NetCDF library can also be found when
  running :program:`GPUMD`.
* Follow the remaining :program:`GPUMD` installation instructions

Following these steps will enable the :ref:`dump_netcdf keyword <kw_dump_netcdf>`.


.. _plumed_setup:
.. index::
   single: PLUMED setup

PLUMED setup
============

To use `PLUMED <https://www.plumed.org/>`_ (see :ref:`plumed keyword <kw_plumed>`) with :program:`GPUMD`, a few extra steps must be taken before building :program:`GPUMD`.
First, you must download and install PLUMED.

The setup instructions are below:

* Download `the latest version of PLUMED <https://github.com/plumed/plumed2/releases/>`_, e.g. the `plumed-src-2.8.2.tgz <https://github.com/plumed/plumed2/releases/download/v2.8.2/plumed-src-2.8.2.tgz>`_ tarball.
* Configure and build PLUMED.
  It is best to follow the `instructions <https://www.plumed.org/doc-v2.8/user-doc/html/_installation.html>`_, but for a quick installation, you may use the following setup:

  .. code:: bash

     ./configure --prefix=<path> --disable-mpi --enable-openmp --enable-modules=all

  Here, the :attr:`--prefix` determines the output directory of the build. Then make and install PLUMED:

  .. code:: bash

     make -j6 && make install

  Then update your environment variables (e.g., add the following lines to your bashrc file):

  .. code:: bash

     export PLUMED_KERNEL=<path>/lib/libplumedKernel.so
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path>/lib
     export PATH=$PATH:<path>/bin

  where :attr:`<path>` should be replaced with the installation path for PLUMED (defined in :attr:`--prefix` of the ``./configure`` command). Finally, reopen your shell to apply changes.
* Enable the PLUMED functionality.
  To do this, one must enable the :attr:`USE_PLUMED` flag.
  In the makefile, this will look as follows:

  .. code:: make

     CFLAGS += -DUSE_PLUMED

  In addition to that line the makefile must also be updated to the following:

  .. code:: make

     INC += -I<path>/include
     LDFLAGS += -L<path>/lib
     LIBS += -lplumed -lplumedKernel

  where :attr:`<path>` should be replaced with the installation path for PLUMED (defined in :attr:`--prefix` of the ``./configure`` command).
* Follow the remaining :program:`GPUMD` installation instructions

Following these steps will enable the :ref:`plumed keyword <kw_plumed>`.

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

Program introduction
--------------------

This is the beginning of :program:`GPUMD` support for NNAP in jse.

jse (Java Simulation Environment, GitHub: `liqa1024/jse <https://github.com/liqa1024/jse>`_)
is a high-performance, extensible simulation framework for atomistic and materials modeling.
It provides the latest support for NNAP (Neural-Network Atomic Potentials), including direct execution, `ASE`, `LAMMPS`, and :program:`GPUMD` as described in this document.
:program:`GPUMD` invokes dedicated NNAP support in jse via JNI to run simulations with NNAP potentials.

Necessary instructions
----------------------

* This is a development version.
* The NNAP potential file (``.json`` / ``.jnn``) and the corresponding GPUMD
  setting file (``.txt``) must be correctly prepared before running
  `GPUMD-NNAP`.
* The element order in setting file must be consistent with that in the
  NNAP potential file.

Installation dependencies
-------------------------

To compile and run `GPUMD-NNAP`, the following requirements must be
satisfied:

* The new version (``>= 4.1.0``) of jse must be installed and able to pass JNI build ``jse --jnibuild``.
* The installation requirements of :program:`GPUMD` itself must be met, including
  a working CUDA compiler and a compatible NVIDIA GPU.

Installation details
--------------------

If you have any questions, please contact Qing'an Li (liqa1024@vip.qq.com) and Ke Xu
(twtdq@qq.com).

This section describes how to compile :program:`GPUMD` with NNAP support on Linux.
For an introduction to NNAP/jse and complete installation and usage guides, refer to
`liqa1024/jse <https://github.com/liqa1024/jse>`_ and
`liqa1024/jse-skill <https://github.com/liqa1024/jse-skill>`_.

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

Notice
------

The element list in the NNAP setting file and the NNAP potential file must be the
same. Otherwise, the atom types will be mapped incorrectly during the
simulation.

References
~~~~~~~~~~
* jse: https://github.com/liqa1024/jse
* jse-skill: https://github.com/liqa1024/jse-skill
* jsex-NNAP: https://github.com/liqa1024/jsex-NNAP
