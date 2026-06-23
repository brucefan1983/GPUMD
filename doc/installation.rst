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

      To compile (and run) :program:`GPUMD` one requires an Nvidia GPU card with compute capability no less than 3.5 and CUDA toolkit 9.0 or newer.
      On Linux systems, one also needs a C++ compiler supporting at least the C++11 standard.
      On Windows systems, one also needs the ``cl.exe`` compiler from Microsoft Visual Studio and a `64-bit version of make.exe <http://www.equation.com/servlet/equation.cmd?fa=make>`_.

      Alternatively, :program:`GPUMD` can be built for AMD GPUs through ROCm/HIP (see :ref:`Building for AMD GPUs <build_amd_hip>` below); this requires a ROCm installation.

   .. tab:: CMake (pre-release)

      To compile (and run) :program:`GPUMD` one requires an Nvidia GPU card and CUDA toolkit.
      On Linux systems, one also needs a C++ compiler supporting the C++17 standard.
      On Windows systems, one also needs the ``cl.exe`` compiler from Microsoft Visual Studio and CMake 3.24 or newer.

.. _compilation:

Compilation
===========

.. tabs::

   .. tab:: Make

      In the ``src`` directory run ``make``, which generates two executables, ``nep`` and ``gpumd``.
      Please check the comments in the beginning of the makefile for some compiling options.

   .. tab:: CMake (pre-release)

      Configure and compile from the project root:

      .. code-block:: bash

        cmake -S . -B build
        cmake --build build --config Release --parallel

      The executables (``gpumd`` and ``nep``) will be placed in the ``build`` directory.
      Omit ``--parallel`` to fall back to serial compilation.

      To target a specific GPU architecture, add ``-D`` at configure time:

      .. code-block:: bash

        -D CMAKE_CUDA_ARCHITECTURES=value   # 80, 90, ... (default native)


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


.. _netcdf_setup:
.. index::
   single: NetCDF setup
   
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

        cmake --build build --target gnep

The usage of the ``gnep`` executable is similar to that of the ``nep`` executable.
The major difference is that training hyperparameters are written in ``gnep.in`` instead of ``nep.in``'
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
  batch        2            # same usage as in nep.in but favors small values
  epoch        50           # one epoch equals #structures/#batchsize training steps

NetCDF setup
============

To use `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ (see :ref:`dump_netcdf keyword <kw_dump_netcdf>`) with :program:`GPUMD`, a few extra steps must be taken before building :program:`GPUMD`.
First, you must download and install the correct version of NetCDF.
Currently, :program:`GPUMD` is coded to work with `netCDF-C 4.6.3 <https://github.com/Unidata/netcdf-c/releases/tag/v4.6.3>`_ and it is recommended that this version is used (not newer versions).

The setup instructions are below:

* Download `netCDF-C 4.6.3 <https://github.com/Unidata/netcdf-c/releases/tag/v4.6.3>`_
* Configure and build NetCDF.
  It is best to follow the instructions included with the software but, for the configuration, please use the following flags seen in our example line

  .. code:: bash

     ./configure --prefix=<path> --disable-netcdf-4 --disable-dap

  Here, the :attr:`--prefix` determines the output directory of the build. Then make and install NetCDF:

  .. code:: bash

     make -j && make install

* Enable the NetCDF functionality.
  To do this, one must enable the :attr:`USE_NETCDF` flag.
  In the makefile, this will look as follows:

  .. code:: make

     CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -DUSE_NETCDF

  In addition to that line the makefile must also be updated to the following:

  .. code:: make

     INC = -I<path>/include -I./
     LDFLAGS = -L<path>/lib
     LIBS = -lcublas -lcusolver -l:libnetcdf.a

  where :attr:`<path>` should be replaced with the installation path for NetCDF (defined in :attr:`--prefix` of the ``./configure`` command).
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

     CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -DUSE_PLUMED

  In addition to that line the makefile must also be updated to the following:

  .. code:: make

     INC = -I<path>/include -I./
     LDFLAGS = -L<path>/lib -lplumed -lplumedKernel

  where :attr:`<path>` should be replaced with the installation path for PLUMED (defined in :attr:`--prefix` of the ``./configure`` command).
* Follow the remaining :program:`GPUMD` installation instructions

Following these steps will enable the :ref:`plumed keyword <kw_plumed>`.

.. _use_dp_in_gpumd:
.. index::
   single: Deep Potential

DP potential support
====================

Program introduction
--------------------
This is the beginning of :program:`GPUMD` support for other machine learned interatomic potentials.

Supported model formats
~~~~~~~~~~~~~~~~~~~~~~~
- ``.pb``: TensorFlow backend, generated by ``dp --tf freeze``
- ``.pth``: PyTorch TorchScript backend (DPA2/DPA3), generated by ``dp --pt freeze``
- ``.pt2``: PyTorch AOTInductor backend (DPA4), generated by ``dp --pt freeze``

.. note::
   The DeePMD-kit C++ API auto-detects the backend from the file extension.
   No changes to ``run.in`` are needed when switching between backends.

Installation dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~
- You must ensure that the new version of DP is installed and can run normally. This program contains DP-related dependencies.
- The installation environment requirements of GPUMD itself must be met.

Installation details
--------------------
Use the instance in AutoDL for testing. If one need testing use AutoDL, please contact Ke Xu (twtdq@qq.com).

And we have created an image in `AutoDL <https://www.autodl.com/>`_ that can run GPUMD-DP directly, which can be shared with the account that provides the user ID. Then, you will not require the following process and can be used directly.

GPUMD-DP installation (Offline version)
---------------------------------------

DP installation (Offline version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the latest version of DP installation steps::

    >> $ # Copy data and unzip files.
    >> $ cd /root/autodl-tmp/
    >> $ wget https://mirror.nju.edu.cn/github-release/deepmodeling/deepmd-kit/v3.0.0/deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 -O deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0
    >> $ wget https://mirror.nju.edu.cn/github-release/deepmodeling/deepmd-kit/v3.0.0/deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 -O deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1
    >> $ cat deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 > deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh
    >> $ # rm deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.0 deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh.1 # Please use with caution "rm"
    >> $ sh deepmd-kit-3.0.0-cuda126-Linux-x86_64.sh -p /root/autodl-tmp/deepmd-kit -u # Just keep pressing Enter/yes.
    >> $ source /root/autodl-tmp/deepmd-kit/bin/activate /root/autodl-tmp/deepmd-kit
    >> $ dp -h

After running according to the above steps, using ``dp -h`` can successfully display no errors.

GPUMD-DP installation
~~~~~~~~~~~~~~~~~~~~~

The GitHub link is `Here <https://github.com/Kick-H/GPUMD/tree/7af5267f4d8ba720830c154f11634a1942b66b08>`_.
::

    >> $ wget https://codeload.github.com/Kick-H/GPUMD/zip/7af5267f4d8ba720830c154f11634a1942b66b08
    >> $ cd ${GPUMD}/src-v0.1

Modify ``makefile`` as follows:

- Line 19 is changed from ``CUDA_ARCH=-arch=sm_60`` to ``CUDA_ARCH=-arch=sm_89`` (for RTX 4090). Modify according to the corresponding graphics card model.
- Line 25 is changed from ``INC = -I./`` to ``INC = -I./ -I/root/miniconda3/deepmd-kit/source/build/path_to_install/include/deepmd``
- Line 27 is changed from ``LIBS = -lcublas -lcusolver`` to ``LIBS = -lcublas -lcusolver -L/root/miniconda3/deepmd-kit/source/build/path_to_install/lib -ldeepmd_cc``

Then run the following installation command::

    >> $ sudo echo "export LD_LIBRARY_PATH=/root/miniconda3/deepmd-kit/source/build/path_to_install/lib:$LD_LIBRARY_PATH" >> /root/.bashrc
    >> $ source /root/.bashrc
    >> $ make gpumd -j

Running tests
~~~~~~~~~~~~~
::

    >> $ cd /root/miniconda3/GPUMD-bu0/tests/dp
    >> $ ../../src/gpumd

GPUMD-DP installation (Online version)
--------------------------------------

Introduction
~~~~~~~~~~~~
This is to use the online method to install the `GPUMD-DP` version, you need to connect the machine to the Internet and use github and other websites.

Conda environment
~~~~~~~~~~~~~~~~~
Create a new conda environment with Python and activate it::

    >> $ conda create -n tf-gpu2  python=3.9
    >> $ conda install -c conda-forge cudatoolkit=11.8
    >> $ pip install --upgrade tensorflow

Download deep-kit and install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download DP source code and compile the source files following DP docs. Here are the cmake commands::

    >> $ git clone https://github.com/deepmodeling/deepmd-kit.git
    >> $ cd deepmd-kit/source
    >> $ mkdir build && cd build
    >> $ cmake -DENABLE_TENSORFLOW=TRUE -DUSE_CUDA_TOOLKIT=TRUE -DCMAKE_INSTALL_PREFIX=`path_to_install` -DUSE_TF_PYTHON_LIBS=TRUE ../
    >> $ make -j && make install

We just need the DP C++ interface, so we don't source all DP environment. The libraries will be installed in ``path_to_install``.

Configure the GPUMD makefile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The GitHub link is `Here <https://github.com/Kick-H/GPUMD/tree/7af5267f4d8ba720830c154f11634a1942b66b08>`_.
::

    >> $ wget https://codeload.github.com/Kick-H/GPUMD/zip/7af5267f4d8ba720830c154f11634a1942b66b08
    >> $ cd ${GPUMD}/src
    >> $ vi makefile

Configure the makefile of GPUMD. The DP code is included by macro definition ``USE_DEEPMD``. So add it to ``CFLAGS``:

``CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -DUSE_DEEPMD``

Then link the DP C++ libraries. Add the following two lines to update the include and link paths and compile GPUMD:

``INC += -Ipath_to_install/include/deepmd``

``LDFLAGS += -Lpath_to_install/lib -ldeepmd_cc``

Then, you can install it using the following command::

    >> $ make gpumd -j


GPUMD-DP with PyTorch backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DeePMD-kit C++ interface installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PyTorch backend requires the DeePMD-kit C++ interface libraries (``libdeepmd_cc``, ``libdeepmd_c``) and PyTorch (LibTorch). Two installation methods are available:

**Method 1: pip install (recommended)**

Install DeePMD-kit with PyTorch support::

    >> $ pip install deepmd-kit[torch]

After installation, the C++ interface files are located under the Python package directory::

    >> $ python -c "import deepmd; print(deepmd.__path__[0] + '/lib')"
    # Output example: /path/to/python/site-packages/deepmd/lib
    # Contains: include/deepmd/*.h, libdeepmd_cc.so, libdeepmd_c.so

LibTorch libraries are bundled with PyTorch::

    >> $ python -c "import torch; print(torch.__path__[0] + '/lib')"
    # Contains: libtorch.so, libtorch_cpu.so, libc10.so, libtorch_cuda.so, etc.

Verify the installation::

    >> $ dp --version
    >> $ python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

**Method 2: Build from source**

Clone and compile the DeePMD-kit C++ interface with PyTorch backend enabled::

    >> $ git clone https://github.com/deepmodeling/deepmd-kit.git
    >> $ cd deepmd-kit/source
    >> $ mkdir build && cd build
    >> $ cmake -DENABLE_PYTORCH=TRUE \
              -DUSE_CUDA_TOOLKIT=TRUE \
              -DCMAKE_INSTALL_PREFIX=/path/to/install \
              ../
    >> $ make -j$(nproc) && make install

The installed files will be at::

    /path/to/install/include/deepmd/   # Header files
    /path/to/install/lib/              # libdeepmd_cc.so, libdeepmd_c.so

.. note::
   Method 1 is sufficient for most users. Method 2 is needed only if you require a custom build
   (e.g., specific CUDA version, debug symbols, or a development branch of DeePMD-kit).

Compile GPUMD
^^^^^^^^^^^^^

With the C++ interface installed, configure the GPUMD makefile. Enable DP support with ``-DUSE_DEEPMD``:

``CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -DUSE_DEEPMD``

Link against the DeePMD-kit C interface and PyTorch/LibTorch libraries::

    INC += -I$(shell python -c "import deepmd; print(deepmd.__path__[0])")/lib/include/deepmd
    LDFLAGS += -ldeepmd_cc -ldeepmd_c
    LDFLAGS += $(shell python -c "import torch; print(' '.join(['-L'+p for p in torch.__path__] + ['-L'+p+'/lib' for p in torch.__path__]))")
    LDFLAGS += -ltorch -ltorch_cpu -lc10

For GPU support, also link CUDA-related torch libraries::

    LDFLAGS += -ltorch_cuda -lc10_cuda

Then compile::

    >> $ make gpumd -j

Freeze a PyTorch model
^^^^^^^^^^^^^^^^^^^^^^

To obtain a frozen model for the PyTorch backend::

    >> $ dp --pt freeze -o frozen_model.pth   # DPA2/DPA3 -> .pth
    >> $ dp --pt freeze -o frozen_model.pth   # DPA4 -> produces .pt2 automatically

For pre-trained universal models (e.g., DPA-2.3.1 from AIS Square), download from https://aissquare.com and freeze as above.

The type map can be extracted from the frozen model::

    >> $ python -c "from deepmd.infer import DeepPot; print(' '.join(DeepPot('frozen_model.pt2').get_type_map()))"

Run GPUMD
~~~~~~~~~

When running GPUMD, if an error occurs stating that the DP libraries could not be found, add the library path temporarily with::

    LD_LIBRARY_PATH=path_to_install/lib:$LD_LIBRARY_PATH

Or add the environment permanently to the ``~/.bashrc``::

    >> $ sudo echo "export LD_LIBRARY_PATH=/root/miniconda3/deepmd-kit/source/build/path_to_install/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
    >> $ source ~/.bashrc

Run Test
~~~~~~~~

This DP interface requires two files: a setting file and a DP potential file. The setting file specifies the number of atom types and their names. For example, the ``dp_settings.txt`` for a water system::

    dp 2 O H

Use in ``run.in``::

    potential dp_settings.txt frozen_model.pb    # TensorFlow backend
    potential dp_settings.txt frozen_model.pth   # PyTorch TorchScript (DPA2/DPA3)
    potential dp_settings.txt frozen_model.pt2   # PyTorch AOTInductor (DPA4)

Notice
~~~~~~
The type list in the setting file and the potential file must be the same.

Example
~~~~~~~
* Some water simulations using the ``DP`` model in ``GPUMD``: https://github.com/brucefan1983/GPUMD/discussions

References
~~~~~~~~~~
* DeePMD-kit: https://github.com/deepmodeling/deepmd-kit

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

:program:`GPUMD` requires the following paths to be set manually; they can be detected automatically by jse:

.. code:: bash
  
  jse -t 'println(jse.code.OS.JAR_PATH)'
  jse -t 'println(jse.clib.JVM.INCLUDE_DIR)'
  jse -t 'println(jse.clib.JVM.LLIB_DIR)'

As a demonstration, the paths are exported here as environment variables:

.. code:: bash
  
  export JSE_JAR_PATH=$(jse -t 'println(jse.code.OS.JAR_PATH)')
  export JVM_INCLUDE=$(jse -t 'println(jse.clib.JVM.INCLUDE_DIR)')
  export JVM_LLIB_DIR=$(jse -t 'println(jse.clib.JVM.LLIB_DIR)')


Configure the GPUMD makefile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable NNAP support and add the jse class path by modifying ``CFLAGS``:

.. code:: make
  
  CFLAGS = -std=c++14 -O3 -arch=sm_60 -DUSE_NNAP -DJVM_CLASS_PATH=\"-Djava.class.path=$(JSE_JAR_PATH)\"

Add the JVM header paths by modifying ``INC``:

.. code:: make
  
  INC = -I./ \
        -I$(JVM_INCLUDE) \
        -I$(JVM_INCLUDE)/linux

Here ``$(JVM_INCLUDE)/linux`` corresponds to Linux; for other systems, replace it with the corresponding platform directory.

Add the JVM library path and runtime path by modifying ``LIBS``:

.. code:: make
  
  LIBS = -lcublas -lcusolver -lcufft \
         -L$(JVM_LLIB_DIR) -ljvm \
         -Xlinker -rpath -Xlinker $(JVM_LLIB_DIR)

Here, ``JSE_JAR_PATH``, ``JVM_INCLUDE``, and ``JVM_LLIB_DIR`` should be
replaced by the actual paths obtained from the jse commands above, or exported
as environment variables before running ``make``.

Compile GPUMD-NNAP
~~~~~~~~~~~~~~~~~~

Compile the executable files:

.. code:: bash
  
  make -j
  ls gpumd nep

If the compilation is successful, the executables ``gpumd`` and ``nep`` should
be generated.

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

