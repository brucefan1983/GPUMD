.. index:: Installation

Installation
************

Download
========

The source code is hosted on `github <https://github.com/brucefan1983/GPUMD>`_.


Prerequisites
=============

To compile (and run) :program:`GPUMD` one requires an Nvidia GPU card with compute capability no less than 3.5 and CUDA toolkit 9.0 or newer.
On Linux systems, one also needs a C++ compiler supporting at least the C++11 standard.
On Windows systems, one also needs the ``cl.exe`` compiler from Microsoft Visual Studio and a `64-bit version of make.exe <http://www.equation.com/servlet/equation.cmd?fa=make>`_.


Compilation
===========

In the ``src`` directory run ``make``, which generates two executables, ``nep`` and ``gpumd``.
Please check the comments in the beginning of the makefile for some compiling options.


Examples
========

You can find several examples for how to use both the ``gpumd`` and ``nep`` executables in `the examples directory <https://github.com/brucefan1983/GPUMD/tree/master/examples>`_ of the :program:`GPUMD` repository.


.. _netcdf_setup:
.. index::
   single: NetCDF setup

NetCDF Setup Instructions
=========================

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

PLUMED Setup Instructions
=========================

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

GPUMD supports DP potential project
=====================================

- Author: 徐克 (kickhsu[at]gmail.com)
- Author: 卜何凯 (hekai_bu[at]whu.edu.cn)

0 Program Introduction
----------------------

0.1 Necessary instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- This is a test version.
- Only potential function files ending with ``.pb`` in deepmd are supported, that is, the potential function files of the tensorflow version generated using ``dp --tf`` freeze.

0.2 Installation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- You must ensure that the new version of DP is installed and can run normally. This program contains DP-related dependencies.
- The installation environment requirements of GPUMD itself must be met.

1 Installation details
----------------------
Use the instance in AutoDL for testing. If one need testing use AutoDL, please contact 徐克.

And we have created an image in `AutoDL <https://www.autodl.com/>`_ that can run GPUMD-DP directly, which can be shared with the account that provides the user ID. Then, you will not require the following process and can be used directly.

2 GPUMD-DP installation (Offline version)
-------------------------------------------

2.0 DP installation (Offline version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

2.1 GPUMD-DP installation
~~~~~~~~~~~~~~~~~~~~~~~~~
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

2.2 Running Tests
~~~~~~~~~~~~~~~~~
::

    >> $ cd /root/miniconda3/GPUMD-bu0/tests/dp
    >> $ ../../src/gpumd

3 GPUMD-DP installation (Online version)
-------------------------------------------

3.0 DP installation (Online version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.1 Conda environment
~~~~~~~~~~~~~~~~~~~~~
Create a new conda environment with Python and activate it::

    >> $ conda create -n tf-gpu2  python=3.9
    >> $ conda install -c conda-forge cudatoolkit=11.8
    >> $ pip install --upgrade tensorflow

3.3 download deep-kit and install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download DP source code and compile the source files following DP docs. Here are the cmake commands::

    >> $ git clone https://github.com/deepmodeling/deepmd-kit.git
    >> $ cd deepmd-kit/source
    >> $ mkdir build
    >> $ cd build
    >> $ cmake -DENABLE_TENSORFLOW=TRUE -DUSE_CUDA_TOOLKIT=TRUE -DCMAKE_INSTALL_PREFIX=`path_to_install` -DUSE_TF_PYTHON_LIBS=TRUE ../
    >> $ make -j
    >> $ make install

We just need the DP C++ interface, so we don't source all DP environment. The libraries will be installed in ``path_to_install``.

3.4 Configure the makefile of ``GPUMD``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The GitHub link is `Here <https://github.com/Kick-H/GPUMD/tree/7af5267f4d8ba720830c154f11634a1942b66b08>`_.
::

    >> $ wget https://codeload.github.com/Kick-H/GPUMD/zip/7af5267f4d8ba720830c154f11634a1942b66b08
    >> $ cd ${GPUMD}/src
    >> $ vi makefile

Configure the makefile of GPUMD. The DP code is included by macro definition ``USE_TENSORFLOW``. So add it to ``CFLAGS``:

``CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -DUSE_TENSORFLOW``
