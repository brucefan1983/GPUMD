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

     ./configure --prefix=/home/alex/netcdf --disable-netcdf-4 --disable-dap

  Here, the :attr:`--prefix` determines the output directory of the build.
* Enable the NetCDF functionality.
  To do this, one must enable the :attr:`USE_NETCDF` flag.
  In the makefile, this will look as follows:

  .. code:: make

     CFLAGS = -std=c++11 -O3 -arch=sm_75 -DUSE_NETCDF

  In addition to that line the makefile must also be updated to the following:

  .. code:: make

     INC = -I<path>/netcdf/include
     LDFLAGS = -L<path>/netcdf/lib
     LIBS = -l:libnetcdf.a

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

     ./configure --prefix=/home/user/plumed --disable-mpi --enable-openmp --enable-modules=all

  Here, the :attr:`--prefix` determines the output directory of the build. Then make and install PLUMED:

  .. code:: bash

     make && make install

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

     CFLAGS = -std=c++11 -O3 -arch=sm_75 -DUSE_PLUMED

  In addition to that line the makefile must also be updated to the following:

  .. code:: make

     INC = -I<path>/include
     LDFLAGS = -L<path>/lib -lplumed -lplumedKernel

  where :attr:`<path>` should be replaced with the installation path for PLUMED (defined in :attr:`--prefix` of the ``./configure`` command).
* Follow the remaining :program:`GPUMD` installation instructions

Following these steps will enable the :ref:`plumed keyword <kw_plumed>`.
