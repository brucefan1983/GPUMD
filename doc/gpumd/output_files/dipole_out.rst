.. _gpumd_dipole_out:
.. index::
   single: dipole.out (output file)

``dipole.out``
==============

This file contains the total dipole vector of the simulation cell, predicted by the response model specified by :ref:`dump_dipole <kw_dump_dipole>`.
This file is generated when the :ref:`dump_dipole keyword <kw_dump_dipole>` is invoked, which also controls the frequency of the output.

File format
-----------

The output mode for this file is append. Each run begins with comment headers starting with ``#``. The numerical columns are as follows::
  
  column    1         2    3    4 
  quantity  time_step mu_x mu_y mu_z

The first column is the zero-based step index within the current run. For an output interval of 100, the first value is 99.
The dipole components are sums over all atoms, in the units used to train the response model.
