.. _gpumd_polarizability_out:
.. index::
   single: polarizability.out (output file)

``polarizability.out``
======================

This file contains the total polarizability tensor of the simulation cell, predicted by the response model specified by :ref:`dump_polarizability <kw_dump_polarizability>`.
This file is generated when the :ref:`dump_polarizability keyword <kw_dump_polarizability>` is invoked, which also controls the frequency of the output.

File format
-----------

The output mode for this file is append. Each run begins with comment headers starting with ``#``. The numerical columns are as follows::
  
  column    1         2    3    4    5    6    7
  quantity  time_step p_xx p_yy p_zz p_xy p_yz p_zx

The first column is the zero-based step index within the current run. For an output interval of 100, the first value is 99.
The polarizability components are sums over all atoms, in the units used to train the response model.
