.. _polarizability_out:
.. index::
   single: polarizability.out (output file)

``polarizability.out``
================

This file contains the global thermodynamic quantities sampled at a given frequency, for each of the specified potentials.
This file is generated when the :ref:`dump_polarizability keyword <kw_dump_polarizability>` is invoked, which also controls the frequency of the output.

File format
-----------

The output mode for this file is append. The file format is as follows::
  
  column    1         2    3    4    5    6    7
  quantity  time_step p_xx p_yy p_zz p_xy p_yz p_zx

