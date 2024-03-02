.. _dipole_out:
.. index::
   single: dipole.out (output file)

``dipole.out``
==============

This file contains the global thermodynamic quantities sampled at a given frequency, for each of the specified potentials.
This file is generated when the :ref:`dump_dipole keyword <kw_dump_dipole>` is invoked, which also controls the frequency of the output.

File format
-----------

The output mode for this file is append. The file format is as follows::
  
  column    1         2    3    4 
  quantity  time_step mu_x mu_y mu_z
