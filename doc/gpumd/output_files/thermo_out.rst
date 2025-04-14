.. _thermo_out:
.. index::
   single: thermo.out (output file)

``thermo.out``
==============

This file contains the global thermodynamic quantities sampled at a given frequency.
The frequency of the output is controlled via the :ref:`dump_thermo keyword <kw_dump_thermo>`.

File format
-----------

There are 18 columns in this output file, each containing the values of a quantity at increasing time points::

  column   1 2 3 4  5  6  7   8   9   10 11 12 13 14 15 16 17 18
  quantity T K U Pxx Pyy Pzz Pyz Pxz Pxy ax ay az bx by bz cx cy cz

* :attr:`T` is the temperature (in units of K)
* :attr:`K` is the kinetic energy (in units of eV) of the system
* :attr:`U` is the potential energy (in units of eV) of the system
* :attr:`Pxx` is the pressure (in units of GPa) in the xx direction
* :attr:`Pyy` is the pressure (in units of GPa) in the yy direction
* :attr:`Pzz` is the pressure (in units of GPa) in the zz direction
* :attr:`Pyz` is the pressure (in units of GPa) in the yz direction
* :attr:`Pxz` is the pressure (in units of GPa) in the xz direction
* :attr:`Pxy` is the pressure (in units of GPa) in the xy direction
* :attr:`ax ay az bx by bz cx cy cz` are the components (in units of Ångstrom) of the triclinic box matrix formed by the following vectors:

  .. math::
     
     \boldsymbol{a} &= a_x \boldsymbol{e}_x + a_y \boldsymbol{e}_y + a_z \boldsymbol{e}_z \\
     \boldsymbol{b} &= b_x \boldsymbol{e}_x + b_y \boldsymbol{e}_y + b_z \boldsymbol{e}_z \\
     \boldsymbol{c} &= c_x \boldsymbol{e}_x + c_y \boldsymbol{e}_y + c_z \boldsymbol{e}_z

Caveats
-------

* The data in this file are also valid for PIMD-related runs, but note that in this case the output temperature is just the target one. The energy and pressure contain the virial-estimator contributions. 