.. _compute_out:
.. index::
   single: compute.out (output file)

``compute.out``
===============

This file contains space and time averaged quantities.
It is produced when invoking the :ref:`compute keyword <kw_compute>`.

File format
-----------

Assuming that the system is divided into :math:`M` groups according to the grouping method used by the :ref:`compute keyword <kw_compute>` keyword, then:

  * if temperature is computed, there are :math:`M` columns of group temperatures (in units of K) from left to right
  * if potential is computed, there are :math:`M` columns of group potentials (in units of eV) from left to right
  * if force is computed: there are :math:`3M` columns of group forces (in units of eV/A) from left to right in the following form::
      
      fx_1 ... fx_M fy_1 ... fy_M fz_1 ... fz_M

  * if virial is computed, there are :math:`3M` columns of group virials (in units of eV) from left to right in a form similar to that for force
  * if the potential part of the heat current is computed, there are :math:`3M` columns of group potential heat currents (in units of eV\ :math:`^{3/2}` amu\ :math:`^{-1/2}`) from left to right in a form similar to that for force
  * if the kinetic part of the heat current is computed, there are :math:`3M` columns of group kinetic heat currents (in units of eV\ :math:`^{3/2}` amu\ :math:`^{-1/2}`) from left to right in a form similar to that for force
  * if momentum is computed, there are :math:`3M` columns of momenta (in units of amu\ :math:`^{1/2}` eV\ :math:`^{1/2}`) from left to right in a form similar to that for force
  * if temperature is computed, the last second column is the total energy of the thermostat coupling to the heat source region (in units of eV) and the last column is the total energy of the thermostat coupling to the heat sink region (in units of eV)

Note that regardless of the order of properties in the :ref:`compute keyword <kw_compute>`, the order of the output data is fixed as given above.
