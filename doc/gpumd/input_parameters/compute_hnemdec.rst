.. _kw_compute_hnemdec:
.. index::
   single: compute_hnemdec (keyword in run.in)

:attr:`compute_hnemdec`
=======================

This keyword is used to calculate the multicomponent system thermal conductivity using the :ref:`homogeneous non-equilibrium molecular dynamics Evans-Cummings algorithm <hnemdec>` (:term:`HNEMDEC`) method.
The results are written to the :ref:`onsager.out output file <onsager_out>`.

Syntax
------

.. code::

   compute_hnemdec <driving_force> <output_interval> <Fe_x> <Fe_y> <Fe_z>

:attr:`driving_force` determines which type of driving force to use. It could be zero or non zero positive integer, which means thermal driving force or diffusive driving force.
In the case of zero, heat flux is equilvalent to dissipative flux, with driving force :math:`F_e` in the units of Å\ :sup:`-1`.
For a non zero positive integer such as :math:`i`, a momentum flux of the :math:`ith` element in the order of the first line of nep.txt is produced as dissipative flux, with driving force :math:`F_e` in the units of eV/Å.

:attr:`output_interval` is the interval to output the onsager coefficients.

:attr:`Fe_x` is the :math:`x` direction component of the external driving force :math:`F_e`.

:attr:`Fe_y` is the :math:`y` direction component of the external driving force :math:`F_e`.

:attr:`Fe_z` is the :math:`z` direction component of the external driving force :math:`F_e`.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   compute_hnemdec 0 1000 0.00001 0 0

This means that

* you want to calculate the onsager coefficients using the :term:`HNEMDEC` method with heat flux as dissipative flux;
* the onsager coefficients will be averaged and output every 1000 steps (the heat current and momentum current is sampled for every step);
* the external driving force is along the :math:`x` direction and has a magnitude of :math:`10^{-5}` Å\ :sup:`-1`. 

Note that one should control the temperature when using this keyword.
Otherwise, the system will be heated up by the external driving force.

**Important:**
For this purpose, the :ref:`Nose-Hoover chain thermostat <nose_hoover_chain_thermostat>` is recommended.
The :ref:`Langevin thermostat <langevin_thermostat>` cannot be used for this purpose because it will affect the dynamics of the system.

Example 2
^^^^^^^^^

.. code::

   compute_hnemdec 2 1000 0 0.00001 0

The external driving force is diffusive driving force that will produced a momentum flux of the second element as dissipative flux and has a magnitude of :math:`10^{-5}` eV/Å. The force is applied along the :math:`y` direction.
