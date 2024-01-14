.. _kw_dump_piston:
.. index::
   single: dump_piston (keyword in run.in)

:attr:`dump_piston`
=====================

In shock wave piston simulations, it's often crucial to compute thermo information at different regions, both before and after the shock wave passage.

Piston simulation often involves millions of atoms. If we dump all the virial, velocity of each atom, the output file will become very large, and data process will be cumbersome. This dump can calculate the spatial thermo information during the simulation.Piston simulations commonly involve millions of atoms. Dumping all the virial and velocity data for each atom can lead to excessively large output files, making data processing cumbersome. The `dump_piston` command addresses this by calculating spatial thermo information during the simulation.

This feature calculates the spatial distribution of partical velocity (km/h), stress (GPa), temperature and density (g/cm3).

Syntax
------

.. code::

   dump_position interval <time_interval> direction <direction> bin_size <size of each bin>

- The :attr:`interval` parameter sets the output interval (number of steps).
- The :attr:`direction` parameter specifies the shock wave direction (x, y, or z).
- The :attr:`bin_size` parameter, optional with a default value of 10, defines the thickness of each histogram bin, measured in Angstroms.

Examples
--------

To output spatial thermo information every 1000 steps for a run in the x-direction, with a bin size of 20 Angstroms, include the following before the :ref:`run keyword <kw_run>`:

.. code::

  dump_position interval 1000 direction x bin_size 20