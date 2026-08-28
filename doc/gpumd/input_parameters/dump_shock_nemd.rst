.. _kw_dump_shock_nemd:
.. index::
   single: dump_shock_nemd (keyword in run.in)

:attr:`dump_shock_nemd`
=======================

In shock wave piston simulations, it's often crucial to compute thermo information at different regions, both before and after the shock wave passage.

Piston simulations commonly involve millions of atoms. Dumping all the virial and velocity data for each atom can lead to excessively large output files, making data processing cumbersome. The `dump_shock_nemd` command addresses this by calculating spatial thermo information during the simulation.

This feature calculates the spatial distributions of particle velocity, normal stress, temperature, and density along the *x* direction.

Syntax
------

.. code::

   dump_shock_nemd interval <time_interval> bin_size <bin_size>

- The :attr:`interval` parameter sets the output interval (number of steps).
- The :attr:`bin_size` parameter, optional with a default value of 10, defines the thickness of each histogram bin in Å.

Output files
------------

The following files are created. Each line corresponds to one output time, and the values on that line correspond to consecutive bins along the *x* direction.

* ``temperature_hist.txt``: temperature in K.
* ``pxx_hist.txt``: :math:`P_{xx}` in GPa.
* ``pyy_hist.txt``: :math:`P_{yy}` in GPa.
* ``pzz_hist.txt``: :math:`P_{zz}` in GPa.
* ``density_hist.txt``: mass density in g/cm\ :math:`^3`.
* ``vp_hist.txt``: mass-weighted mean particle velocity in the *x* direction in km/s.

Examples
--------

To output spatial thermo information every 1000 steps for a run in the x-direction, with a bin size of 20 Angstroms, include the following before the :ref:`run keyword <kw_run>`:

.. code::

  dump_shock_nemd interval 1000 bin_size 20
