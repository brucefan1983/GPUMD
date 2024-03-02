.. _kw_dump_observer:
.. index::
   single: dump_observer (keyword in run.in)

:attr:`dump_observer`
=====================

Writes atomistic properties such as positions, velocities and forces for each of the supplied NEP potentials in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_. Takes the same arguments as :ref:`dump_exyz keyword <kw_dump_exyz>`, and additionally a keyword `mode`. `mode` can either be set to `observe` or `average`.

If set to `observe`, the first of the supplied NEP potentials will be used to propagate the molecular dynamics run, and the remaining potentials will be evaluated every `interval_thermo` and `interval_exyz` time steps. Every `interval_thermo` timesteps files in the style of :ref:`thermo.out <thermo_out>` will be written, and every `interval_exyz` timesteps extended XYZ-files will be written. The files are named according to the following convention:

* **.out**: `observer0.out`, `observer1.out`, ..., `observer(N-1).out` for `N` supplied potentials.
* **.xyz**: `observer0.xyz`, `observer1.xyz`, ..., `observer(N-1).xyz` for `N` supplied potentials.

The index of these `observer(index)` files correspond to the index of each potential in the `run.in` file. Thus, `observer0` corresponds to the first potential, `observer1` to the second and so on. In this mode, `observer0` corresponds to the main potential.

If set to `average`, all supplied NEP potentials will be evaluated at every timestep, with the average of all potentials used to propagate the molecular dynamics. 
In this case, two files will be written: `observer.out` every `interval_thermo` timesteps, and `observer.xyz` every `interval_exyz` timesteps. These files contains the thermo and atomistic properties as calculated with the average potential. 

Note that the supplied potentials must have their atomic species written in the same order, i.e. the line `nep* n_species species0 species1` must be the same in all potential files.

Syntax
------

.. code::

   dump_observer <mode> <interval_thermo> <interval_exyz> <has_velocity> <has_force>

:attr:`mode` corresponds to the two cases described above, and can be either `observe` or `average`.
:attr:`interval_thermo` parameter is the output interval (number of steps) for writing thermo files.
:attr:`interval_exyz` parameter is the output interval (number of steps) of writing exyz files.
:attr:`has_velocity` can be 1 or 0, which means the velocities will or will not be included in the exyz output.
:attr:`has_force` can be 1 or 0, which means the forces will or will not be included in the exyz output.

Examples
--------

Example 1
^^^^^^^^^
To use one NEP potential to propagate the MD (`nep0`), and another (`nep1`) to observe thermo properties every 100 steps and write exyz files every 1000 steps, write::

  potential nep0
  potential nep1  
  ...
  dump_observer observe 100 1000 1 1

before the :ref:`run keyword <kw_run>`. This will generate four output files, `observer0.xyz`, `observer0.out`, `observer1.xyz` and `observer1.out`, containing thermo properties and positions, velocities and forces as calculated with  `nep0` and `nep1` respectively.


Example 2
^^^^^^^^^
To run MD with an average of two NEP potentials, `nep0` and `nep1`, and dump the positions, velocities and forces every 1000 steps, write::

  potential nep0
  potential nep1  
  ...
  dump_observer average 100 1000 1 1

before the :ref:`run keyword <kw_run>`. This will generate two output files, `observer.out` containing thermo properties  and `observer.xyz`, containing positions, velocities and forces as calculated with the average of `nep0` and `nep1`. `observer.out` will be written every 100 timesteps, and `observer.xyz` every 1000 timesteps.


Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* If `mode` is set to `observe`, then the output file has an appending behavior and will result in two files, `observer(index).out` and `observer(index).xyz` file for each potential no matter how many times the simulation is run.
* If `mode` is set to `average`, then the output file has an appending behavior and will result in a single `observer.xyz` file no matter how many times the simulation is run.
