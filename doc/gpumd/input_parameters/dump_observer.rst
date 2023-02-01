.. _kw_dump_observer:
.. index::
   single: dump_observer (keyword in run.in)

:attr:`dump_observer`
=====================

Writes atomistic properties such as positions, velocities and forces for each of the supplied NEP potentials in the  `extended XYZ format <https://github.com/libAtoms/extxyz>`_. Takes the same arguments as :ref:`dump_exyz keyword <kw_dump_exyz>`, and additionally a keyword `mode`. `mode` can either be set to `observe` or `average`.

If set to `observe`, the first of the supplied NEP potentials will be used to propagate the molecular dynamics run, and the remaining potentials will be evaluated every `interval` time steps. The results will be written to separate extended XYZ-files, following the naming convention: `observer0.xyz`, `observer1.xyz`, ..., `observer(N-1).xyz` for `N` supplied potentials. The index of these `observer(index).xyz` files correspond to the index of each potential in the `run.in` file. Thus, `observer0.xyz` corresponds to the first potential, `observer1.xyz` to the second and so on. In this mode, `observer0.xyz` corresponds to the main potential.

If set to `average`, all supplied NEP potentials will be evaluated at every timestep, with the average of all potentials used to propagate the molecular dynamics. In this case, only a single file called `observer.xyz` will be written at every `interval` time steps, containing the atomistic properties as calculated with the average potential.

Syntax
------

.. code::

   dump_observer <mode> <interval> <has_velocity> <has_force>

:attr:`mode` corresponds to the two cases described above, and can be either `observe` or `average`.
:attr:`interval` parameter is the output interval (number of steps) of the data.
:attr:`has_velocity` can be 1 or 0, which means the velocities will or will not be included in the output.
:attr:`has_force` can be 1 or 0, which means the forces will or will not be included in the output.

Examples
--------

Example 1
^^^^^^^^^
To use one NEP potential to propagate the MD (`nep0`), and another to observe every 1000 steps (`nep1`), write::

  potential nep0
  potential nep1  
  ...
  dump_observer observe 1000 1 1

before the :ref:`run keyword <kw_run>`. This will generate two output files, `observer0.xyz` and `observer1.xyz`, containing positions, velocities and forces as calculated with  `nep0` and `nep1` respectively.


Example 2
^^^^^^^^^
To run MD with an average of two NEP potentials, `nep0` and `nep1`, and dump the positions, velocities and forces every 1000 steps, write::

  potential nep0
  potential nep1  
  ...
  dump_observer average 1000 1 1

before the :ref:`run keyword <kw_run>`. This will a single output file, `observer.xyz`, containing positions, velocities and forces as calculated with the average of `nep0` and `nep1`.


Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* If `mode` is set to `observe`, then the output file has an appending behavior and will result in a single `observer(index).xyz` file for each potential no matter how many times the simulation is run.
* If `mode` is set to `average`, then the output file has an appending behavior and will result in a single `observer.xyz` file no matter how many times the simulation is run.
