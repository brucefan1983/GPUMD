.. _kw_dump_position:
.. index::
   single: dump_position (keyword in run.in)

:attr:`dump_position`
=====================

Write the atomic positions (coordinates) to the :ref:`movie.xyz output file <movie_xyz>`.

Syntax
------

.. code::

   dump_position interval <options>

The :attr:`interval` parameter is the output interval (number of steps) of the atom positions.

The :attr:`<options>` can be :attr:`group` or :attr:`precision`, which can be in any order.

The option :attr:`group` shoud have two parameters::

  group grouping_method group_id

which means only dumping positions of atoms in group :attr:`group_id` within the grouping method :attr:`grouping_method`.
If this option is not used, positions will be dumped for all the atoms.

The option :attr:`precision` should have one parameter which can only be ``single`` or ``double``::

  precision single # output data with %0.9f format
  precision double # output data with %0.17f format

If this option is not used, data will be output with the ``%g`` format.


Examples
--------

Example 1
^^^^^^^^^
To dump all the positions every 1000 steps for a run, one can add::

  dump_position 1000

before the :ref:`run keyword <kw_run>`.


Example 2
^^^^^^^^^
Similar to the above example, but only for atoms in group 1 within grouping method 2::

  dump_position 1000 group 2 1

  
Example 3
^^^^^^^^^
Similar to the above example, but using double precision::

  dump_position 1000 group 2 1 precision double

or equivalently::

  dump_position 1000 precision double group 2 1


Caveats
-------  
This keyword is not propagating.
That means, its effect will not be passed from one run to the next.

The output file has an appending behavior and will result in a single :ref:`movie.xyz output file <movie_xyz>` file no matter how many times the simulation is run.
