.. _kw_dump_xyz:
.. index::
   single: dump_xyz (keyword in run.in)

:attr:`dump_xyz`
================

Write per-atom data into user-specified file(s) in `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

Syntax
------

.. code::

   dump_xyz <grouping_method> <group_id> <inverval> <filename> {<property_1> <property_2> ...}

* :attr:`grouping_method` and :attr:`group_id` are the grouping method and the related group ID to be used.

If :attr:`grouping_method` is negative, :attr:`group_id` will be ignored and data for the whole system will be output.

* :attr:`interval` is the output interval (number of steps) of the data.

* :attr:`filename` is the output file.

If it is ended by a star (*), the data for one frame will be output to one file, named by changing the star to the step number.

* Then one can write the properties to be output, and the allowed properties include: :attr:`mass`, :attr:`velocity`, :attr:`force`, :attr:`potential`, :attr:`virial`, :attr:`group`, and :attr:`unwrapped_position`.

* The wrapped positions will always be included in the output.


Examples
--------

.. code::

    ensemble xxx # some ensemble

    # dump positions every 1000 steps, for the whole system:
    dump_xyz -1 1 1000 positions.xyz

    # dump many other quantities every 100 steps, for atoms in group 0 of grouping method 1:
    dump_xyz 1 0 100 properties.xyz mass velocity potential force virial    

    run 1000000

Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* The output file has an appending behavior.
* Different from many of the other keywords, this keyword is allowed to be invoked multiple times within one run.
