.. _kw_dump_xyz:
.. index::
   single: dump_xyz (keyword in run.in)

:attr:`dump_xyz`
================

Write per-atom data into user-specified file(s) in `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

Syntax
------

.. code::

   dump_xyz <interval> <filename> [group <grouping_method> <group_id>] [precision <single|double>] {<property_1> <property_2> ...}

* :attr:`interval` is the output interval (number of steps) of the data.

* :attr:`filename` is the output file.

If it is ended by a star (*), the data for one frame will be output to one file, named by changing the star to the step number.

* The :attr:`group` option restricts the output to the atoms in group :attr:`group_id` of grouping method :attr:`grouping_method`.

If it is omitted, data for the whole system will be output.

* The :attr:`precision` option controls the number of significant digits written for every floating-point value.

:attr:`single` (the default) writes nine significant digits, which is enough to recover a single-precision value exactly.
:attr:`double` writes 17 significant digits, which is enough to recover a double-precision value exactly.

* Then one can write the properties to be output, and the allowed properties include: :attr:`mass`, :attr:`velocity`, :attr:`force`, :attr:`potential`, :attr:`virial`, :attr:`charge`, :attr:`bec`, :attr:`group_labels`, and :attr:`unwrapped_position`.

The :attr:`group_labels` property writes one integer column per grouping method, holding the label of the group each atom belongs to.

* The wrapped positions will always be included in the output.


Examples
--------

.. code::

    ensemble xxx # some ensemble

    # dump positions every 1000 steps, for the whole system:
    dump_xyz 1000 positions.xyz

    # dump many other quantities every 100 steps, for atoms in group 0 of grouping method 1:
    dump_xyz 100 properties.xyz group 1 0 mass velocity potential force virial

    # dump forces with the full precision of the underlying double-precision values:
    dump_xyz 100 forces.xyz precision double force

    run 1000000

Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* The output file has an appending behavior.
* Different from many of the other keywords, this keyword is allowed to be invoked multiple times within one run.
* For qNEP models, the charge values dumped out are predicted by the qNEP models.
  For other models, the charge values are those specified in :attr:`model.xyz` via :attr:`charge:R:1`.
* The Born effective charge (:term:`BEC`) is only meaningful for a qNEP model trained with target :term:`BEC`.
