.. _kw_compute:
.. index::
   single: compute (keyword in run.in)

``compute``
===============

This keyword is used to compute and output space and time averaged quantities.

Syntax
------

::

  compute <grouping_method> <sample_interval> <output_interval> {<quantity>}

Parameters
----------

The first parameter ``grouping_method`` refers to the grouping method defined in the :ref:`simulation model file <model_xyz>`.
This parameter should be an integer and a number :math:`m` means the :math:`m`-th grouping method (we count from 0) in the :ref:`simulation model file <model_xyz>`.

The second parameter ``sample_interval`` means sampling the quantities every so many time steps.

The third parameter ``output_interval`` means averaging over so many sampled data before giving one output.

Starting from the fourth parameter, one can list the quantities to be computed.
The allowed names for ``quantity`` are:

* ``temperature``, which yields the temperature
* ``potential``, which yields the potential energy
* ``force``, which yields the force vector
* ``virial``, which yields all virial components
* ``jp``, which yields the potential part of the heat current vector
* ``jk``, which yields the kinetic part of the heat current vector
* ``momentum``, which yields the momentum

One can write one or more (distinct) names in any order.

Output
------

The results are written to the :ref:`compute.out output file <compute_out>`.

Examples
--------

For example::

  compute 0 100 10 temperature

means using the 0-th grouping method defined in the :ref:`simulation model file <model_xyz>`, sampling ``temperature`` every 100 time steps and averaging over 10 data points before writing to file.
That is, there is only one output every :math:`100 \times 10=1000` time steps.

Notes
-----

None.
