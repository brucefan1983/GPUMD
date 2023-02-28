.. _kw_compute:
.. index::
   single: compute (keyword in run.in)

:attr:`compute`
===============

This keyword is used to compute and output space and time averaged quantities. 
The results are written to the :ref:`compute.out output file <compute_out>`.


Syntax
------
It is used in the following way::

  compute <grouping_method> <sample_interval> <output_interval> {<quantity>}

The first parameter :attr:`grouping_method` refers to the grouping method defined in the :ref:`simulation model file <model_xyz>`.
This parameter should be an integer and a number :math:`m` means the :math:`m`-th grouping method (we count from 0) in the :ref:`simulation model file <model_xyz>`.

The second parameter :attr:`sample_interval` means sampling the quantities every so many time steps.

The third parameter :attr:`output_interval` means averaging over so many sampled data before giving one output.

Starting from the fourth parameter, one can list the quantities to be computed.

The allowed names for the quantities are:

* :attr:`temperature`, which is the temperature
* :attr:`potential`, which is the potential energy
* :attr:`force`, which is the force vector
* :attr:`virial`, which is the diagonal part of the virial
* :attr:`jp`, which is the potential part of the heat current vector
* :attr:`jk`, which is the kinetic part of the heat current vector

One can write one or more (distinct) names in any order.

Example
-------

For example::
  
  compute 0 100 10 temperature

means using the 0-th grouping method defined in the :ref:`simulation model file <model_xyz>`, sampling :attr:`temperature` every 100 time steps and averaging over 10 data points before writing to file.
That is, there is only one output every :math:`100 \times 10=1000` time steps.
