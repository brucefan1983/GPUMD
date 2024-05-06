.. _kw_compute_shc:
.. index::
   single: compute_shc (keyword in run.in)

:attr:`compute_shc`
===================

The :attr:`compute_shc` keyword is used to compute the non-equilibrium virial-velocity correlation function :math:`K(t)` and the spectral heat current (:term:`SHC`) :math:`J_q(\omega)`, in a given direction, for a group of atoms, as defined in Eq. (18) and the left part of Eq. (20) of [Fan2019]_.
The results are written to the :ref:`shc.out output file <shc_out>`.

  
Syntax
------

.. code::

   compute_shc <sample_interval> <Nc> <transport_direction> <num_omega> <max_omega> [{<optional_arg>}]

:attr:`sample_interval` is the sampling interval (number of steps) between two correlation steps.
This parameter must be an integer that is :math:`\geq 1` and :math:`\leq 10`. 

:attr:`Nc` is the total number of correlation steps.
This parameter must be an integer that is :math:`\geq 100` and :math:`\leq 1000`. 

:attr:`transport_direction` is the direction of heat transport to be measured.
It can only be 0, 1, and 2, corresponding to the :math:`x`, :math:`y`, and :math:`z` directions, respectively.

:attr:`num_omega` is the number of frequency points one wants to consider. 

:attr:`max_omega` is the maximum angular frequency (in units of THz) one wants to consider.
The angular frequency data will be :attr:`max_omega/num_omega, 2*max_omega/num_omega, ..., max_omega`.

:attr:`<optional_arg>` can only be :attr:`group`, which requires two parameters::

   group <grouping_method> <group_id>

This means that :math:`K(t)` will be calculated for atoms in group :attr:`group_id` of grouping method :attr:`grouping_method`.
Usually, :attr:`group_id` should be :math:`\geq 0` and smaller than the number of groups in grouping method :attr:`grouping_method`.
If :attr:`grouping_method` is assigned and :attr:`group_id` is -1, it means to calculate the :math:`K(t)` for every :attr:`group_id` except for :attr:`group_id` 0 in the assigned :attr:`grouping_method`.
Since it is very time and memory consuming to calculate the all group :math:`K(t)` for a large system, so one can assign the part that don't want to calculate to :attr:`group_id` 0.
Also, grouping method :attr:`grouping_method` must be defined in the :ref:`simulation model input file <model_xyz>`.
If this option is missing, it means computing :math:`K(t)` for the whole system.

Examples
--------

Example 1
^^^^^^^^^

The command::

  compute_shc 2 250 0 1000 400.0

means that

* you want to calculate :math:`K(t)` for the whole system
* the sampling interval is 2
* the maximum number of correlation steps is 250
* the transport direction is :math:`x`
* you want to consider 1000 frequency points
* the maximum angular frequency is 400 THz

Example 2
^^^^^^^^^

The command::

  compute_shc 1 500 1 500 200.0 group 0 4

means that

* you want to calculate :math:`K(t)` for atoms in group :attr:`4` defined in grouping method :attr:`0`
* the sampling interval is 1 (sample the data at each time step)
* the maximum number of correlation steps is 500
* the transport direction is :math:`y`
* you want to consider 500 frequency points
* the maximum angular frequency is 200 THz

Example 3
^^^^^^^^^

The command::

  compute_shc 1 500 1 500 200.0 group 1 -1

means that

* you want to calculate :math:`K(t)` for all :attr:`group_id` except for :attr:`group_id` 0 defined in grouping method :attr:`1`
* the sampling interval is 1 (sample the data at each time step)
* the maximum number of correlation steps is 500
* the transport direction is :math:`y`
* you want to consider 500 frequency points
* the maximum angular frequency is 200 THz

Caveats
-------
This computation can be memory consuming.

If you want to use the in-out decomposition for 2D materials, you need to make the basal plane in the :math:`xy` directions.


Related tutorial
----------------

The use of this keyword is illustrated in the tutorial on the :ref:`thermal transport from NEMD and HNEMD simulations <tutorials>`.
