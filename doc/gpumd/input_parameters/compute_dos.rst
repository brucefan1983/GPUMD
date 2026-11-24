.. _kw_compute_dos:
.. index::
   single: compute_dos (keyword in run.in)

``compute_dos``
===================

This keyword computes the phonon density of states (:term:`PDOS`) using the mass-weighted velocity autocorrelation (:term:`VAC`) function.
The output is normalized such that the integral of the :term:`PDOS` over all frequencies equals :math:`3N`, where :math:`N` is the number of atoms.
If this keyword appears in a run, the mass-weighted :term:`VAC` function will be computed and directly used to compute the :term:`PDOS`.

Syntax
------

::

  compute_dos <sample_interval> <Nc> <omega_max> [{<optional_arg>}]

Parameters
----------

* ``sample_interval``: Sampling interval of the velocity data
* ``Nc``: Maximum number of correlation steps
* ``omega_max``: Maximum angular frequency :math:`\omega_{max}=2\pi\nu_{max}` used in the :term:`PDOS` calculation

The optional arguments (``optional_arg``) provide additional functionality by allowing special keywords.
The keywords for this function are ``group`` and ``num_dos_points``.
These keywords can be used in any order but the parameters associated with each must follow directly.

The option ``group`` has two parameters::

  group <group_method> <group>

where ``group_method`` is the grouping method to use for computation and ``group`` is the index of the group to use.
If ``group`` is -1, it means to calculate the DOS for every group in the ``group_method``. If each atom is in one group, one can get the per-atom DOS and then calculate the phonon participation ratio.

The option ``num_dos_points`` has one parameter::

  num_dos_points <points>

where ``points`` is the number of frequency points to be used in the DOS calculation.
It defaults to ``Nc`` if the ``num_dos_points`` option is not specified.

Output
------

The results of these calculations will be written to :ref:`mvac.out <mvac_out>` (for the normalized mass-weighted :term:`VAC` function) and :ref:`dos.out <dos_out>` (for the :term:`DOS`).

Examples
--------

An example for the use of this keyword is::

  compute_dos 5 200 400.0 group 1 2 num_dos_points 300

This means that you

* want to calculate the :term:`PDOS`
* the velocity data will be recorded every 5 steps
* the maximum number of correlation steps is 200
* the maximum angular frequency you want to consider is :math:`\omega_{max} = 2\pi\nu_{max} =` 400 THz
* you would like to compute only over group 2 in group method 1
* you would like the maximum angular frequency to be evenly divided into 300 points for output.

Notes
-----

None.
