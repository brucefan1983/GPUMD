.. _kw_compute_dos:
.. index::
   single: compute_dos (keyword in run.in)

:attr:`compute_dos`
===================

This keyword computes the phonon density of states (:term:`PDOS`) using the mass-weighted velocity autocorrelation (:term:`VAC`) function.
The output is normalized such that the integral of the :term:`PDOS` over all frequencies equals :math:`3N`, where :math:`N` is the number of atoms.
If this keyword appears in a run, the mass-weighted :term:`VAC` function will be computed and directly used to compute the :term:`PDOS`.

The results of these calculations will be written to :ref:`mvac.out <mvac_out>` (for the mass-normalized :term:`VAC` function) and :ref:`dos.out <dos_out>` (for the :term:`DOS`).

Syntax
------
For this keyword, the command looks like::

  compute_dos sample_interval Nc omega_max <optional_args>

with parameters defined as:

* :attr:`sample_interval`: Sampling interval of the velocity data
* :attr:`Nc`: Maximum number of correlation steps
* :attr:`omega_max`: Maximum angular frequency :math:`\omega_{max}=2\pi\nu_{max}` used in the :term:`PDOS` calculation

The :attr:`optional_args` provide additional functionality by allowing special keywords.
The keywords for this function are :attr:`group` and :attr:`num_dos_points`.
These keywords can be used in any order but the parameters associated with each must follow directly.
The parameters are:

* :attr:`group group_method group`

 * :attr:`group_method`: The grouping method to use for computation
 * :attr:`group`: The group in the grouping method to use

* :attr:`num_dos_points points`

 * :attr:`points`: Number of frequency points to be used in the DOS calculation (:attr:`Nc` if option not selected)

Example
-------

An example of this keyword is::
  
  compute_dos 5 200 400.0 group 1 1 num_dos_points 300

This means that you

* want to calculate the :term:`PDOS`
* the velocity data will be recorded every 5 steps
* the maximum number of correlation steps is 200
* the maximum angular frequency you want to consider is :math:`\omega_{max} = 2\pi\nu_{max} =` 400 THz
* you would like to compute only over group 1 in group method 1
* you would like the maximum angular frequency to be cut in to 300 points for output.


Caveats
-------
This keyword cannot be used in the same run as the :ref:`compute_sdc keyword <kw_compute_sdc>`.

Related tutorial
----------------
The use of this keyword is illustrated in the :ref:`tutorial on the density of states <tutorials>`.
