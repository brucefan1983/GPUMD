.. _kw_compute_hnema:
.. index::
   single: compute_hnema (keyword in run.in)

:attr:`compute_hnema`
=====================

The :attr:`compute_hnema` keyword is used to calculate the modal thermal conductivity using the :ref:`homogeneous non-equilibrium modal analysis <hnema>` (:term:`HNEMA`) method [Gabourie2021]_.
The results are written to the :ref:`kappamode.out output file <kappamode_out>`.

Syntax
------

.. code::

   compute_hnema sample_interval output_interval Fe_x Fe_y Fe_z first_mode last_mode bin_option size

:attr:`sample_interval` is the sampling interval (in number of steps) used to compute the heat modal heat current.
Must be a divisor of :attr:`output_interval`.
      
:attr:`output_interval` is the interval to output the modal thermal conductivity. Each modal thermal conductivity output is averaged over all samples per output interval.

:attr:`Fe_x` is the :math:`x` direction component of the external driving force :math:`F_e` in units of Å\ :sup:`-1`.

:attr:`Fe_y` is the :math:`y` direction component of the external driving force :math:`F_e` in units of Å\ :sup:`-1`.

:attr:`Fe_z` is the :math:`z` direction component of the external driving force :math:`F_e` in units of Å\ :sup:`-1`.

:attr:`first_mode` and :attr:`last_mode` are the first and last mode, respectively, in the :ref:`eigenvector.in input file <eigenvector_in>` to include in the calculation.

:attr:`bin_option` determines which binning technique to use.
The options are :attr:`bin_size` and :attr:`f_bin_size`.

:attr:`size` defines how the modes are added to each bin.
If :attr:`bin_option` is ``bin_size``, then this is an integer describing how many modes are included per bin.
If :attr:`bin_option` is ``f_bin_size``, then binning is by frequency and this is a float describing the bin size in THz.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   compute_hnema 10 1000 0.000008 0 0 1 27216 f_bin_size 1.0

This means that

* you want to calculate the modal thermal conductivity with the :term:`HNEMA` method
* the modal thermal conductivity will be sampled every 10 steps
* the average of all sampled modal thermal conductivities will be output every 1000 time steps
* the external driving force is along the :math:`x` direction and has a magnitude of :math:`0.8 \times 10^{-5}` Å\ :sup:`-1`
* the range of modes you want to include of calculations are from 1 to 27216
* you want to bin the modes by frequency with a bin size of 1 THz.

Example 2
^^^^^^^^^

.. code::

   compute_hnema 10 1000 0.000008 0 0 1 27216 bin_size 1

This example is identical to Example 1, except the modes are binned by count.
Here, each bin only has one mode (i.e. all modes are included in the output).

Example 3
^^^^^^^^^

.. code::

   compute_hnema 10 1000 0.000008 0 0 1 27216 bin_size 10

This example is identical to Example 2, except each bin has 10 modes.

Caveats
-------
This computation can be very memory intensive.
The memory requirements are comparable to the size of the :ref:`eigenvector.in input file <eigenvector_in>`.

This keyword cannot be used in the same run as the :ref:`compute_gkma keyword <kw_compute_gkma>`.
The keyword used last will be used in the run.
