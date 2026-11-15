.. _kw_compute_ic:
.. index::
   single: compute_ic (keyword in run.in)

:attr:`compute_ic`
===================

This keyword computes the Nernst--Einstein estimate of the ionic conductivity (:term:`IC`) for the selected atom type from its self mean-square displacement (:term:`MSD`).

Syntax
------

For this keyword, the command looks like::

  compute_ic <sample_interval> <Nc> <type_iex> <charge>

Parameters
----------

with parameters defined as

* :attr:`sample_interval`: Sampling interval of the position data
* :attr:`Nc`: Maximum number of correlation steps
* :attr:`type_iex`: Type index of the atoms to be calculated. For a compacted multi-element NEP potential, this refers to the active atom types printed at initialization; otherwise, it refers to the type index in the potential file.
* :attr:`charge`: Charge :math:`z_s` of each selected ion in units of the elementary charge :math:`e`.

Output
------

The results will be written to :ref:`ic.out output file <ic_out>`.

Examples
--------

An example of this function is::

  compute_ic 5 200 0 1

This means that you

* want to calculate the ionic conductivity from the mean-square displacement function
* the position data will be recorded every 5 steps
* the maximum number of correlation steps is 200
* you would like to compute only for the atom with type 0 according to the applicable type mapping and charge 1.

Notes
-----

Definition
^^^^^^^^^^

For :math:`N_s` ions of type :math:`s`, each with charge :math:`z_s e`, the output in direction :math:`\alpha` is

.. math::

   \sigma_{s,\alpha\alpha}^{\rm NE}(t)
   =\frac{N_s(z_s e)^2}{2Vk_{\rm B}T}\frac{d}{dt}
   \left[\frac{1}{N_s}\sum_{i\in s}
   \left\langle [r_i^{\alpha}(t)-r_i^{\alpha}(0)]^2\right\rangle\right],

where :math:`V` is the system volume, :math:`T` is the temperature, and unwrapped positions are used.
In the diffusive regime, this approaches :math:`N_s(z_s e)^2D_{s,\alpha}/(Vk_{\rm B}T)`, where :math:`D_{s,\alpha}` is the directional self-diffusion coefficient.
Displacement correlations between different ions are omitted, so this estimate generally differs from the collective ionic conductivity.
