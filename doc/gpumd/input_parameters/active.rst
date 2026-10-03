.. _kw_active:
.. index::
   single: active (keyword in run.in)

:attr:`active`
=====================

Run on-the-fly active learning, based on committee uncertainty estimates over a group of supplied NEP potentials. Note that this mode is only supported with NEP potentials. Furthermore, the molecular dynamics simulation is propagated using the first NEP potential specified in :ref:`run.in <run_in>`, or using the average of the potentials when `dump_observer` is given in `average` mode.

The uncertainty :math:`\sigma_f` is estimated as the maximum over the atoms :math:`i` of the standard deviation of the force over the :math:`M` models,

.. math::
        \sigma_f = \textrm{max}_i \sqrt{ \sigma_{i,x}^2 + \sigma_{i, y}^2 + \sigma_{i, z}^2  },

where

.. math::
        \sigma_{i,k}^2 = \frac{1}{M} \sum_{m=1}^{M} \left( F_{i,k}^{(m)} - \bar{F}_{i,k} \right)^2

is the variance of the :math:`k` Cartesian component of the force on atom :math:`i`, :math:`F_{i,k}^{(m)}` is that component for model :math:`m`, and :math:`\bar{F}_{i,k}` is its mean over the models. If the uncertainty exceeds the specified threshold, :math:`\sigma_f>\delta`, for a structure in a step of an molecular dynamics simulation, then that structure is appended to the file `active.xyz` in the `extended XYZ format <https://github.com/libAtoms/extxyz>`_. Additionally, the simulation time :math:`t` and :math:`\sigma_f` are written to the file `active.out` regardless of if :math:`\sigma_f>\delta`.
If the uncertainty of any atom is NaN, which a model with forces that are not finite produces, :math:`\sigma_f` is NaN and the structure is appended to `active.xyz`.
The energy, virial and forces in `active.xyz` are those of the main potential alone.
The `stress` field holds the pressure tensor in eV/Å³, which adds the kinetic contribution of the velocities to the virial of the main potential.
The forces exclude those that other keywords add to the atoms, such as `add_force` and `add_efield`.
Checking the uncertainty leaves the molecular dynamics run unchanged.

`active` takes five arguments. The first four sets the interval for uncertainty estimation and what per atom quantities are outputted in `active.xyz`, the fifth keyword sets the threshold :math:`\delta` in units of eV/Å.
      

Syntax
------

.. code::

   active <interval> <has_velocity> <has_force> <has_uncertainty> <threshold>

:attr:`interval` is the interval (number of steps) between checking the uncertainty. If set to 1, the uncertainty will be computed for every step of the MD simulation.
:attr:`has_velocity` can be 1 or 0, which means the velocities will or will not be included in the exyz output.
:attr:`has_force` can be 1 or 0, which means the forces will or will not be included in the exyz output.
:attr:`has_uncertainty` can be 1 or 0, which means the per atom uncertainties will or will not be included in the exyz output.
:attr:`threshold` is a non-negative float, and corresponds to the threshold :math:`\delta` in units of eV/Å.

Examples
--------

Example 1
^^^^^^^^^
To run on-the-fly active learning using a committee of 5 NEP potentials, checking if the uncertainty exceeds 0.01 eV/Å every tenth MD step, write::

  potential nep0
  potential nep1  
  potential nep2
  potential nep3
  potential nep4
  ...
  active 10 1 1 1 0.01

before the :ref:`run keyword <kw_run>`. This will generate two output files, `active.xyz` and `active.out`. 

Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* Molecular dynamics will be run with the first potential specified, or with the average of the potentials under `dump_observer` in `average` mode.
* If the system has exploded, unphysical structures may be saved since no upper bound is set on the uncertainty :math:`\sigma_f`.
  Ensure that the resulting structures in `active.xyz` are physical. 
