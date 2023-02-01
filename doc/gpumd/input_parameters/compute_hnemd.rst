.. _kw_compute_hnemd:
.. index::
   single: compute_hnemd (keyword in run.in)

:attr:`compute_hnemd`
=====================

This keyword is used to calculate the thermal conductivity using the :ref:`homogeneous non-equilibrium molecular dynamics <hnemd>` (:term:`HNEMD`) method [Fan2019]_.
The results are written to the :ref:`kappa.out output file <kappa_out>`.

Syntax
------

.. code::

   compute_hnemd <output_interval> <Fe_x> <Fe_y> <Fe_z>

The first parameter is the output interval.

The next three parameters are the :math:`x`, :math:`y`, and :math:`z` components of the external driving force :math:`\boldsymbol{F}_e` in units of Å\ :sup:`-1`.

Usually, there should be only one nonzero component of :math:`\boldsymbol{F}_e`.
According to Eq. (8) of [Fan2019]_:

* Using a nonzero :math:`x` component of :math:`\boldsymbol{F}_e`, one can obtain the :math:`xx`, :math:`yx` and :math:`zx` components of the thermal conductivity tensor.
* Using a nonzero :math:`y` component of :math:`\boldsymbol{F}_e`, one can obtain the :math:`xy`, :math:`yy` and :math:`zy` components of the thermal conductivity tensor.
* Using a nonzero :math:`z` component of :math:`\boldsymbol{F}_e`, one can obtain the :math:`xz`, :math:`yz` and :math:`zz` components of the thermal conductivity tensor.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   compute_hnemd 1000 0.00001 0 0

This means that

* you want to calculate the thermal conductivity using the :term:`HNEMD` method;
* the thermal conductivity will be averaged and output every 1000 steps (the heat current is sampled for every step);
* the external driving force is along the :math:`x` direction and has a magnitude of :math:`10^{-5}` Å\ :sup:`-1`. 

Note that one should control the temperature when using this keyword.
Otherwise, the system will be heated up by the external driving force.

**Important:**
For this purpose, the :ref:`Nose-Hoover chain thermostat <nose_hoover_chain_thermostat>` is recommended.
The :ref:`Langevin thermostat <langevin_thermostat>` cannot be used for this purpose because it will affect the dynamics of the system.

Example 2
^^^^^^^^^

.. code::

   compute_hnemd 1000 0 0.00001 0

This is similar to the above example, but the external driving force is applied along the :math:`y` direction.

Related tutorial
----------------

The use of the :attr:`compute_hnemd` keyword is illustrated in :ref:`the tutorial on calculating the thermal conductivity via HNEMD simulations <tutorials>`.
