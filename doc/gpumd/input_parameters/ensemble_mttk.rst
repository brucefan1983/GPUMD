.. _mttk:
.. _kw_ensemble_mttk:
.. index::
   single: npt_mttk (keyword in run.in)
   single: nph_mttk (keyword in run.in)
   single: MTTK integrator

:attr:`ensemble` (MTTK)
=======================

The variants of the :attr:`ensemble` keyword described on this page implement the Nosé-Hoover thermostat [Hoover1996]_ and the Parrinello-Rahman barostat [Parrinello1981]_.
Both the thermostat and barostat are enhanced by the :ref:`Nosé-Hoover chain method <nose_hoover_chain_thermostat>` as recommended in [Martyna1994]_. 
The resulting so-called Martyna-Tuckerman-Tobias-Klein (:term:`MTTK`) integrators enable one to perform simulations in the isothermal-isobaric (NPT) or isenthalpic (NPH) ensembles.

This implementation of the :term:`MTTK` integrator provides more fine-grained control than :ref:`"standard" ensembles <kw_ensemble_standard>`.
The style of the :attr:`npt_mttk` and :attr:`nph_mttk` keywords is therefore slightly different.

Syntax
------

The parameters for running in the isothermal-isobaric ensemble (NPT) can be specified as follows::

    ensemble npt_mttk temp <T_1> <T_2> <tau_temp> <direction> <p_1> <p_2> <tau_press>

:attr:`<T_1>` and :attr:`<T_2>` specify the initial and final temperature, respectively.
The temperature will vary linearly from :attr:`<T_1>` to :attr:`<T_2>` during the simulation process.
The optional :attr:`<tau_temp>` parameter, which defaults to ``100``, determines the period of the thermostat in units of the timestep.
It determines how strongly the system is coupled to the thermostat.

The :attr:`<direction>` parameter can assume one or more of the following values: ``iso``, ``aniso``, ``tri``, ``x``, ``y``, ``z``, ``xy``, ``yz``, ``xz``.
Here, ``iso``, ``aniso``, and ``tri`` use hydrostatic pressure as the target pressure.
``iso`` updates the simulation cell isotropically.
``aniso`` updates the dimensions of the simulation cell along :math:`x`, :math:`y`, and :math:`z` independently.
``tri`` updatess all six degrees of freedom of the simulation cell.
Using ``x``, ``y``, ``z``, ``xy``, ``yz``, ``xz`` allows one to specify each stress component independently.

The parameters :attr:`<p_1>` and :attr:`<p_2>` specify the initial and final pressure, respectively.
Finally, the optional parameter :attr:`<tau_press>`, which defaults to ``1000``, determines the period of the barostat in units of the timestep.
It determines how strongly the system is coupled to the barostat.

The :attr:`nph_mttk` keyword can be used in analoguous fashion to run simulations in the isenthalpic (NPH) ensemble::

    ensemble nph_mttk <direction> <p_1> <p_2> <tau_press>


Examples
--------

Below follow some examples of how to use these keywords for different ensembles.

NPT Ensemble
^^^^^^^^^^^^

.. code-block:: rst

    ensemble npt_mttk temp 300 300 iso 10 10

This command sets the target temperature to 300 K and the target pressure to 10 GPa.
The cell shape will not change during the simlation but only the volume.
These conditions are suitable for simulating liquids.
If not constrained, the cell shape may undergo extreme changes since liquids have a vanishing shear modulus (in the long-time limit).

.. code-block:: rst

    ensemble nvt_mttk temp 300 1000 iso 100 100

This command ramps the temperature from 300 K to 1000 K, while keeping the pressure at 100 GPa.

.. code-block:: rst

    ensemble npt_mttk temp 300 300 aniso 10 10

This command replaces `iso` with `ansio`.
The three dimensions of the cell thus change independently, but `xy`, `xz` and `yz` remain unchanged.

.. code-block:: rst

    ensemble npt_mttk temp 300 300 tri 10 10

All six degrees of freedom of the simulation cell are allowed to change.
The simulated system will converge to fully hydrostatic pressure. 
Note that with `iso` and `aniso`, there is no guarantee that the pressure is hydrostatic, as the system is constrained.

.. code-block:: rst

    ensemble npt_mttk temp 300 300 x 5 5 y 0 0 z 0 0

Using these settings one applies a pressure of 5 GPa along the :math:`x` direction, and 0 GPa along the :math:`y` and :math:`z` directions.

.. code-block:: rst

    ensemble npt_mttk temp 300 300 x 5 5

Using this setup one applies 5 GPa of pressure along the :math:`x` direction while fixing the cell dimensions along the other directions.


NPH Ensemble
^^^^^^^^^^^^

.. code-block:: rst

    ensemble nph_mttk iso 10 10

When using this command one performs a NPH simulation at 10 GPa, allowing only changes in the volume but not the cell shape.
