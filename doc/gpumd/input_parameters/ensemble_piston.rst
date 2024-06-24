.. _piston:
.. _kw_ensemble_wall_piston:
.. index::
   single: piston (keyword in run.in)
   single: piston integrator

:attr:`ensemble` (piston)
=========================

This keyword is employed to configure a piston shock wave simulation, where a fixed wall of atoms is displaced at a specified velocity to generate a shock wave.

We recommand to use it with the :ref:`dump_piston keyword <kw_dump_piston>`.

Syntax
------

The parameters are specified as follows::

    ensemble piston direction <direction> vp <vp> thickness <thickness>

- :attr:`<direction>`: Specifies the direction of the shock wave (`x`, `y`, or `z`).
- :attr:`<vp>`: Indicates the velocity of the moving piston in km/s.
- :attr:`<thickness>`: Defines the thickness of the wall in Angstroms. Note that the other side of the piston is also fixed to prevent atoms from escaping. This keyword is optional with the default value 20.

Example
--------

.. code-block:: rst

    ensemble piston direction x vp 5 thickness 30

This command makes the piston moving at 5 km/s in x direction. The thickness of the wall is 30. 

