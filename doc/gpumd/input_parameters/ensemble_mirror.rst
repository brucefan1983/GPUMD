.. _mirror:
.. _kw_ensemble_mirror:
.. index::
   single: mirror (keyword in run.in)
   single: mirror integrator

:attr:`ensemble` (mirror)
=========================

This keyword is employed to configure a momentum mirror shock wave simulation, where atoms are deflected by a moving momentum mirror to generate a shock wave.

The direction of the shock wave is along the x axis.

We recommand to use it with the :ref:`dump_piston keyword <kw_dump_piston>`.

Syntax
------

The parameters are specified as follows::

    ensemble mirror vp <vp> thickness <thickness>

- :attr:`<vp>`: Indicates the velocity of the moving mirror in km/s.
- :attr:`<thickness>`: Defines the thickness of the fixed region in the other side of the cell. Its unit is Ang. This keyword is optional with the default value 20.

Example
--------

.. code-block:: rst

    ensemble mirror vp 5 thickness 30

This command makes the mirror moving at 5 km/s in x direction. The thickness of the fixed region is 30. 