.. _velocity_out:
.. index::
   single: velocity.out (output file)

``velocity.out``
================

This file contains the velocities of the atoms sampled at a given frequency.
It produced when invoking the :ref:`dump_velocity keyword <kw_dump_velocity>`.


File format
-----------

.. code::

    vx_1(1) vy_1(1) vz_1(1)
    vx_2(1) vy_2(1) vz_2(1)
    ...
    vx_N(1) vy_N(1) vz_N(1)
    vx_1(2) vy_1(2) vz_1(2)
    vx_2(2) vy_2(2) vz_2(2)
    ...
    vx_N(2) vy_N(2) vz_N(2)
    ...

* There are three columns, corresponding to the :math:`x`, :math:`y`, and :math:`z` components of the velocity.
* Each :attr:`N` consecutive lines correspond to one frame at a given time point.
* The number of lines equals the number the number of frames times :attr:`N`.
* :attr:`vx_m(n)` is the :math:`x` component of the :attr:`m`-th atom in the :attr:`n`-th frame.
* :attr:`vy_m(n)` is the :math:`y` component of the :attr:`m`-th atom in the :attr:`n`-th frame.
* :attr:`vz_m(n)` is the :math:`z` component of the :attr:`m`-th atom in the :attr:`n`-th frame.
* Velocity components are in units of Å/fs.
