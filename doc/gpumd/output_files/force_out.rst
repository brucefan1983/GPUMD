.. _force_out:
.. index::
   single: force.out (output file)

``force.out``
=============

This file contains the per-atom forces of the atoms sampled at a given frequency.
It is produced when invoking :ref:`dump_force keyword <kw_dump_force>`.

File format
-----------

.. code::

   fx_1(1) fy_1(1) fz_1(1)
   fx_2(1) fy_2(1) fz_2(1)
   ...
   fx_N(1) fy_N(1) fz_N(1)
   fx_1(2) fy_1(2) fz_1(2)
   vx_2(2) fy_2(2) fz_2(2)
   ...
   fx_N(2) fy_N(2) fz_N(2)
   ...

* There are three columns, corresponding to the :math:`x`, :math:`y`, and :math:`z` components of the force.
* Each :attr:`N` consecutive lines correspond to one frame at a given time point.
* The number of lines equals the number of frames times :attr:`N`.
* :attr:`fx_m(n)` is the :math:`x` component of the :attr:`m`-th atom in the :attr:`n`-th frame.
* :attr:`fy_m(n)` is the :math:`y` component of the :attr:`m`-th atom in the :attr:`n`-th frame.
* :attr:`fz_m(n)` is the :math:`z` component of the :attr:`m`-th atom in the :attr:`n`-th frame.
* Force components are in units of eV/Å.
