.. _kw_potential:
.. index::
   single: potential (keyword in run.in)

:attr:`potential`
=================

This keyword is used to specify the interatomic potential model for the system.

Available potential models
--------------------------

* :ref:`Tersoff-1989 potential <tersoff_1989>`
* :ref:`Tersoff-1988 potential <tersoff_1988>`
* :ref:`Tersoff mini potential <tersoff_mini>`
* :ref:`Embedded atom method (EAM) potential <eam>`
* :ref:`Force constant potential (FCP) <fcp>`
* :ref:`Lennard-Jones (LJ) potential <lennard_jones_potential>`
* :ref:`Neuroevolution potential (NEP) <nep>`

Syntax
------

This keyword needs one parameter, :attr:`potential_filename`, which is the filename (including relative or absolute path) of the potential file to be used.

Example
-------

.. code::

   potential Si_NEP.txt
