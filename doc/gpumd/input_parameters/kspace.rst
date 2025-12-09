.. _kw_kspace:
.. index::
   single: kspace (keyword in run.in)

:attr:`kspace`
==============

This keyword is used to set the computation method for the reciprocal space contribution to the electristatic energy.

Syntax
------

This keyword is used as follows::

  kspace <method>

where :attr:`<method>` can be either `ewald` or `pppm`.

Example
-------

To use the particle-particle particle-mesh (PPPM) method use::

   kspace pppm
