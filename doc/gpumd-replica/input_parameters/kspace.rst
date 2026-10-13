.. _replica_kw_kspace:
.. index::
   single: kspace (keyword in gpumd_replica run.in)

:attr:`!kspace`
===============

Syntax
------

::

  kspace <method>

Select ``pppm`` (the default) or ``ewald`` for charge NEP electrostatics.
This setting has no effect for ordinary NEP models.

Example
-------

::

  kspace pppm
