.. _replica_kw_kspace:
.. index::
   single: kspace (keyword in replica run.in)

:attr:`!kspace`
===============

Syntax
------

::

  kspace <method> [<spacing>]

Select ``pppm`` (the default) or ``ewald`` for charge NEP electrostatics, as by the :ref:`kspace keyword <kw_kspace>` of ``gpumd``.
This setting has no effect for ordinary NEP models.

Example
-------

::

  kspace pppm
