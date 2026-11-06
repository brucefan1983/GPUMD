.. _replica_kw_potential:
.. index::
   single: potential (keyword in replica run.in)

:attr:`!potential`
==================

Syntax
------

Specify one potential file::

  potential <filename>

Supported are energy NEP models and charge NEP (qNEP) models, read as by the :ref:`potential keyword <kw_potential>` of ``gpumd``.
Temperature-dependent NEP models are not supported.
Charge NEP models require periodic boundaries in all three directions.

Example
-------

::

  potential nep.txt
