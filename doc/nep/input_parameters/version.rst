.. _kw_version:
.. index::
   single: version (keyword in nep.in)

:attr:`version`
===============

This keyword selects which :term:`NEP` version should be used.

Syntax
------

::

  version <version_number>

Parameters
----------

Here, :attr:`<version_number>` must be an integer, which can only be 4, corresponding to NEP4.
This selects the model version, independently of ``format_version 1`` in :ref:`nep.txt <nep_txt>`.

Output
------

None.

Examples
--------

::

  version 4

Notes
-----

More information about the :term:`NEP` formalism can be found :ref:`here <nep_formalism>`.
