.. _kw_type:
.. index::
   single: type (keyword in nep.in)

:attr:`type`
============

This is a *mandatory* keyword that specifies the chemical species, for which a model is to be constructed.

Syntax
------

::

  type <number_of_species> [<species>]

Parameters
----------

:attr:`<number_of_species>` must be an integer and :attr:`[<species>]` must be a list of :attr:`<number_of_species>` chemical symbols.
The latter are case-sensitive and must correspond to species in the periodic table.

Output
------

None.

Examples
--------

::

  type 2 Si C

Notes
-----

None.
