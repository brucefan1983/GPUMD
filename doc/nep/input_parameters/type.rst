.. _kw_type:
.. index::
   single: type (keyword in nep.in)

:attr:`type`
============

This is a *mandatory* keyword that species the chemical species, for which a model is to be constructed.

The syntax is::

  type <number_of_species> [<species>]

where :attr:`<number_of_species>` must be an integer and :attr:`[<species>]` must be a list of :attr:`<number_of_species>` chemical symbols.
The latter are case-sensitive and must correspond to species in the periodic table.
