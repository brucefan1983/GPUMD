.. _kw_type_weight:
.. index::
   single: type_weight (keyword in nep.in)

:attr:`type_weight`
===================

This keyword allows one to specify different weights for different chemical species.
These weights are employed in the calculation of the force term in the loss function.
Different weights for different species can be useful if the number of atoms per species are very different, e.g., for a few impurity atoms embedded in a host.

The syntax is as follows::

  type_weigth [<weight>]

Here, :attr:`[<weight>]` must be a list of :math:`N_\mathrm{typ}` non-negative real numbers representing the relative force weights for the different atomic species.
By default all species carry the same weight (``1.0``).
