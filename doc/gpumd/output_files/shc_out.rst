.. _shc_out:
.. index::
   single: shc.out (output file)

``shc.out``
===========

This file contains the non-equilibrium virial-velocity correlation function :math:`K(t)` and the spectral heat current (:term:`SHC`) :math:`J_q(\omega)`, in a given direction, for a group of atoms, as defined in Eq. (18) and the left part of Eq. (20) of [Fan2019]_.
It is generated when invoking the :ref:`compute_shc keyword <kw_compute_shc>`.

File format
-----------

For each run, there are 3 columns and :attr:`2*Nc-1 + num_omega` rows.
Here, :attr:`Nc` is the number of correlation steps and :attr:`num_omega` is the number of frequency points.

In the first :attr:`2*Nc-1` rows:

* column 1: correlation time :math:`t` from negative to positive, in units of ps
* column 2: :math:`K^{\rm in}(t)` in units of Å eV/ps
* column 3: :math:`K^{\rm out}(t)` in units of Å eV/ps

:math:`K^{\rm in}(t) + K^{\rm out}(t) = K(t)` is exactly the expression in Eq. (18) of [Fan2019]_.
The in-out decomposition follows the definition in [Fan2017]_, which is useful for 2D materials but is not necessary for 3D materials.

In the next :attr:`num_omega` rows:

* column 1: angular frequency :math:`\omega` in units of THz
* column 2: :math:`J_q^{\rm in}(\omega)` in units of Å eV/ps/THz
* column 3: :math:`J_q^{\rm out}(\omega)` in units of Å eV/ps/THz
 
:math:`J_q^{\rm in}(\omega) + J_q^{\rm out}(\omega) = J_q(\omega)` is exactly the left expression in Eq. (20) of [Fan2019]_.

Only the potential part of the heat current has been included.

If :attr: 'group_id' is -1, then the file follows the above rules and will contain :math:`K(t)` and :math:`J_q(\omega)` for each group id except for group id 0. And the contents of the :attr: 'group_id' are arranged from smallest to largest.