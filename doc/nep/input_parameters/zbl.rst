.. _kw_zbl:
.. index::
   single: zbl (keyword in nep.in)

:attr:`zbl`
===========

This keyword can be used to spline the pair potential to the universal :term:`ZBL` potential [Ziegler1985]_ at short distances.
This is useful, for example, for models to be used in simulations of ion irradiation or extreme compression, when the interatomic distances can become very short.

The syntax is as follows::

  zbl <cutoff>

Here, :attr:`<cutoff>` is a real number that specifies the "outer" cutoff :math:`r_\mathrm{c}^\mathrm{ZBL-outer}`, below which the :term:`NEP` pair potential is being splined to the :term:`ZBL` potential.
The "inner" cutoff of the :term:`ZBL` potential, below which value the pair interaction is completely given by the :term:`ZBL` potential, is fixed to half of the outer cutoff, :math:`r_\mathrm{c}^\mathrm{ZBL-inner} = r_\mathrm{c}^\mathrm{ZBL-outer} /2`, which we have empirically found to be a reasonable choice.

When this keyword is absent, the :term:`ZBL` potential will not be enabled and the value of :math:`r_\mathrm{c}^\mathrm{ZBL-outer}` is irrelevant.

Permissible values are 1 Å :math:`\leq r_\mathrm{c}^\mathrm{ZBL-outer} \leq` 2.5 Å
