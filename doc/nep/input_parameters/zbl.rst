.. _kw_zbl:
.. index::
   single: zbl (keyword in nep.in)

:attr:`zbl`
===========

This keyword can be used to spline the pair potential to the universal :term:`ZBL` potential [Ziegler1985]_ at short distances.
This is useful, for example, for models to be used in simulations of ion irradiation or extreme compression, when the interatomic distances can become very short.

The syntax is as follows::

  zbl <cutoff>

Here, :attr:`<cutoff>` is a real number that specifies the "outer" cutoff :math:`r_\mathrm{c}^\mathrm{ZBL-outer}`, beyond which the :term:`ZBL` potential is zero.
The "inner" cutoff of the :term:`ZBL` potential, below which value the pair interaction is completely given by the :term:`ZBL` potential, is fixed to half of the outer cutoff, :math:`r_\mathrm{c}^\mathrm{ZBL-inner} = r_\mathrm{c}^\mathrm{ZBL-outer} /2`, which we have empirically found to be a reasonable choice.

When this keyword is absent, the :term:`ZBL` potential will not be enabled and the value of :math:`r_\mathrm{c}^\mathrm{ZBL-outer}` is irrelevant.

The ``nep_dipole`` and ``nep_polarizability`` models reject this keyword and
:ref:`use_typewise_cutoff_zbl <kw_use_typewise_cutoff_zbl>`.

NEP-spin supports universal ZBL but not flexible ZBL. With ZBL enabled,
remove ``zbl.in`` for :ref:`spin_version 1 <kw_spin_version>` training.
ZBL contributes to energy, force, and virial, but not magnetic force or torque.

Permissible values are 1 Å :math:`\leq r_\mathrm{c}^\mathrm{ZBL-outer} \leq` 4 Å.

Only ordinary NEP (``model nep``, including its default and equivalent legacy
selection) supports flexible ZBL parameters. To use them, enable this keyword
and provide a ``zbl.in`` file in the working directory; the :attr:`<cutoff>`
parameter is still needed but will not be used. Other models reject flexible
ZBL. Without this keyword, a ``zbl.in`` file is ignored. In prediction mode,
ZBL parameters are read from ``nep.txt`` rather than ``zbl.in``.
For a :math:`n`-species system, there should be :math:`n(n+1)/2` lines in the `zbl.in` files.
Each line represents a unique pair of species. 
Counting from 1, the order of the lines is 1-1, 1-2, ..., 1-:math:`n`, 2-2, 2-3, ..., 2-:math:`n`, :math:`n`-:math:`n`.
For each pair of species, there are 10 parameters to be listed from left to right: the first two are the inner and outer cutoff radii, respectively; the next 8 are the parameters :math:`a_1` to :math:`a_8` in the ZBL function :math:`\phi(x)=a_1e^{-a_2x}+a_3e^{-a_4x}+a_5e^{-a_6x}+a_7e^{-a_8x}`.
