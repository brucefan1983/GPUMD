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

One can also use flexible ZBL parameters by providing a `zbl.in` file in the working directory, in which case the :attr:`<cutoff>` parameter is still needed but will not be used.
For a :math:`n`-species system, there should be :math:`n(n+1)/2` lines in the `zbl.in` files.
Each line represents a unique pair of species. 
Counting from 1, the order of the lines is 1-1, 1-2, ..., 1-:math:`n`, 2-2, 2-3, ..., 2-:math:`n`, :math:`n`-:math:`n`.
For each pair of species, there are 10 parameters to be listed from left to right: the first two are the inner and outer cutoff radii, respectively; the next 8 are the parameters :math:`a_1` to :math:`a_8` in the ZBL function :math:`\phi(x)=a_1e^{-a_2x}+a_3e^{-a_4x}+a_5e^{-a_6x}+a_7e^{-a_8x}`.
