.. _sw_ilp:
.. index::
   single: SW ILP

Hybrid SW+ILP potential
========================

The hybrid :term:`SW` + :term:`ILP` potential [Jiang2025]_ in :program:`GPUMD` combines
the Stillinger-Weber potential (:term:`SW`) [Stillinger1985]_ for intralayer
interactions and the interlayer potential (:term:`ILP`) [Ouyang2018]_ [Ouyang2020]_
for interlayer interactions to simulate homostructures based on
transition metal dichalcogenides layered materials. The modified SW term is described in [Jiang2015]_ and [Jiang2019]_.

Potential form
--------------

The ILP energy, taper function, and parameter definitions are given in :ref:`ILP potential form <ilp_potential_form>`.
The ILP contribution acts between atoms in different layers.

The site potential of modified :term:`SW` can be written as:

.. math::

   U_i^{\mathrm{SW}} =& \frac{1}{2}\sum_{j\neq i}\phi_2\left(r_{ij}\right)+
   \frac{1}{2}\sum_{j\neq i}\sum_{k\neq i,j}\phi_3 \left(r_{ij}, r_{ik}, \theta_{ijk}\right)\\
   \phi_2\left(r_{ij}\right) =& A_{ij}\epsilon_{ij}\left[B_{ij}
   \left(\frac{\sigma_{ij}}{r_{ij}} \right)^{p_{ij}} - \left(\frac{\sigma_{ij}}{r_{ij}}
   \right)^{q_{ij}} \right] \exp \left(\frac{\sigma_{ij}}{r_{ij}-a_{ij}\sigma_{ij}} \right)\\
   \phi_3\left(r_{ij}, r_{ik}, \theta_{ijk} \right) =& \lambda_{ijk}
   \epsilon_{ijk} \left[f_C(\delta) \delta \right]^2
   \exp \left(\frac{\gamma_{ij}\sigma_{ij}}{r_{ij}-a_{ij}\sigma_{ij}} \right)
   \exp \left(\frac{\gamma_{ik}\sigma_{ik}}{r_{ik}-a_{ik}\sigma_{ik}} \right) \\
   \delta =& \cos \theta_{ijk} - \cos \theta_{0ijk}\\

where :math:`\phi_2` and :math:`\phi_3` are two-body and three-body terms.
The sums run over intralayer neighbors; the factors of :math:`1/2` account for the two directions of each bond and the two orders of each neighbor pair.
The cutoff of :term:`SW` potential is :math:`a\cdot\sigma`. For some materials, such as borophene and
transition metal dichalcogenides, some unnecessary angle types should be excluded in the
three-body interaction by multiplying the cutoff function :math:`f_C(\delta)`.
It equals 1 for :math:`|\delta|\leq0.25`, 0 for :math:`|\delta|\geq0.35`, and
:math:`[1+\cos(\pi(|\delta|-0.25)/0.10)]/2` between these limits.

File format
-----------

This hybrid potential requires 2 potential files: :term:`ILP` potential file and
:term:`SW` potential file. We have adopted the :term:`ILP` file format that is similar
but not identical to that used by `lammps <https://docs.lammps.org/pair_ilp_graphene_hbn.html>`_.
To identify the layers, it's required to set two :attr:`group_methods`
in ``model.xyz`` file. :attr:`group_method` 0 is used to identify the different layers and :attr:`group_method` 1
is used to identify different sublayers. One transition metal dichalcogenide layer has three sublayers,
i.e., one :math:`MoS_2` layer has one Mo sublayer and two S sublayers. For atoms in the same layer,
the :attr:`group_id` of :attr:`group_method` 0 must be the same and for atoms in the same sublayer,
the :attr:`group_id` of :attr:`group_method` 1 must be the same.
For :math:`\mathrm{MX_2}`, M is a transition metal (such as Mo or W), and X is S, Se, or Te.

In ``run.in`` file, the :attr:`potential` setting is as::

  potential <ilp file> <sw file>

where :attr:`ilp file` and :attr:`sw file` are the filenames of
the :term:`ILP` potential file and :term:`SW` potential file.
:attr:`ilp file` is similar to other empirical potential files in :program:`GPUMD`::

  sw_ilp <number of atom types> <list of elements>
  beta alpha delta epsilon C d sR reff C6 S rcut1 rcut2
  ...

* :attr:`sw_ilp` is the name of this hybrid potential.
* :attr:`number of atom types` is the number of atom types defined in the ``model.xyz``.
  Here, this value must be set to **2** for transition metal dichalcogenide homostructures.
* :attr:`list of element` is a list of all the elements in the potential.
* The last line(s) is(are) parameters of :term:`ILP`. :attr:`rcut1` is used for calculating the normal vectors
  and :attr:`rcut2` is the cutoff of :term:`ILP`, usually 16Å.

More specifically, for :math:`\mathrm{MX_2}`, the :attr:`ilp file` is required to set as::

  sw_ilp 2 M X
  beta_MM alpha_MM delta_MM epsilon_MM C_MM d_MM sR_MM reff_MM C6_MM S_MM rcut1_MM rcut2_MM
  beta_MX alpha_MX delta_MX epsilon_MX C_MX d_MX sR_MX reff_MX C6_MX S_MX rcut1_MX rcut2_MX
  beta_XM alpha_XM delta_XM epsilon_XM C_XM d_XM sR_XM reff_XM C6_XM S_XM rcut1_XM rcut2_XM
  beta_XX alpha_XX delta_XX epsilon_XX C_XX d_XX sR_XX reff_XX C6_XX S_XX rcut1_XX rcut2_XX

The :attr:`sw file` use the same atomic type list as the :attr:`ilp file` and just contains parameters
of :term:`SW`. The potential file reads, specifically::

  A_MM B_MM a_MM sigma_MM gamma_MM
  A_MX B_MX a_MX sigma_MX gamma_MX
  A_XX B_XX a_XX sigma_XX gamma_XX
  lambda_MMM cos0_MMM
  lambda_MMX cos0_MMX
  lambda_MXM cos0_MXM
  lambda_MXX cos0_MXX
  lambda_XMM cos0_XMM
  lambda_XMX cos0_XMX
  lambda_XXM cos0_XXM
  lambda_XXX cos0_XXX
