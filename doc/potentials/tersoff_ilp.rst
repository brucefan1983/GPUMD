.. _tersoff_ilp:
.. index::
   single: Tersoff ILP

Hybrid Tersoff+ILP potential
============================

The hybrid Tersoff + :term:`ILP` potential in :program:`GPUMD` combines the Tersoff (1988) potential [Tersoff1988]_ for intralyer interactions and the interlayer potential (:term:`ILP`) [Ouyang2018]_ [Ouyang2020]_  for interlayer interactions to simulate homo- and heterostructures based on graphene and :math:`h`-BN layered materials.
The Tersoff term of this hybrid potential uses the same implementation as :ref:`Tersoff (1988) potential <tersoff_1988>`.

Potential form
--------------

The site potential of :term:`ILP` can be written as:

.. math::
   
   U_i^{\mathrm{ILP}}=  \mathrm{Tap}(r_{ij}) \left[U^{\mathrm{att}}(r_{ij})+U^{\mathrm{rep}}(r_{ij}, \boldsymbol{n}_i, \boldsymbol{n}_j)\right]

The function :math:`\mathrm{Tap}(r_{ij})` is a cutoff function and takes the following form in the intermediate region:

.. math::

   \mathrm{Tap}(r_{ij})=20{\left(\frac{r_{ij}}{R_{\mathrm{cut}}}\right)}^7-
   70{\left(\frac{r_{ij}}{R_{\mathrm{cut}}}\right)}^6+84{\left(\frac{r_{ij}}{R_{\mathrm{cut}}}\right)}^5-
   35{\left(\frac{r_{ij}}{R_{\mathrm{cut}}}\right)}^4+1

The repulsive term and the attractive term take the following forms:

.. math::

   U^{\mathrm{att}}(r_{ij})&=-\frac{1}{1+e^{-d_{ij}\left[r_{ij}/(S_{\mathrm{R},ij}\cdot r_{ij}^{\mathrm{eff}})-1\right]}}\frac{C_{6,ij}}{r_{ij}^{6}}.
   
   U^{\mathrm{rep}}(r_{ij}, \boldsymbol{n}_i, \boldsymbol{n}_j)&=e^{\alpha_{ij}\left(1-\frac{\gamma_{ij}}{\beta_{ij}}\right)} \left\{\epsilon_{ij}+C_{ij}\left[e^{-{(\frac{\rho_{ij}}{\gamma_{ij}})}^2}+e^{-{\left(\frac{\rho_{ji}}{\gamma_{ij}}\right)}^2}\right]\right\}.

where :math:`\boldsymbol n_i` and :math:`\boldsymbol n_j` are normal vectors of atom :math:`i` and :math:`j`,
:math:`\rho_{ij}` and :math:`\rho_{ji}` are the lateral interatomic distance, which can be expressed as:

.. math::

   \rho_{ij}^{2}&= r_{ij}^2-{(\boldsymbol r_{ij} \cdot \boldsymbol n_i)}^2\\
   \rho_{ji}^{2}&= r_{ij}^2-{(\boldsymbol r_{ij} \cdot \boldsymbol n_j)}^2

Other variables are all fitting parameters.

The site potential of Tersoff can be written as:

.. math::
   
   U_i^{\mathrm{Tersoff}} =  \frac{1}{2} \sum_{j \neq i} f_C(r_{ij}) \left[ f_R(r_{ij}) - b_{ij} f_A(r_{ij}) \right].

The function :math:`f_{C}` is a cutoff function, which is 1 when :math:`r_{ij}<R` and 0 when :math:`r_{ij}>S` and takes the following form in the intermediate region:

.. math::

   f_{C}(r_{ij}) = \frac{1}{2}
   \left[
   1 + \cos \left( \pi \frac{r_{ij} - R}{S - R} \right)
   \right].

The repulsive function :math:`f_{R}` and the attractive function :math:`f_{A}` take the following forms:

.. math::

   f_{R}(r) &= A e^{-\lambda r_{ij}} \\
   f_{A}(r) &= B e^{-\mu r_{ij}}.

The bond-order is

.. math::

   b_{ij} = \left(1 + \beta^{n} \zeta^{n}_{ij}\right)^{-\frac{1}{2n}},

where

.. math::
   
   \zeta_{ij} &= \sum_{k\neq i, j}f_C(r_{ik}) g_{ijk} e^{\alpha(r_{ij} - r_{ik})^{m}} \\
   g_{ijk} &= \gamma\left( 1 + \frac{c^2}{d^2} - \frac{c^2}{d^2+(h-\cos\theta_{ijk})^2} \right).

File format
-----------

This hybrid potential requires 2 potential files: :term:`ILP` potential file and 
Tersoff potential file. We have adopted the :term:`ILP` file format that similar 
but not identical to that used by `lammps <https://docs.lammps.org/pair_ilp_graphene_hbn.html>`_.
To identify the different layers, it's required to set one :attr:`group_methods`
in ``model.xyz`` file.
Now this hybrid potential could be only used to simulate homo- and heterostructures of graphene and :math:`h`-BN.

In ``run.in`` file, the :attr:`potential` setting is as::
  
  potential <ilp file> <tersoff file>

where :attr:`ilp file` and :attr:`tersoff file` are the filenames of 
the :term:`ILP` potential file and Tersoff potential file.
:attr:`ilp file` is similar to other empirical potential files in :program:`GPUMD`::

  tersoff_ilp <number of atom types> <list of elements>
  <group_method for layers>
  beta alpha delta epsilon C d sR reff C6 S rcut1 rcut2
  ...

* :attr:`tersoff_ilp` is the name of this hybrid potential.
* :attr:`number of atom types` is the number of atom types defined in the ``model.xyz``.
* :attr:`list of element` is a list of all the elements in the potential.
* :attr:`group_method for layers` is the :attr:`group_method` set in ``model.xyz`` 
  to identify different layers. For example, monolayer graphene and monolayer 
  :math:`h`-BN are both single layer so for the atoms in each layer 
  the :attr:`group_id` of :attr:`group_method for layers` are the same.
* The last line(s) is(are) parameters of :term:`ILP`. :attr:`rcut1` is used for calculating the normal vectors 
  and :attr:`rcut2` is the cutoff of :term:`ILP`, usually 16Å.

More specifically, for graphene, if :attr:`group_method` 0 is used for different layers, the :attr:`ilp file` is required to set as::

  tersoff_ilp 1 C
  0
  beta_CC alpha_CC delta_CC epsilon_CC C_CC d_CC sR_CC reff_CC C6_CC S_CC rcut1_CC rcut2_CC

The :attr:`tersoff file` use the same atomic type list as the :attr:`ilp file` and just contains parameters 
of Tersoff potential. The potential file reads, specifically for graphene::

  A_CCC B_CCC lambda_CCC mu_CCC beta_CCC n_CCC c_CCC d_CCC h_CCC R_CCC S_CCC m_CCC alpha_CCC gamma_CCC

More parameter details are in :ref:`Tersoff (1988) potential <tersoff_1988>` and :ref:`NEP+ILP potential <nep_ilp>`.

