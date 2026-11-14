.. _tersoff_ilp:
.. index::
   single: Tersoff ILP

Hybrid Tersoff+ILP potential
============================

The hybrid Tersoff + :term:`ILP` potential in :program:`GPUMD` combines the Tersoff (1988) potential [Tersoff1988]_ for intralayer interactions and the interlayer potential (:term:`ILP`) [Ouyang2018]_ [Ouyang2020]_  for interlayer interactions to simulate homo- and heterostructures based on graphene and :math:`h`-BN layered materials.

Potential form
--------------

The ILP energy, taper function, and parameter definitions are given in :ref:`ILP potential form <ilp_potential_form>`.
The ILP contribution acts between atoms in different layers.

The intralayer contribution uses the :ref:`Tersoff (1988) potential form <tersoff_1988>`.
Only atoms in the same layer contribute to that term.

File format
-----------

This hybrid potential requires 2 potential files: :term:`ILP` potential file and
Tersoff potential file. We have adopted the :term:`ILP` file format that is similar
but not identical to that used by `lammps <https://docs.lammps.org/pair_ilp_graphene_hbn.html>`_.
Define a grouping method in ``model.xyz`` to identify the layers.

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
