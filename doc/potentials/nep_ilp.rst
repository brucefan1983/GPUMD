.. _nep_ilp:
.. index::
   single: NEP ILP

Hybrid NEP+ILP potential
========================

The hybrid :term:`NEP` + :term:`ILP` potential [Bu2025]_ in :program:`GPUMD` combines the neuroevolution potential 
(:term:`NEP`), [Fan2022b]_ (NEP3), and [Song2024]_ (NEP4), for intralyer
interactions and the interlayer potential (:term:`ILP`) [Ouyang2018]_ [Ouyang2020]_ 
for interlayer interactions to simulate van der Waals materials. 
Now this hybrid potential supports to simulate homo- and heterostructures based on 
graphene, :math:`h`-BN and transition metal dichalcogenides (TMDs) layered materials. 
The :term:`nep` potential here doesn't support :attr:`USE_TABLE` flag to accelerate now.

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

The site potential of :term:`NEP` can be written as:

.. math:: 

   U_i^{\mathrm{NEP}} = \sum_{\mu=1}^{N_\mathrm{neu}}w^{(1)}_{\mu}\tanh\left(\sum_{\nu=1}^{N_\mathrm{des}} w^{(0)}_{\mu\nu} q^i_{\nu} - b^{(0)}_{\mu}\right) - b^{(1)},

More details of :term:`NEP` potential are in :ref:`Neuroevolution potential <nep_formalism>`. Note that in hybrid :term:`NEP` + :term:`ILP` potential, the :term:`NEP` potential will
just compute the intralayer interactions.

File format
-----------

This hybrid potential requires 3 kinds of files: one for :term:`ILP` potential, 
one for :term:`NEP` potential and the other for mapping :term:`NEP` potential to groups in model file.
We have adopted the :term:`ILP` file format that similar but not identical to that used by `lammps <https://docs.lammps.org/pair_ilp_graphene_hbn.html>`_.
The :term:`NEP` potential file is not required to modify, while to make the :term:`ILP` and :term:`NEP` potentials identify the layers, it's required to set some groups
in ``model.xyz`` file.

In ``run.in`` file, the :attr:`potential` setting is as::
  
  potential <ilp file> <nep map file>

where :attr:`ilp file` and :attr:`nep map file` are the filenames of 
the :term:`ILP` potential file and :term:`NEP` mapping file.

:attr:`ilp file` is similar to other empirical potential files in :program:`GPUMD`.
But in addition, :term:`ILP` uses different :attr:`group_ids` to identify the different layers, so 
you need to add two :attr:`group_methods` in :attr:`ilp file`::

  nep_ilp <number of atom types> <list of elements>
  <group_method for layers> <group_method for sublayers>
  beta alpha delta epsilon C d sR reff C6 S rcut1 rcut2
  ...

* :attr:`nep_ilp` is the name of this hybrid potential.
* :attr:`number of atom types` is the number of atom types defined in the ``model.xyz``.
* :attr:`list of element` is a list of all the elements in the potential (can be in any order).
* :attr:`group_method for layers` is the :attr:`group_method` set in ``model.xyz`` 
  to identify different layers. For example, monolayer graphene and monolayer 
  :math:`\mathrm{MoS}_2` are both single layer so for the atoms in each layer 
  the :attr:`group_id` of :attr:`group_method for layers` are the same.
* :attr:`group_method for sublayers` is used to identify the different sublayers.
  For example, monolayer graphene contains one sublayer while monolayer :math:`\mathrm{MoS}_2` 
  contains three sublayers, one Mo sublayer and two S sublayers. For the atoms in each sublayer 
  the :attr:`group_id` of :attr:`group_method for sublayers` are the same.
* The last line(s) is(are) parameters of :term:`ILP`. :attr:`rcut1` is used for calculating the normal vectors 
  and :attr:`rcut2` is the cutoff of :term:`ILP`, usually 16Å.

:attr:`nep_map_file` can map one or more :term:`NEP` potential files to
different layers. The setting is as::

  <group_method for layers> <number of NEP files> <list of NEP files>
  <number of groups>
  <NEP_id for group_0>
  <NEP_id for group_1>
  ...

* :attr:`group_method for layers` is the same as the setting in :attr:`ilp file`.
* :attr:`number of NEP files` is the number of :term:`NEP` files used in your 
  simulation.
* :attr:`list of NEP files` is a list of all the :term:`NEP` filenames. Note 
  that the first file will be identified as :attr:`NEP_0` and then :attr:`NEP_1` and so on.
* :attr:`number of groups` is the number of groups in :attr:`group_method for layers`.
* The last :attr:`number of groups` lines map the :term:`NEP` to each group.
  If :attr:`NEP_id for group_0` is set to 0, the intralayer interactions between 
  atoms within :attr:`group_id` 0 are computed by the first :term:`NEP` file (:attr:`NEP_0`)
  in :attr:`list of NEP files`. If set to 1, then computed by the second :term:`NEP` file (:attr:`NEP_1`) and so on.



Examples
--------

Example 1: bilayer graphene
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assume your have three files: :term:`ILP` potential file (``C.ilp``), 
:term:`NEP` potential file (``C.nep``) and :term:`NEP` mapping file (``map.nep``). The potential 
setting in ``run.in`` file is as::
  
  potential C.ilp map.nep

Assume that the first line in ``C.nep`` is::
  
  nep3 1 C 

and :attr:`group_method` 0 is used to identify the different layers. Then ``C.ilp``
is required to set as::
  
  nep_ilp 1 C
  0 0
  beta_CC alpha_CC delta_CC epsilon_CC C_CC d_CC sR_CC reff_CC C6_CC S_CC rcut1_CC rcut2_CC

The first **0** in the second line represents :term:`ILP` potential uses :attr:`group_method` 0 to identify different
layers. The second **0** represents :attr:`group_method` 0 is used to identify the sublayers. For 
the system with only graphene and :math:`h`-BN, just set it the same as the previous number.


Then, ``map.nep`` file required to set as::
  
  0 1 C.nep
  2
  0
  0

The first **0** in the first line represents :term:`NEP` potential uses :attr:`group_method` 0 to identify different
layers. 
The next **1** represents there is just one :term:`NEP` potential file. The number in the second
line represents there are two groups in the :attr:`group_method` 0.
The last two lines represent the :attr:`group_0`  and :attr:`group_1` in :attr:`group_method` 0 will use
``C.nep`` potential file (:attr:`NEP_0`).



Example 2: bilayer :math:`h`-BN / :math:`\mathrm{MoS}_2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assume your have four files: :term:`ILP` potential file (``BNMoS.ilp``), 
:term:`NEP` potential files (``BN.nep``, ``MoS.nep``) and :term:`NEP` mapping file (``map.nep``). 
The potential setting in ``run.in`` file is as::
  
  potential BNMoS.ilp map.nep

Assume the first line in ``BN.nep`` is::
  
  nep4 2 B N

and in ``MoS.nep`` is::

  nep4 2 Mo S

We also assume the :attr:`group_method` 0 is used to identify the different layers and 
:attr:`group_method` 1 is used to identify the different sublayers for :term:`ILP`. In :attr:`group_method` 1, 
atoms in the sublayers of Mo and S should be set as the different
:attr:`group_id`. Then ``BNMoS.ilp`` is required to set as::
  
  nep_ilp 4 B N Mo S
  0 1
  beta_BB alpha_BB delta_BB epsilon_BB C_BB d_BB sR_BB reff_BB C6_BB S_BB rcut1_BB rcut2_BB
  beta_BN alpha_BN delta_BN epsilon_BN C_BN d_BN sR_BN reff_BN C6_BN S_BN rcut1_BN rcut2_BN
  beta_BMo alpha_BMo delta_BMo epsilon_BMo C_BMo d_BMo sR_BMo reff_BMo C6_BMo S_BMo rcut1_BMo rcut2_BMo
  beta_BS alpha_BS delta_BS epsilon_BS C_BS d_BS sR_BS reff_BS C6_BS S_BS rcut1_BS rcut2_BS
  ...
  beta_SS alpha_SS delta_SS epsilon_SS C_SS d_SS sR_SS reff_SS C6_SS S_SS rcut1_SS rcut2_SS

Assume :attr:`group_id` of :math:`\mathrm{MoS}_2` is 0 and of :math:`h`-BN is 1.
Then ``map.nep`` file is set as::
  
  0 2 BN.nep MoS.nep
  2
  1
  0

The **1** in the third line means :attr:`group_0` (:math:`\mathrm{MoS}_2`) uses ``MoS.nep`` potential file (:attr:`NEP_1`) 
and the last **0** means :attr:`group_1` (:math:`h`-BN) uses ``BN.nep`` potential file (:attr:`NEP_0`).