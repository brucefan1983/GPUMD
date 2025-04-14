.. _ILP_NEP:
.. index::
   single: ILP NEP

ILP+NEP potential
=================

The hybrid ILP+NEP potential in :program:`GPUMD` combines the NEP potential for intralyer
interactions and the ILP potential for interlayer interactions to simulate vdw materials,
graphene, :math:`h`-BN and TMDs.

Potential form
--------------

The site potential of ILP can be written as:

.. math::
   
   U_i^{\mathrm{ILP}}=  \mathrm{Tap}(r_{ij}) [U^{\mathrm{att}}(r_{ij})+U^{\mathrm{rep}}(r_{ij}, \boldsymbol{n}_i, \boldsymbol{n}_j)]

The function :math:`\mathrm{Tap}(r_{ij})` is a cutoff function and takes the following form in the intermediate region:

.. math::

   \mathrm{Tap}(r_{ij})=20{(\frac{r_{ij}}{R_{\mathrm{cut}}})}^7-70{(\frac{r_{ij}}{R_{\mathrm{cut}}})}^6+84{(\frac{r_{ij}}{R_{\mathrm{cut}}})}^5-35{(\frac{r_{ij}}{R_{\mathrm{cut}}})}^4+1

The repulsive term and the attractive term take the following forms:

.. math::

   U^{\mathrm{att}}(r_{ij})=-\frac{1}{1+e^{-d_{ij}[r_{ij}/(S_{\mathrm{R},ij}\cdot r_{ij}^{\mathrm{eff}})-1]}}\frac{C_{6,ij}}{r_{ij}^{6}}.
   
   U^{\mathrm{rep}}(r_{ij}, \boldsymbol{n}_i, \boldsymbol{n}_j)=e^{\alpha_{ij}(1-\frac{\gamma_{ij}}{\beta_{ij}})} \{\epsilon_{ij}+C_{ij}[e^{-{(\frac{\rho_{ij}}{\gamma_{ij}})}^2}+e^{-{(\frac{\rho_{ji}}{\gamma_{ij}})}^2}]\}.

where :math:`\boldsymbol n_i` and :math:`\boldsymbol n_j` are normal vectors of atom :math:`i` and :math:`j`,
:math:`\rho_{ij}` and :math:`\rho_{ji}` are the lateral interatomic distance, which can be expressed as:

.. math::

   \rho_{ij}^{2}&= r_{ij}^2-{(\boldsymbol r_{ij} \cdot \boldsymbol n_i)}^2\\
   \rho_{ji}^{2}&= r_{ij}^2-{(\boldsymbol r_{ij} \cdot \boldsymbol n_j)}^2

Other variables are all fitting parameters.

The site potential of NEP can be written as:

.. math:: 

   U_i^{\mathrm{NEP}} = \sum_{\mu=1}^{N_\mathrm{neu}}w^{(1)}_{\mu}\tanh\left(\sum_{\nu=1}^{N_\mathrm{des}} w^{(0)}_{\mu\nu} q^i_{\nu} - b^{(0)}_{\mu}\right) - b^{(1)},

More details of NEP potential are in :ref:`Neuroevolution potential <NEP>`. Note that in ILP+NEP potential, the NEP potential will
just calculate the intralayer interactions.

File format
-----------

This hybrid potential requires 3 kinds of files to set the interactions: one for ILP potential, 
one for NEP potential and the other for mapping NEP potential to groups in model file.
We have adopted the ILP file format that similar but not identical to that used by `lammps <https://docs.lammps.org/pair_ilp_graphene_hbn.html>`_.
The NEP potential is not required to modify, while to make the ILP and NEP potentials identify the layers, it's required to set some groups
in model file.

Here, we take 2 examples, bilayer graphene homostructure and
:math:`h`-BN / :math:`\mathrm{MoS}_2` heterostructure, to explain the potential settings.

For bilayer graphene, assume that the first line in your NEP potential file (C.nep) is::
  nep3 1 C 

and :attr:`group_method` 0 is to identify the different layers. Then your ILP potential file (C.ILP)
is required to set as::
  ilp_nep 1 C
  0 0
  beta_CC alpha_CC delta_CC epsilon_CC C_CC d_CC sR_CC reff_CC C6_CC S_CC rcut1_CC rcut2_CC

The first zero in the second line represents ILP potential uses :attr:`group_method` 0 to identify different
layers, which means you are required to set the same :attr:`group_id` for atoms in the same layer.
The second zero represents :attr:`group_method` to identify the sublayer for TMDs. For graphene and
:math:`h`-BN, just set it the same as the previous number. The last line is parameters.
:attr:`rcut1` is used for calculating the normal vectors while :attr:`rcut2` is the cutoff of ILP potential,
usually :math:`16\AA`.

Then, to ensure NEP potential get the messages of layers, the map.nep file required to set as::
  0 1 C.nep
  2
  0
  0

The first zero in the first line represents NEP potential uses :attr:`group_method` 0 to identify different
layers, which could be different from the :attr:`group_method` of ILP. This hybrid potential will
calculate the interlayer interactions for different groups in the :attr:`group_method` of ILP and
the intralayer interactions for the atoms at the same group in the :attr:`group_method` of NEP.
The next one represents there is just one NEP potential file. The number in the second
line represents the total number of layers in the :attr:`group_method` of NEP, here :attr:`group_method` 0.
The remaining lines are mapping each layer to NEP potential(s) set in the first line. 
Here, the last two lines represent the :attr:`group_id` 0 and 1 in group_method 0 will use
C.nep potential file (NEP 0).

The potential setting in run.in file requires ILP potential file and NEP mapping file::
  potential C.ilp map.nep

For bilayer :math:`h`-BN and :math:`\mathrm{MoS}_2`, 
assume that you have two NEP potential files, BN.nep and MoS.nep, and the first line in BN.nep is::
  nep4 2 B N

while and in MoS.nep is::
  nep4 2 Mo S

We also assume the :attr:`group_method` 0 is used to identify the different layers for ILP and NEP and 
:attr:`group_method` 1 is used to identify the different sublayers for ILP. In :attr:`group_method` 1, sublayers of
Mo and S should be set as the different :attr:`group_id`. Then your ILP potential file (BNMoS.ILP)
is required to set as::
  ilp_nep 4 B N Mo S
  0 1
  beta_BB alpha_BB delta_BB epsilon_BB C_BB d_BB sR_BB reff_BB C6_BB S_BB rcut1_BB rcut2_BB
  beta_BN alpha_BN delta_BN epsilon_BN C_BN d_BN sR_BN reff_BN C6_BN S_BN rcut1_BN rcut2_BN
  beta_BMo alpha_BMo delta_BMo epsilon_BMo C_BMo d_BMo sR_BMo reff_BMo C6_BMo S_BMo rcut1_BMo rcut2_BMo
  beta_BS alpha_BS delta_BS epsilon_BS C_BS d_BS sR_BS reff_BS C6_BS S_BS rcut1_BS rcut2_BS
  ...
  beta_SS alpha_SS delta_SS epsilon_SS C_SS d_SS sR_SS reff_SS C6_SS S_SS rcut1_SS rcut2_SS

Then, the map.nep file may set as::
  0 2 BN.nep MoS.nep
  2
  0
  1

which means :attr:`group_id` 0 of :attr:`group_method` 0 will use BN.nep potential file (NEP 0) 
and :attr:`group` 1 of :attr:`group_method` 0 use MoS.nep potential file (NEP 1).