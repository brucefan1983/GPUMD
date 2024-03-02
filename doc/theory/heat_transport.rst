.. index::
   single: Heat transport
   
Heat transport
==============

.. _green_kubo_method:
.. _heat_current_autocorrelation:
.. _running_thermal_conductivity:
.. index::
   single: EMD method
   single: Green-Kubo method
   single: heat current autocorrelation

EMD method
----------

A popular approach for computing the lattice thermal conductivity is to use equilibrium molecular dynamics (:term:`EMD`) simulations and the Green-Kubo (:term:`GK`) formula.
In this method, the running thermal conductivity (:term:`RTC`) along the :math:`x`-direction (similar expressions apply to other directions) can be expressed as an integral of the heat current autocorrelation (:term:`HAC`) function:

.. math::
   
   \kappa_{xx}(t) = \frac{1}{k_BT^2V} \int_0^{t} dt' \text{HAC}_{xx}(t').

Here, :math:`k_{\rm B}` is Boltzmann's constant, :math:`V` is the volume of the simulated system, :math:`T` is the absolute temperature, and :math:`t` is the correlation time.
The :term:`HAC` is

.. math::
   
   \text{HAC}_{xx}(t)=\langle J_{x}(0)J_{x}(t)\rangle,

where :math:`J_{x}(0)` and :math:`J_{x}(t)` are the total heat current of the system at two time points separated by an interval of :math:`t`.
The symbol :math:`\langle \rangle` means that the quantity inside will be averaged over different time origins.

* Related keyword in the :ref:`run.in file <run_in>`: :ref:`compute_hac <kw_compute_hac>`
* Related output file: :ref:`hac.out <hac_out>`
* Related tutorial: :ref:`Thermal transport from EMD <tutorials>`

We only used the potential part of the heat current.
If you are studying fluids, you need to output the heat currents (potential and kinetic part) using the :ref:`compute <kw_compute>` keyword and calculated the :term:`HAC` by yourself.

We have decomposed the potential part of the heat current into in-plane and out-of-plane components [Fan2017]_.
If you do not need this decomposition, you can simply sum up some components in the :ref:`hac.out <hac_out>` file.


.. _nemd:
.. index::
   single: NEMD method
   single: Non-equilibrium molecular dynamics

NEMD method
-----------
 
Non-equilibrium molecular dynamics (:term:`NEMD`) can be used to study thermal transport.
In this method, two local thermostats at different temperatures are used to generate a non-equilibrium steady state with a constant heat flux. 

If the temperature difference between the two thermostats is :math:`\Delta T` and the heat flux is :math:`Q/S`, the thermal conductance :math:`G` between the two thermostats can be calculated as

.. math::
   
   G = \frac{Q/S}{\Delta T}.

Here, :math:`Q` is the energy transfer rate between the thermostat and the thermostated region and :math:`S` is the cross-sectional area perpendicular to the transport direction.

We can also calculate an effective thermal conductivity (also called the apparent thermal conductivity) :math:`\kappa(L)` for the finite system:

.. math::
   
   \kappa(L) = GL = \frac{Q/S}{\Delta T/L}.

where :math:`L` is the length between the heat source and the heat sink.
This is to say that the temperature gradient should be calculated as :math:`\Delta T/L`, rather than that extracted from the linear part of the temperature profile away from the local thermostats.
This is an important conclusion in [Li2019]_.

To generate the non-equilibrium steady state, one can use a pair of local thermostats.
Based on [Li2019]_, the Langevin thermostatting method is recommended.
Therefore, the :ref:`ensemble <kw_ensemble>` keyword with the first parameter of :attr:`heat_lan` should be used to generate the heat current.

* The :ref:`compute <kw_compute>` keyword should be used to compute the temperature profile and the heat transfer rate :math:`Q`.
* Related output file: :ref:`compute.out <compute_out>`
* Related tutorial: :ref:`Thermal transport from NEMD and HNEMD <tutorials>`


.. _hnemd:
.. index::
   single: HNEMD method
   single: Homogeneous non-equilibrium molecular dynamics
   
HNEMD method
------------

The homogeneous non-equilibrium molecular dynamics (:term:`HNEMD`) method for heat transport by Evans has been generalized to general many-body potentials [Fan2019]_.
This method is physically equivalent to the :term:`EMD` method but can be computationally faster. 

In this method, an external force of the form [Fan2019]_

.. math::
   
   \boldsymbol{F}_{i}^{\rm ext}
   = E_i \boldsymbol{F}_{\rm e} + \sum_{j \neq i} \left(\frac{\partial U_j}{\partial \boldsymbol{r}_{ji}} \otimes \boldsymbol{r}_{ij}\right) \cdot \boldsymbol{F}_{\rm e}

is added to each atom :math:`i`, driving the system out of equilibrium. According to [Gabourie2021]_, it can also be written as

.. math::
   
   \boldsymbol{F}_{i}^{\rm ext}
   = E_i \boldsymbol{F}_{\rm e} + \boldsymbol{F}_{\rm e} \cdot \mathbf{W}_i

Here, 
:math:`E_i` is the total energy of particle :math:`i`.
:math:`U_i` is the potential energy of particle :math:`i`.
:math:`\mathbf{W}_i` is the per-atom virial.
:math:`\boldsymbol{r}_{ij}\equiv\boldsymbol{r}_{j}-\boldsymbol{r}_{i}`, and :math:`\boldsymbol{r}_i` is the position of particle :math:`i`.

The parameter :math:`\boldsymbol{F}_{\rm e}` is of the dimension of inverse length and should be small enough to keep the system within the linear response regime. 
The driving force will induce a non-equilibrium heat current :math:`\langle \boldsymbol{J} \rangle_{\rm ne}` linearly related to :math:`\boldsymbol{F}_{\rm e}`:

.. math::
   
   \frac{\langle J^{\mu}(t)\rangle_{\rm ne}}{TV} = \sum_{\nu} \kappa^{\mu\nu}  F^{\nu}_{\rm e},

where :math:`\kappa^{\mu\nu}` is the thermal conductivity tensor, :math:`T` is the system temperature, and :math:`V` is the system volume. 

A global thermostat should be applied to control the temperature of the system.
For this, we recommend using the Nose-Hoover chain thermostat.
So one should use the :ref:`ensemble <kw_ensemble>` keyword with the first parameter of :attr:`nvt_nhc`.

* The :ref:`compute_hnemd <kw_compute_hnemd>` keyword should be used to add the driving force and calculate the thermal conductivity.
* The computed results are saved to the :ref:`kappa.out <kappa_out>` file.
* Related tutorial: :ref:`Thermal transport from NEMD and HNEMD <tutorials>`


.. index::
   single: Spectral heat current

Spectral heat current
---------------------

In the framework of the :term:`NEMD` and :term:`HNEMD` methods, one can also calculate spectrally decomposed thermal conductivity (or conductance).
In this method, one first calculates the following virial-velocity correlation function [Gabourie2021]_:

.. math::
   
   \boldsymbol{K}(t) = \sum_{i} 
   \left\langle
   \mathbf{W}_i(0) \cdot \boldsymbol{v}_i (t)
   \right\rangle,

which reduces to the non-equilibrium heat current when :math:`t=0`. 

Then one can define the following Fourier transform pairs [Fan2017]_:

.. math::

   \tilde{\boldsymbol{K}}(\omega) = \int_{-\infty}^{\infty} dt e^{i\omega t} K(t)

where

.. math::
   
   \boldsymbol{K}(t) = \int_{-\infty}^{\infty} \frac{d\omega}{2\pi} e^{-i\omega t}
   \tilde{\boldsymbol{K}}(\omega)

By setting :math:`t=0` in the equation above, we can get the following spectral decomposition of the non-equilibrium heat current:

.. math::
   
   \boldsymbol{J} = \int_{0}^{\infty} \frac{d\omega}{2\pi}
   \left[2\tilde{\boldsymbol{K}}(\omega)\right].

From the spectral decomposition of the non-equilibrium heat current, one can deduce the spectrally decomposed thermal conductance in the :term:`NEMD` method:

.. math::
   
   G(\omega) = \frac{2\tilde{\boldsymbol{K}}(\omega)}{V\Delta T}

with

.. math::
   
   G = \int_{0}^{\infty} \frac{d\omega}{2\pi} G(\omega).

where :math:`\Delta T` is the temperature difference between the two thermostats and :math:`V` is the volume of the considered system or subsystem.

One can also calculate the spectrally decomposed thermal conductivity in the :term:`HNEMD` method:

.. math::
   
   \kappa(\omega) = \frac{2\tilde{\boldsymbol{K}}(\omega)}{VTF_{\rm e}}

with

.. math::
   
   \kappa = \int_{0}^{\infty} \frac{d\omega}{2\pi} \kappa(\omega).

where :math:`F_{\rm e}` is the magnitude of the driving force parameter in the :term:`HNEMD` method.

This calculation is invoked by the :ref:`compute_shc <kw_compute_shc>` keyword and the results are saved to the :ref:`shc.out <shc_out>` file.

* Related tutorial: :ref:`Thermal transport from NEMD and HNEMD <tutorials>`


.. index::
   single: Modal analysis methods

Modal analysis methods
----------------------

A system with :math:`N` atoms will have :math:`3N` vibrational modes.
Using lattice dynamics, the vibrational modes (or eigenmodes) of the system can be found.
The heat flux can be decomposed into contributions from each vibrational mode and the thermal conductivity can be written in terms of those contributions [Lv2016]_.
To calculate the modal heat current in GPUMD, the velocities must first be decomposed into their modal contributions:

.. math::

   \boldsymbol{v}_i (t) = \frac{1}{\sqrt{m_i}} \sum_{n}  \boldsymbol{e}_{i,n} \cdot \boldsymbol{\dot{X}}_n(t)  

Here,
:math:`\boldsymbol{\dot{X}}_n` is the normal mode coordinates of the velocity of mode :math:`n`
:math:`m_i` is the mass of atom :math:`i`
:math:`\boldsymbol{e}_{i,n}` is the eigenvector that gives the magnitude and direction of mode :math:`n` for atom :math:`i`
:math:`\boldsymbol{v}_i` is the velocity of atom :math:`i`

The heat current can be rewritten in terms of the modal velocity to be:

.. math::
   
   \boldsymbol{J}^{\text{pot}} = \sum_{i} \mathbf{W}_i  \cdot \left[ \frac{1}{\sqrt{m_i}} \sum_{n}  \boldsymbol{e}_{i,n} \cdot \boldsymbol{\dot{X}}_n(t) \right]
   = \sum_{n} \left(\sum_{i} \frac{1}{\sqrt{m_i}} \mathbf{W}_i  \cdot \boldsymbol{e}_{i,n} \right) \cdot \boldsymbol{\dot{X}}_n(t)

This means that the modal heat current can be written as:

.. math::
   
   \boldsymbol{J}^{\text{pot}}_n = \left(\sum_{i} \frac{1}{\sqrt{m_i}} \mathbf{W}_i  \cdot \boldsymbol{e}_{i,n} \right) \cdot \boldsymbol{\dot{X}}_n(t)

This modal heat current can be used to extend the capabilities of the :term:`EMD` and :term:`HNEMD` methods.
The extended methods are called Green-Kubo modal analysis (:term:`GKMA`) [Lv2016]_ and homogeneous non-equilibrium modal analysis (:term:`HNEMA`) [Gabourie2021]_.


.. _green_kubo_modal_analysis:
.. index::
   single: GKMA
   single: Green-Kubo modal analysis

Green-Kubo modal analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

The Green-Kubo Modal Analysis (:term:`GKMA`) calculates the modal contributions to thermal conductivity by using [Lv2016]_ [Gabourie2021]_:

.. math::
   
   \kappa_{xx,n}(t) = \frac{1}{k_BT^2V} \int_0^{t} dt' \langle J_{x,n}(t')J_{x}(0)\rangle.

Here, :math:`k_{\rm B}` is Boltzmann's constant, :math:`V` is the volume of the simulated system, :math:`T` is the absolute temperature, and :math:`t` is the correlation time. :math:`J_{x}(0)` is the total heat current and and:math:`J_{x,n}(t')` is the mode-specific heat current of the system at two time points separated by an interval of :math:`t'`.
The symbol :math:`\langle \rangle` means that the quantity inside will be averaged over different time origins.

* Related input file: :ref:`eigenvector.in <eigenvector_in>`
* Related keyword in the :ref:`run.in file <run_in>`: :ref:`compute_gkma <kw_compute_gkma>`
* Related output file: :ref:`heatmode.out <heatmode_out>`

For the :term:`GKMA` method, we only used the potential part of the heat current.


.. _hnema:
.. index::
   single: HNEMA
   single: Homogeneous non-equilibrium modal analysis

Homogeneous non-equilibrium modal analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The homogeneous non-equilibrium modal analysis (:term:`HNEMA`) method calculates the modal contributions of thermal conductivity using [Gabourie2021]_:

.. math::
   
   \frac{\langle J_n^{\mu}(t)\rangle_{\rm ne}}{TV} = \sum_{\nu} \kappa_n^{\mu\nu}  F^{\nu}_{\rm e},

Here, :math:`\kappa_n^{\mu\nu}` is the thermal conductivity tensor of mode :math:`n`, :math:`T` is the system temperature, and :math:`V` is the system volume.
The mode-specific non-equilibrium heat current is :math:`\langle J_n^{\mu}(t)\rangle_{\rm ne}` and the driving force parameter is :math:`\boldsymbol{F}_{\rm e}`.

* Related input file: :ref:`eigenvector.in <eigenvector_in>`
* Related keyword in the run.in file: :ref:`compute_hnema <kw_compute_hnema>`
* Related output file: :ref:`kappamode.out <kappamode_out>`

For the :term:`HNEMA` method, we only used the potential part of the heat current.
A global thermostat should be applied to control the temperature of the system.
For this, we recommend using the Nose-Hoover chain thermostat. So one should use the :ref:`ensemble <kw_ensemble>` keyword with the first parameter of :attr:`nvt_nhc`.


.. _hnemdec:
.. index::
   single: HNEMDEC
   single: Homogeneous non-equilibrium molecular dynamics Evans-Cummings algorithm

HNEMDEC method
----------------------

A system with :math:`M` components has :math:`M` independent fluxes: the heat flux and any :math:`M-1` momentum fluxes of the :math:`M` component.
The central idea of Evans-Cummings algorithm is designing such a driving force that produce a dissipative flux that is equivalent to heat flux or momentum flux.
By measuring the heat current and momentum current, we obtain onsager coefficents that can be used to derive the thermal conductivity. 

In the case of heat flux :math:`\boldsymbol{J}_{q}` as dissipative flux, for the :math:`i` atom belonging to :math:`\alpha` component:

.. math::

   \boldsymbol{F}_{i}^{\alpha,\rm ext}
   =( \mathbf{S}_{i}^{\alpha}
   -\frac{m_{\alpha}}{M}\mathbf{S}
   +k_BT\frac{M_{tot}-Nm_{\alpha}}{M_{tot}N}\mathbf{I})\cdot \boldsymbol{F}_{\rm e} 

where :math:`\mathbf{S}_{i}^{\alpha}=E_{i}^{\alpha}\mathbf{I}+\mathbf{W}_{i}^{\alpha}`, :math:`\mathbf{S}=\sum_{\beta=1}^{M}\sum_{i=1}^{N_{\beta}}\mathbf{S}_{i}^{\beta}`, :math:`M_{tot}` is the total mass of the system, :math:`N` is the atom number of the system. Any physical quantity :math:`A(t)` is related to driving force by correlation funtion:

.. math::

   \langle \boldsymbol{A}(t) \rangle
   =\langle A(0) \rangle + (\int_{0}^{t}dt'\frac{\langle \boldsymbol{A}(t') \otimes \boldsymbol{J}_{q}(0) \rangle}{k_BT})\cdot \boldsymbol{F}_{\rm e}

In the case of momentum flux :math:`\boldsymbol{J}_{\gamma}` of :math:`\gamma` component as dissipative flux:

.. math::

   \boldsymbol{F}_{i}^{\alpha,\rm ext}=c_{\alpha}\boldsymbol{F}_{\rm e}

where :math:`c_{\gamma}=\frac{N}{N_{\gamma}}`, and :math:`c_{\beta}=-\frac{Nm_{\beta}}{M'}`, :math:`M'=\sum_{\epsilon=1,\epsilon\neq\gamma}^{M}N_{\epsilon}m_{\epsilon}`.
Similar to the former case,

.. math::

   \langle \boldsymbol{A}(t) \rangle
   =\langle A(0) \rangle 
   + (\frac{N}{M'}+\frac{N}{N_{\gamma}m_{\gamma}})(\int_{0}^{t}dt'\frac{\langle \boldsymbol{A}(t') \otimes \boldsymbol{J}_{\gamma}(0) \rangle}{k_BT})\cdot \boldsymbol{F}_{\rm e}

Then we can obtain any matrix element :math:`\Lambda_{ij}` of onsager matrix by:

.. math::

   \Lambda_{ij}
   =\frac{1}{k_BV}\int_{0}^{t}dt'\langle \boldsymbol{J}_{i}(t') \otimes \boldsymbol{J}_{j}(0) \rangle


The onsager matrix is arranged as: 

.. math::

   \begin{array}{cccc}
   \Lambda_{qq}
   & \Lambda_{q1}
   & \cdots
   & \Lambda_{q(M-1)}
   \\
   \Lambda_{1q}
   & \Lambda_{11}
   & \cdots
   & \Lambda_{1(M-1)}
   \\
   \vdots
   & \vdots
   & \ddots
   & \vdots
   \\
   \Lambda_{(M-1)q}
   & \Lambda_{(M-1)1}
   & \cdots
   & \Lambda_{(M-1)(M-1)}
   \end{array}

The thermal conductivity could be derived from onsager matrix: 

.. math::

   \kappa=\frac{1}{T^{2}(\Lambda^{-1})_{00}}

* The :ref:`compute_hnemdec <kw_compute_hnemdec>` keyword should be used to add the driving force and calculate the thermal conductivity.
* The computed results are saved to the :ref:`onsager.out <onsager_out>` file.

