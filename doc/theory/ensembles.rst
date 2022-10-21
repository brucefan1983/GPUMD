.. _ensembles:
.. index::
   single: Ensembles

Ensembles
=========

The aim of the time evolution in :term:`MD` simulations is to find the phase trajectory

.. math::
   \{ \boldsymbol{r}_i(t_1), ~\boldsymbol{v}_{i}(t_1)\}_{i=1}^N,~
   \{ \boldsymbol{r}_i(t_2), ~\boldsymbol{v}_{i}(t_2)\}_{i=1}^N,~
   \cdots

starting from the initial phase point

.. math::

   \{ \boldsymbol{r}_i(t_0), ~\boldsymbol{v}_{i}(t_0)\}_{i=1}^N.

The time interval between two time points :math:`\Delta t=t_1-t_0=t_2-t_1=\cdots` is called the time step.

The algorithm for integrating by one step depends on the ensemble type and other external conditions.
There are many ensembles used in MD simulations, but we only consider the following 3 in the current version:


.. index::
   single: NVE ensemble
   single: Microcanonical ensemble

NVE ensemble
------------

The NVE ensmeble is also called the micro-canonical ensemble.
We use the velocity-Verlet integration method with the following equations:

.. math::
   
   \boldsymbol{v}_i(t_{m+1}) \approx \boldsymbol{v}_i(t_{m}) +
   \frac{\boldsymbol{F}_i(t_m)+\boldsymbol{F}_i(t_{m+1})}{2m_i}\Delta t
   \\
   \boldsymbol{r}_i(t_{m+1}) \approx \boldsymbol{r}_i(t_{m}) +
   \boldsymbol{v}_i(t_m) \Delta t
   + \frac{1}{2} \frac{\boldsymbol{F}_i(t_m)}{m_i} (\Delta t)^2.

Here,
:math:`\boldsymbol{v}_i(t_{m})` is the velocity vector of particle :math:`i` at time :math:`t_{m}`.
:math:`\boldsymbol{r}_i(t_{m})` is the position vector of particle :math:`i` at time :math:`t_{m}`.
:math:`\boldsymbol{F}_i(t_{m})` is the force vector of particle :math:`i` at time :math:`t_{m}`.
:math:`m_i` is the mass of particle :math:`i` and
:math:`\Delta t` is the time step for integration.


.. index::
   single: NVT ensemble
   single: Canonical ensemble

NVT ensemble
------------

The NVT ensemble is also called the canonical ensemble.
We have implemented several thermostats for the NVT ensemble.


.. _berendsen_thermostat:
.. index::
   single: Berendsen thermostat

Berendsen thermostat
^^^^^^^^^^^^^^^^^^^^

The velocities are scaled in the Berendsen thermostat [Berendsen1984]_ as follows:

.. math::

   \boldsymbol{v}_i^{\text{scaled}}
   = \boldsymbol{v}_i
   \sqrt{1 + \frac{\Delta t}{\tau_T} \left(\frac{T_0}{T} - 1\right)}.

Here, :math:`\tau_T` is the coupling time (relaxation time) parameter, :math:`T_0` is the target temperature, and :math:`T` is the instant temperature calculated from the current velocities.
We require that :math:`\tau_T/\Delta t \geq 1`. 

When :math:`\tau_T/\Delta t = 1`, the above formula reduces to the simple velocity-scaling formula:

.. math::
   
   \boldsymbol{v}_i^{\text{scaled}} = \boldsymbol{v}_i \sqrt{\frac{T_0}{T}}.

A larger value of :math:`\tau_T/\Delta t` represents a weaker coupling between the system and the thermostat.
We recommend a value of :math:`\tau_T/\Delta t \approx 100`.
The above re-scaling is applied at each time step after the velocity-Verlet integration.
This thermostat is usually used for reaching equilibrium and is *not recommended* for sampling the canonical ensemble.


.. _nose_hoover_chain_thermostat:
.. index::
   single: Nose-Hoover chain thermostat

Nose-Hoover chain thermostat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Nose-Hoover chain (:term:`NHC`) method [Tuckerman2010]_, the equations of motion for the particles in the thermostatted region are (those for the thermostat variables are not presented):

.. math::
   \frac{d \boldsymbol{r}_i}{dt} = \frac{\boldsymbol{p}_i}{m_i} \\
   \frac{d \boldsymbol{p}_i}{dt} = \boldsymbol{F}_i - \frac{\pi_0}{Q_0} \boldsymbol{p}_i.

Here,
:math:`\boldsymbol{r}_i` is the position of atom :math:`i`.
:math:`\boldsymbol{p}_i` is the momentum of atom :math:`i`.
:math:`m_i` is the mass of atom  :math:`i`.
:math:`\boldsymbol{F}_i` is the total force on atom :math:`i` resulting from the potential model used.
:math:`Q_0=N_{\rm f} k_{\rm B} T_0 \tau_T^2` is the ''mass'' of the thermostat variable directly coupled to the system and :math:`\pi_0` is the corresponding ''momentum''. 
:math:`N_{\rm f}` is the degree of freedom in the thermostatted region. 
:math:`k_{\rm B}` is Boltzmann's constant and :math:`T_0` is the target temperature.
:math:`\tau_T` is a time parameter, and we suggest a value of :math:`\tau_T/\Delta t \approx 100`, where :math:`\Delta t` is the time step.

We use a fixed chain length of 4.


.. _langevin_thermostat:
.. index::
   single: Langevin thermostat

Langevin thermostat
^^^^^^^^^^^^^^^^^^^

In the Langevin method, the equations of motion for the particles in the thermostatted region are

.. math::

   \frac{d \boldsymbol{r}_i}{dt} = \frac{\boldsymbol{p}_i}{m_i} \\
   \frac{d \boldsymbol{p}_i}{dt} = \boldsymbol{F}_i - \frac{\boldsymbol{p}_i}{\tau_T} + \boldsymbol{f}_i,

Here,
:math:`\boldsymbol{r}_i` is the position of atom :math:`i`.
:math:`\boldsymbol{p}_i` is the momentum of atom :math:`i`.
:math:`m_i` is the mass of atom  :math:`i`.
:math:`\boldsymbol{F}_i` is the total force on atom :math:`i` resulted from the potential model used.
:math:`\boldsymbol{f}_i` is a random force with a variation determined by the fluctuation-dissipation relation to recover the canonical ensemble distribution with the target temperature.
:math:`\tau_T` is a time parameter, and we suggest a value of :math:`\tau_T/\Delta t \approx 100`, where :math:`\Delta t` is the time step.
We implemented the integrator proposed in [Bussi2007a]_.


.. _bdp_thermostat:
.. _svr_thermostat:
.. index::
   single: Bussi-Donadio-Parrinello thermostat

Bussi-Donadio-Parrinello thermostat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Berendsen thermostat does not generate a true NVT ensemble.
As an extension of the Berendsen thermostat, the Bussi-Donadio-Parrinello (:term:`BDP`) thermostat [Bussi2007b]_ incorporates a proper randomness into the velocity re-scaling factor and generates a true NVT ensemble.
It is also called the stochastic velocity rescaling (:term:`SVR`) thermostat.

In the :term:`BDP` thermostat, the velocities are scaled in the following way: 

.. math::
   
   \boldsymbol{v}_i^{\text{scaled}} = \alpha \boldsymbol{v}_i

where

.. math::
   
   \alpha^2=
   e^{-\Delta t/\tau_T} + 
   \frac{T_0}{TN_f} \left( 1-e^{-\Delta t/\tau_T} \right) \left( R_1^2 + \sum_{i=2}^{N_f}R_i^2 \right) +
   2e^{-\Delta t/2\tau_T} R_1 \sqrt{\frac{T_0}{TN_f} \left( 1-e^{-\Delta t/\tau_T} \right) }.

Here,
:math:`\boldsymbol{v}_i` is the velocity of atom :math:`i` before the re-scaling.
:math:`N_{\rm f}` is the degree of freedom in the thermostatted region. 
:math:`T` is instant temperature and :math:`T_0` is the target temperature.
:math:`\Delta t` is the time step for integration.
:math:`\tau_T` is a time parameter, and we suggest a value of :math:`\tau_T/\Delta t \approx 100`, where :math:`\Delta t` is the time step.
:math:`\{R_i\}_{i=1}^{N_f}` are :math:`N_{\rm f}` Gaussian distributed random numbers with zero mean and unit variance.


.. index::
   single: NPT ensemble
   single: Isothermal-isobaric ensemble

NPT ensemble
------------

The NPT ensemble is also called the isothermal-isobaric ensemble.


.. _berendsen_barostat:
.. index::
   single: Berendsen barostat

Berendsen barostat
^^^^^^^^^^^^^^^^^^

The Berendsen barostat is used with the Berendsen thermostat discussed above.
The barostat scales the box and positions as follows:

.. math::

   \left(
   \begin{array}{ccc}
   a_x^{\rm scaled} & b_x^{\rm scaled} & c_x^{\rm scaled} \\
   a_y^{\rm scaled} & b_y^{\rm scaled} & c_y^{\rm scaled} \\
   a_z^{\rm scaled} & b_z^{\rm scaled} & c_z^{\rm scaled} 
   \end{array}
   \right)
   =
   \left(
   \begin{array}{ccc}
   \mu_{xx} & \mu_{xy} & \mu_{xz} \\
   \mu_{yx} & \mu_{yy} & \mu_{yz} \\
   \mu_{zx} & \mu_{zy} & \mu_{zz} \\
   \end{array}
   \right)
   \left(
   \begin{array}{ccc}
   a_x & b_x & c_x \\
   a_y & b_y & c_y \\
   a_z & b_z & c_z 
   \end{array}
   \right)

where

.. math::

   \left(
   \begin{array}{c}
   x^{\rm scaled}_i \\
   y^{\rm scaled}_i \\
   z^{\rm scaled}_i
   \end{array}
   \right)
   =
   \left(
   \begin{array}{ccc}
   \mu_{xx} & \mu_{xy} & \mu_{xz} \\
   \mu_{yx} & \mu_{yy} & \mu_{yz} \\
   \mu_{zx} & \mu_{zy} & \mu_{zz} \\
   \end{array}
   \right)
   \left(
   \begin{array}{c}
   x_i \\
   y_i \\
   z_i
   \end{array}
   \right).

We consider the following three pressure-controlling conditions:

* *Condition 1*:
  The simulation box is *orthogonal* and only the hydrostatic pressure (trace of the pressure tensor) is controlled.
  The simulation box must be periodic in all three directions.
  The scaling matrix only has nonzero diagonal components and the diagonal components can be written as:

  .. math::

     \mu_{xx}=\mu_{yy}=\mu_{zz}= 1-\frac{\beta_{\rm hydro} \Delta t}{3 \tau_p} (p^{\rm target}_{\rm hydro} - p^{\rm instant}_{\rm hydro}).

* *Condition 2*:
  The simulation box is *orthogonal* and the three diagonal pressure components are controlled independently.
  The simulation box can be periodic or non-periodic in any of the three directions.
  Pressure is only controlled for periodic directions.
  The diagonal components of the scaling matrix can be written as:

  .. math::

     \mu_{xx}= 1-\frac{\beta_{xx} \Delta t}{3 \tau_p} (p^{\rm target}_{xx} - p^{\rm instant}_{xx}) \\
     \mu_{yy}= 1-\frac{\beta_{yy} \Delta t}{3 \tau_p} (p^{\rm target}_{yy} - p^{\rm instant}_{yy}) \\
     \mu_{zz}= 1-\frac{\beta_{zz} \Delta t}{3 \tau_p} (p^{\rm target}_{zz} - p^{\rm instant}_{zz}).

* *Condition 3*:
  The simulation box is *triclinic* and the 6 nonequivalent pressure components are controlled independently. The
  simulation box must be periodic in all three directions.
  The scaling matrix components are:

  .. math::

     \mu_{\alpha\beta}= 1-\frac{\beta_{\alpha\beta} \Delta t}{3 \tau_p} (p^{\rm target}_{\alpha\beta} - p^{\rm instant}_{\alpha\beta}).

The parameter :math:`\beta_{\alpha\beta}` is the isothermal compressibility, which is the inverse of the elastic modulus.
:math:`\Delta t` is the time step and :math:`\tau_p` is the pressure coupling time (relaxation time).


.. _stochastic_cell_rescaling:
.. index::
   single: Stochastic cell rescaling barostat

Stochastic cell rescaling barostat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Berendsen method does not generate a true NPT ensemble.
As an extension of the Berendsen method, the stochastic cell rescaling (:term:`SCR`) barostat, combined with the :term:`BDP` thermostat, incorporates a proper randomness into the box and position rescaling factor and generates a true NPT ensemble.

In the :term:`SCR` barostat, the scaling matrix is a sum of the scaling matrix as in the Berendsen barostat and a stochastic one.
The stochastic scaling matrix components are

.. math::

   \mu^{\rm stochastic}_{\alpha\beta} 
   = \sqrt{
   \frac{1}{D_{\rm couple}}}
   \sqrt{ 
   \frac{\beta_{\alpha\beta} \Delta t}{3\tau_p} 
   \frac{2k_{\rm B} T^{\rm target}}{V} 
   } R_{\alpha\beta}.

Here,
:math:`\beta_{\alpha\beta}`, :math:`\Delta t`, and :math:`\tau_p` have the same meanings as in the Berendsen barostat.
:math:`k_{\rm B}` is Boltzmann's constant.
:math:`T^{\rm target}` is the target temperature.
:math:`V` is the current volume of the system.
:math:`R_{\alpha\beta}` is a Gaussian random number with zero mean and unit variance.
:math:`D_{\rm couple}` is the number of directions that are coupled together.
It is 3, 1, and 1, respectively, for *condition 1*, *condition 2*, and *condition 3* as discussed above.
