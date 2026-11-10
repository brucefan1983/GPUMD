Forces and stresses
===================

In classical molecular dynamics, the total potential energy :math:`U` of a system can be written as the sum of site potentials :math:`U_i`:

.. math::
   
   U=\sum_{i=1}^N U_i(\{\boldsymbol{r}_{ij}\}_{j\neq i}).

Here, :math:`\boldsymbol{r}_{ij} \equiv \boldsymbol{r}_j - \boldsymbol{r}_i` is the position difference vector used throughout this manual.


.. index::
   single: Forces

Interatomic forces
------------------

A well-defined force expression for general many-body potentials that explicitly respects (the weak version of) Newton's third law has been derived in [Fan2015]_:

.. math::
   
   \boldsymbol{F}_{i} = \sum_{j \neq i} \boldsymbol{F}_{ij}

where

.. math::
   \boldsymbol{F}_{ij} = - \boldsymbol{F}_{ji} =
   \frac{\partial U_{i}}{\partial \boldsymbol{r}_{ij}} -
   \frac{\partial U_{j}}{\partial \boldsymbol{r}_{ji}} =
   \frac{\partial \left(U_{i} + U_{j}\right) }{\partial \boldsymbol{r}_{ij}}.

Here, :math:`\partial U_{i}/\partial \boldsymbol{r}_{ij}` is a shorthand notation for a vector with Cartesian components :math:`\partial U_{i}/\partial x_{ij}`, :math:`\partial U_{i}/\partial y_{ij}`, and :math:`\partial U_{i}/\partial z_{ij}`.

Starting from the above force expression, one can derive expressions for the stress tensor and the heat current.

.. index::
   single: Stress tensor

Stress tensor
-------------

The pressure tensor is an important quantity in MD simulations.
It consists of two parts: a virial part which is related to the force and an ideal gas part which is related to the temperature.
The virial part must be calculated along with force evaluation.

The validity of Newton's third law is crucial to derive the following expression of the virial tensor [Fan2015]_:

.. math::
   
   \mathbf{W} = \sum_{i} \widetilde{\mathbf{W}}_i

where

.. math::
   
   \widetilde{\mathbf{W}}_i = -\frac{1}{2} \sum_{j \neq i} \boldsymbol{r}_{ij} \otimes \boldsymbol{F}_{ij},

where only relative positions :math:`\boldsymbol{r}_{ij}` are involved. 

By exchanging the summation indices, the same total virial can be written using the per-atom partition adopted in GPUMD [Gabourie2021]_

.. math::
   
   \mathbf{W} = \sum_{i} \mathbf{W}_i

where

.. math::

   \mathbf{W}_i = \sum_{j \neq i} \boldsymbol{r}_{ij} \otimes \frac{\partial U_j}{\partial \boldsymbol{r}_{ji}}.

The two partitions give the same total virial, but in general :math:`\widetilde{\mathbf{W}}_i \neq \mathbf{W}_i` for an individual atom.
The latter definition is used to calculate the per-atom heat current.

The ideal gas part of the pressure is:

.. math::
   
   \frac{\sum_{i} m_i v_i^{\alpha} v_i^{\beta} }{V},

where :math:`m_i` is the mass of particle :math:`i`, :math:`v_i^{\alpha}` is the velocity of particle :math:`i`, and :math:`V` is the volume of the system.

The total pressure tensor, as reported in :ref:`thermo.out <thermo_out>` for classical MD, is thus

.. math::
   
   P^{\alpha\beta} = \frac{W^{\alpha\beta} + \sum_{i} m_i v_i^{\alpha} v_i^{\beta}}{V}.

The diagonal pressure components are positive under compression.
The tensile-positive Cauchy stress has the opposite sign: :math:`\sigma^{\alpha\beta}=-P^{\alpha\beta}`.
