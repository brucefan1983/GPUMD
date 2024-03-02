.. index::
   single: Heat current

Heat current
============

Using the force expression, one can derive the following expression for the heat current for the whole system (:math:`E_i` is the total energy of atom :math:`i`) [Fan2015]_:

.. math::

   \boldsymbol{J} = \boldsymbol{J}^{\text{pot}} + \boldsymbol{J}^{\text{kin}} = \sum_{i} \boldsymbol{J}^{\text{pot}}_i + \sum_{i} \boldsymbol{J}^{\text{kin}}_i;

where

.. math::

   \boldsymbol{J}^{\text{kin}}_i = \boldsymbol{v}_i E_i;

and

.. math::
   \boldsymbol{J}^{\text{pot}}_i = -\frac{1}{2}\sum_{j \neq i} \boldsymbol{r}_{ij}
   \left(\frac{\partial U_i}{\partial \boldsymbol{r}_{ij}} \cdot \boldsymbol{v}_j
   -\frac{\partial U_j}{\partial \boldsymbol{r}_{ji}} \cdot \boldsymbol{v}_i\right).

The potential part of the per-atom heat current can also be written in the following equivalent forms [Fan2015]_:

.. math::
   \boldsymbol{J}^{\text{pot}}_i = -\sum_{j \neq i} \boldsymbol{r}_{ij}
   \left(\frac{\partial U_i}{\partial \boldsymbol{r}_{ij}} \cdot \boldsymbol{v}_j\right);

and

.. math::

   \boldsymbol{J}^{\text{pot}}_i = \sum_{j \neq i} \boldsymbol{r}_{ij}
   \left(\frac{ \partial U_j} {\partial \boldsymbol{r}_{ji}} \cdot \boldsymbol{v}_i\right).

Therefore, the per-atom heat current can also be expressed in terms of the per-atom virial [Gabourie2021]_:

.. math::

   \boldsymbol{J}^{\text{pot}}_i = \mathbf{W}_i  \cdot \boldsymbol{v}_i.

where the per-atom virial tensor cannot be assumed to be symmetric and the full tensor with 9 components should be used [Gabourie2021]_.
This result has actually been clear from the derivations in [Fan2015]_ but it was wrongly stated there that the potential part of the heat current *cannot* be expressed in terms of the per-atom virial.

One can also derive the following expression for the heat current from a subsystem :math:`A` to a subsystem :math:`B` [Fan2017]_:

.. math::
   
   Q_{A \rightarrow B} = -\sum_{i \in A} \sum_{j \in B}
   \left(\frac{\partial U_i}{\partial \boldsymbol{r}_{ij}} \cdot \boldsymbol{v}_j
   -\frac{\partial U_j}{\partial \boldsymbol{r}_{ji}} \cdot \boldsymbol{v}_i\right).
