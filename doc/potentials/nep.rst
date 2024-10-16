.. _nep_formalism:
.. index::
   single: Neuroevolution potential

Neuroevolution potential
************************

The neuroevolution potential (:term:`NEP`) approach was proposed in [Fan2021]_ (NEP1) and later improved in [Fan2022a]_ (NEP2) and [Fan2022b]_ (NEP3).
Currently, :program:`GPUMD` supports NEP3, NEP4 (to be published), and NEP5 (to be published).
All versions have comparable accuracy for single-component systems.
For multi-component systems, NEP4 and NEP5 usually have higher accuracy, if all the other hyperparameters are the same.

:program:`GPUMD` not only allows one to carry out simulations using :term:`NEP` models via the :ref:`gpumd executable <gpumd_executable>` but even the construction of such models via the :ref:`nep executable <nep_executable>`.

The neural network model
========================

:term:`NEP` uses a simple feedforward neural network (:term:`NN`) to represent the site energy of atom :math:`i` as a function of a descriptor vector with :math:`N_\mathrm{des}` components,

.. math::
   
   U_i(\mathbf{q}) = U_i \left(\{q^i_{\nu}\}_{\nu =1}^{N_\mathrm{des}}\right).

There is a single hidden layer with :math:`N_\mathrm{neu}` neurons in the NN and we have

.. math::
   
   U_i = \sum_{\mu=1}^{N_\mathrm{neu}}w^{(1)}_{\mu}\tanh\left(\sum_{\nu=1}^{N_\mathrm{des}} w^{(0)}_{\mu\nu} q^i_{\nu} - b^{(0)}_{\mu}\right) - b^{(1)},

where :math:`\tanh(x)` is the activation function in the hidden layer, :math:`\mathbf{w}^{(0)}` is the connection weight matrix from the input layer (descriptor vector) to the hidden layer, :math:`\mathbf{w}^{(1)}` is the connection weight vector from the hidden layer to the output node, which is the energy :math:`U_i`, :math:`\mathbf{b}^{(0)}` is the bias vector in the hidden layer, and :math:`b^{(1)}` is the bias for node :math:`U_i`.

The descriptor
==============

The descriptor for atom :math:`i` consists of a number of radial and angular components as described below. 

The **radial descriptor** components are defined as

.. math::

   q^i_{n} = \sum_{j\neq i} g_{n}(r_{ij})

with

.. math::
   
   0\leq n\leq n_\mathrm{max}^\mathrm{R},

where the summation runs over all the neighbors of atom :math:`i` within a certain cutoff distance.
There are thus :math:`n_\mathrm{max}^\mathrm{R}+1` radial descriptor components.

For the **angular descriptor** components, we consider 3-body to 5-body ones.
The formulation is similar but not identical to the atomic cluster expansion (:term:`ACE`) approach [Drautz2019]_.
For 3-body ones, we define (:math:`0\leq n\leq n_\mathrm{max}^\mathrm{A}`, :math:`1\leq l \leq l_\mathrm{max}^\mathrm{3b}`)

.. math::
   
   q^i_{nl} = \sum_{m=-l}^l (-1)^m A^i_{nlm} A^i_{nl(-m)},

where

.. math::

   A^i_{nlm} = \sum_{j\neq i} g_n(r_{ij}) Y_{lm}(\theta_{ij},\phi_{ij}),

and :math:`Y_{lm}(\theta_{ij},\phi_{ij})` are the spherical harmonics as a function of the polar angle :math:`\theta_{ij}` and the azimuthal angle :math:`\phi_{ij}`.
For expressions of the 4-body and 5-body descriptor components, we refer to [Fan2022b]_.

The radial functions :math:`g_n(r_{ij})` appear in both the radial and the angular descriptor components.
In the radial descriptor components,

.. math::
   
   g_n(r_{ij}) = \sum_{k=0}^{N_\mathrm{bas}^\mathrm{R}} c^{ij}_{nk} f_k(r_{ij}),

with 

.. math::
   
   f_k(r_{ij}) = \frac{1}{2}
   \left[T_k\left(2\left(r_{ij}/r_\mathrm{c}^\mathrm{R}-1\right)^2-1\right)+1\right]
   f_\mathrm{c}(r_{ij}),

and

.. math::
   
   f_\mathrm{c}(r_{ij}) 
   = \begin{cases}
   \frac{1}{2}\left[
   1 + \cos\left( \pi \frac{r_{ij}}{r_\mathrm{c}^\mathrm{R}} \right) 
   \right],& r_{ij}\leq r_\mathrm{c}^\mathrm{R}; \\
   0, & r_{ij} > r_\mathrm{c}^\mathrm{R}.
   \end{cases}

In the angular descriptor components, :math:`g_n(r_{ij})` have similar forms but with :math:`N_\mathrm{bas}^\mathrm{R}` changed to :math:`N_\mathrm{bas}^\mathrm{A}` and with :math:`r_\mathrm{c}^\mathrm{R}` changed to :math:`r_\mathrm{c}^\mathrm{A}`.

Model dimensions
================

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Number of ...
     - Count
   * - atom types
     - :math:`N_\mathrm{typ}`
   * - radial descriptor components
     - :math:`n_\mathrm{max}^\mathrm{R}+1`
   * - 3-body angular descriptor components
     - :math:`(n_\mathrm{max}^\mathrm{A}+1) l_\mathrm{max}^\mathrm{3b}`
   * - 4-body angular descriptor components
     - :math:`(n_\mathrm{max}^\mathrm{A}+1)` or zero (if not used)
   * - 5-body angular descriptor components
     - :math:`(n_\mathrm{max}^\mathrm{A}+1)` or zero (if not used)
   * - descriptor components
     - :math:`N_\mathrm{des}` is the sum of the above numbers of descriptor components
   * - trainable parameters :math:`c_{nk}^{ij}` in the descriptor
     - :math:`N_\mathrm{typ}^2 [(n_\mathrm{max}^\mathrm{R}+1)(N_\mathrm{bas}^\mathrm{R}+1)+(n_\mathrm{max}^\mathrm{A}+1)(N_\mathrm{bas}^\mathrm{A}+1)]`
   * - trainable :term:`NN` parameters
     - :math:`N_\mathrm{nn} = (N_\mathrm{des} +2) N_\mathrm{neu}+1` (NEP3)
   * -
     - :math:`N_\mathrm{nn} = (N_\mathrm{des} +2) N_\mathrm{neu} N_\mathrm{typ}+1` (NEP4)
   * -
     - :math:`N_\mathrm{nn} = ((N_\mathrm{des} +2) N_\mathrm{neu} + 1) N_\mathrm{typ}+1` (NEP5)

The total number of trainable parameters is the sum of the number of trainable descriptor parameters and the number of :term:`NN` parameters :math:`N_\mathrm{nn}`.


.. _nep_loss_function:
.. _nep_optimization_procedure:
.. index::
   single: NEP loss function   

Optimization procedure
======================

The name of the :term:`NEP` model is owed to the use of the separable natural evolution strategy (:term:`SNES`) that is used for the optimization of the parameters [Schaul2011]_.
The interested reader is referred to [Schaul2011]_ and [Fan2021]_ for details.

The key quantity in the optimization procedure is the loss (or objective) function, which is being minimized.
It is defined as a weighted sum over the loss terms associated with energies, forces and virials as well as the :math:`\mathcal{L}_1` and :math:`\mathcal{L}_2` norms of the parameter vector.

.. math::
   
   L(\boldsymbol{z}) 
   &= \lambda_\mathrm{e} \left( 
   \frac{1}{N_\mathrm{str}}\sum_{n=1}^{N_\mathrm{str}} \left( U^\mathrm{NEP}(n,\boldsymbol{z}) - U^\mathrm{tar}(n)\right)^2
   \right)^{1/2} \nonumber \\
   &+  \lambda_\mathrm{f} \left( 
   \frac{1}{3N}
   \sum_{i=1}^{N} \left( \boldsymbol{F}_i^\mathrm{NEP}(\boldsymbol{z}) - \boldsymbol{F}_i^\mathrm{tar}\right)^2
   \right)^{1/2} \nonumber \\
   &+  \lambda_\mathrm{v} \left( 
   \frac{1}{6N_\mathrm{str}}
   \sum_{n=1}^{N_\mathrm{str}} \sum_{\mu\nu} \left( W_{\mu\nu}^\mathrm{NEP}(n,\boldsymbol{z}) - W_{\mu\nu}^\mathrm{tar}(n)\right)^2
   \right)^{1/2} \nonumber \\
   &+  \lambda_1 \frac{1}{N_\mathrm{par}} \sum_{n=1}^{N_\mathrm{par}} |z_n| \nonumber \\
   &+  \lambda_2 \left(\frac{1}{N_\mathrm{par}} \sum_{n=1}^{N_\mathrm{par}} z_n^2\right)^{1/2}.

Here, :math:`N_\mathrm{str}` is the number of structures in the training data set (if using a full batch) or the number of structures in a mini-batch (see the :ref:`batch keyword <kw_batch>` in the :ref:`nep.in input file <nep_in>`) and :math:`N` is the total number of atoms in these structures.
:math:`U^\mathrm{NEP}(n,\boldsymbol{z})` and :math:`W_{\mu\nu}^\mathrm{NEP}(n,\boldsymbol{z})` are the per-atom energy and virial tensor predicted by the :term:`NEP` model with parameters :math:`\boldsymbol{z}` for the :math:`n^\mathrm{th}` structure, and :math:`\boldsymbol{F}_i^\mathrm{NEP}(\boldsymbol{z})` is the predicted force for the :math:`i^\mathrm{th}` atom.
:math:`U^\mathrm{tar}(n)`, :math:`W_{\mu\nu}^\mathrm{tar}(n)`, and :math:`\boldsymbol{F}_i^\mathrm{tar}` are the corresponding target values.
That is, the loss terms for energies, forces, and virials are defined as the respective :term:`RMSE` values between the :term:`NEP` predictions and the target values.
The last two terms represent :math:`\mathcal{L}_1` and :math:`\mathcal{L}_2` regularization terms of the parameter vector.
The weights :math:`\lambda_\mathrm{e}`, :math:`\lambda_\mathrm{f}`, :math:`\lambda_\mathrm{v}`, :math:`\lambda_1`, and :math:`\lambda_2` are tunable hyper-parameters (see the eponymous keywords in the :ref:`nep.in input file <nep_in>`).
When calculating the loss function, we use eV/atom for energies and virials and eV/Å for force components.
