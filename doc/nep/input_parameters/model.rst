.. _kw_model:
.. index::
   single: model (keyword in nep.in)

:attr:`model`
=============

This keyword selects the model used for training or prediction::

  model <name>

The names match the ``model`` field in ``nep.txt``. If no model-selection
keyword is given, the default is ``nep``.

.. list-table::
   :header-rows: 1

   * - Name
     - Model
   * - ``nep``
     - Standard energy NEP
   * - ``nep_dipole``
     - TNEP dipole
   * - ``nep_polarizability``
     - TNEP polarizability
   * - ``nep_temperature``
     - Temperature-dependent NEP
   * - ``nep_spin``
     - Spin-dependent energy NEP
   * - ``qnep1``
     - Charge NEP with real- and reciprocal-space electrostatics
   * - ``qnep2``
     - Charge NEP with reciprocal-space electrostatics
   * - ``qnep3``
     - qNEP mode 2 with charge self-energy correction
   * - ``nep_vdw``
     - NEP with learned long-range vdW
   * - ``qnep3_vdw``
     - qNEP mode 3 with learned long-range vdW

``qnep3`` uses the same parameter layout as ``qnep2``, but adds the
self-energy correction to the long-range charge energy.

Specify ``model`` at most once.

For ``model nep_spin``, :ref:`spin_version <kw_spin_version>` may still be
specified and defaults to ``1``. It is an error to combine ``spin_version``
with any other explicit ``model``.

Using ``spin_version`` without ``model`` also remains supported with a warning;
``spin_version`` itself remains the spin descriptor version setting.

See :ref:`flip_charge <kw_flip_charge>` for the restart option.

.. _nep_long_range_models:

Long-range models
-----------------

The models ``qnep1``, ``qnep2``, and ``qnep3`` [Fan2026]_ predict environment-dependent atomic charges in addition to the local NEP energy.
The ``qnep1`` and ``qnep2`` models follow the approaches of Song et al. [Song2024b]_ and Cheng [Cheng2025]_, respectively.
For periodic systems, their electrostatic contributions are

.. math::

   E_{\mathrm{el}}^{\mathrm{qnep1}} &= E_{\mathrm{real}}+E_{\mathrm{k}}+E_{\mathrm{self}},\\
   E_{\mathrm{el}}^{\mathrm{qnep2}} &= E_{\mathrm{k}},\\
   E_{\mathrm{el}}^{\mathrm{qnep3}} &= E_{\mathrm{k}}+E_{\mathrm{self}}.

Here, :math:`E_{\mathrm{real}}` and :math:`E_{\mathrm{k}}` are the real- and reciprocal-space Ewald terms.
The correction :math:`E_{\mathrm{self}}=-k_{\mathrm e}\alpha\sum_i q_i^2/\sqrt{\pi}` removes the charge self interaction, where :math:`k_{\mathrm e}` is the Coulomb prefactor and :math:`\alpha` is the Ewald splitting parameter.
In ``qnep2`` and ``qnep3``, the local NEP energy represents the remaining short-range interactions.

The ``nep_vdw`` model predicts environment-dependent dispersion coefficients and adds the long-range part of the attractive :math:`r^{-6}` interaction.
For periodic systems, this part is evaluated in reciprocal space with its self interaction removed; the local NEP energy describes the short-range interactions.
The ``qnep3_vdw`` model combines this dispersion contribution with the charge contribution used by ``qnep3``.

Training data
^^^^^^^^^^^^^

Use the energy, force, and optional virial or stress labels described in :ref:`train.xyz and test.xyz <train_test_xyz>`.
Atomic charge and dispersion-coefficient labels are not required.
For the charge models, ``charge=`` specifies the total charge of each structure in units of e and defaults to zero.
Optional ``bec:R:9`` labels provide target Born effective charges.
Fitting Born effective charges is usually appropriate for a single material in a single phase, since a constant high-frequency dielectric constant is assumed for the whole training dataset.
The weights of the total-charge and Born-effective-charge loss terms are set by :ref:`lambda_q <kw_lambda_q>` and :ref:`lambda_z <kw_lambda_z>`, respectively.

Boundary conditions and MD use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For training and prediction with ``nep``, all five long-range models support fully periodic (``pbc="T T T"``) and fully nonperiodic (``pbc="F F F"``) structures.
The default is fully periodic; mixed boundary conditions are not supported.
For nonperiodic structures, the long-range contribution is evaluated by direct summation without periodic images.

For MD with ``gpumd``, these models require periodic boundaries in all three directions and use a single GPU.
The charge models enforce zero total charge during MD.
Load the trained model in ``run.in`` using the :ref:`potential keyword <kw_potential>`::

  potential nep.txt

The model is selected from the contents of ``nep.txt``.
The :ref:`kspace keyword <kw_kspace>` selects PPPM (the default) or Ewald summation for the reciprocal-space charge and dispersion contributions.
