.. _kw_deform:
.. index::
   single: deform (keyword in run.in)

:attr:`deform`
==============

This keyword is used to continuously deform the simulation box during a run.
It can be used for tensile, compressive, and shear deformation.

Syntax
------

The :attr:`deform` keyword supports two legacy forms and one general form.

**4-parameter legacy form** ::

  deform <A_per_step> <deform_x> <deform_y> <deform_z>

Here, :attr:`A_per_step` specifies the increment of the box length per simulation step, in units of Ångstrom/step.
For example, suppose the box length in a given direction at the beginning of a run is 100 Ångstrom and this parameter is :math:`10^{-5}` Ångstrom/step. A run with :math:`10^{6}` steps will then change the box length by 10%.
This gives a strain rate of :math:`10^{8}` s :math:`^{-1}` if the time step is 1 fs.
The parameters :attr:`deform_x`, :attr:`deform_y`, and :attr:`deform_z` can be 0 or 1, where 1 enables deformation in the corresponding direction and 0 disables it.

In this form, the same deformation rate is applied to all directions that are flagged as deformed.

**6-parameter legacy form** ::

  deform <A_per_step_x> <A_per_step_y> <A_per_step_z> <deform_x> <deform_y> <deform_z>

Here, :attr:`A_per_step_x`, :attr:`A_per_step_y`, and :attr:`A_per_step_z` specify the increments of the box lengths in the x, y, and z directions per simulation step, respectively, in units of Ångstrom/step.
The parameters :attr:`deform_x`, :attr:`deform_y`, and :attr:`deform_z` enable or disable deformation in each direction.

This form allows independent deformation rates in the x, y, and z directions.

**General form** ::

  deform <component_1> <A_per_step_1> [<component_2> <A_per_step_2> ...]

The available components are

.. code-block:: text

  xx yy zz xy xz yz

Each rate specifies the increment of the corresponding GPUMD box-matrix component per simulation step, in units of Ångstrom/step. Positive and negative values can be used.
The components :attr:`xy`, :attr:`xz`, and :attr:`yz` are geometric box/tilt components and are not symmetric Voigt shear-strain components.
Each component can only be specified once in a :attr:`deform` command.

Examples
--------

For a uniaxial tensile test, one can first equilibrate the system and then deform the box.
For example, using the legacy syntax::

  # equilibration stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  run 1000000

  # production stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  deform 0.00001 1 0 0
  run 1000000

The equivalent general syntax is::

  deform xx 0.00001

The general syntax can also be used for shear deformation. For example::

  ensemble nvt_nhc 300 300 100
  deform xy 0.00001
  run 1000000

Multiple box components can be deformed simultaneously. For example::

  deform xx 0.00001 yy -0.000005 xy 0.00002

Compatibility and caveats
-------------------------

* :attr:`deform` can be used with NVE and the standard NVT ensembles.
* With NPT-Berendsen or NPT-SCR, isotropic pressure control cannot be combined with :attr:`deform`.
* With 3 target pressure components, only diagonal :attr:`xx`, :attr:`yy`, and :attr:`zz` deformation is allowed, and the box must be orthogonal.
* With 6 target pressure components, the general form of :attr:`deform` can be used. Pressure control is disabled for the box components controlled by :attr:`deform`.
* The legacy forms cannot be combined with 6-component NPT pressure control and only support orthogonal boxes.
* :attr:`deform` is currently not supported with MTTK pressure-control ensembles or other special box-changing ensembles.
* Every direction involved in a deformation component must be periodic. For example, :attr:`xx` requires periodicity in x, while :attr:`xy` requires periodicity in both x and y.
* Automatic lattice reduction or box flipping is not performed for very large cumulative shear. The simulation box should therefore not be allowed to become excessively skewed.
