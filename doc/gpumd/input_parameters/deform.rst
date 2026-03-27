.. _kw_deform:
.. index::
   single: deform (keyword in run.in)

:attr:`deform`
==============

This keyword is used to deform the simulation box, which can be used to do tensile tests.

Syntax
------

The :attr:`deform` keyword supports two forms.

**4-parameter form** ::

  deform <A_per_step> <deform_x> <deform_y> <deform_z>

Here, :attr:`A_per_step` specifies the speed of the increase of the box length, which is in units of Ångstrom/step.
For example, suppose the box length (in a given direction) in the beginning of a run is 100 Ångstrom and this parameter is :math:`10^{-5}` Ångstrom/step, then a run with :math:`10^{6}` steps will change the box length by 10%.
This gives a strain rate of :math:`10^{8}` s :math:`^{-1}` if the time step is 1 fs.
The second parameter :attr:`deform_x` can be 0 or 1, where 0 means do not deform the :math:`x` direction and 1 means deform the :math:`x` direction.
The last two parameters have similar meanings for the :math:`y` and :math:`z` directions.

In this form, the same deformation rate is applied to all directions that are flagged as deformed.

**6-parameter form** ::

  deform <A_per_step_x> <A_per_step_y> <A_per_step_z> <deform_x> <deform_y> <deform_z>

Here, :attr:`A_per_step_x`, :attr:`A_per_step_y`, :attr:`A_per_step_z` specify the increment (or decrement) of the box length in the x, y, and z directions per simulation step, respectively. The unit is Ångstrom/step. For example, if the box length in the x-direction is initially 100 Å and :attr:`A_per_step_x` is set to :math:`10^{-5}` Å/step, then after :math:`10^{6}` steps, the box length will change by 10%. With a time step of 1 fs, this corresponds to a strain rate of :math:`10^{8}` s :math:`^{-1}`.Use :attr:`deform_x`, :attr:`deform_y`, :attr:`deform_z` to enable or disable deformation in each direction. A value of `1` enables deformation, while `0` disables it. For a direction where deformation is disabled, the box length remains constant throughout the simulation.

This form allows independent deformation rates in the x, y, and z directions.

Example
-------

For uniaxial tensile test, one can first equilibrate the system and then deform the box.

**Using the 4-parameter form**::

  # equilibration stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  run 1000000

  # production stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  deform 0.00001 1 0 0
  run 1000000

**Using the 6-parameter form**::

  # equilibration stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  run 1000000

  # production stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  deform 0.00001 0 0 1 0 0
  run 1000000

Both examples achieve the same deformation (x-direction stretched at 0.00001 Å/step).

Caveats
-------
* Currently, one must use the NPT ensemble when using this keyword.
  That is, the code assumes that the pressure components in the directions which are not deformed will be controlled.
  If one does not want to control the pressure (box) in a direction that is not deformed, 
  one can set the elastic constant for that direction to be larger than 2000 GPa.
* In the equilibration stage, it is also recommended to use the NPT ensemble to obtain the zero strain state before applying the deformation.
* One must control the three pressure components independently when using this keyword.
* The 6-parameter form is more flexible and is recommended for new input files.
