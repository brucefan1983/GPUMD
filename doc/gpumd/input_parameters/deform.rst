.. _kw_deform:
.. index::
   single: deform (keyword in run.in)

:attr:`deform`
==============

This keyword is used to deform the simulation box, which can be used to do tensile tests.

Syntax
------

The :attr:`deform` keyord requires 4 parameters::

  deform A_per_step deform_x deform_y deform_z

Here, :attr:`A_per_step` specifies the speed of the increase of the box length, which is in units of Ångstrom/step.
For example, suppose the box length (in a given direction) in the beginning of a run is 100 Ångstrom and this parameter is :math:`10^{-5}` Ångstrom/step, then a run with :math:`10^{6}` steps will change the box length by 10%.
This gives a strain rate of :math:`10^{8}` s:math:`^{-1}` if the time step is 1 fs.
The second parameter :attr:`deform_x` can be 0 or 1, where 0 means do not deform the :math:`x` direction and 1 means deform the :math:`x` direction.
The last two parameters have similar meanings for the :math:`y` and :math:`z` directions.


Example
-------

For uniaxial tensile test, one can first equilibrate the system and then deform the box::

  # equilibration stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  run 1000000

  # production stage
  ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000
  deform 0.00001 1 0 0
  run 1000000

Caveats
-------
* Currently, one must use the NPT ensemble when using this keyword.
  That is, the code assumes that the pressure components in the directions which are not deformed will be controlled.
* In the equilibration stage, it is also recommended to use the NPT ensemble to obtain the zero strain state before applying the deformation.
* One must control the three pressure components independently when using this keyword.
