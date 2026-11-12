.. _kw_change_box:
.. index::
   single: change_box (keyword in run.in)

:attr:`change_box`
==================

This keyword changes the simulation box and atomic positions immediately when it is read from ``run.in``.
For gradual deformation during a run, use :ref:`deform <kw_deform>`.
The box variables and the atom positions are changed according to the following equations:

.. math::

   \left(
   \begin{array}{ccc}
   a_x^{\rm new} & b_x^{\rm new} & c_x^{\rm new} \\
   a_y^{\rm new} & b_y^{\rm new} & c_y^{\rm new} \\
   a_z^{\rm new} & b_z^{\rm new} & c_z^{\rm new} 
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
   a_x^{\rm old} & b_x^{\rm old} & c_x^{\rm old} \\
   a_y^{\rm old} & b_y^{\rm old} & c_y^{\rm old} \\
   a_z^{\rm old} & b_z^{\rm old} & c_z^{\rm old} 
   \end{array}
   \right);
   \\
   \left(
   \begin{array}{c}
   x^{\rm new}_i \\
   y^{\rm new}_i \\
   z^{\rm new}_i
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
   x_i^{\rm old} \\
   y_i^{\rm old} \\
   z_i^{\rm old}
   \end{array}
   \right).

The deformation matrix :math:`\mu_{\alpha\beta}` is specified by the parameters below.
The transformation preserves the fractional atomic coordinates; atomic velocities are unchanged.

Syntax
------

This keyword accepts 1 or 3 or 6 parameters.

In the case of 1 parameter :math:`\delta` (in units of Ångstrom)::

 change_box <delta>

we have

.. math::
   \left(
   \begin{array}{ccc}
   \mu_{xx} & \mu_{xy} & \mu_{xz} \\
   \mu_{yx} & \mu_{yy} & \mu_{yz} \\
   \mu_{zx} & \mu_{zy} & \mu_{zz} \\
   \end{array}
   \right) 
   = 
   \left(
   \begin{array}{ccc}
   \frac{a_x^{\rm old} + \delta}{a_x^{\rm old}} & 0 & 0 \\
   0 & \frac{b_y^{\rm old} + \delta}{b_y^{\rm old}} & 0 \\
   0 & 0 & \frac{c_z^{\rm old} + \delta}{c_z^{\rm old}} \\
   \end{array}
   \right)

In the case of 3 parameters, :math:`\delta_{xx}` (in units of Ångstrom), :math:`\delta_{yy}` (in units of Ångstrom), and :math:`\delta_{zz}` (in units of Ångstrom)::
  
   change_box <delta_xx> <delta_yy> <delta_zz>

we have

.. math::
   
   \left(
   \begin{array}{ccc}
   \mu_{xx} & \mu_{xy} & \mu_{xz} \\
   \mu_{yx} & \mu_{yy} & \mu_{yz} \\
   \mu_{zx} & \mu_{zy} & \mu_{zz} \\
   \end{array}
   \right) 
   = 
   \left(
   \begin{array}{ccc}
   \frac{a_x^{\rm old} + \delta_{xx}}{a_x^{\rm old}} & 0 & 0 \\
   0 & \frac{b_y^{\rm old} + \delta_{yy}}{b_y^{\rm old}} & 0 \\
   0 & 0 & \frac{c_z^{\rm old} + \delta_{zz}}{c_z^{\rm old}} \\
   \end{array}
   \right)

In the case of 6 parameters (the box type must be triclinic), :math:`\delta_{xx}` (in units of Ångstrom), :math:`\delta_{yy}` (in units of Ångstrom), :math:`\delta_{zz}` (in units of Ångstrom), :math:`\epsilon_{yz}` (dimensionless strain), :math:`\epsilon_{xz}` (dimensionless strain), and :math:`\epsilon_{xy}` (dimensionless strain)::

  change_box <delta_xx> <delta_yy> <delta_zz> <epsilon_yz> <epsilon_xz> <epsilon_xy>

we have

.. math::
   
   \left(
   \begin{array}{ccc}
   \mu_{xx} & \mu_{xy} & \mu_{xz} \\
   \mu_{yx} & \mu_{yy} & \mu_{yz} \\
   \mu_{zx} & \mu_{zy} & \mu_{zz} \\
   \end{array}
   \right) 
   = 
   \left(
   \begin{array}{ccc}
   \frac{a_x^{\rm old} + \delta_{xx}}{a_x^{\rm old}} & \epsilon_{xy} & \epsilon_{xz} \\
   \epsilon_{yx} & \frac{b_y^{\rm old} + \delta_{yy}}{b_y^{\rm old}} & \epsilon_{yz} \\
   \epsilon_{zx} & \epsilon_{zy} & \frac{c_z^{\rm old} + \delta_{zz}}{c_z^{\rm old}} \\
   \end{array}
   \right)

Here, :math:`\epsilon_{yx}=\epsilon_{xy}`, :math:`\epsilon_{zx}=\epsilon_{xz}`, and :math:`\epsilon_{zy}=\epsilon_{yz}`: each shear input sets both corresponding off-diagonal elements of :math:`\mu`.
These inputs are dimensionless. In the small-strain limit, the engineering shear strain is :math:`\gamma_{\alpha\beta}=2\epsilon_{\alpha\beta}` for :math:`\alpha\ne\beta`.
The diagonal inputs :math:`\delta_{xx}`, :math:`\delta_{yy}`, and :math:`\delta_{zz}` are lengths in Ångstrom.

Example
-------

For an orthogonal box with :math:`a_x=100` Å, the command::

  change_box 1 0 0

increases the x box length to 101 Å and scales the x coordinates by 1.01, while leaving the y and z box lengths and coordinates unchanged.
