.. _kw_change_box:
.. index::
   single: change_box (keyword in run.in)

:attr:`change_box`
==================

This keyword is used to change the simulation box. The box variables and the atom positions are changed according to the following equations:

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

The deformation matrix :math:`\mu_{\alpha\beta}` will be specified by the parameters of this keyword, as we detail below.

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
