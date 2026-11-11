.. _dpdt_out:
.. index::
   single: dpdt.out (output file)

``dpdt.out``
============

This file contains the time derivative of the total dipole moment of the simulation cell and its time integral.
The derivative is computed as

.. math::

   \dot{M}_{\alpha}(t) = \sum_i \sum_{\beta} Z^*_{i,\alpha\beta}(t) v_{i,\beta}(t),

where :math:`Z^*_{i,\alpha\beta}` is the Born effective charge of atom :math:`i` in units of e and :math:`v_{i,\beta}` is its velocity.
No division by the cell volume is applied.
For a fixed cell volume :math:`V`, the polarization change is :math:`\Delta P_\alpha = \Delta M_\alpha / V`.
The integral starts from zero and gives a dipole-moment change, rather than an absolute dipole moment.
It is produced when invoking :ref:`compute_dpdt keyword <kw_compute_dpdt>` in the :ref:`run.in input file <run_in>`.

File format
-----------
Each run appends a block beginning with comment lines marked by ``#``.
These record the sampling interval, format version, atom count, initial cell vectors, output time interval, and column labels.
The column labels are ``time_fs dpdt_x dpdt_y dpdt_z P_x P_y P_z``; the quantities labeled ``P_x``, ``P_y``, and ``P_z`` are the integrated dipole-moment changes described below.

Each data row contains:

* column 1: time (in units of fs)
* column 2: total dipole-moment derivative :math:`dM_x/dt` (in units of e Å / fs)
* column 3: total dipole-moment derivative :math:`dM_y/dt` (in units of e Å / fs)
* column 4: total dipole-moment derivative :math:`dM_z/dt` (in units of e Å / fs)
* column 5: integrated dipole-moment change :math:`\Delta M_x` (in units of e Å)
* column 6: integrated dipole-moment change :math:`\Delta M_y` (in units of e Å)
* column 7: integrated dipole-moment change :math:`\Delta M_z` (in units of e Å)

For a constant MD time step :math:`\Delta t` and sampling interval :math:`s`, the output interval is :math:`\delta t = s\Delta t`.
The integral uses the derivative at the end of each output interval:

.. math::

   \Delta M_\alpha(t_n) = \delta t \sum_{j=1}^{n} \dot{M}_\alpha(t_j),
   \qquad t_j = j\delta t.

Time and the accumulated integral restart from zero for each run; the first data row is at :math:`t_1 = \delta t`.
