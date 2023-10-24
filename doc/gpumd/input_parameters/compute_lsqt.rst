.. _kw_compute_lsqt:
.. index::
   single: compute_lsqt (keyword in run.in)

:attr:`compute_lsqt`
====================

This keyword is used to compute the electronic transport properties using the linear-scaling quantum transport (:term:`LSQT`) [Fan2021b]_ approach.
:term:`LSQT` and :term:`MD` are coupled to account for electron-phonon scattering. 
Currently, only a tight-binding model for carbon is supported (hard coded).
The results will be written into the files lsqt_dos.out, lsqt_velocity.out, and lsqt_sigma.out.
This feature is preliminary and changes might be made in the near future.

Syntax
------

This keyword is used as follows::

  compute_lsqt <transport_direction> <num_moments> <num_energies> <E_1> <E_2> <E_max>

* :attr:`transport_direction` is the transport direction, which can take values x, y, and z.
* :attr:`num_moments` is the number of the Chebyshev moments for the energy resolution operator.
* The :attr:`num_energies` energy points increase linearly from :attr:`E_1` (eV) to :attr:`E_2` (eV).
* :attr:`E_max` (eV) is an energy value taht should be (slightly) larger than the maximum of the absolute energy of the tight-binding model. This can be determined by trial and error.

Example
-------

   compute_lsqt x 3000 10001 -8.1 8.1 8.2
