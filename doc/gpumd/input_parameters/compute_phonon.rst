.. _kw_compute_phonon:
.. index::
   single: compute_phonon (keyword in run.in)

:attr:`compute_phonon`
======================

This keyword can be used to compute the phonon dispersions using the finite-displacement method.
The results are written to the :ref:`D.out output file <D_out>` and the :ref:`omega2.out output file <omega2_out>`.

To use this keyword, please make sure the following files have been preprared in the input directory:

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Input filename
     - Brief description
   * - ``kpoints.in``
     - Specify the :math:`\boldsymbol{k}`-points along the high-symmetry paths


To use this keyword, :attr:`replicate` keywords must write head in the :attr:`run.in` file.

Syntax
------

This keyword is used as follows::

  compute_phonon <displacement>

:attr:`displacement` is the displacement (in units of Å) for calculating the force constants using the finite-displacement method.

Example
-------

For example, the command::

    compute_phonon 0.01

means that one wants to compute the phonon dispersion using a displacement of 0.01 Å.

Caveats
-------


This keyword should occur after all the :attr:`potential` keywords.

For two-body potentials, the cutoff distance for force constants can be the same as that for force evaluations.
However, for many-body potentials, the cutoff distance for force constants usually needs to be twice of the potential cutoff distance.
Also make sure that the box size in any direction is at least twice of this force constant cutoff distance.

Related tutorial
----------------

The use of the :attr:`calculate_phonon` keyword is illustrated in `this tutorial <https://github.com/brucefan1983/GPUMD/blob/master/examples/empirical_potentials/phonon_dispersion/Phonon%20Dispersion.ipynb>`_.
