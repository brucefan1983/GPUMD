.. _kw_dftd3:
.. index::
   single: dftd3 (keyword in run.in)

:attr:`dftd3`
=============

This keyword is used to add the DFT-D3 dispersion correction to NEP.
It has no effect when any other potential is used.
For theoretical background on DFT-D3, see [Grimme2010]_ and [Grimme2011]_.

Syntax
------

  dftd3 <functional> <potential_cutoff> <coordination_number_cutoff>

where :attr:`functional` is the exchange-correlation functional used in generating the reference data for training the NEP model, :attr:`potential_cutoff` is the cutoff radius (in units of Angstrom) for the D3 potential, :attr:`coordination_number_cutoff` is the cutoff radius (in units of Angstrom) for calculating the coordination numbers. 

This keyword can be put anywhere in the ``run.in`` input file.

We have only implemented the Becke-Johnson (BJ) damping, and a full list of supported functionals can be found from the following link: https://www.chemiebn.uni-bonn.de/pctc/mulliken-center/software/dft-d3/functionalsbj

Example
-------

.. code::

  dftd3 pbe 12 6
  
This will add the DFT-D3 dispersion correction to NEP with the PBE functional and a cutoff radius of 12 Angstrom for the D3 potential and a cutoff radius of 6 Angstrom for the coordination number.

Tips
----

* It usually requires to test the convergence with respect to the cutoff radii.

* The user is responsible for not double counting the dispersion correction, i.e., it is not a good idea to add the DFT-D3 correction to a NEP model that has been trained against a dataset containing dispersion correction (no matter what flavor it is).
