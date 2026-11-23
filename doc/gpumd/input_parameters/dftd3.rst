.. _kw_dftd3:
.. index::
   single: dftd3 (keyword in run.in)

:attr:`dftd3`
=============

This keyword is used to add the DFT-D3 dispersion correction to NEP.
It must not be combined with ``nep_spin``, ``nep_vdw``, or ``qnep3_vdw`` models.
For theoretical background on DFT-D3, see [Grimme2010]_ (zero damping) and [Grimme2011]_ (BJ damping).

Syntax
------

The syntax is::

  dftd3 <functional> <potential_cutoff> <coordination_number_cutoff> [zero|bj]

Parameters
----------

where :attr:`functional` is the exchange-correlation functional used in generating the reference data for training the NEP model, :attr:`potential_cutoff` is the cutoff radius (in units of Angstrom) for the D3 potential, and :attr:`coordination_number_cutoff` is the cutoff radius (in units of Angstrom) for calculating the coordination numbers.
The optional last token selects the damping function: ``bj`` (the default) is the Becke-Johnson damping and ``zero`` is the zero damping, D3(0).

This keyword can be put anywhere in the ``run.in`` input file.

Both dampings are implemented for the two-body term only; the three-body (Axilrod-Teller-Muto) term is not included.
With ``bj``, a full list of supported functionals can be found from the following link: https://www.chemiebn.uni-bonn.de/pctc/mulliken-center/software/dft-d3/functionalsbj

With ``zero``, the parameters are those of Grimme's DFT-D3 as in the dftd3-lib project, and only the elements H to Rn (atomic number up to 86) are supported, because for the elements Fr to Pu (87 to 94) the available parameter tables disagree with each other and the correct values are not verified.
The check is made at the first force evaluation, when the atom types of the system are known; the elements listed in the model file do not matter.
The functional name must be in lower case and one of the following (``hf`` is Hartree-Fock): ``slaterdiracexchange``, ``blyp``, ``bp86``, ``b97d``, ``revpbe``, ``pbe``, ``pbesol``, ``rpw86pbe``, ``rpbe``, ``tpss``, ``b3lyp``, ``pbe0``, ``hse06``, ``revpbe38``, ``pw6b95``, ``tpss0``, ``b2plyp``, ``pwpb95``, ``b2gpplyp``, ``ptpss``, ``hf``, ``mpwlyp``, ``bpbe``, ``bhlyp``, ``tpssh``, ``pwb6k``, ``b1b95``, ``bop``, ``olyp``, ``opbe``, ``ssb``, ``revssb``, ``otpss``, ``b3pw91``, ``revpbe0``, ``pbe38``, ``mpw1b95``, ``mpwb1k``, ``bmk``, ``camb3lyp``, ``lcwpbe``, ``m05``, ``m052x``, ``m06l``, ``m06``, ``m062x``, ``m06hf``, ``hcth120``.
If the functional is not available for the chosen damping, or the system contains an element that is not supported, the run stops with an error message; there is no fallback to the other damping.

Output
------

None.

Examples
--------

.. code::

  dftd3 pbe 12 6

This will add the DFT-D3 dispersion correction (BJ damping) to NEP with the PBE functional and a cutoff radius of 12 Angstrom for the D3 potential and a cutoff radius of 6 Angstrom for the coordination number.

.. code::

  dftd3 pbe 12 6 zero

This is the same, but with the zero damping, D3(0). See ``examples/gpumd_dftd3_zero``.

Notes
-----

* It usually requires to test the convergence with respect to the cutoff radii, and a balance between accuracy and efficiency must be achieved.

* The user is responsible for not double counting the dispersion correction, i.e., it is not a good idea to add the DFT-D3 correction to a NEP model that has been trained against a dataset containing dispersion correction (no matter what flavor it is).

* The zero damping is available for NEP models only; it is not available for the qNEP charge models (``qnep1``, ``qnep2``, ``qnep3``), where ``dftd3`` accepts only the three parameters (BJ damping).

* Currently cannot use DFT-D3 for the multi-GPU version of NEP.
