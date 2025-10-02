.. _adp:
.. index::
   single: Angular Dependent Potential

Angular Dependent Potential (ADP)
==================================

:program:`GPUMD` supports the Angular Dependent Potential (ADP), which is an extension of the embedded atom method (:term:`EAM`) that includes angular forces through dipole and quadrupole distortions of the local atomic environment.

The ADP was developed to provide a more accurate description of directional bonding and angular forces in metallic systems, particularly for materials where traditional EAM potentials fail to capture the full complexity of atomic interactions. The ADP formalism is especially useful for modeling complex crystal structures, defects, and phase transformations in metals and alloys.

Potential form
--------------

General form
^^^^^^^^^^^^

The ADP is described in detail by Mishin et al. [Mishin2005]_ and has been successfully applied to various metallic systems including the Cu-Ta system [Pun2015]_, U-Mo alloys [Starikov2018]_, etc. The total energy of atom :math:`i` is given by:

.. math::
   
   E_i = F_\alpha\left(\sum_{j\neq i} \rho_\beta(r_{ij})\right) + \frac{1}{2} \sum_{j\neq i} \phi_{\alpha\beta}(r_{ij}) + \frac{1}{2} \sum_s (\mu_{is})^2 + \frac{1}{2} \sum_{s,t} (\lambda_{ist})^2 - \frac{1}{6} \nu_i^2

where:

- :math:`F_\alpha` is the embedding energy as a function of the total electron density at atom :math:`i`
- :math:`\rho_\beta(r_{ij})` is the electron density contribution from atom :math:`j` at distance :math:`r_{ij}`
- :math:`\phi_{\alpha\beta}(r_{ij})` is the pair potential interaction between atoms of types :math:`\alpha` and :math:`\beta`
- :math:`\alpha` and :math:`\beta` are the element types of atoms :math:`i` and :math:`j`
- :math:`s` and :math:`t` are indices running over Cartesian coordinates (:math:`x, y, z`)
- :math:`\mu_{is}` is the dipole distortion tensor (3 components)
- :math:`\lambda_{ist}` is the quadrupole distortion tensor (6 independent components)
- :math:`\nu_i` is the trace of the quadrupole tensor

Angular terms
^^^^^^^^^^^^^

The dipole distortion tensor represents the first moment of the local environment:

.. math::
   
   \mu_{is} = \sum_{j\neq i} u_{\alpha\beta}(r_{ij}) r_{ij}^s

where :math:`u_{\alpha\beta}(r)` is a tabulated function and :math:`r_{ij}^s` is the :math:`s`-component of the vector from atom :math:`i` to atom :math:`j`.

The quadrupole distortion tensor represents the second moment of the local environment:

.. math::
   
   \lambda_{ist} = \sum_{j\neq i} w_{\alpha\beta}(r_{ij}) r_{ij}^s r_{ij}^t

where :math:`w_{\alpha\beta}(r)` is another tabulated function. The trace of the quadrupole tensor is:

.. math::
   
   \nu_i = \lambda_{ixx} + \lambda_{iyy} + \lambda_{izz}

The angular terms :math:`\mu` and :math:`\lambda` introduce directional dependence into the potential energy, allowing the ADP to capture angular forces that are absent in the traditional EAM formalism. These terms are essential for accurately modeling materials with complex bonding environments.


File format
-----------

General structure
^^^^^^^^^^^^^^^^^

The ADP potential file follows the extended DYNAMO setfl format, which is compatible with LAMMPS and other molecular dynamics codes. The file structure consists of:

**Header section** (lines 1-5):

- Lines 1-3: Comment lines (can contain any text, typically author and date information)
- Line 4: :attr:`Nelements` :attr:`Element1` :attr:`Element2` ... :attr:`ElementN`

  * :attr:`Nelements`: Number of elements in the potential
  * :attr:`Element1`, :attr:`Element2`, etc.: Element symbols (e.g., Cu, Ta, Mo)

- Line 5: :attr:`Nrho` :attr:`drho` :attr:`Nr` :attr:`dr` :attr:`cutoff`

  * :attr:`Nrho`: Number of points in the embedding function :math:`F(\rho)` tabulation
  * :attr:`drho`: Spacing between tabulated :math:`\rho` values
  * :attr:`Nr`: Number of points in the pair potential and density function tabulations
  * :attr:`dr`: Spacing between tabulated :math:`r` values
  * :attr:`cutoff`: Cutoff distance for all functions (in Angstroms)

**Per-element sections** (repeated :attr:`Nelements` times):

Each element section contains:

- Line 1: :attr:`atomic_number` :attr:`mass` :attr:`lattice_constant` :attr:`lattice_type`

  * :attr:`atomic_number`: Atomic number of the element
  * :attr:`mass`: Atomic mass (in amu)
  * :attr:`lattice_constant`: Equilibrium lattice constant (in Angstroms)
  * :attr:`lattice_type`: Crystal structure (e.g., fcc, bcc, hcp)

- Next :attr:`Nrho` values: Embedding function :math:`F(\rho)` 

  * Tabulated values of :math:`F` at :math:`\rho = 0, \Delta\rho, 2\Delta\rho, ..., (N_\rho-1)\Delta\rho`
  * Units: eV

- Next :attr:`Nr` values: Electron density function :math:`\rho(r)`

  * Tabulated values at :math:`r = 0, \Delta r, 2\Delta r, ..., (N_r-1)\Delta r`
  * Units: electron density

**Pair potential section**:

For all element pairs :math:`(i, j)` with :math:`i \geq j` (upper triangular, since :math:`\phi_{ij} = \phi_{ji}`):

- :attr:`Nr` values: Pair potential :math:`\phi_{ij}(r)`

  * Tabulated as :math:`r \times \phi(r)` (scaled by distance)
  * Units: eV·Angstrom
  * Order: (1,1), (2,1), (2,2), (3,1), (3,2), (3,3), etc.

**Dipole function section**:

For all element pairs :math:`(i, j)` with :math:`i \geq j`:

- :attr:`Nr` values: Dipole function :math:`u_{ij}(r)`

  * Tabulated as :math:`u(r)` (NOT scaled by distance)
  * Units: electron density·Angstrom
  * Same ordering as pair potentials

**Quadrupole function section**:

For all element pairs :math:`(i, j)` with :math:`i \geq j`:

- :attr:`Nr` values: Quadrupole function :math:`w_{ij}(r)`

  * Tabulated as :math:`w(r)` (NOT scaled by distance)  
  * Units: electron density·Angstrom²
  * Same ordering as pair potentials


Usage
-----

Syntax
^^^^^^

To use an ADP potential in GPUMD, specify it in the :file:`run.in` input file with the following syntax::

    potential adp <potential_file> [list of elements]

where:

- :attr:`<potential_file>`: Path to the ADP potential file (required)
- :attr:`[list of elements]`: Space-separated list of element symbols for explicit mapping (optional)

The element list, if provided, determines the mapping between atom types in your structure file and elements in the potential file. The number of element types is automatically inferred from the number of elements specified.

Basic usage
^^^^^^^^^^^

For a single-element or auto-detected system::

    potential adp Ta.adp

GPUMD will automatically read the number of elements from the potential file and map them in order.

Single-element system
^^^^^^^^^^^^^^^^^^^^^

For a pure metal system (e.g., pure tantalum)::

    potential adp Ta.adp

If not specifying element names, GPUMD will use default mapping (all atoms map to the first element in the ADP file).

You can also explicitly specify the element::

    potential adp Ta.adp Ta

This explicitly maps all atoms in the system to the Ta element from the potential file.

Multi-element system
^^^^^^^^^^^^^^^^^^^^

For binary alloys, you have several options depending on your system.

**Example 1: Copper-Tantalum (Cu-Ta)**

Simple usage::

    potential adp Cu_Ta.adp

Without explicit element mapping, GPUMD uses default mapping (all atoms map to the first element in the potential file).

Explicit element mapping::

    potential adp Cu_Ta.adp Cu Ta

This maps atom type 0 → Cu and atom type 1 → Ta in your structure file.

Reversed element order::

    potential adp Cu_Ta.adp Ta Cu

This maps atom type 0 → Ta and atom type 1 → Cu.

**Example 2: Uranium-Molybdenum (U-Mo)**

Simple usage::

    potential adp U_Mo.adp

Explicit element mapping::

    potential adp U_Mo.adp U Mo

This maps atom type 0 → U and atom type 1 → Mo in your structure file.

Reversed element order::

    potential adp U_Mo.adp Mo U

This is useful when your structure file has atom types in a different order than desired.

.. note::

   When using explicit element mapping, the order of elements in the :attr:`[list of elements]` determines how atom types in your structure file correspond to elements in the potential file. The number of elements must match the number of atom types in your system.

References
----------

.. [Mishin2005] Y. Mishin, M. J. Mehl, and D. A. Papaconstantopoulos, "Phase stability in the Fe–Ni system: Investigation by first-principles calculations and atomistic simulations," Acta Mater. 53, 4029 (2005).

.. [Apostol2011] F. Apostol and Y. Mishin, "Interatomic potential for the Al-Cu system," Phys. Rev. B 83, 054116 (2011).

.. [Pun2015] G. P. Pun, K. A. Darling, L. J. Kecskes, and Y. Mishin, "Angular-dependent interatomic potential for the Cu–Ta system and its application to structural stability of nano-crystalline alloys," Acta Mater. 100, 377 (2015).

.. [Starikov2018] S. V. Starikov, L. N. Kolotova, A. Y. Kuksin, D. E. Smirnova, and V. I. Tseplyaev, "Atomistic simulation of cubic and tetragonal phases of U-Mo alloy: Structure and thermodynamic properties," J. Nucl. Mater. 499, 451 (2018).
