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

    potential adp <potential_file>

where:

- :attr:`<potential_file>`: Path to the ADP potential file (required)

Element types are automatically detected from :file:`model.xyz` based on the element symbols. The potential file header specifies which elements are available, and GPUMD automatically matches atoms in :file:`model.xyz` to the corresponding element parameters in the potential file.

Basic usage
^^^^^^^^^^^

For any system (single-element or multi-element)::

    potential adp Ta.adp

GPUMD will automatically:

1. Read the element list from the ADP potential file header
2. Match atoms in :file:`model.xyz` based on their element symbols
3. Assign the correct potential parameters to each atom

Single-element system
^^^^^^^^^^^^^^^^^^^^^

For a pure metal system (e.g., pure tantalum)::

    potential adp Ta.adp

All atoms labeled as "Ta" in :file:`model.xyz` will automatically use the Ta parameters from the potential file.

Multi-element system
^^^^^^^^^^^^^^^^^^^^

For binary or multi-component alloys, the same simple syntax applies.

**Example 1: Copper-Tantalum (Cu-Ta)**

Usage::

    potential adp Cu_Ta.adp

GPUMD will:

- Read that the potential file contains Cu and Ta parameters
- Automatically assign Cu parameters to atoms labeled "Cu" in :file:`model.xyz`
- Automatically assign Ta parameters to atoms labeled "Ta" in :file:`model.xyz`

**Example 2: Uranium-Molybdenum (U-Mo)**

Usage::

    potential adp U_Mo.adp

GPUMD will:

- Read that the potential file contains U and Mo parameters  
- Automatically assign U parameters to atoms labeled "U" in :file:`model.xyz`
- Automatically assign Mo parameters to atoms labeled "Mo" in :file:`model.xyz`

**Example 3: Pure Mo from a U-Mo potential file**

Usage::

    potential adp U_Mo.adp

If your :file:`model.xyz` only contains Mo atoms (no U atoms), GPUMD will automatically use only the Mo parameters from the potential file. This is useful for testing pure element properties using multi-element potential files.

.. note::

   Element detection is fully automatic based on the element symbols in :file:`model.xyz`. The element symbols must match those defined in the ADP potential file header (line 4). This behavior is consistent with other potentials in GPUMD (e.g., NEP).

References
----------

.. [Mishin2005] Y. Mishin, M. J. Mehl, and D. A. Papaconstantopoulos, "Phase stability in the Fe–Ni system: Investigation by first-principles calculations and atomistic simulations," Acta Mater. 53, 4029 (2005).

.. [Apostol2011] F. Apostol and Y. Mishin, "Interatomic potential for the Al-Cu system," Phys. Rev. B 83, 054116 (2011).

.. [Pun2015] G. P. Pun, K. A. Darling, L. J. Kecskes, and Y. Mishin, "Angular-dependent interatomic potential for the Cu–Ta system and its application to structural stability of nano-crystalline alloys," Acta Mater. 100, 377 (2015).

.. [Starikov2018] S. V. Starikov, L. N. Kolotova, A. Y. Kuksin, D. E. Smirnova, and V. I. Tseplyaev, "Atomistic simulation of cubic and tetragonal phases of U-Mo alloy: Structure and thermodynamic properties," J. Nucl. Mater. 499, 451 (2018).
