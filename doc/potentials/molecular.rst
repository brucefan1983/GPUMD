.. _molecular_potential:

Fixed-topology molecular force field
************************************

Use two files to assign Lennard-Jones (LJ) interactions, harmonic bonds and
harmonic angles to a fixed list of atoms::

  potential forcefield.in topology.in

The first file defines the force-field types and parameters. The second assigns
a type to each atom in the order of the coordinates in ``model.xyz``.
When using :ref:`kw_replicate`, supply the topology for all atoms in the
replicated model, in their expanded order.
Put the force-field header on the first line. Below it and in the topology,
blank lines and whitespace-separated ``#`` comments are allowed. Both files
are read at initialization. Use three periodic directions with each
perpendicular box height greater than twice the LJ cutoff throughout the run.

Force-field parameters
======================

A file for a single argon type reads::

  molecular 1 Ar
  cutoff 8.5
  types 1
  0 Ar
  pairs 1
  0 0 3.4 0.0103

The first line names the model, the number of chemical elements and their
symbols, as for an ordinary GPUMD potential file. The cutoff and
:math:`\sigma` are in angstroms and :math:`\epsilon` is in eV.
Each pair record contains the two force-field type IDs, :math:`\sigma`
and :math:`\epsilon`, in that order. The ordinary pair potential is
:math:`4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6]` for :math:`r<r_c`
and zero for :math:`r\ge r_c`. The cutoff is unshifted.

Type IDs run from zero to the number of force-field types minus one.
Several types may share a chemical element. Supply one upper-triangular
pair record for each type combination, including the diagonal. Set
:math:`\epsilon=0` for a pair with no LJ interaction. Specify the final
cross parameters in the file. For example, two carbon types can have
independent interaction parameters::

  molecular 1 C
  cutoff 10
  types 2
  0 C
  1 C
  pairs 3
  0 0 3.5 0.003
  0 1 3.6 0.004
  1 1 3.8 0.005

Topology
========

For two argon atoms using the first force field, the topology reads::

  atoms 2
  0 0
  0 0

Each atom row gives its force-field type ID and charge in elementary charges.
The declared element of the type must match that atom's species in
``model.xyz``. This model currently supports neutral LJ sites. Use zero
charges in the topology and omit the optional XYZ charge column or set it
to zero.

Special LJ pairs
================

Append an optional ``special_pairs`` section after the atom rows.
List each unordered pair once using zero-based atom indices in XYZ order.
The last two fields give the LJ and Coulomb weights, both in the range
from zero to one. With neutral sites only the LJ weight affects this model.

For example, these records exclude atoms 0 and 1 and scale the 0–3
interaction by one half in a topology with at least four atoms::

  special_pairs 2
  0 1 0.0 0.0
  0 3 0.5 0.5

A listed pair is removed from the ordinary cutoff calculation. A nonzero LJ
weight multiplies the full LJ potential at the pair's periodic displacement,
including beyond the ordinary cutoff. In a triclinic cell GPUMD wraps each
fractional displacement component to its nearest image. An intended 1–4
distance below half the smallest perpendicular box height guarantees
that this image stays unambiguous. Supply all desired
exclusions and scaled pairs explicitly.

Harmonic bonds
==============

To define a bond type, append a ``bond_types`` section after the LJ pair table
in the force-field file. For example::

  bond_types 1
  0 20.0 1.5

The columns are the bond type ID, force constant :math:`k` in
:math:`\mathrm{eV}/\text{Å}^2`, and equilibrium length :math:`r_0` in
angstroms. Bond type IDs run from zero to the number of bond types minus one.
Each bond contributes :math:`U = \tfrac12 k(r-r_0)^2`, with :math:`k \geq 0`
and :math:`r_0 > 0`.

In the topology file, list each bond once as two zero-based atom indices
followed by its bond type ID. The following example bonds atoms 0 and 1
and excludes their LJ interaction::

  bonds 1
  0 1 0
  special_pairs 1
  0 1 0.0 0.0

Both sections are optional and may appear in either order after the atom
rows. A bonded pair retains its ordinary LJ interaction unless explicitly
excluded or scaled by ``special_pairs``. Bonds are evaluated at every
distance, including beyond the LJ cutoff.

The bond vector uses the periodic image obtained by wrapping its fractional
displacement. Keep the bond shorter than half the smallest perpendicular
box height to ensure an unambiguous image. Each endpoint receives half of
the bond energy and half of the bond virial. If the endpoints coincide,
the bond contributes :math:`\tfrac12 k r_0^2` in total and zero force and
virial because the bond direction is undefined.

Harmonic angles
===============

After the LJ pair table in the force-field file, an optional ``angle_types``
section defines the bending parameters. It can precede or follow ``bond_types``.
For example::

  angle_types 1
  0 3.0 105.0

The columns give the zero-based angle type ID, force constant
:math:`k_\theta` in eV/rad², and equilibrium angle :math:`\theta_0` in
degrees. Supply each type ID from zero through the number of types minus
one exactly once. The potential is
:math:`U_\theta=\tfrac12 k_\theta(\theta-\theta_0)^2`, evaluated in
radians. The force constant must be nonnegative and the equilibrium angle
must lie between 0 and 180 degrees, inclusive.

List angles in the topology file after the atom rows. Each record gives three
atom indices and the angle type ID. **The middle atom is the vertex.** For
atoms in O, H, H order, the following record defines the H–O–H angle::

  angles 1
  1 0 2 0

The three atom indices must be distinct. Reversing the endpoints gives the
same angle, so supply it only once. The ``angles``, ``bonds`` and
``special_pairs`` sections may appear in any order. An angle can be used on
its own. Specify bond stretching and any desired 1–3 LJ exclusion separately.

For an angle A–B–C, GPUMD takes the periodic displacements from B to A and
from B to C independently. Keep each arm shorter than half the smallest
perpendicular box height for an unambiguous periodic image. With positive
stiffness each arm must have nonzero length. An exactly collinear geometry
has zero bending force at equilibrium. Away from equilibrium its bending
direction is undefined and the run stops.

The vertex B receives the entire angle energy and feels the force
:math:`\mathbf F_B=-\mathbf F_A-\mathbf F_C`. With this site-energy
assignment the endpoint virials contributed by this angle are
:math:`\mathbf r_{BA}\otimes\mathbf F_A` and
:math:`\mathbf r_{BC}\otimes\mathbf F_C`, respectively, and the
vertex virial contribution is zero. Here :math:`\mathbf r_{BA}` points
from B to A. GPUMD retains all nine per-atom virial components for the
heat current.
