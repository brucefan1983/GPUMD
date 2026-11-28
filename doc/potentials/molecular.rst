.. _molecular_potential:

Fixed-topology molecular force field
************************************

Use two files to assign Lennard-Jones (LJ) interactions to a fixed list of atoms::

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
to zero. Bond and angle interactions require additional force terms.

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
