.. _kw_compute_react_count:
.. index::
   single: compute_react_count (keyword in run.in)

:attr:`compute_react_count`
===========================

This keyword computes distance-based transient-bond statistics for a
variable-topology simulation:

* species counts, grouped by chemical formula;
* bond counts, grouped by element pair;
* coordination-number distributions;
* optional reaction events, balanced at the formula level.

The analysis is performed on the host at the requested interval.  With a
periodic cell list the cost is linear in the number of atoms plus the number
of transient bonds; no extra CUDA kernel and no MPI communication are
involved.  A non-periodic direction uses a single cell in that direction, and a fully
non-periodic system or a slab can therefore approach O(N^2) distance checks;
increase the interval (or use a periodic box) for large non-periodic systems.  The action is usable with any GPUMD
potential, including NEP models trained for reactive systems.

Syntax
------

::

  compute_react_count <interval> <output_base> [keyword value ...]

``interval`` is a positive integer.  The analysis is performed every
``interval`` MD steps.  ``output_base`` is a path prefix; four files are
created:

::

  <output_base>.species.csv
  <output_base>.bond.csv
  <output_base>.coord.csv
  <output_base>.events.csv

The CSV files are appended when a run restarts with the same output prefix or
when a new run uses an existing prefix.  A header is written only when a file
is created.  Remove the files first if a fresh output set is wanted.  Their
columns are:

.. list-table::
   :header-rows: 1

   * - file
     - columns
   * - ``.species.csv``
     - ``step,formula,count``
   * - ``.bond.csv``
     - ``step,pair,count``
   * - ``.coord.csv``
     - ``step,element,coordination,count``
   * - ``.events.csv``
     - ``step,event_id,n_reactants,n_products,reactants,products,balanced,atom_conserved``

Bond labels use the lexicographically larger element first, for
example ``O-H`` and ``H-C``.  This is the ReaxTools convention; it is not atomic-mass
ordering.

Bond perception
---------------

For atoms :math:`i` and :math:`j` with covalent radii :math:`r_i` and
:math:`r_j`, a transient bond exists when

.. math::

   d_{ij} < \frac{1}{2}(r_i+r_j)\,\eta,

where :math:`\eta` is ``cutoff`` (default 1.2).  The comparison is strict and
is performed with the same float rounding as ReaxTools: coordinates are
converted to single precision, the minimum-image displacement is accumulated
in double precision, and the squared distance is stored as a float.

Atoms are visited in ascending atom index (which is also the GPUMD atom order
and therefore the ReaxTools atom-ID order).  Candidates are sorted by the
normalized squared distance

.. math::

   \tilde d_{ij}^2 = \frac{d_{ij}^2}{r_{ij}^2},

where the bond radius is half the sum of the two covalent radii times the
cutoff; ties are broken by atom index.  Each atom accepts its nearest
candidates first, subject to its valence.  A candidate is rejected when the
other atom is already saturated; this one-sided greedy rule reproduces
ReaxTools exactly.  ``max_neigh`` limits the candidate list before the greedy
step.

With ``perception mutual``, every atom first selects its nearest candidates
and a bond is accepted only when both atoms select each other.  This mode is
symmetric; when valence capacity is saturated, conflicts are resolved
in ascending atom-index order, so it is not strictly rank-invariant in
that limit.  It is chemically similar but not bitwise identical to
ReaxTools; use ``perception reax`` when ReaxTools parity
is required.

Species are identified by formula, for example ``C2H4O2``.  No SMILES, ring
perception or reaction graph is constructed.

Reaction events
---------------

When ``events yes`` is given, each molecule is represented by a 64-bit hash
of its atom indices and by its formula.  Between two analysis frames, atoms
whose molecule hash changes are collected, the old and new molecule hashes are
joined with a union-find, and each connected component is reported as one
event.  The reactant and product sides are written as formula multisets, for
example ``C2H6O+O2+O2``.  ``balanced`` compares the element counts and
``atom_conserved`` compares the total number of atoms; both are written as
``0`` or ``1``.

The first analysis frame only initializes the previous-frame state; no events
are emitted for it.  Events are emitted immediately; there is no ``stable``
delay.

Default element table
--------------------

Radii are in Angstrom and are used as the covalent radii above.  Elements
not in the table have radius 0 (no bonds) and valence 4.

========  ======  =======
Element   radius  valence
========  ======  =======
H          1.05        1
He         1.40        0
Li         1.82        1
Be         2.00        2
B          2.00        3
C          1.70        4
N          1.55        4
O          1.70        2
F          1.35        1
Ne         1.54        0
Na         1.36        1
Mg         1.18        2
Al         2.00        3
Si         2.10        4
P          1.80        6
S          1.80        6
Cl         2.27        7
Ar         1.88        0
K          1.76        1
Ca         1.37        2
Fe         2.00        6
Co         2.00        3
Ni         1.63        2
Cu         1.90        2
Zn         1.39        2
Br         1.85        7
I          1.98        7
========  ======  =======

Keywords
--------

``cutoff <real>``
   Distance prefactor :math:`\eta`.  Default: ``1.2``.

``element <s1,s2,...>``
   Comma-separated element symbols, in GPUMD type order.  If omitted, the
   symbols are taken from ``model.xyz``.  An explicit list is useful when the
   potential file uses a different type order.

``radius <E:r>``
   Override the covalent radius of element ``E`` in Angstrom.  May be repeated.
   Unknown elements have radius 0 and cannot form bonds.

``valence <E:v>``
   Override the maximum valence of element ``E``.  May be repeated.
   ``valence *:0`` applies the value to every element and takes precedence
   over individual E:v entries; when 0, all coordination rows collapse to 0.

``max_neigh <int>``
   Maximum number of nearest candidates considered per atom.  Default: ``10``.

``order <s1,s2,...>``
   Formula element order.  Default: ``C,H,N,O``.  Elements not listed are
   appended in alphabetical order.

``events <yes|no>``
   Write the reaction-event CSV.  Default: ``no``.

``require_balance <yes|no>``
   If ``yes``, events whose element counts do not balance are not written.
   Implies ``events yes``.  Default: ``no``.

``perception <reax|mutual>``
   Bond-perception rule.  Default: ``reax``.

``restart <filename> <interval>``
   Optional binary sidecar state file and its write interval.  ``interval``
   must be a multiple of the analysis interval and must equal the interval
   used by ``dump_restart``.  The sidecar is written only at this interval;
   it stores the event counter, the previous molecule registry, the previous
   per-atom molecule hashes, the last displayed step, and a hash of the
   coordinates/types/box.  On restart the hash must match the coordinates
   loaded from ``restart.xyz``; a mismatch is an error.  This prevents the
   analysis state from a later sample being combined with older restart
   coordinates.  The sidecar also stores the cutoff, max_neigh, perception, element symbols, radii and valences, and rejects a restart with different analysis parameters.  The file is not a GPUMD restart file and must be kept
   alongside the trajectory.  Atom order in ``model.xyz`` must be preserved.

Example
-------

::

  compute_react_count 10 react/combustion cutoff 1.2 element C,H,O order C,H,O events yes restart react/combustion.state 1000

This analyzes every 10 MD steps and writes ``react/combustion.species.csv``,
``react/combustion.bond.csv``, ``react/combustion.coord.csv`` and
``react/combustion.events.csv``.

Restrictions
------------

* Not intended for path-integral MD; the centroid position is used.
* Every atom must be present in every analysis frame.  Atom insertion or
  deletion is not modelled.
* ``restart`` requires the same atom order as the run that wrote the state and
  an interval equal to dump_restart; the coordinate hash is checked.
* The action is host-side.  For very large systems, increase ``interval`` to
  amortize the position/type copies.
