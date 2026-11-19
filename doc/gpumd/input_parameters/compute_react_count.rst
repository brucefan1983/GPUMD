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

The analysis is performed on the host at the requested interval.  The cost is
linear in the number of atoms plus the number of transient bonds; no extra
CUDA kernel and no MPI communication are involved.  This makes the action
usable with any GPUMD potential, including NEP models trained for reactive
systems.

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

The CSV files are overwritten when a new run starts.  Their columns are:

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

Bond labels use the heavier/lexicographically larger element first, for
example ``O-H`` and ``H-C``.  This is the convention used by ReaxTools and by
the LAMMPS ``fix react/count`` implementation.

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
and therefore the ReaxTools atom-ID order).  Each atom accepts its nearest
candidates first, subject to its valence.  A candidate is rejected when the
other atom is already saturated; this one-sided greedy rule reproduces
ReaxTools exactly.  ``max_neigh`` limits the candidate list before the greedy
step.

With ``perception mutual``, every atom first selects its nearest candidates
and a bond is accepted only when both atoms select each other.  This mode is
symmetric and independent of atom order.  It is chemically similar but not
bitwise identical to ReaxTools; use ``perception reax`` when ReaxTools parity
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
   ``valence *:0`` disables bonding for all elements.

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

``restart <filename>``
   Optional binary sidecar state file.  It stores the event counter, the
   previous molecule registry and the previous per-atom molecule hash.  On a
   new run the file is read if it exists; events then continue from the
   previous frame, and the ``step`` column continues from the last step
   stored in the sidecar.  The file is not a GPUMD restart file and must be
   kept alongside the trajectory.  Atom order in ``model.xyz`` must be
   preserved.

Example
-------

::

  compute_react_count 10 react/combustion cutoff 1.2 element C,H,O order C,H,O events yes restart react/combustion.state

This analyzes every 10 MD steps and writes ``react/combustion.species.csv``,
``react/combustion.bond.csv``, ``react/combustion.coord.csv`` and
``react/combustion.events.csv``.

Restrictions
------------

* Not intended for path-integral MD; the centroid position is used.
* Every atom must be present in every analysis frame.  Atom insertion or
  deletion is not modelled.
* ``restart`` requires the same atom order as the run that wrote the state.
* The action is host-side.  For very large systems, increase ``interval`` to
  amortize the position/type copies.
