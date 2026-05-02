.. _kw_compute_chunk:
.. index::
   single: compute_chunk (keyword in run.in)

:attr:`compute_chunk`
=====================

This keyword computes space- and time-averaged per-chunk quantities based on dynamic spatial binning.
Unlike the :ref:`compute keyword <kw_compute>`, which uses static group assignments defined in the model file, :attr:`compute_chunk` assigns atoms to spatial bins (chunks) based on their real-time coordinates at each sampling step.
This is particularly useful for systems where atoms diffuse across spatial regions during the simulation, such as low-density, porous, or amorphous materials in NEMD simulations.

The results are written to the ``compute_chunk.out`` output file.


Syntax
------

For 1D binning::

  compute_chunk <sample_interval> <output_interval> bin/1d <dim> <origin> <delta> {<quantity>}

For 2D binning::

  compute_chunk <sample_interval> <output_interval> bin/2d <dim1> <origin1> <delta1> <dim2> <origin2> <delta2> {<quantity>}

For 3D binning::

  compute_chunk <sample_interval> <output_interval> bin/3d <dim1> <origin1> <delta1> <dim2> <origin2> <delta2> <dim3> <origin3> <delta3> {<quantity>}

The parameters are defined as follows:

* :attr:`sample_interval`: Sampling interval in time steps. The quantities are sampled every this many steps.

* :attr:`output_interval`: Number of samples to average before producing one output. This is consistent with the :ref:`compute keyword <kw_compute>`: one output is produced every :math:`\text{sample\_interval} \times \text{output\_interval}` time steps.

* :attr:`bin/1d`, :attr:`bin/2d`, :attr:`bin/3d`: The binning style, specifying 1D, 2D, or 3D spatial binning respectively.

* :attr:`dim` (or :attr:`dim1`, :attr:`dim2`, :attr:`dim3`): The axis along which to bin. Must be :attr:`x`, :attr:`y`, or :attr:`z`. For :attr:`bin/2d` and :attr:`bin/3d`, the specified axes must be distinct.

* :attr:`origin` (or :attr:`origin1`, :attr:`origin2`, :attr:`origin3`): The bin origin. Currently only :attr:`lower` is supported, meaning bins start from the lower boundary of the simulation box (coordinate 0).

* :attr:`delta` (or :attr:`delta1`, :attr:`delta2`, :attr:`delta3`): The bin width in Ångströms. The number of bins along each axis is automatically calculated as :math:`\lceil L / \delta \rceil`, where :math:`L` is the box length along that axis. The last bin may be narrower than :math:`\delta` if :math:`L` is not an integer multiple of :math:`\delta`; its volume and center coordinate are computed using the actual width.

Starting after the binning parameters, one can list the quantities to be computed.
The allowed names for :attr:`quantity` are:

* :attr:`temperature`, which yields the temperature of each chunk
* :attr:`density/number`, which yields the number density (atoms per volume) of each chunk
* :attr:`density/mass`, which yields the mass density of each chunk
* :attr:`vx`, which yields the average velocity in the x direction
* :attr:`vy`, which yields the average velocity in the y direction
* :attr:`vz`, which yields the average velocity in the z direction
* :attr:`fx`, which yields the average force in the x direction
* :attr:`fy`, which yields the average force in the y direction
* :attr:`fz`, which yields the average force in the z direction

One can write one or more (distinct) names in any order.

For triclinic (non-orthogonal) simulation boxes, the box length along each axis is computed as the geometric thickness :math:`L = V / A`, where :math:`V` is the box volume and :math:`A` is the cross-sectional area perpendicular to that axis. This is consistent with GPUMD's internal geometry calculations.


Example
-------

Example 1: 1D temperature profile along z
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  compute_chunk 10 100 bin/1d z lower 1.0 temperature

This means:

* sample the temperature every 10 time steps
* average over 100 samples before writing output (one output every :math:`10 \times 100 = 1000` steps)
* bin atoms along the z axis with 1.0 Å bin width, starting from the lower boundary
* compute the temperature of each bin

Example 2: 1D temperature and density profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  compute_chunk 10 100 bin/1d z lower 1.0 temperature density/number

Same as above, but also computes the number density of each bin.

Example 3: 2D velocity field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  compute_chunk 10 100 bin/2d x lower 2.0 z lower 2.0 vx vz

This creates a 2D grid of bins in the x-z plane with 2.0 Å bin width in each direction, and computes the average x and z velocity components in each bin.

Example 4: NEMD temperature profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A typical use case for NEMD simulations::

  ensemble heat_lan 300 300 50 0 1
  compute_chunk 10 100 bin/1d z lower 1.0 temperature density/number
  run 1000000

This computes the temperature and density profiles along z during a Langevin heat bath NEMD simulation.


Output file
-----------

The results are written to ``compute_chunk.out``.
The file is appended to if it already exists.
For each output time, there are :math:`N_\text{chunk}` consecutive lines (one per chunk), with the following format per line::

  <chunk_id> <coord1> [<coord2>] [<coord3>] <avg_count> <value1> [<value2>] ...

The columns are:

* :attr:`chunk_id`: the zero-based index of the chunk (dimensionless integer).
* :attr:`coord1` (and :attr:`coord2`, :attr:`coord3` for 2D/3D binning): the center coordinates (in units of Å) of the chunk along the corresponding binning axis.
* :attr:`avg_count`: the time-averaged number of atoms in the chunk (dimensionless).
* :attr:`value1`, :attr:`value2`, ...: the time-averaged values of the requested quantities, in the same order as specified in the command. The units for each quantity are:

  - :attr:`temperature`: K
  - :attr:`density/number`: Å\ :sup:`-3` (number of atoms per volume)
  - :attr:`density/mass`: amu/Å\ :sup:`3` (total atomic mass per volume)
  - :attr:`vx`, :attr:`vy`, :attr:`vz`: Å/fs
  - :attr:`fx`, :attr:`fy`, :attr:`fz`: eV/Å

The temperature is computed from the kinetic energy of the atoms in each chunk as :math:`T = 2 E_\text{k} / (3 k_\text{B} N)`, where :math:`E_\text{k}` is the total kinetic energy, :math:`N` is the atom count, and :math:`k_\text{B}` is the Boltzmann constant.
The velocity and force values are per-atom averages within each chunk.

For example, the command::

  compute_chunk 10 100 bin/1d z lower 2.0 temperature density/number

produces output lines of the form::

  0 1.000000 52.0 3.0012345678e+02 2.5000000000e-02
  1 3.000000 48.0 2.9876543210e+02 2.4000000000e-02
  ...

Here, each line contains: chunk index, bin center along z (Å), average atom count, temperature (K), and number density (Å\ :sup:`-3`).
The output blocks for successive time points are written consecutively with no blank line separator; each block contains exactly :math:`N_\text{chunk}` lines.


Caveats
-------

* The :attr:`origin` parameter currently only accepts :attr:`lower`. Numeric origin values are not supported.
* This keyword addresses the measurement/analysis side of dynamic spatial binning. The NEMD heat baths (:attr:`ensemble heat_*` source/sink regions) still use static group assignments. Dynamic heat bath regions would require separate modifications to the integrator.
* Multiple :attr:`compute_chunk` commands can be used in the same run.
