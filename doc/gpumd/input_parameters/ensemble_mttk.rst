:attr:`nvt_mttk`, :attr:`npt_mttk`, and :attr:`nph_mttk`.
^^^^^^^^^^^

These keywords implement the Nosé-Hoover thermostat [Hoover1996]_ and the Parrinello-Rahman barostat [Parrinello1981]_. 
Both the thermostat and barostat are enhanced by the the :ref:`Nose-Hoover chain method <nose_hoover_chain_thermostat>`., 
as recommended in the following research papers [Martyna1994]_.

Depending on the input parameters, they can be used to perform simulations in different ensembles:

- Canonical Ensemble (NVT)
- Isothermal-Isobaric Ensemble (NPT)
- Isoenthalpic Ensemble (NPH)

Please note that currently there is another :attr:`nvt_nhc` method in GPUMD.
For historic reasons, :attr:`nvt_nhc` is equivalent to :attr:`nvt_mttk`. 
You may choose either of them to perform your simulation.

**Thermostat Parameters**

The thermostat parameters can be specified as follows:

.. code-block:: rst

    ensemble nvt_mttk temp T_1 T_2 tperiod t

- `T_1` and `T_2`: Initial and final temperature. The temperature will vary linearly from `T1` to `T2` during the simulation process.
- `t` (optional, default: 100): Determines the thermostat's period. The period will be `t` times the timestep. A typical value is 100.

**Barostat Parameters**

The barostat parameters can be specified as follows:

.. code-block:: rst

    ensemble npt_mttk direction p_1 p_2 pperiod t

- `direction`: One or more of the following values: `iso`, `aniso`, `tri`, `x`, `y`, `z`, `xy`, `yz`, `xz`.
- `iso`, `aniso`, and `tri` use hydrostatic pressure as the target pressure. `iso` changes the box isotropicly. `aniso` changes the dimensions of the box along x, y, and z independently. `tri` change all 6 degrees of freedom of the box.
- `x`, `y`, `z`, `xy`, `yz`, `xz`: Specify each stress component independently.
- `p_1` and `p_2`: Initial and final pressure.
- `t` (optional, default: 1000): Determines the barostat's period. The period will be `t` times the timestep. A typical value is 1000.

**Examples**

Here are some examples of how to use these keywords for different ensembles:

**NVT Ensemble Example:**

.. code-block:: rst

    ensemble nvt_mttk temp 300 300

This command set the target temperature to 300 K.

.. code-block:: rst

    ensemble nvt_mttk temp 300 1000

This command increases the simulated system's temperature from 300 K to 1000 K.

**NPT Ensemble Example:**

.. code-block:: rst

    ensemble npt_mttk temp 300 300 iso 10 10

This command set the target temperature to 300 K and the target pressure to 10 GPa.
The cell's shape will not change during the simlation, only the volume will change.
It is suitable for simualting liquid.
If not constrained, the cell shape may undergo extreme changes since liquids are not elastic.

**NPT Ensemble Example:**

.. code-block:: rst

    ensemble npt_mttk temp 300 300 aniso 10 10

This command replaces `iso` with `ansio`. The three dimensions of the cell can change independently,
but `xy`, `xz` and `yz` will not be changed.

.. code-block:: rst

    ensemble npt_mttk temp 300 300 tri 10 10

All six degrees of freedom are allowed to change. The simulated system will converge to fully hydrostatic pressure. 
Note that with `iso` and `aniso`, there will be no guarantee that the pressure is hydrostatic, 
as the system is constrained.

.. code-block:: rst

    ensemble npt_mttk temp 300 300 x 5 5 y 0 0 z 0 0

Apply 5 GPa to x direction, and 0 GPa to y and z directions.

.. code-block:: rst

    ensemble npt_mttk temp 300 300 x 5 5

Apply 5 GPa to x direction but fix other directions.

**NPH Ensemble Example:**

.. code-block:: rst

    ensemble nph_mttk iso 10 10

NPH is the same as NPT but without thermostat parameters.