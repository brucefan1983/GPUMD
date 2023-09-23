:attr:`npt_mttk` and :attr:`nph_mttk`.
^^^^^^^^^^^

These keywords implement the Nosé-Hoover thermostat [Hoover1996]_ and the Parrinello-Rahman barostat [Parrinello1981]_. 
Both the thermostat and barostat are enhanced by the the :ref:`Nosé-Hoover chain method <nose_hoover_chain_thermostat>`, 
as recommended in the following research papers [Martyna1994]_. 
MTTK is an abbreviation for Martyna-Tuckerman-Tobias-Klein.

Depending on the input parameters, they can be used to perform simulations in different ensembles:

- Isothermal-Isobaric Ensemble (NPT)
- Isoenthalpic Ensemble (NPH)

Please note that for canonical ensemble (NVT), the barostat is turned off,
and it becomes equivalent to the Nosé-Hoover chain thermostat.
You can use the :attr:`nvt_nhc` method in GPUMD to perform NVT simulations.
Also, the style of these keywords is slightly different from other ensembles in GPUMD,
since it allows more flexible control over the ensemble.

**Parameters**

The thermostat parameters can be specified as follows:

.. code-block:: rst

    ensemble npt_mttk temp T_1 T_2 tperiod t_temp direction p_1 p_2 pperiod t_press

- `T_1` and `T_2`: Initial and final temperature. The temperature will vary linearly from `T1` to `T2` during the simulation process.
- `t_temp` (optional, default: 100): Determines the thermostat's period. The period will be `t` times the timestep. A typical value is 100.
- `direction`: One or more of the following values: `iso`, `aniso`, `tri`, `x`, `y`, `z`, `xy`, `yz`, `xz`.
- `iso`, `aniso`, and `tri` use hydrostatic pressure as the target pressure. `iso` changes the box isotropicly. `aniso` changes the dimensions of the box along x, y, and z independently. `tri` change all 6 degrees of freedom of the box.
- `x`, `y`, `z`, `xy`, `yz`, `xz`: Specify each stress component independently.
- `p_1` and `p_2`: Initial and final pressure.
- `t_press` (optional, default: 1000): Determines the barostat's period. The period will be `t` times the timestep. A typical value is 1000.

.. code-block:: rst

    ensemble nph_mttk direction p_1 p_2 pperiod t

The parameters of :attr:`nph_mttk` is the same as :attr:`npt_mttk`, 
but without thermostat parameters.

**Examples**

Here are some examples of how to use these keywords for different ensembles:

**NPT Ensemble Example:**

.. code-block:: rst

    ensemble npt_mttk temp 300 300 iso 10 10

This command set the target temperature to 300 K and the target pressure to 10 GPa.
The cell's shape will not change during the simlation, only the volume will change.
It is suitable for simualting liquid.
If not constrained, the cell shape may undergo extreme changes since liquids are not elastic.

.. code-block:: rst

    ensemble nvt_mttk temp 300 1000 iso 100 100

This command increases the simulated system's temperature from 300 K to 1000 K,
while keeps the pressure at 100 GPa.

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

Perform a NPH simualtion at 10 GPa.