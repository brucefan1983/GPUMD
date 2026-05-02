#!/usr/bin/env python3
"""
VASP MDI Driver for GPUMD

This script acts as an MDI DRIVER that controls a GPUMD ENGINE.
It launches VASP for QM calculations and communicates forces back to GPUMD.
"""

import sys
import os
import argparse
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import numpy as np

try:
    import mdi
except ImportError:
    print("ERROR: MDI Library not found. Install with: pip install mdi")
    sys.exit(1)

import logging.handlers

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

log_dir = ".gpumd_vasp_tmp"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "vasp_driver.log")
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
)
file_handler.setFormatter(file_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger("VASP_MDI_Driver")
logger.info(f"Logging to file: {log_file}")

KBAR_TO_EV_PER_ANG3 = 0.00062415091


class VASPCalculator:
    """Wrapper around VASP calculations."""

    def __init__(self, vasp_cmd, poscar_template, timeout=3600):
        self.vasp_cmd = vasp_cmd
        self.poscar_template_path = poscar_template
        self.timeout = timeout
        self.natoms = 0
        self.elements = []
        self.natoms_per_type = []
        self.ntypes = 0

        if not os.path.exists(poscar_template):
            raise FileNotFoundError(f"POSCAR_template not found: {poscar_template}")

        self._parse_template(poscar_template)
        logger.info(
            f"Initialized VASP calculator: {self.natoms} atoms, types={self.elements}"
        )

    def _parse_template(self, poscar_path):
        """Parse POSCAR_template to get atom counts and types."""
        try:
            with open(poscar_path, "r") as f:
                lines = f.readlines()

            self.elements = lines[5].split()
            self.natoms_per_type = [int(x) for x in lines[6].split()]
            self.ntypes = len(self.natoms_per_type)
            self.natoms = sum(self.natoms_per_type)

            logger.info(f"Parsed POSCAR: {self.natoms} atoms, types={self.elements}")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to parse POSCAR_template: {e}")

    def get_cell_from_template(self, poscar_path=None):
        """Extract cell parameters from POSCAR template."""
        if poscar_path is None:
            poscar_path = self.poscar_template_path

        try:
            with open(poscar_path, "r") as f:
                lines = f.readlines()

            scaling = float(lines[1].strip())
            cell = []
            for i in range(2, 5):
                cell.extend([float(x) * scaling for x in lines[i].split()])

            return np.array(cell)
        except (IndexError, ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to get cell from POSCAR: {e}")
            return np.array([20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0])

    def write_poscar(self, coords, cell_params, atom_types=None):
        """Write POSCAR file from atomic coordinates and cell."""
        if len(coords) != self.natoms * 3:
            raise ValueError(
                f"Coordinate mismatch: expected {self.natoms * 3}, got {len(coords)}"
            )

        coords = np.array(coords, dtype=np.float64)
        cell_params = np.array(cell_params, dtype=np.float64)

        logger.debug(
            f"write_poscar(): Input cell_params shape={cell_params.shape}, dtype={cell_params.dtype}"
        )

        if cell_params.shape == (9,):
            cell_3x3 = cell_params.reshape(3, 3)
        elif cell_params.shape == (3, 3):
            cell_3x3 = cell_params
        else:
            raise ValueError(f"Invalid cell shape: {cell_params.shape}")

        cell_volume = np.linalg.det(cell_3x3)
        logger.debug(f"write_poscar(): Cell determinant={cell_volume:.10e}")

        if abs(cell_volume) < 1e-10:
            logger.error(f"Cell volume is zero or near-zero: {cell_volume}")
            raise ValueError(f"Invalid cell: volume = {cell_volume}")

        logger.info(f"Cell volume: {cell_volume:.6f} A^3")

        with open(self.poscar_template_path, "r") as f:
            template_lines = f.readlines()

        poscar_lines = []
        poscar_lines.append(template_lines[0])
        poscar_lines.append(template_lines[1])

        for i in range(3):
            poscar_lines.append(
                f"  {cell_3x3[i, 0]:15.10f} {cell_3x3[i, 1]:15.10f} {cell_3x3[i, 2]:15.10f}\n"
            )

        poscar_lines.append(template_lines[5])
        poscar_lines.append(template_lines[6])
        poscar_lines.append("Cartesian\n")

        coords_3d = coords.reshape(self.natoms, 3)

        for i in range(self.natoms):
            x, y, z = coords_3d[i]
            poscar_lines.append(f"  {x:15.10f} {y:15.10f} {z:15.10f}\n")

        poscar_path = "POSCAR"
        with open(poscar_path, "w") as f:
            f.writelines(poscar_lines)

        logger.info(f"Wrote POSCAR file")
        return poscar_path

    def run_vasp(self):
        """Execute VASP and wait for completion."""
        logger.info(f"Running VASP: {self.vasp_cmd}")

        try:
            result = subprocess.run(
                self.vasp_cmd,
                shell=True,
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"VASP exited with code {result.returncode}")
                logger.warning(f"STDOUT: {result.stdout[-500:]}")
                logger.warning(f"STDERR: {result.stderr[-500:]}")
                return False

            logger.info("VASP completed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"VASP timeout after {self.timeout} seconds")
            return False

    def read_vasprun(self):
        """Parse vasprun.xml to extract energy, forces, and stress."""
        vasprun_path = "vasprun.xml"

        if not os.path.exists(vasprun_path):
            logger.error("vasprun.xml not found")
            return None, None, None, False

        try:
            tree = ET.parse(vasprun_path)
            root = tree.getroot()

            calcs = root.findall("calculation")
            if not calcs:
                logger.error("No calculation found in vasprun.xml")
                return None, None, None, False

            calc = calcs[-1]

            energy_elem = calc.find("energy/e_0_energy")
            if energy_elem is None:
                energy_elem = calc.find("energy/i[@name='e_0_sigma->0 VASP']")
            if energy_elem is None:
                energy_elem = calc.find("energy/i")
            if energy_elem is None:
                logger.error("Energy tag not found in vasprun.xml")
                return None, None, None, False

            energy = float(energy_elem.text)
            logger.info(f"Extracted energy: {energy:.6f} eV")

            forces_elem = calc.find("varray[@name='forces']")
            if forces_elem is None:
                logger.error("Forces array not found in vasprun.xml")
                return None, None, None, False

            forces = []
            for v in forces_elem.findall("v"):
                force = [float(x) for x in v.text.split()]
                forces.append(force)

            forces = np.array(forces).reshape(-1)
            logger.info(f"Extracted {len(forces) // 3} force vectors")

            stress_elem = calc.find("varray[@name='stress']")
            if stress_elem is not None:
                stress = []
                for v in stress_elem.findall("v"):
                    stress_row = [float(x) for x in v.text.split()]
                    stress.append(stress_row)
                stress = np.array(stress) * KBAR_TO_EV_PER_ANG3
                logger.info(f"Extracted stress tensor")
            else:
                stress = np.zeros((3, 3))

            converged = True
            return energy, forces, stress, converged

        except ET.ParseError as e:
            logger.error(f"Failed to parse vasprun.xml: {e}")
            return None, None, None, False

    def run_step(self, coords, cell_params, atom_types=None):
        """Run a single VASP calculation step."""
        try:
            coords = np.array(coords, dtype=np.float64)
            cell_params = np.array(cell_params, dtype=np.float64)

            if len(coords) != self.natoms * 3:
                logger.error(
                    f"Coordinate size mismatch: {len(coords)} vs expected {self.natoms * 3}"
                )
                return 0.0, np.zeros(self.natoms * 3), np.zeros((3, 3))

            self.write_poscar(coords, cell_params, atom_types)

        except Exception as e:
            logger.error(f"Failed to write POSCAR: {e}")
            import traceback

            traceback.print_exc()
            return 0.0, np.zeros(self.natoms * 3), np.zeros((3, 3))

        success = self.run_vasp()
        if not success:
            logger.warning("VASP calculation failed, returning zeros")
            return 0.0, np.zeros(self.natoms * 3), np.zeros((3, 3))

        energy, forces, stress, converged = self.read_vasprun()

        if not converged or forces is None:
            logger.warning("VASP did not converge, returning zeros")
            return 0.0, np.zeros(self.natoms * 3), np.zeros((3, 3))

        return energy, forces, stress


class VASPMDIDriver:
    """MDI DRIVER that controls a GPUMD ENGINE."""

    def __init__(self, vasp_calc, port=8021, steps=100):
        self.vasp = vasp_calc
        self.port = port
        self.steps = steps
        self.mdi_comm = None
        self.natoms = vasp_calc.natoms

    def initialize_mdi(self):
        """Initialize MDI as DRIVER and connect to GPUMD ENGINE."""
        mdi_options = f"-role DRIVER -name vasp_driver -method TCP -port {self.port}"
        logger.info(f"Initializing MDI with: {mdi_options}")

        try:
            mdi.MDI_Init(mdi_options)
            logger.info("MDI initialized as DRIVER")
        except Exception as e:
            logger.error(f"MDI_Init failed: {e}")
            raise

        logger.info("Requesting communicator to 'gpumd' engine...")
        try:
            self.mdi_comm = mdi.MDI_Accept_Communicator()
            logger.info(f"Accepted communicator: {self.mdi_comm}")
        except Exception as e:
            logger.error(f"Failed to accept communicator: {e}")
            raise

    def run_md_loop(self):
        """Main MDI communication loop."""
        if self.mdi_comm is None:
            raise RuntimeError("MDI not initialized. Call initialize_mdi() first.")

        logger.info(f"Starting MD loop for {self.steps} steps")

        try:
            logger.info("Requesting natoms from GPUMD")
            mdi.MDI_Send_Command("<NATOMS", self.mdi_comm)
            natoms_result = mdi.MDI_Recv(1, mdi.MDI_INT, self.mdi_comm)
            if isinstance(natoms_result, (list, tuple)):
                natoms_gpumd = natoms_result[0]
            else:
                natoms_gpumd = natoms_result
            logger.info(f"GPUMD reports {natoms_gpumd} atoms (VASP has {self.natoms})")

            if natoms_gpumd != self.natoms:
                logger.error(
                    f"Atom count mismatch: GPUMD={natoms_gpumd}, VASP={self.natoms}"
                )
                return

            for step in range(self.steps):
                logger.info(f"\n--- Step {step + 1}/{self.steps} ---")

                try:
                    logger.info("Requesting coordinates from GPUMD")
                    mdi.MDI_Send_Command("<COORDS", self.mdi_comm)
                    coords = mdi.MDI_Recv(
                        self.natoms * 3, mdi.MDI_DOUBLE, self.mdi_comm
                    )

                    logger.info(f"Received {len(coords)} coordinates")

                    cell_params = self.vasp.get_cell_from_template()

                    if not isinstance(coords, np.ndarray):
                        coords = np.array(coords, dtype=np.float64)
                    else:
                        coords = coords.astype(np.float64)

                    if not isinstance(cell_params, np.ndarray):
                        cell_params = np.array(cell_params, dtype=np.float64)
                    else:
                        cell_params = cell_params.astype(np.float64)

                    if np.allclose(coords, 0.0):
                        logger.error("ALL COORDINATES ARE ZERO!")

                    cell_3x3 = cell_params.reshape(3, 3)
                    cell_volume = abs(np.linalg.det(cell_3x3))
                    logger.info(f"Cell volume (from template): {cell_volume:.10f} A^3")

                    logger.info("Running VASP calculation")
                    energy, forces, stress = self.vasp.run_step(coords, cell_params)
                    logger.info(
                        f"VASP returned: E={energy:.6f} eV, F_norm={np.linalg.norm(forces):.6f}"
                    )
                    stress_total = np.array(stress, dtype=np.float64)

                    logger.info("Sending energy to GPUMD via >ENERGY")
                    mdi.MDI_Send_Command(">ENERGY", self.mdi_comm)
                    mdi.MDI_Send(np.array([energy]), 1, mdi.MDI_DOUBLE, self.mdi_comm)
                    logger.info("Sending stress to GPUMD via >STRESS")
                    mdi.MDI_Send_Command(">STRESS", self.mdi_comm)
                    mdi.MDI_Send(
                        stress_total.reshape(9), 9, mdi.MDI_DOUBLE, self.mdi_comm
                    )
                    logger.info("Sending forces to GPUMD via >FORCES")
                    mdi.MDI_Send_Command(">FORCES", self.mdi_comm)
                    mdi.MDI_Send(forces, self.natoms * 3, mdi.MDI_DOUBLE, self.mdi_comm)

                    logger.info(f"Step {step + 1} complete")

                except Exception as e:
                    logger.error(f"Error in step {step + 1}: {e}")
                    import traceback

                    traceback.print_exc()
                    try:
                        mdi.MDI_Send_Command(">ENERGY", self.mdi_comm)
                        mdi.MDI_Send(np.array([0.0]), 1, mdi.MDI_DOUBLE, self.mdi_comm)
                        mdi.MDI_Send_Command(">STRESS", self.mdi_comm)
                        mdi.MDI_Send(np.zeros(9), 9, mdi.MDI_DOUBLE, self.mdi_comm)
                        logger.warning(
                            "Sending zero forces to GPPUD via >FORCES due to error"
                        )
                        mdi.MDI_Send_Command(">FORCES", self.mdi_comm)
                        mdi.MDI_Send(
                            np.zeros(self.natoms * 3),
                            self.natoms * 3,
                            mdi.MDI_DOUBLE,
                            self.mdi_comm,
                        )
                    except Exception as e2:
                        logger.error(f"Could not send error response to GPUMD: {e2}")
                    if step < self.steps - 1:
                        logger.warning("Continuing to next step despite error")
                    else:
                        logger.error("Last step failed, aborting")
                        break

            logger.info("Sending EXIT command to GPUMD")
            mdi.MDI_Send_Command("EXIT", self.mdi_comm)

        except Exception as e:
            logger.error(f"Error in MD loop: {e}")
            raise

        logger.info("MD loop completed successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VASP MDI Driver for GPUMD QM/MM hybrid simulations"
    )
    parser.add_argument(
        "--vasp-cmd", default="vasp_std", help="VASP executable command"
    )
    parser.add_argument(
        "--poscar-template", default="POSCAR_template", help="Path to POSCAR_template"
    )
    parser.add_argument("--port", type=int, default=8021, help="MDI TCP port")
    parser.add_argument("--steps", type=int, default=100, help="Number of MD steps")
    parser.add_argument(
        "--timeout", type=int, default=3600, help="VASP timeout in seconds"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("VASP MDI Driver for GPUMD")
    logger.info("=" * 70)
    logger.info(f"VASP command: {args.vasp_cmd}")
    logger.info(f"POSCAR template: {args.poscar_template}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Steps: {args.steps}")

    try:
        logger.info("\nInitializing VASP calculator...")
        vasp_calc = VASPCalculator(args.vasp_cmd, args.poscar_template, args.timeout)

        logger.info("\nInitializing MDI driver...")
        driver = VASPMDIDriver(vasp_calc, port=args.port, steps=args.steps)
        driver.initialize_mdi()

        logger.info("\nStarting MD loop...")
        driver.run_md_loop()

        logger.info("\n" + "=" * 70)
        logger.info("SUCCESS: VASP MDI Driver completed")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\nFATAL ERROR: {e}")
        logger.error("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
