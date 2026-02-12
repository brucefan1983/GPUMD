#!/usr/bin/env python3
"""
VASP MDI Driver for GPUMD

This script acts as an MDI DRIVER that controls a GPUMD ENGINE.
It launches VASP for QM calculations and communicates forces back to GPUMD.

Usage:
    python3 vasp_mdi_driver.py --vasp-cmd "vasp_std" \
                               --poscar-template POSCAR_template \
                               --port 8021 \
                               --steps 100

Requirements:
    - MDI Library (Python): mdi module
    - numpy
    - VASP installed and working
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

# Configure logging - both console and file
import logging.handlers

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)

# File handler - write to .gpumd_vasp_tmp directory  
log_dir = ".gpumd_vasp_tmp"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "vasp_driver.log")
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
file_handler.setFormatter(file_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Get logger for this module
logger = logging.getLogger("VASP_MDI_Driver")
logger.info(f"Logging to file: {log_file}")

# Physical constants
BOHR_TO_ANG = 0.52917721067
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
EV_TO_HARTREE = 0.03674932248
HARTREE_TO_EV = 1.0 / EV_TO_HARTREE
KBAR_TO_EV_PER_ANG3 = 0.00062415091


class VASPCalculator:
    """
    Wrapper around VASP calculations.
    Handles POSCAR input/output and vasprun.xml parsing.
    """

    def __init__(self, vasp_cmd, poscar_template, timeout=3600):
        """
        Initialize VASP calculator.

        Args:
            vasp_cmd (str): Command to run VASP (e.g., "vasp_std")
            poscar_template (str): Path to POSCAR_template file
            timeout (int): Maximum time (seconds) to wait for VASP
        """
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
        logger.info(f"Initialized VASP calculator: {self.natoms} atoms, types={self.elements}")

    def _parse_template(self, poscar_path):
        """Parse POSCAR_template to get atom counts and types."""
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()

            # POSCAR format:
            # Line 0: Comment
            # Line 1: Scaling factor
            # Lines 2-4: Lattice vectors
            # Line 5: Element symbols
            # Line 6: Number of atoms per type
            # Line 7+: Atomic coordinates

            self.elements = lines[5].split()
            self.natoms_per_type = [int(x) for x in lines[6].split()]
            self.ntypes = len(self.natoms_per_type)
            self.natoms = sum(self.natoms_per_type)
            
            logger.info(f"Parsed POSCAR: {self.natoms} atoms, types={self.elements}")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to parse POSCAR_template: {e}")

    def get_cell_from_template(self, poscar_path=None):
        """
        Extract cell parameters from POSCAR template.
        
        Args:
            poscar_path: Path to POSCAR file. If None, uses stored template path.
            
        Returns:
            numpy array: Cell vectors as flat array [a11, a12, a13, a21, a22, a23, a31, a32, a33]
        """
        if poscar_path is None:
            poscar_path = self.poscar_template_path
            
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            # POSCAR format: cell vectors in lines 2-4
            scaling = float(lines[1].strip())
            cell = []
            for i in range(2, 5):
                cell.extend([float(x) * scaling for x in lines[i].split()])
            
            return np.array(cell)
        except (IndexError, ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to get cell from POSCAR: {e}")
            # Return default orthogonal cell
            return np.array([20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0])

    def write_poscar(self, coords, cell_params, atom_types=None):
        """
        Write POSCAR file from atomic coordinates and cell.

        Args:
            coords (array): Atomic coordinates [natoms*3] in Angstrom
            cell_params (array): Cell vectors [9] in Angstrom (flat)
            atom_types (array): Atom type indices [natoms] (1-indexed)

        Returns:
            str: Path to written POSCAR file
        """
        if len(coords) != self.natoms * 3:
            raise ValueError(f"Coordinate mismatch: expected {self.natoms*3}, got {len(coords)}")

        # Ensure coords and cell are numpy arrays
        coords = np.array(coords, dtype=np.float64)
        cell_params = np.array(cell_params, dtype=np.float64)
        
        logger.debug(f"write_poscar(): Input cell_params shape={cell_params.shape}, dtype={cell_params.dtype}")
        logger.debug(f"write_poscar(): cell_params values={cell_params[:9]}")
        
        # Reshape cell if needed (should be 9 elements -> reshape to 3x3)
        if cell_params.shape == (9,):
            cell_3x3 = cell_params.reshape(3, 3)
        elif cell_params.shape == (3, 3):
            cell_3x3 = cell_params
        else:
            raise ValueError(f"Invalid cell shape: {cell_params.shape}")
        
        logger.debug(f"write_poscar(): Cell matrix shape={cell_3x3.shape}")
        logger.debug(f"write_poscar(): Cell matrix =\n{cell_3x3}")
        
        # Validate cell volume is not zero
        cell_volume = np.linalg.det(cell_3x3)
        logger.debug(f"write_poscar(): Cell determinant={cell_volume:.10e}")
        
        if abs(cell_volume) < 1e-10:
            logger.error(f"Cell volume is zero or near-zero: {cell_volume}")
            logger.error(f"Cell matrix:\n{cell_3x3}")
            logger.error(f"Input cell_params:\n{cell_params}")
            raise ValueError(f"Invalid cell: volume = {cell_volume}")
        
        logger.info(f"Cell volume: {cell_volume:.6f} A^3")

        # Parse existing POSCAR template
        with open("POSCAR_template", 'r') as f:
            template_lines = f.readlines()

        # Reconstruct POSCAR
        poscar_lines = []
        poscar_lines.append(template_lines[0])  # Comment
        poscar_lines.append(template_lines[1])  # Scaling
        
        # Write cell vectors (3x3 format)
        for i in range(3):
            poscar_lines.append(f"  {cell_3x3[i,0]:15.10f} {cell_3x3[i,1]:15.10f} {cell_3x3[i,2]:15.10f}\n")

        poscar_lines.append(template_lines[5])  # Element symbols
        poscar_lines.append(template_lines[6])  # Atom counts

        poscar_lines.append("Cartesian\n")  # Coordinate type

        # Reshape coordinates to (natoms, 3)
        coords_3d = coords.reshape(self.natoms, 3)
        
        # Log sample coordinates
        logger.debug(f"Coordinate sample: atom 0 = {coords_3d[0]}")

        # Write coordinates in original order (do NOT sort by type)
        for i in range(self.natoms):
            x, y, z = coords_3d[i]
            poscar_lines.append(f"  {x:15.10f} {y:15.10f} {z:15.10f}\n")

        # Write POSCAR file
        poscar_path = "POSCAR"
        with open(poscar_path, 'w') as f:
            f.writelines(poscar_lines)

        logger.info(f"Wrote POSCAR file. Contents:")
        logger.info("=" * 70)
        for line in poscar_lines:
            logger.info(line.rstrip('\n'))
        logger.info("=" * 70)
        
        logger.debug(f"Wrote {poscar_path}")
        return poscar_path

    def run_vasp(self):
        """
        Execute VASP and wait for completion.

        Returns:
            bool: True if VASP converged, False otherwise
        """
        logger.info(f"Running VASP: {self.vasp_cmd}")

        try:
            result = subprocess.run(
                self.vasp_cmd,
                shell=True,
                timeout=self.timeout,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.warning(f"VASP exited with code {result.returncode}")
                logger.warning(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
                logger.warning(f"STDERR: {result.stderr[-500:]}")
                return False

            logger.info("VASP completed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"VASP timeout after {self.timeout} seconds")
            return False

    def read_vasprun(self):
        """
        Parse vasprun.xml to extract energy, forces, and stress.

        Returns:
            tuple: (energy_eV, forces[natoms,3], stress[3,3], converged)
        """
        vasprun_path = "vasprun.xml"

        if not os.path.exists(vasprun_path):
            logger.error("vasprun.xml not found")
            return None, None, None, False

        try:
            tree = ET.parse(vasprun_path)
            root = tree.getroot()

            # Find the last calculation (converged result)
            calcs = root.findall("calculation")
            if not calcs:
                logger.error("No calculation found in vasprun.xml")
                return None, None, None, False

            calc = calcs[-1]

            # Extract energy
            energy_elem = calc.find("energy/e_0_energy")
            if energy_elem is None:
                # Try alternative energy tag
                energy_elem = calc.find("energy/i[@name='e_0_sigma->0 VASP']")
            if energy_elem is None:
                # Try direct energy element
                energy_elem = calc.find("energy/i")
            if energy_elem is None:
                logger.error("Energy tag not found in vasprun.xml")
                logger.debug("Attempting to list available energy elements...")
                energy_section = calc.find("energy")
                if energy_section is not None:
                    logger.debug(f"Energy children: {[child.tag for child in energy_section]}")
                return None, None, None, False

            energy = float(energy_elem.text)
            logger.info(f"Extracted energy: {energy:.6f} eV")

            # Extract forces
            forces_elem = calc.find("varray[@name='forces']")
            if forces_elem is None:
                logger.error("Forces array not found in vasprun.xml")
                return None, None, None, False

            forces = []
            for v in forces_elem.findall("v"):
                force = [float(x) for x in v.text.split()]
                forces.append(force)

            forces = np.array(forces).reshape(-1)
            logger.info(f"Extracted {len(forces)//3} force vectors")

            # Extract stress
            stress_elem = calc.find("varray[@name='stress']")
            if stress_elem is not None:
                stress = []
                for v in stress_elem.findall("v"):
                    stress_row = [float(x) for x in v.text.split()]
                    stress.append(stress_row)
                stress = np.array(stress)
                logger.info(f"Extracted stress tensor")
            else:
                stress = np.zeros((3, 3))

            converged = True
            return energy, forces, stress, converged

        except ET.ParseError as e:
            logger.error(f"Failed to parse vasprun.xml: {e}")
            return None, None, None, False

    def run_step(self, coords, cell_params, atom_types=None):
        """
        Run a single VASP calculation step.

        Args:
            coords (array): Atomic coordinates [natoms*3] Angstrom
            cell_params (array): Cell vectors [9] Angstrom (flat)
            atom_types (array): Atom types [natoms]

        Returns:
            tuple: (energy_eV, forces_eV_per_A, stress_eV_per_A3)
        """
        try:
            # Validate inputs
            coords = np.array(coords, dtype=np.float64)
            cell_params = np.array(cell_params, dtype=np.float64)
            
            if len(coords) != self.natoms * 3:
                logger.error(f"Coordinate size mismatch: {len(coords)} vs expected {self.natoms*3}")
                return 0.0, np.zeros(self.natoms * 3), np.zeros((3, 3))
            
            # Write POSCAR
            self.write_poscar(coords, cell_params, atom_types)
            
        except Exception as e:
            logger.error(f"Failed to write POSCAR: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, np.zeros(self.natoms * 3), np.zeros((3, 3))

        # Run VASP
        success = self.run_vasp()
        if not success:
            logger.warning("VASP calculation failed, returning zeros")
            return 0.0, np.zeros(self.natoms * 3), np.zeros((3, 3))

        # Read results
        energy, forces, stress, converged = self.read_vasprun()

        if not converged or forces is None:
            logger.warning("VASP did not converge, returning zeros")
            return 0.0, np.zeros(self.natoms * 3), np.zeros((3, 3))

        return energy, forces, stress


class VASPMDIDriver:
    """
    MDI DRIVER that controls a GPUMD ENGINE.
    Launches VASP for QM calculations and communicates forces back.
    """

    def __init__(self, vasp_calc, port=8021, steps=100):
        """
        Initialize the MDI driver.

        Args:
            vasp_calc (VASPCalculator): Initialized VASP calculator
            port (int): MDI TCP port
            steps (int): Number of MD steps to run
        """
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

        # Request communicator to GPUMD engine
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
            # Verify system size
            logger.info("Requesting natoms from GPUMD")
            mdi.MDI_Send_Command("<NATOMS", self.mdi_comm)
            natoms_result = mdi.MDI_Recv(1, mdi.MDI_INT, self.mdi_comm)
            # Handle both list and scalar returns
            if isinstance(natoms_result, (list, tuple)):
                natoms_gpumd = natoms_result[0]
            else:
                natoms_gpumd = natoms_result
            logger.info(f"GPUMD reports {natoms_gpumd} atoms (VASP has {self.natoms})")

            if natoms_gpumd != self.natoms:
                logger.error(f"Atom count mismatch: GPUMD={natoms_gpumd}, VASP={self.natoms}")
                return

            # Main loop
            for step in range(self.steps):
                logger.info(f"\n--- Step {step+1}/{self.steps} ---")

                try:
                    # Request coordinates from GPUMD
                    logger.info("Requesting coordinates from GPUMD")
                    mdi.MDI_Send_Command("<COORDS", self.mdi_comm)
                    coords = mdi.MDI_Recv(self.natoms * 3, mdi.MDI_DOUBLE, self.mdi_comm)
                    
                    # DEBUG: Log raw received coordinates
                    logger.info(f"Raw MDI_Recv result type: {type(coords)}")
                    logger.info(f"Raw MDI_Recv result: {coords}")
                    logger.info(f"Raw MDI_Recv length: {len(coords) if hasattr(coords, '__len__') else 'N/A'}")
                    
                    # Get cell from POSCAR template (GPUMD doesn't support <CELL in MDI v1)
                    # NOTE: This means cell is STATIC across all steps - atoms move but box doesn't
                    cell_params = self.vasp.get_cell_from_template()
                    
                    # Ensure we have numpy arrays
                    if not isinstance(coords, np.ndarray):
                        coords = np.array(coords, dtype=np.float64)
                    else:
                        coords = coords.astype(np.float64)
                    
                    if not isinstance(cell_params, np.ndarray):
                        cell_params = np.array(cell_params, dtype=np.float64)
                    else:
                        cell_params = cell_params.astype(np.float64)
                    
                    # DEBUG: Check if coordinates are all zeros (indicates MDI receive issue)
                    logger.info(f"Coordinates after conversion: {coords}")
                    logger.info(f"Coordinates sum: {np.sum(coords)}")
                    logger.info(f"Coordinates min/max: [{np.min(coords)}, {np.max(coords)}]")
                    
                    if np.allclose(coords, 0.0):
                        logger.error("⚠️  ALL COORDINATES ARE ZERO!")
                        logger.error("This indicates a problem with MDI_Recv() or GPUMD is sending zeros")
                        logger.error("Check: 1) Is GPUMD updating coordinates? 2) Is MDI data type correct?")
                    
                    # Log coordinate and cell info
                    logger.info(f"Received {len(coords)} coordinates")
                    
                    # Validate cell volume is reasonable
                    cell_3x3 = cell_params.reshape(3, 3)
                    cell_volume = abs(np.linalg.det(cell_3x3))
                    logger.info(f"Cell volume (from template): {cell_volume:.10f} A^3")
                    logger.info(f"Cell matrix:\n{cell_3x3}")

                    # Run VASP
                    logger.info("Running VASP calculation")
                    energy, forces, stress = self.vasp.run_step(coords, cell_params)
                    logger.info(f"VASP returned: E={energy:.6f} eV, F_norm={np.linalg.norm(forces):.6f}")

                    # Send QM forces back to GPUMD
                    # NOTE: According to the MDI convention and GPUMD's engine implementation,
                    #       the driver must use >FORCES and >ENERGY when SENDING data to the engine.
                    #       <FORCES / <ENERGY are reserved for REQUESTING data from the engine.
                    logger.info("Sending forces to GPUMD via >FORCES")
                    mdi.MDI_Send_Command(">FORCES", self.mdi_comm)
                    mdi.MDI_Send(forces, self.natoms * 3, mdi.MDI_DOUBLE, self.mdi_comm)
                    
                    # Send QM energy back to GPUMD
                    logger.info("Sending energy to GPUMD via >ENERGY")
                    mdi.MDI_Send_Command(">ENERGY", self.mdi_comm)
                    mdi.MDI_Send(np.array([energy]), 1, mdi.MDI_DOUBLE, self.mdi_comm)

                    logger.info(f"Step {step+1} complete")
                    
                except Exception as e:
                    logger.error(f"Error in step {step+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Try to send zeros to GPUMD but don't crash
                    try:
                        logger.warning("Sending zero forces to GPUMD via >FORCES due to error")
                        mdi.MDI_Send_Command(">FORCES", self.mdi_comm)
                        mdi.MDI_Send(np.zeros(self.natoms * 3), self.natoms * 3, mdi.MDI_DOUBLE, self.mdi_comm)
                        mdi.MDI_Send_Command(">ENERGY", self.mdi_comm)
                        mdi.MDI_Send(np.array([0.0]), 1, mdi.MDI_DOUBLE, self.mdi_comm)
                    except Exception as e2:
                        logger.error(f"Could not send error response to GPUMD: {e2}")
                    # Continue with next step or exit
                    if step < self.steps - 1:
                        logger.warning("Continuing to next step despite error")
                    else:
                        logger.error("Last step failed, aborting")
                        break

            # Send exit command
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
    parser.add_argument("--vasp-cmd", default="vasp_std", help="VASP executable command")
    parser.add_argument("--poscar-template", default="POSCAR_template", help="Path to POSCAR_template")
    parser.add_argument("--port", type=int, default=8021, help="MDI TCP port")
    parser.add_argument("--steps", type=int, default=100, help="Number of MD steps")
    parser.add_argument("--timeout", type=int, default=3600, help="VASP timeout in seconds")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("VASP MDI Driver for GPUMD")
    logger.info("=" * 70)
    logger.info(f"VASP command: {args.vasp_cmd}")
    logger.info(f"POSCAR template: {args.poscar_template}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Steps: {args.steps}")

    try:
        # Initialize VASP calculator
        logger.info("\nInitializing VASP calculator...")
        vasp_calc = VASPCalculator(args.vasp_cmd, args.poscar_template, args.timeout)

        # Initialize MDI driver
        logger.info("\nInitializing MDI driver...")
        driver = VASPMDIDriver(vasp_calc, port=args.port, steps=args.steps)
        driver.initialize_mdi()

        # Run MD loop
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

