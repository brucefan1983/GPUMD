from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import multiprocessing
import sys
import shutil
from collections import defaultdict
from ase import Atoms
from ase.units import Bohr
from ase.calculators.calculator import all_changes, Calculator
from ase.calculators.nwchem import NWChem
from ase.calculators.orca import ORCA
from ase.calculators.mixing import SumCalculator
from calorine.calculators import CPUNEP
# from dscribe.descriptors import MBTR
# https://singroup.github.io/dscribe/latest/tutorials/descriptors/mbtr.html


sys.path.append("../dipole-models/gen1/scripts")  # For loading the torch model
sys.path.append("../dipole-models/gen1_gpumd")  # For loading the gpumd model


class TimeDependentCavityCalculator:
    """ This class implements the time-dependent cavity potential

    Parameters
    ----------
    resonance_frequency
        Resonance frequency of cavity in ase units
    coupling_strength
        Coupling strength vector of cavity in ase units
    dipole_v
        Initial dipole that determines the initial cavity displacement
    """

    def __init__(self,
                 resonance_frequency: float,
                 coupling_strength: float,
                 dipole_v: np.ndarray | None = None):

        self.cav_frequency = resonance_frequency
        self.coupling_strength_v = np.array(coupling_strength)

        if dipole_v is not None:
            if self.cav_frequency > 0:
                self._cav_q0 = self.coupling_strength_v @ dipole_v / self.cav_frequency
            else:
                self._cav_q0 = 0
            self._prevdipole_v = dipole_v
        else:
            self._cav_q0 = None
            self._prevdipole_v = None

        self._cos_integral = 0
        self._sin_integral = 0

        self._prevtime = 0
        self._time = 0

    def __str__(self) -> str:
        def indent(s: str, i: int):
            s = '\n'.join([i * ' ' + line for line in s.split('\n')])
            return s

        parameters = '\n'.join([f'{key}: {getattr(self, key)}'
                                for key in ['cav_frequency', 'coupling_strength_v']])
        parameters = indent(parameters, 4)

        s = f'{self.__class__.__name__}\n{parameters}'

        return s

    def initialize_md(self, dyn):
        def time_setter(dyn):
            self._time = dyn.get_time()

        dyn.attach(time_setter, interval=1, dyn=dyn)

    @property
    def cav_q0(self) -> float:
        """ Initial mode displacement"""
        return self._cav_q0

    @property
    def canonical_position(self) -> float:
        """ Cavity position coordinate

        q(t) = sin(ω(t-t₀)) Icos - cos(ω(t-t₀)) Isin + q(t₀) cos(ω(t-t₀))

        where

                t
        Icos = ∫  dt' cos(ωt') λ⋅μ
                t₀

        and

                t
        Isin = ∫  dt' sin(ωt') λ⋅μ
                t₀
        """
        phase = self.cav_frequency * self._time
        cav_q = (np.sin(phase) * self._cos_integral
                 - np.cos(phase) * self._sin_integral
                 + self.cav_q0 * np.cos(phase))

        return cav_q

    @property
    def canonical_momentum(self) -> float:
        """ Cavity momentum coordinate

        p(t) = ω cos(ω(t-t₀)) Icos + ω sin(ω(t-t₀)) Isin - q(t₀) ω sin(ω(t-t₀))

        where

                t
        Icos = ∫  dt' cos(ωt') λ⋅μ
                t₀

        and

                t
        Isin = ∫  dt' sin(ωt') λ⋅μ
                t₀
        """
        phase = self.cav_frequency * self._time
        cav_p = self.cav_frequency * (np.cos(phase) * self._cos_integral
                                      + np.sin(phase) * self._sin_integral
                                      - self.cav_q0 * np.sin(phase))

        return cav_p

    def cavity_potential_energy(self,
                                dipole_v: np.ndarray) -> float:
        """ Potential energy of the cavity

        0.5 (ω q(t) - λ⋅μ(t))²
        """
        pot_energy = 0.5 * (self.cav_frequency * self.canonical_position
                            - self.coupling_strength_v @ dipole_v) ** 2
        return pot_energy

    def cavity_kinetic_energy(self) -> float:
        """ Kinetic energy of the cavity

        0.5 p(t)²
        """
        kin_energy = 0.5 * self.canonical_momentum ** 2

        return kin_energy

    def cavity_force(self,
                     dipole_v: np.ndarray,
                     dipole_jacobian_ivv: np.ndarray) -> np.ndarray:
        """ Force from the cavity

        """
        njdip_iv = dipole_jacobian_ivv @ self.coupling_strength_v
        force_iv = njdip_iv * (self.cav_frequency * self.canonical_position
                               - self.coupling_strength_v @ dipole_v)
        return force_iv

    def calculate(self,
                  results: dict,
                  dipole_v: np.ndarray,
                  dipole_jacobian_ivv: np.ndarray | None,
                  properties: list[str] = ['energy', 'forces']):
        """ Calculate properties

        Parameters
        ----------
        results
            Dictionary where to put the results
        dipole_v
            Dipole moment μ(t)
        dipole_jacobian_ivv
            Jacobian of dipole moment

                   ∂μ
                     ν'
            J    = ---
             iνν'  ∂r
                     iν
        properties
            List of properies to calculate
        """

        if self._cav_q0 is None:
            if self.cav_frequency > 0:
                self._cav_q0 = self.coupling_strength_v @ dipole_v / self.cav_frequency
            else:
                self._cav_q0 = 0

        if 'energy' in properties:
            energy = self.cavity_potential_energy(dipole_v)
            results['energy'] = energy
        if 'forces' in properties:
            assert dipole_jacobian_ivv is not None
            force_iv = self.cavity_force(dipole_v, dipole_jacobian_ivv)
            results['forces'] = force_iv

    def step(self,
             prevtime: float,
             time: float,
             prev_dipole_v: np.ndarray,
             dipole_v: np.ndarray):
        """ Step the time dependent potential by time dt

        Should be called after updating the positions

        Parameters
        ----------
        prevtime
            Previous time
        time
            Current time
        prev_dipole_v
            Dipole at previous time
        dipole_v
            Dipole at current time
        """
        dt = time - prevtime
        prevlmu = self.coupling_strength_v @ prev_dipole_v
        lmu = self.coupling_strength_v @ dipole_v
        self._cos_integral += 0.5 * dt * np.cos(self.cav_frequency * prevtime) * prevlmu
        self._sin_integral += 0.5 * dt * np.sin(self.cav_frequency * prevtime) * prevlmu
        self._cos_integral += 0.5 * dt * np.cos(self.cav_frequency * time) * lmu
        self._sin_integral += 0.5 * dt * np.sin(self.cav_frequency * time) * lmu

    def step_if_time_changed(self,
                             dipole_v: np.ndarray):
        """ If the simulation time has changed, then step the time dependent
        potential

        Parameters
        ----------
        dipole_v
            Current dipole

        Returns
        -------
        True if the time has changed, False otherwise
        """
        changed = False
        if self._time > self._prevtime:
            # The time has changed. We need to step the integrals using the
            # trapezoidal rule
            self.step(self._prevtime, self._time, self._prevdipole_v, dipole_v)
            changed = True

        # Save the time and for use in the trapezoidal rule later
        self._prevtime = self._time
        self._prevdipole_v = dipole_v

        return changed


class DipoleCalculator(Calculator):
    """ Calculates the energies and forces from a cavity on a set of dipoles

    Parameters
    ----------
    dipole_filename
        Path to dipole model. If not provided, only the ionic dipoles will be
        included (i.e. dipole is given by positions)
    resonance_frequency
        Resonance frequency of cavity in ase units
    coupling_strength
        Coupling strength vector of cavity in ase units
    gradient_mode
        'fd' or 'autograd'
    atoms : Atoms
        Atoms to attach the calculator to
    """

    implemented_properties = ['energy', 'energies', 'forces']

    def __init__(self,
                 dipole_filename: str | None,
                 resonance_frequency: float,
                 coupling_strength: float,
                 charge: float = -1.0,
                 atoms: Atoms | None = None,
                 gradient_mode: str = 'fd'):

        parameters = dict()
        parameters['dipole_filename'] = dipole_filename
        parameters['resonance_frequency'] = resonance_frequency
        parameters['coupling_strength'] = coupling_strength
        parameters['charge'] = charge
        parameters['gradient_mode'] = gradient_mode
        self.gpumddipole = False

        super().__init__(atoms=atoms, **parameters)
        # After using super().__init__ parameters can be accessed as self.parameters

        self.atoms = atoms

        if self.parameters.dipole_filename is None:
            self.dipole_calc = None
        else:
            if 'gpumd' in self.parameters.dipole_filename:
                self.gpumddipole = True
                self.calcdipole = CPUNEP(self.parameters.dipole_filename)
                self.dipole_calc = True
            else:
                self.dipole_calc = torch.load(self.parameters.dipole_filename)['model']
                self.dipole_calc.eval()
                self.dipole_calc.to('cpu')

        dipole_v = None
        if atoms is not None:
            dipole_v = self.get_dipole(atoms)
        if np.linalg.norm(coupling_strength) > 0:
            self.td_cav = TimeDependentCavityCalculator(resonance_frequency,
                                                        coupling_strength,
                                                        dipole_v)
        else:
            self.td_cav = None

        self._shiftval = 1e-3
        self.charge = charge

        if gradient_mode == 'autograd':
            assert self.dipole_calc is not None, 'Cannot use autograd without torch model'
            self._use_autograd = True
        elif gradient_mode == 'fd':
            self._use_autograd = False
        else:
            raise ValueError(f"Don't recognize {gradient_mode}")

    def __str__(self) -> str:
        def indent(s: str, i: int):
            s = '\n'.join([i * ' ' + line for line in s.split('\n')])
            return s

        parameters = '\n'.join([f'{key}: {value}' for key, value in self.parameters.items()])
        parameters = indent(parameters, 4)

        if self.gpumddipole:
            using_torch = ('Using GPUMD model' if self.dipole_calc is not None else
                           'Using ionic dipoles')
            using_fd = (sys.exit() if self._use_autograd else
                        'Using finite difference Jacobian')
        else:
            using_torch = ('Using torch model' if self.dipole_calc is not None else
                           'Using ionic dipoles')
            using_fd = ('Using torch autograd Jacobian' if self._use_autograd else
                        'Using finite difference Jacobian')

        s = (f'{self.__class__.__name__}\n{parameters}\n'
             f'{using_torch}\n'
             f'{using_fd}'
             )

        return s

    def initialize_md(self, dyn):
        if self.td_cav is None:
            return
        self.td_cav.initialize_md(dyn)

    def calculate(self,
                  atoms: Atoms | None = None,
                  properties: list[str] = ['energy', 'forces'],
                  system_changes: list[str] = all_changes):
        if self.td_cav is None:
            if 'energy' in properties:
                self.results['energy'] = 0
            if 'forces' in properties:
                self.results['forces'] = np.zeros((len(atoms), 3))
            return

        # Calculate dipole given the atomic positions
        dipole_v = self.get_dipole(atoms)

        # Calculate Jacobian only if needed
        jacobian_ivv = None
        if 'forces' in properties:
            jacobian_ivv = self.get_dipole_jacobian(atoms)

        # Add the next term to the time dependent integrals
        if self.td_cav.step_if_time_changed(dipole_v):
            # The time has changed so the results are now invalid
            self.results.clear()

        if atoms is None:
            atoms = self.atoms.copy()

        self.td_cav.calculate(self.results, dipole_v, jacobian_ivv, properties)

    def get_dipole(self,
                   atoms: Atoms) -> np.ndarray:
        if self.dipole_calc is None:
            return self.get_ionic_dipole(atoms)
        else:
            if self.gpumddipole:
                atoms_copy = atoms.copy()
                atoms_copy.set_positions(atoms.get_positions() - atoms.get_center_of_mass())
                gpumd_dipole = (self.calcdipole.get_dipole_moment(atoms_copy) * Bohr +
                                self.charge * atoms.get_center_of_mass())
                # atoms.calc = self.calcdipole
                # gpumd_dipole = (atoms.get_dipole_moment() * Bohr +
                #                 self.charge * atoms.get_center_of_mass())
                return gpumd_dipole
            else:
                return self.get_torch_dipole(atoms)

    def get_dipole_jacobian(self, atoms):
        if self._use_autograd and not self.gpumddipole:
            return self.get_torch_dipole_jacobian(atoms)
        else:
            if self.dipole_calc is None:
                # Use the ionic dipole function
                return self.get_ionic_dipole_jacobian(atoms)
            elif self.dipole_calc is not None and self.gpumddipole:
                # Use the CPUNEP version
                # return self.get_gpumd_dipole_jacobian_generic(atoms)
                return self.get_gpumd_dipole_jacobian_CPUNEP(atoms)
            else:
                # Use the batched version
                return self.get_torch_dipole_jacobian_batch(atoms)

    def get_torch_dipole(self,
                         atoms: Atoms) -> np.ndarray:
        # CHECK AGAIN if the predictions make sense, i.e., that I correctly loaded the dipole-model.
        com_v = atoms.get_center_of_mass()
        pos_iv = atoms.get_positions() - com_v

        descriptors = np.array(pos_iv.flatten())
        desc_tensor = torch.from_numpy(descriptors).to(torch.float)
        with torch.inference_mode():
            y_pred = self.dipole_calc(desc_tensor)
        dipole_v = y_pred.detach().numpy()
        dipole_v *= Bohr  # Convert a.u. to eÅ
        dipole_v += self.charge * atoms.get_center_of_mass()
        return dipole_v

    def get_torch_dipole_batch(self,
                               descriptors_xiv: np.ndarray) -> np.ndarray:
        """ Get dipole using torch model for a batch of descriptors

        Parameters
        ----------
        descriptors_xiv
            Array of shape (..., Ni, Nv)
            The first dimension(s) correspond to different configurations

        Returns
        -------
        Dipoles for each of the configurations
        Numpy array of shape (..., Nv), where the ... are the same as in descriptors_xiv
        """
        Ni = descriptors_xiv.shape[-2]
        assert descriptors_xiv.shape[-1] == 3, 'Not 3 Cartesian directions'
        shapex = descriptors_xiv.shape[:-2]
        desc_tensor = torch.from_numpy(descriptors_xiv.reshape((-1, 3*Ni))).to(torch.float)
        with torch.inference_mode():
            y_pred = self.dipole_calc(desc_tensor)

        dipole_xv = y_pred.detach().numpy().reshape(shapex + (3, ))
        dipole_xv *= Bohr  # Convert a.u. to eÅ
        return dipole_xv

    def get_ionic_dipole(self,
                         atoms: Atoms) -> np.ndarray:
        pos_iv = atoms.get_positions()
        charge_i = atoms.get_atomic_numbers()

        dipole_v = charge_i @ pos_iv

        return dipole_v

    def get_ionic_dipole_jacobian(self,
                                  atoms: Atoms) -> np.ndarray:
        charge_i = atoms.get_atomic_numbers()
        jacobian_ivv = charge_i[:, None, None] * np.eye(3)[None, ...]

        return jacobian_ivv

    def get_torch_dipole_jacobian_batch(self,
                                        atoms: Atoms) -> np.ndarray:
        """ Compute the jacobian as a forward difference. The neural network is evaluated in batch

        """
        pos_iv = atoms.get_positions()
        dipole_v = self.get_dipole(atoms)

        atoms_copy = atoms.copy()
        descriptors_iviv = np.zeros(pos_iv.shape + pos_iv.shape)
        shift_iviv = np.zeros(pos_iv.shape + pos_iv.shape)
        com_ivv = np.zeros(pos_iv.shape + (3, ))
        for i in range(len(pos_iv)):
            for v in range(3):
                shift_iviv[i, v, i, v] = self._shiftval
                atoms_copy.set_positions(pos_iv + shift_iviv[i, v])
                com_ivv[i, v] = atoms_copy.get_center_of_mass()

        descriptors_iviv = pos_iv[None, None, :, :] + shift_iviv - com_ivv[:, :, None, :]
        forward_ivv = self.get_torch_dipole_batch(descriptors_iviv)
        forward_ivv += self.charge * com_ivv
        jacobian_ivv = (forward_ivv - dipole_v) / self._shiftval

        return jacobian_ivv

    def get_gpumd_dipole_jacobian_generic(self,
                                          atoms: Atoms) -> np.ndarray:
        pos_iv = atoms.get_positions()
        dipole_v = self.get_dipole(atoms)
        dipole_jacobian_ivv = np.zeros(pos_iv.shape + (3,))

        atoms_copy = atoms.copy()
        for i in range(len(pos_iv)):
            for v in range(3):
                shift_iv = np.zeros_like(pos_iv)
                shift_iv[i, v] = self._shiftval
                atoms_copy.set_positions(pos_iv + shift_iv)
                diff_v = (self.get_dipole(atoms_copy) - dipole_v) / self._shiftval
                dipole_jacobian_ivv[i, v, :] = diff_v
        return dipole_jacobian_ivv

    def get_gpumd_dipole_jacobian_CPUNEP(self,
                                         atoms: Atoms) -> np.ndarray:
        calc = CPUNEP(
               self.parameters.dipole_filename,
               atoms=atoms.copy(),
        )
        gradient_forward_cpp_CPUNEP = calc.get_dipole_gradient(
            displacement=1e-3, method="second order central difference", charge=self.charge/Bohr
        )
        return gradient_forward_cpp_CPUNEP * Bohr

    def get_torch_dipole_jacobian(self, atoms):
        # Uses the autograd jacobian given by pytorch - it seems to give
        # qualitative agreement but the deviations are on the 10% order, maybe
        # the finite-difference that I use for comparison is not really
        # accurate. The energies are quite different also. njdip nearly doubles
        # the energy while nablalambdadip lowers the energy, as would probably make more sense.
        com_v = atoms.get_center_of_mass()
        pos_iv = atoms.get_positions() - com_v
        descriptors = np.array(pos_iv.flatten())
        # desc_tensor = torch.from_numpy(descriptors).to(torch.float) # previous version
        desc_tensor = torch.tensor(descriptors, requires_grad=True).to(
            torch.float)  # needed to use the gradcheck
        jacobian = torch.autograd.functional.jacobian(self.dipole_calc, desc_tensor)
        jacobian_ivv = jacobian.detach().numpy().T.reshape((-1, 3, 3))

        torch.autograd.gradcheck(self.dipole_calc, desc_tensor, eps=1e-03, atol=1e-02)

        return jacobian_ivv * Bohr

    def get_dipole_jacobian_generic(self,
                                    atoms: Atoms) -> np.ndarray:
        pos_iv = atoms.get_positions()
        dipole_v = self.get_dipole(atoms)
        dipole_jacobian_ivv = np.zeros(pos_iv.shape + (3,))

        atoms_copy = atoms.copy()
        for i in range(len(pos_iv)):
            for v in range(3):
                shift_iv = np.zeros_like(pos_iv)
                shift_iv[i, v] = self._shiftval
                atoms_copy.set_positions(pos_iv + shift_iv)
                diff_v = (self.get_dipole(atoms_copy) - dipole_v) / self._shiftval
                dipole_jacobian_ivv[i, v, :] = diff_v

        return dipole_jacobian_ivv

    def get_crossrotdip(self, atoms):
        # ## CHECK AGAIN if I really need this term. I should carefully  ## #
        # ## check the Hamilton equation again.                          ## #
        # TODO this function is now broken but I don't think we really need it
        pos_iv = atoms.get_positions()
        atoms_copy = atoms.copy()
        dipole_v = self.get_dipole(atoms)
        crossrotdip_iv = np.zeros_like(pos_iv)
        for i in range(len(pos_iv)):
            diffdipole_vv = np.zeros((3, 3))
            for v in range(3):
                shift_iv = np.zeros_like(pos_iv)
                shift_iv[i, v] = self._shiftval
                atoms_copy.set_positions(pos_iv + shift_iv)
                diffdipole_vv[:, v] = (self.get_dipole(atoms_copy) - dipole_v) / self._shiftval

            rotdip_v = np.array([diffdipole_vv[2, 1] - diffdipole_vv[1, 2],
                                 diffdipole_vv[0, 2] - diffdipole_vv[2, 0],
                                 diffdipole_vv[1, 0] - diffdipole_vv[0, 1]])
            crossrotdip_iv[i, :] = np.cross(self.coupling_strength_v, rotdip_v)
        return crossrotdip_iv * Bohr


class CavityCalculator(SumCalculator):

    """This class provides an ASE calculator.

    Parameters
    ----------
    potential_filename
        Path to nep.txt potential
    dipole_filename
        Path to dipole model. If not provided, only the ionic dipoles will be
        included (i.e. dipole is given by positions)
    resonance_frequency
        Resonance frequency of cavity
    coupling_strength
        Coupling strength vector of cavity
    atoms : Atoms
        Atoms to attach the calculator to
    """

    def __init__(self,
                 nep_calc: CPUNEP | dict[str, Any],
                 dipole_calc: DipoleCalculator | dict[str, Any]):

        if isinstance(nep_calc, dict):
            nep_calc = CPUNEP(**nep_calc)
        if isinstance(dipole_calc, dict):
            dipole_calc = DipoleCalculator(**dipole_calc)

        self.nep_calc = nep_calc
        self.dipole_calc = dipole_calc

        calcs = [self.nep_calc, self.dipole_calc]

        super().__init__(calcs)

    def __str__(self) -> str:
        def indent(s: str, i: int):
            s = '\n'.join([i * ' ' + line for line in s.split('\n')])
            return s

        nepcalc = indent(str(self.nep_calc), 4)
        dipolecalc = indent(str(self.dipole_calc), 4)

        s = (f'{self.__class__.__name__}\n'
             f'NEP Calculator:\n{nepcalc}\n\n'
             f'Dipole Calculator:\n{dipolecalc}')
        return s

    def todict(self) -> dict[str, Any]:
        dct = dict()
        dct['nep_calc'] = self.nep_calc
        dct['dipole_calc'] = self.dipole_calc

        return dct

    @classmethod
    def from_parameters(cls,
                        potential_filename: str,
                        dipole_filename: str,
                        resonance_frequency: float,
                        coupling_strength: float,  # XXX type is vector
                        charge: float,
                        gradient_mode: bool,
                        atoms: Atoms | None = None) -> CavityCalculator:

        nep_calc = CPUNEP(potential_filename, atoms=atoms)
        dipole_calc = DipoleCalculator(dipole_filename=dipole_filename,
                                       resonance_frequency=resonance_frequency,
                                       coupling_strength=coupling_strength,
                                       charge=charge,
                                       gradient_mode=gradient_mode,
                                       atoms=atoms)
        return cls(nep_calc=nep_calc, dipole_calc=dipole_calc)


class AbInitioCavityCalculator(Calculator):
    """ Calculates the energy, forces and dipole from DFT. Adds the contributions
    from a cavity using the dipole """

    implemented_properties = ['dipole', 'energy', 'energies', 'forces']
    all_dft_calculators = ['nwchem', 'orca']

    def __init__(self,
                 resonance_frequency: float,
                 coupling_strength: float,  # XXX type is vector
                 charge: float,
                 atoms: Atoms | None = None,
                 xc: str = 'PBE',
                 basis: str = 'def2-TZVP',
                 calcid: str = '',  # Necessary for restarts
                 calculator: str = 'nwchem',
                 ):
        parameters = dict()
        parameters['resonance_frequency'] = resonance_frequency
        parameters['coupling_strength'] = coupling_strength
        parameters['charge'] = charge
        parameters['xc'] = xc
        parameters['basis'] = basis
        parameters['calculator'] = calculator.lower()

        super().__init__(atoms=atoms, **parameters)
        # After using super().__init__ parameters can be accessed as self.parameters

        assert self.parameters.calculator in self.all_dft_calculators

        self.calcid = calcid
        self.dft_calc = None
        self.dft_calc_jacobian_iv = None
        self.charge = charge

        if np.linalg.norm(coupling_strength) > 0:
            self.td_cav = TimeDependentCavityCalculator(resonance_frequency, coupling_strength,
                                                        atoms)
        else:
            self.td_cav = None

        self._shiftval = 1e-3

        self.timings = defaultdict(float)

    def __str__(self) -> str:
        def indent(s: str, i: int):
            s = '\n'.join([i * ' ' + line for line in s.split('\n')])
            return s

        parameters = '\n'.join([f'{key}: {value}' for key, value in self.parameters.items()])
        parameters = indent(parameters, 4)

        s = f'{self.__class__.__name__}\n{parameters}'

        return s

    def initialize_md(self, dyn):
        if self.td_cav is None:
            return
        self.td_cav.initialize_md(dyn)

    def setup_dft_calc(self,
                       first_start: bool,
                       i: int | None = None,
                       v: int | None = None) -> Calculator:
        """
        Parameters
        ----------
        first_start
            Whether this is the calculator that should converge the first density
        i
            Index in Jacobian calculation, or None if this calculator is not used to
            calculate the Jacobian
        v
            Index in Jacobian calculation, or None if this calculator is not used to
            calculate the Jacobian
        """
        xc = self.parameters.xc
        basis = self.parameters.basis
        if self.parameters.calculator == 'nwchem':
            if xc == 'PBE':
                xc = 'PBE96'
            kwargs = dict(label=f'tmp/nwchemcalc_{self.calcid}',
                          memory='1024 mb',
                          basis=basis,
                          charge=self.parameters.charge,
                          dft=dict(xc=xc,
                                   convergence=dict(energy=1e-5,
                                                    density=1e-4,
                                                    gradient=5e-3),
                                   iterations=150,
                                   ),
                          )
            kwargs['restart_kw'] = 'start' if first_start else 'restart'

            return NWChem(**kwargs)

        if self.parameters.calculator == 'orca':
            moread = ''
            moinp = ''
            if i is not None:
                xyz = 'xyz'[v]
                gbwfile = Path(f'tmp/orcacalc_{self.calcid}') / 'orca.gbw'
                directory = Path(f'tmp/orcacalc_{self.calcid}_jacobian_{i:03.0f}{xyz}')
                moread = 'MORead '
                moinp = f' %moinp "{str(gbwfile.absolute())}"'
                nprocs = 1
            else:
                assert v is None
                directory = Path(f'tmp/orcacalc_{self.calcid}')
                nprocs = multiprocessing.cpu_count()

            if first_start:
                # Remove old files
                (directory / 'orca.gbw').unlink(missing_ok=True)
                (directory / 'orca.densities').unlink(missing_ok=True)
            calc = ORCA(directory=str(directory),
                        orcasimpleinput=f'{moread}{xc} {basis} TIGHTSCF EnGrad',
                        orcablocks=f'%pal nprocs {nprocs} end{moinp}',
                        charge=self.parameters.charge,
                        )
            if tuple(calc.profile.argv) == ('orca',):
                # ORCA needs a full path for parallel runs
                orcapath = shutil.which('orca.run')
                if orcapath is None:
                    raise RuntimeError('orca binary not found. Load ORCA module')
                calc.profile.argv[0] = orcapath

            return calc

        raise RuntimeError(f'calculator {self.parameters.calculator} must be '
                           f'one of {self.all_dft_calculators}')

    def init_calc(self,
                  atoms: Atoms):
        import time
        t0 = time.time()

        # Start new calculation
        self.dft_calc = self.setup_dft_calc(first_start=True)

        # Converge the density (this will take some time)
        self.dft_calc.get_potential_energy(atoms)
        self.timings['converge_initial'] += time.time() - t0

        # Make a new calculator with the restart keyword in the same directory
        self.dft_calc = self.setup_dft_calc(first_start=False)

        if self.parameters.calculator == 'orca':
            # Make a new calculators for the jacobian calculations in parallel
            self.dft_calc_jacobian_iv = [[self.setup_dft_calc(first_start=True, i=i, v=v)
                                          for v in range(3)] for i in range(len(atoms))]

    def get_dipole_jacobian(self,
                            atoms: Atoms,
                            serial: bool = False) -> np.ndarray:
        import time
        pos_iv = atoms.get_positions()
        dipole_v = self.get_dipole(atoms)
        dipole_jacobian_ivv = np.zeros(pos_iv.shape + (3,))
        t0 = time.time()

        if serial or self.parameters.calculator == 'nwchem':
            # Serial version
            for i in range(len(atoms)):
                for v in range(3):
                    dipole_shift_v = self.get_dipole_shift_single(atoms, i, v)
                    diff_v = (dipole_shift_v - dipole_v) / self._shiftval
                    dipole_jacobian_ivv[i, v, :] = diff_v
        else:
            # Parallel version
            args = [(atoms, i, v) for v in range(3) for i in range(len(atoms))]
            with multiprocessing.Pool() as pool:
                iterator = pool.imap_unordered(self.get_dipole_shift_single_packed, args)
                for i, v, dipole_shift_v in iterator:
                    diff_v = (dipole_shift_v - dipole_v) / self._shiftval
                    dipole_jacobian_ivv[i, v, :] = diff_v
        self.timings['calculate_jacobian'] += time.time() - t0

        return dipole_jacobian_ivv

    def get_dipole_shift_single_packed(self,
                                       args: tuple[Atoms, int, int]) -> tuple[int, int, np.ndarray]:
        # Unpack arguments
        atoms, i, v = args
        return i, v, self.get_dipole_shift_single(atoms, i, v)

    def get_dipole_shift_single(self,
                                atoms: Atoms,
                                i: int,
                                v: int) -> np.ndarray:
        atoms_copy = atoms.copy()

        # Shift
        pos_iv = atoms.get_positions()
        shift_iv = np.zeros_like(pos_iv)
        shift_iv[i, v] = self._shiftval
        atoms_copy.set_positions(pos_iv + shift_iv)

        # Calculate
        dipole_v = self._get_dipole(atoms_copy, i=i, v=v)

        return dipole_v

    def get_dipole(self,
                   atoms: Atoms) -> np.ndarray:
        """ Calculate the dipole and measure timings

        If necessary, initialize the calculator.
        """
        import time
        if self.dft_calc is None:
            assert atoms is not None
            self.init_calc(atoms)

        t0 = time.time()
        dipole_v = self._get_dipole(atoms)
        self.timings['calculate'] += time.time() - t0

        return dipole_v

    def _get_dipole(self,
                    atoms: Atoms,
                    i: int | None = None,
                    v: int | None = None) -> np.ndarray:
        """ Calculate the dipole using the DFT calculator

        Parameters
        ----------
        atoms
            Atoms object
        i
            Index in Jacobian calculation, or None if Jacobian is not being computed
        v
            Index in Jacobian calculation, or None if Jacobian is not being computed
        """
        if self.parameters.calculator == 'nwchem':
            dft_calc = self.dft_calc
        elif self.parameters.calculator == 'orca':
            # For ORCA I have implemented calculations in parallel
            if i is None:
                assert v is None
                dft_calc = self.dft_calc
            else:
                assert v is not None
                dft_calc = self.dft_calc_jacobian_iv[i][v]
        else:
            raise RuntimeError(f'calculator {self.parameters.calculator} must be '
                               f'one of {self.all_dft_calculators}')

        dipole_v = dft_calc.get_dipole_moment(atoms)

        if self.parameters.calculator == 'orca':
            # ORCA gives the dipole moment in the center of mass frame of reference
            dipole_v += atoms.get_center_of_mass() * self.charge
        return dipole_v

    def calculate(self,
                  atoms: Atoms | None = None,
                  properties: list[str] = ['energy', 'forces'],
                  system_changes: list[str] = all_changes):
        import time
        if self.td_cav is None:
            if 'energy' in properties:
                self.results['energy'] = 0
            if 'forces' in properties:
                self.results['forces'] = np.zeros((len(atoms), 3))
            return

        # Calculate dipole given the atomic positions
        if self.dft_calc is None:
            assert atoms is not None
            self.init_calc(atoms)

        t0 = time.time()
        dft_force_iv = self.dft_calc.get_forces(atoms)
        dft_energy = self.dft_calc.get_potential_energy(atoms)
        self.timings['calculate'] += time.time() - t0
        dipole_v = self.get_dipole(atoms)

        # Calculate Jacobian only if needed
        jacobian_ivv = None
        if 'forces' in properties:
            jacobian_ivv = self.get_dipole_jacobian(atoms)

        # Add the next term to the time dependent integrals
        if self.td_cav.step_if_time_changed(dipole_v):
            # The time has changed so the results are now invalid
            self.results.clear()

        if atoms is None:
            atoms = self.atoms.copy()

        self.td_cav.calculate(self.results, dipole_v, jacobian_ivv, properties)

        if 'energy' in properties:
            self.results['energy'] += dft_energy
        if 'forces' in properties:
            self.results['forces'] += dft_force_iv
