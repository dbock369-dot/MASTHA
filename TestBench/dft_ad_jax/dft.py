import copy
import jax.numpy as jnp
from jax import jit, grad, Array, ShapeDtypeStruct
from tqdm import tqdm
from typing import List, Tuple

from .helmholtz_functionals.fmt import FMT, FMTAntiSym, FMTPure, FMTAntiSymPure
from .helmholtz_functionals.hardchain import HardChain, HardChainPure
from .helmholtz_functionals.dispersion import Dispersion, DispersionPure
from .helmholtz_functionals.pure_pcsaft import PurePcSaft, PurePcSaftAntiSym

from .grid import Grid
from .parameters import PcSaftParameters


class DFT:
    def __init__(self, grid: Grid, parameters: PcSaftParameters, temperature: float):
        """Utility class collecting Helmholtz energy density, Helmholtz energy,
        functional derivative for DFT calculations. Furthermore executes Picard
        iterations to solve for the equilibrium density profiles.

        Parameters
        ----------
        grid : Grid
            Representation of 3-D grid of cDFT domain.
        parameters : PcSaftParameters
            Representation of PC-SAFT parameters.
        temperature : float
            Temperature in Kelvin at which the DFT calculation if performed.

        Methods
        -------
        weighted_densities(density):

        helmholtz_energy_density(density):

        helmholtz_energy(density):

        df_drho(density):

        helmholtz_energy_contributions(density):

        df_drho_contributions(density):

        rhs:

        rhs_log:

        picard_iteration

        picard_iteration_log

        fmt

        hard_chain

        dispersion
        """
        self.parameters = parameters
        self.grid = grid
        self.temperature = temperature
        self.contributions = [
            FMTAntiSym(parameters, grid),
            HardChain(parameters, grid),
            Dispersion(parameters, grid)
        ]

        self.weight_functions = [contribution._weight_functions(
            self.temperature) for contribution in self.contributions]

        # Just-in-time compilation of functional derivative
        jit_shape_dtype = ShapeDtypeStruct(
            shape=[self.parameters.n_comp] + self.grid.n_grid, dtype=jnp.dtype('float64'))
        self.df_drho = jit(self.df_drho).lower(jit_shape_dtype).compile()

    def weighted_densities(self, density: Array) -> Array:
        """Generator for weighted densities for each contribution."""
        for contribution, weight_function in zip(self.contributions, self.weight_functions):
            yield contribution.weighted_density(density, weight_function)

    def helmholtz_energy_density(self, density: Array) -> Array:
        """Generator for Helmholtz energy density for each contribution."""
        for contribution, weighted_densities in zip(self.contributions, self.weighted_densities(density)):
            yield contribution.helmholtz_energy_density(weighted_densities, self.temperature)

    def helmholtz_energy(self, density: Array) -> float:
        """Sum over Helmholtz energy density for each contribution."""
        # f = density * (torch.log(density) - 1) # ideal gas
        f = 0.0
        for contribution, weighted_densities in zip(self.contributions, self.weighted_densities(density)):
            f += jnp.sum(contribution.helmholtz_energy_density(
                weighted_densities, self.temperature))
        return f * self.grid.dv

    def df_drho(self, density: Array) -> Array:
        """Functional derivative of the Helmholtz energy w.r.t. density."""
        return grad(self.helmholtz_energy)(density) / self.grid.dv

    def grand_potential(self, density: Array,  density_bulk: Array, chemical_potential_residual_bulk: Array, external_potential: Array) -> float:
        """Grand potential density for each contribution.

        Parameters
        ----------
        density : Array
            Density profile for each component.
        density_bulk : Array
            Density of the bulk reservoir for each component.
        chemical_potential_residual_bulk : Array
            Reduced residual chemical potential of the corresponding bulk density in the reservoir for each component.
        external_potential : Array
            Reduced external potential for each component.

        Returns
        -------
        Reduced grand potential density.
        """
        helmholtz_energy_ideal_gas_segments = jnp.sum(self.parameters.m[(...,) + (None,)*self.grid.dim] * density * (
            jnp.log(density / density_bulk) - 1.0)) * self.grid.dv
        chemical_external_potential_contribution = jnp.sum(
            density * (external_potential - chemical_potential_residual_bulk)) * self.grid.dv

        return helmholtz_energy_ideal_gas_segments + self.helmholtz_energy(density) + chemical_external_potential_contribution

    def domega_drho(self, density: Array, density_bulk: Array, chemical_potential_residual_bulk: Array, external_potential: Array) -> Array:
        """Functional derivative of the Helmholtz energy w.r.t. density."""
        # Derivative is only calculated w.r.t. the first argument (`argnums=0` is obsolete here but clarifies)
        return grad(self.grand_potential, argnums=0)(density, density_bulk, chemical_potential_residual_bulk, external_potential) / self.grid.dv

    def d2omega_drho2_vector(self, density: Array, density_bulk: Array, chemical_potential_residual_bulk: Array, external_potential: Array, vector: Array) -> Array:
        """Hessian"""
        # return grad(lambda rho: jnp.vdot(vector, self.domega_drho(rho, density_bulk, chemical_potential_residual_bulk, external_potential)))(density)  # dimensions of both `vectors`

    # def helmholtz_energy_contributions(self, density: Array) -> Array:
    #     """Sum over Helmholtz energy density for each contribution."""
    #     for contribution, weighted_densities in zip(self.contributions, self.weighted_densities(density)):
    #         yield contribution.helmholtz_energy_density(weighted_densities, self.temperature).sum() * self.grid.dv

    def df_drho_contributions(self, density: Array) -> Tuple[Array]:
        """Functional derivative of each Helmholtz energy contribution w.r.t. density."""
        df_drho_contributions = list()

        for contribution, weight_function in zip(self.contributions, self.weight_functions):
            # Intermediate function calculating the Helmholtz energy from the density directly
            def _helmholtz_energy_contributions(rho):
                weighted_densities = contribution.weighted_density(
                    rho, weight_function)
                return jnp.sum(contribution.helmholtz_energy_density(weighted_densities, self.temperature)) * self.grid.dv

            # Get gradient from intermediate function
            df_drho_contributions.append(
                grad(_helmholtz_energy_contributions)(density) / self.grid.dv)

        return tuple(df_drho_contributions)

    @ staticmethod
    def rhs(dft, density: Array, density_bulk: Array, chemical_potential_residual_bulk: Array, external_potential: Array) -> Array:
        res = (
            density_bulk[(...,) + (None,)*dft.grid.dim] * jnp.exp(  # density_bulk[(...,) + (None,)*dim]
                (chemical_potential_residual_bulk[(...,) + (None,)*dft.grid.dim]
                 - dft.df_drho(density)
                 - external_potential) / dft.parameters.m[(...,) + (None,)*dft.grid.dim]) - density
        )
        return res

    @ staticmethod
    def rhs_log(dft, density: Array, density_bulk: Array, chemical_potential_residual_bulk: Array, external_potential: Array) -> Array:
        return (
            (chemical_potential_residual_bulk[(...,) + (None,)*dft.grid.dim]
             - dft.df_drho(density)
             - external_potential) / dft.parameters.m[(...,) + (None,)*dft.grid.dim] - jnp.log(density / density_bulk)
        )

    def picard_iteration(self,
                         initial_density: Array,
                         density_bulk: Array,
                         chemical_potential_residual_bulk: Array,
                         external_potential: Array,
                         max_iter: int = 1000,
                         tol: float = 1e-13,
                         damping_constant: float = 0.01,
                         max_change: float = jnp.inf,
                         verbosity: bool = False) -> Tuple[Array, float, Array]:
        # Leave initial density unmodified
        density = copy.deepcopy(initial_density)
        error = self.rhs(self, density, density_bulk,
                         chemical_potential_residual_bulk, external_potential)

        # divided by square root of number of total grid points (cells) multiplied with number of components
        res = jnp.linalg.norm(error, ord=None) / jnp.sqrt(error.size)
        if jnp.isnan(res):
            raise InvalidIteration('initial state')
        res_iter = jnp.zeros(max_iter)
        res_iter = res_iter.at[0].set(res)

        for i in tqdm(range(max_iter)):
            error = jnp.where(external_potential +
                              2.220446049250313e-16 >= 50.0, 0.0,  error)
            density += jnp.clip(damping_constant * error,
                                a_max=max_change * density_bulk[(...,) + (None,)*self.grid.dim])
            error = self.rhs(self, density, density_bulk,
                             chemical_potential_residual_bulk, external_potential)
            res = jnp.linalg.norm(error, ord=None) / jnp.sqrt(error.size)
            if jnp.isnan(res):
                raise InvalidIteration(i)
            res_iter = res_iter.at[i].set(res)

            # If tolerance is met: write residuals to output array
            if res <= tol:
                res_iter = res_iter[:i+1]
                break

        return density, res, res_iter

    def picard_iteration_log(self,
                             initial_density: Array,
                             density_bulk: Array,
                             chemical_potential_residual_bulk: Array,
                             external_potential: Array,
                             max_iter: int = 1000,
                             tol: float = 1e-13,
                             damping_constant: float = 0.01,
                             max_change: float = jnp.inf,
                             verbosity: bool = False) -> Tuple[Array, float, Array]:
        # Leave initial density unmodified
        density = copy.deepcopy(initial_density)
        error = self.rhs_log(self, density, density_bulk,
                             chemical_potential_residual_bulk, external_potential)
        res = jnp.linalg.norm(error, ord=None) / jnp.sqrt(error.size)
        if jnp.isnan(res):
            raise InvalidIteration('initial state')
        res_iter = jnp.zeros(max_iter)
        res_iter = res_iter.at[0].set(res)

        for i in tqdm(range(max_iter)):
            density *= jnp.exp(damping_constant * error)
            error = self.rhs_log(self, density, density_bulk,
                                 chemical_potential_residual_bulk, external_potential)
            res = jnp.linalg.norm(error, ord=None) / jnp.sqrt(error.size)
            if jnp.isnan(res):
                raise InvalidIteration(i)
            res_iter = res_iter.at[i].set(res)

            # If tolererance is met: write residuals to output array
            if res <= tol:
                res_iter = res_iter[:i+1]
                break

        return density, res, res_iter

    @property
    def fmt(self):
        """FMT (hard-sphere) contribution."""
        return self.contributions[0]

    @property
    def hard_chain(self):
        """Hard-chain contribution."""
        return self.contributions[1]

    @property
    def dispersion(self):
        """Dispersion contribution."""
        return self.contributions[2]


class InvalidIteration(Exception):
    """Raised when DFT iteration produces a NaN."""

    def __init__(self, iteration):
        self.message = f"Iteration `{iteration}` produced a NaN.."
        super().__init__(self.message)
