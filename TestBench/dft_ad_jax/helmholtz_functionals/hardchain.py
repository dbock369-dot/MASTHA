import jax.numpy as jnp
from jax import Array
from scipy.special import spherical_jn

from . import HelmholtzFunctional
from ..grid import Grid
from ..parameters import PcSaftParameters


class HardChain(HelmholtzFunctional):
    """PC-SAFT hard-chain Helmholtz energy functional.

    Attributes
    ----------
    parameters : PcSaftParameters
        PC-SAFT parameters.
    grid : Grid
        Information about the numerical grid.

    Methods
    -------
    weight_functions(grid, temperature):
        Calculates the Fourier transform of the weight functions.
    weighted_density(density, weight_functions):
        Calculates the weighted densities from density and weight functions.
    helmholtz_energy_density(grid, weighted_density, _temperature):
        Calculates the Helmholtz energy density from weighted densities.
    """

    def __init__(self, parameters: PcSaftParameters, grid: Grid):
        self.parameters = parameters
        self.grid = grid

    def _weight_functions(self, temperature: float) -> Array:
        radius = self.parameters._hard_sphere_radius(
            temperature)[(...,) + (None,)*self.grid.dim]  # [(...,) + (None,)*self.grid.dim]

        j0 = jnp.array(spherical_jn(0, 2.0 * radius * self.grid.k_abs))
        j2 = jnp.array(spherical_jn(2, 2.0 * radius * self.grid.k_abs))

        omega = jnp.stack([
            j0,
            jnp.pi / 6.0 *
            self.parameters.m[(...,) + (None,)*self.grid.dim] *
            (2.0 * radius)**2 * (j0 + j2),
            jnp.pi / 6.0 *
            self.parameters.m[(...,) + (None,)*self.grid.dim] *
            (2.0 * radius)**3 * (j0 + j2)
        ], axis=1)

        if self.grid.lanczos_sigma:
            omega *= self.grid.sigma

        return omega

    def weighted_density(self, density: Array, weight_functions: Array) -> Array:
        rho = jnp.fft.rfftn(density, s=self.grid.n_grid)
        lambd = rho * weight_functions[:, 0]
        zeta2 = (rho * weight_functions[:, 1]).sum(axis=0)[None]
        zeta3 = (rho * weight_functions[:, 2]).sum(axis=0)[None]
        return jnp.concatenate([density, jnp.fft.irfftn(jnp.concatenate([lambd, zeta2, zeta3], axis=0), s=self.grid.n_grid)], axis=0)

    def helmholtz_energy_density(self, weighted_density: Array, temperature: float) -> Array:
        weighted_density = self.non_negative_weighted_density(
            weighted_density)  # essential here for stable iteration

        n_comp = self.parameters.n_comp
        diameter = 2.0 * self.parameters._hard_sphere_radius(
            temperature)[(...,) + (None,)*self.grid.dim]
        _density = weighted_density[:n_comp]
        _lambd = weighted_density[n_comp:2*n_comp]
        _zeta2 = weighted_density[-2]
        _inv_1zeta3 = 1.0 / (1.0 - weighted_density[-1])
        y_dd = _inv_1zeta3 + (0.5 * diameter * _zeta2 * _inv_1zeta3**2
                              * (3.0 + diameter * _zeta2 * _inv_1zeta3))

        # add for F^hc:  + ((self.parameters.m[:, None, None, None] - 1.0) * _density * (jnp.log(_density) - 1.0)).sum(axis=0)
        return (-(self.parameters.m[(...,) + (None,)*self.grid.dim] - 1.0) * _density * (jnp.log(y_dd * _lambd) - 1.0)).sum(axis=0)


class HardChainPure(HardChain):
    """PC-SAFT hard-chain Helmholtz energy functional.

    Attributes
    ----------
    parameters : PcSaftParameters
        PC-SAFT parameters.
    grid : Grid
        Information about the numerical grid.

    Methods
    -------
    weight_functions(grid, temperature):
        Calculates the Fourier transform of the weight functions.
    weighted_density(density, weight_functions):
        Calculates the weighted densities from density and weight functions.
    helmholtz_energy_density(grid, weighted_density, _temperature):
        Calculates the Helmholtz energy density from weighted densities.
    """

    def _weight_functions(self, temperature: float) -> Array:
        radius = self.parameters._hard_sphere_radius(
            temperature)[(...,) + (None,)*self.grid.dim]  # [(...,) + (None,)*self.grid.dim]

        j0 = jnp.array(spherical_jn(0, 2.0 * radius * self.grid.k_abs))
        j2 = jnp.array(spherical_jn(2, 2.0 * radius * self.grid.k_abs))

        # Only lambda & zeta3
        omega = jnp.stack([
            j0,
            jnp.pi / 6.0 *
            self.parameters.m[(...,) + (None,)*self.grid.dim] *
            (2.0 * radius)**3 * (j0 + j2)
        ], axis=1)

        if self.grid.lanczos_sigma:
            omega *= self.grid.sigma

        return omega

    def weighted_density(self, density: Array, weight_functions: Array) -> Array:
        rho = jnp.fft.rfftn(density, s=self.grid.n_grid)
        lambd = rho * weight_functions[:, 0]
        zeta3 = (rho * weight_functions[:, 1]).sum(axis=0)[None]
        return jnp.concatenate([density, jnp.fft.irfftn(jnp.concatenate([lambd, zeta3], axis=0), s=self.grid.n_grid)], axis=0)

    def helmholtz_energy_density(self, weighted_density: Array, temperature: float) -> Array:
        # Essential here for stable iteration
        n = self.non_negative_weighted_density(weighted_density)

        _inv_1zeta3 = 1.0 / (1.0 - n[2])
        y_dd = _inv_1zeta3 + \
            (0.5 * n[2] * _inv_1zeta3**2 * (3.0 + n[2] * _inv_1zeta3))

        # add for F^hc:  + ((self.parameters.m[:, None, None, None] - 1.0) * _density * (jnp.log(_density) - 1.0)).sum(axis=0)
        return -(self.parameters.m - 1.0) * n[0] * (jnp.log(y_dd * n[1]) - 1.0)
