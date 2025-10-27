"""White Bear version of fundamental measure theory (hard-sphere) contribution
to the Helmholtz energy functional.
"""
import jax.numpy as jnp
from jax import Array
from scipy.special import spherical_jn

from . import HelmholtzFunctional
from ..grid import Grid
from ..parameters import PcSaftParameters


class FMT(HelmholtzFunctional):
    """White Bear version of fundamental measure theory.

    Attributes
    ----------
    parameters : PcSaftParameters
        PC-SAFT parameters.
    grid : Grid
        Grid information.

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
        """Hard-sphere contribution of PC-SAFT functional.

        Parameters
        ----------
        parameters : PcSaftParameters
            Parameters for PC-SAFT functionals.
        grid : Grid
            Grid information.

        Returns
        -------
        Hard-sphere contribution of PC-SAFT functional.
        """
        self.parameters = parameters
        self.grid = grid

    def __repr__(self):
        return "Helmholtz energy functional: FMT White Bear."

    def _weight_functions(self, temperature: float) -> Array:
        """Weight functions of PC-SAFT Hard-sphere contribution.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin at which the DFT calculation if performed.

        Returns
        -------
        Weight functions of PC-SAFT Hard-sphere contribution.
        """
        radius = self.parameters._hard_sphere_radius(
            temperature)[(...,) + (None,)*self.grid.dim]

        j0 = jnp.array(spherical_jn(0, radius * self.grid.k_abs))
        j2 = jnp.array(spherical_jn(2, radius * self.grid.k_abs))

        omega3 = 4 / 3 * jnp.pi * radius**3 * (j0 + j2)
        omega = jnp.concatenate([
            # expand dimension from [ncomp, ngrid1, ngrid2, ngrid3] to [ncomp, 1, ngrid1, ngrid2, ngrid3]
            j0[:, None],
            (radius * j0)[:, None],
            (4 * jnp.pi * radius**2 * j0)[:, None],
            omega3[:, None],
            # has dimension [ncomp, dim, ngrid1, ngrid2, ngrid3]
            -1j * self.grid.k * 0.25 / jnp.pi * (omega3 / radius)[:, None],
            -1j * self.grid.k * omega3[:, None]
        ], axis=1)

        if self.grid.lanczos_sigma:
            omega *= self.grid.sigma

        return self.parameters.m[(..., None) + (None,)*self.grid.dim] * omega

    def weighted_density(self, density: Array, weight_functions: Array) -> Array:
        """Weighted density of PC-SAFT hard-sphere contribution.

        Parameters
        ----------
        density : Tensor
            Density profile.
        weight_functions : torch.Tensor
            Weight functions of PC-SAFT hard-sphere contribution.

        Returns
        -------
        Weighted densities of PC-SAFT hard-sphere contribution.
        """
        rho = jnp.fft.rfftn(density, s=self.grid.n_grid)[:, None]
        wd = jnp.fft.irfftn(
            (rho * weight_functions).sum(axis=0), s=self.grid.n_grid)  # axes=(1, 2, 3)

        return wd

    # def weighted_density(self, density: Array, weight_functions: Array) -> Array:
    #     """Weighted density of PC-SAFT hard-sphere contribution.

    #     Parameters
    #     ----------
    #     density : Tensor
    #         Density profile.
    #     weight_functions : torch.Tensor
    #         Weight functions of PC-SAFT hard-sphere contribution.

    #     Returns
    #     -------
    #     Weighted densities of PC-SAFT hard-sphere contribution.
    #     """
    #     rho = jnp.fft.rfftn(density, s=self.grid.n_grid)[:, None]
    #     wd = list()
    #     print(weight_functions.shape)
    #     for w in weight_functions:
    #         print(rho.shape, w.shape, jnp.sum(rho * w, axis=0).shape)
    #         wd.append(jnp.fft.irfftn(
    #             jnp.sum(rho * w, axis=0), axes=(1, 2, 3)))

    #     return jnp.stack(wd)

    def helmholtz_energy_density(self, weighted_density: Array, _temperature: float) -> Array:
        """Helmholtz energy density of PC-SAFT hard-sphere contribution.

        Parameters
        ----------
        weighted_density : Tensor
            Weighted densities of hard-sphere contribution.
        temperature : float
            Temperature in Kelvin at which the DFT calculation if performed.

        Returns
        -------
        Helmholtz energy density of PC-SAFT hard-sphere contribution.
        """
        n = self.non_negative_weighted_density(weighted_density[0:4])

        dim = self.grid.dim
        nv1nv2 = jnp.clip(
            (weighted_density[4:4+dim] *
             weighted_density[4+dim:]).sum(axis=0), a_max=n[1] * n[2]
        )
        nv2nv2 = jnp.clip(
            (weighted_density[4+dim:]**2).sum(axis=0), a_max=n[2]**2
        )

        ln_1n3 = jnp.log(1 - n[3])
        inv_1n3 = 1.0 / (1 - n[3])

        f3 = jnp.where(
            n[3] > 1.0e-4,
            (n[3] + (1 - n[3])**2 * ln_1n3) *
            inv_1n3**2 / (36.0 * jnp.pi * n[3]**2),
            (1.0 / jnp.pi * (1.0 / 24.0 + 2.0 *
             n[3] / 27.0 + 5.0 * n[3]**2 / 48.0))
        )

        return (
            -n[0] * ln_1n3
            + (n[1] * n[2] - nv1nv2) * inv_1n3
            + self._phi_fmt_vec(n[2], nv2nv2) * f3
        )

    @staticmethod
    def _phi_fmt_vec(n2, nv2nv2):
        """Produces the original White Bear FMT functional."""
        return n2**3 - 3 * n2 * nv2nv2


class FMTAntiSym(FMT):
    __doc__ = 'Antisymmetrized ' + FMT.__doc__

    @staticmethod
    def _phi_fmt_vec(n2, nv2nv2):
        """Produces the antisymmetrized White Bear FMT functional."""
        return (n2 - nv2nv2 / n2)**3


class FMTPure(FMT):
    """White Bear version of fundamental measure theory.

    Attributes
    ----------
    parameters : PcSaftParameters
        PC-SAFT parameters.
    grid : Grid
        Grid information.

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
        """Hard-sphere contribution of PC-SAFT functional.

        Parameters
        ----------
        parameters : PcSaftParameters
            Parameters for PC-SAFT functionals.
        grid : Grid
            Grid information.

        Returns
        -------
        Hard-sphere contribution of PC-SAFT functional.
        """
        self.parameters = parameters
        self.grid = grid

    def __repr__(self):
        return "Helmholtz energy functional: FMT White Bear."

    def _weight_functions(self, temperature: float) -> Array:
        """Weight functions of PC-SAFT Hard-sphere contribution.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin at which the DFT calculation if performed.

        Returns
        -------
        Weight functions of PC-SAFT Hard-sphere contribution.
        """
        radius = self.parameters._hard_sphere_radius(
            temperature)[(...,) + (None,)*self.grid.dim]

        j0 = jnp.array(spherical_jn(0, radius * self.grid.k_abs))
        j2 = jnp.array(spherical_jn(2, radius * self.grid.k_abs))

        omega3 = 4 / 3 * jnp.pi * radius**3 * (j0 + j2)
        omega = jnp.concatenate([
            # expand dimension from [ncomp, ngrid1, ngrid2, ngrid3] to [ncomp, 1, ngrid1, ngrid2, ngrid3]
            (4 * jnp.pi * radius**2 * j0)[:, None],
            omega3[:, None],
            # has dimension [ncomp, dim, ngrid1, ngrid2, ngrid3]
            -1j * self.grid.k * omega3[:, None]
        ], axis=1)

        if self.grid.lanczos_sigma:
            omega *= self.grid.sigma

        return self.parameters.m[(..., None) + (None,)*self.grid.dim] * omega

    def helmholtz_energy_density(self, weighted_density: Array, temperature: float) -> Array:
        """Helmholtz energy density of PC-SAFT hard-sphere contribution.

        Parameters
        ----------
        weighted_density : Tensor
            Weighted densities of hard-sphere contribution.
        temperature : float
            Temperature in Kelvin at which the DFT calculation if performed.

        Returns
        -------
        Helmholtz energy density of PC-SAFT hard-sphere contribution.
        """
        n = self.non_negative_weighted_density(weighted_density[0:2])

        dim = self.grid.dim

        nv2nv2 = jnp.clip(
            (weighted_density[2:]**2).sum(axis=0), a_max=n[0]**2
        )

        ln_1n3 = jnp.log(1 - n[1])
        inv_1n3 = 1.0 / (1 - n[1])

        f3 = jnp.where(
            n[1] > 1.0e-4,
            (n[1] + (1 - n[1])**2 * ln_1n3) *
            inv_1n3**2 / (36.0 * jnp.pi * n[1]**2),
            (1.0 / jnp.pi * (1.0 / 24.0 + 2.0 *
             n[1] / 27.0 + 5.0 * n[1]**2 / 48.0))
        )
        radius = self.parameters._hard_sphere_radius(temperature)[0]

        return (
            (-n[0] / (4.0 * jnp.pi * radius**2) * ln_1n3
             + (n[0]**2 / (4.0 * jnp.pi * radius) -
                nv2nv2 / (4.0 * jnp.pi * radius)) * inv_1n3
             + self._phi_fmt_vec(n[0], nv2nv2) * f3)
        )

    @staticmethod
    def _phi_fmt_vec(n2, nv2nv2):
        """Produces the original White Bear FMT functional."""
        return n2**3 - 3 * n2 * nv2nv2


class FMTAntiSymPure(FMTPure):
    __doc__ = 'Antisymmetrized ' + FMT.__doc__

    @staticmethod
    def _phi_fmt_vec(n2, nv2nv2):
        """Produces the antisymmetrized White Bear FMT functional."""
        return (n2 - nv2nv2 / n2)**3
