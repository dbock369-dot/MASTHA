"""Pure PC-SAFT Helmholtz energy functional."""
import jax.numpy as jnp
from jax import Array
from scipy.special import spherical_jn

from . import HelmholtzFunctional
from ..grid import Grid
from ..parameters import PcSaftParameters


class PurePcSaft(HelmholtzFunctional):
    """Pure PC-SAFT Helmholtz energy functional.

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
        """Pure PC-SAFT functional.

        Parameters
        ----------
        parameters : PcSaftParameters
            Parameters for PC-SAFT functionals.
        grid : Grid
            Grid information.

        Returns
        -------
        Pure PC-SAFT functional.
        """
        self.parameters = parameters
        self.grid = grid
        self.A0 = jnp.array([0.91056314451539, 0.63612814494991, 2.68613478913903,
                             -26.5473624914884, 97.7592087835073, -159.591540865600,
                             91.2977740839123],)
        self.A1 = jnp.array([-0.30840169182720, 0.18605311591713, -2.50300472586548,
                             21.4197936296668, -65.2558853303492, 83.3186804808856,
                             -33.7469229297323])
        self.A2 = jnp.array([-0.09061483509767, 0.45278428063920, 0.59627007280101,
                             -1.72418291311787, -4.13021125311661, 13.7766318697211,
                             -8.67284703679646])

        self.B0 = jnp.array([0.72409469413165, 2.23827918609380, -4.00258494846342,
                             -21.00357681484648, 26.85564136266150, 206.55133840661881,
                             -355.60235612207947])
        self.B1 = jnp.array([-0.57554980753450, 0.69950955214436, 3.89256733895307,
                             -17.21547164777212, 192.67226446524950, -161.82646164876479,
                             -165.20769345556070])
        self.B2 = jnp.array([0.09768831158356, -0.25575749816100, -9.15585615297321,
                             20.64207597439724, -38.80443005206285, 93.62677407701460,
                             -29.66690558514725])
        self.PSI = 1.3862

    def __repr__(self):
        return "Helmholtz energy functional: Pure component PC-SAFT."

    def _weight_functions(self, temperature: float) -> Array:
        """Weight functions of PC-SAFT pure functional.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin at which the DFT calculation if performed.

        Returns
        -------
        Weight functions of PC-SAFT pure functional.
        """
        radius = self.parameters._hard_sphere_radius(
            temperature)[(...,) + (None,)*self.grid.dim]

        j0_fmt = jnp.array(spherical_jn(0, radius * self.grid.k_abs))
        j2_fmt = jnp.array(spherical_jn(2, radius * self.grid.k_abs))
        j0_hc = jnp.array(spherical_jn(0, 2.0 * radius * self.grid.k_abs))
        j2_hc = jnp.array(spherical_jn(2, 2.0 * radius * self.grid.k_abs))

        # Order: rho, n_2, n_3, n_lambda, n_zeta3, n_disp, n_v2x, n_v2y, n_v2z
        omega = jnp.concatenate([
            # FMT n2, n3; expand dimension from [ngrid1, ngrid2, ngrid3] to [1, ngrid1, ngrid2, ngrid3
            (4 * jnp.pi * radius**2 * j0_fmt)[None],
            (4 / 3 * jnp.pi * radius**3 * (j0_fmt + j2_fmt))[None],
            # HC lambda, zeta3; expand dimension from [ngrid1, ngrid2, ngrid3] to [1, ngrid1, ngrid2, ngrid3
            (j0_hc / self.parameters.m)[None],
            (jnp.pi / 6.0 * (2.0 * radius)**3 * (j0_hc + j2_hc))[None],
            # DISP; expand dimension from [ngrid1, ngrid2, ngrid3] to [1, ngrid1, ngrid2, ngrid3
            jnp.array(spherical_jn(0, 2.0 * self.PSI * radius * self.grid.k_abs) +
                      spherical_jn(2, 2.0 * self.PSI * radius * self.grid.k_abs))[None],
            # FMT nv2; has dimension [dim, ngrid1, ngrid2, ngrid3]
            - 1j * self.grid.k * (4 / 3 * jnp.pi * radius **
                                  3 * (j0_fmt + j2_fmt))[None]
        ], axis=1)

        if self.grid.lanczos_sigma:
            omega *= self.grid.sigma

        return self.parameters.m * omega

    def weighted_density(self, density: Array, weight_functions: Array) -> Array:
        """Weighted densities of the pure PC-SAFT functional.

        Parameters
        ----------
        density : Tensor
            Density profile.
        weight_functions : torch.Tensor
            Weight functions of PC-SAFT pure functional.

        Returns
        -------
        Weighted densities of PC-SAFT pure functional.
        """
        rho = jnp.fft.rfftn(density, s=self.grid.n_grid)[:, None]
        wd = jnp.fft.irfftn(rho * weight_functions, s=self.grid.n_grid)

        return jnp.concatenate([density[None], wd], axis=1)

    def helmholtz_energy_density(self, weighted_density: Array, temperature: float) -> Array:
        """Helmholtz energy density of PC-SAFT pure functional.

        Parameters
        ----------
        weighted_density : Tensor
            Weighted densities of the PC-SAFT pure functional.
        temperature : float
            Temperature in Kelvin at which the DFT calculation if performed.

        Returns
        -------
        Helmholtz energy density of PC-SAFT pure functional.
        """
        n = self.non_negative_weighted_density(weighted_density[0:6])[0]

        # FMT
        nv2nv2 = jnp.clip(
            (weighted_density[0, 6:]**2).sum(axis=0), a_max=n[1]**2
        )

        ln_1n3 = jnp.log(1 - n[2])
        inv_1n3 = 1.0 / (1 - n[2])

        f3 = jnp.where(
            n[2] > 1.0e-4,
            (n[2] + (1 - n[2])**2 * ln_1n3) *
            inv_1n3**2 / (36.0 * jnp.pi * n[2]**2),
            (1.0 / jnp.pi * (1.0 / 24.0 + 2.0 *
             n[2] / 27.0 + 5.0 * n[2]**2 / 48.0))
        )

        radius = self.parameters._hard_sphere_radius(temperature)[
            0, None, None, None]
        f_fmt = (-n[1] / (4.0 * jnp.pi * radius**2) * ln_1n3
                 + (n[1]**2 / (4.0 * jnp.pi *
                               radius) - nv2nv2 / (4.0 * jnp.pi * radius)) * inv_1n3
                 + self._phi_fmt_vec(n[1], nv2nv2) * f3)

        # Hard Chain
        _inv_1zeta3 = 1.0 / (1.0 - n[4])
        y_dd = _inv_1zeta3 + \
            (0.5 * n[4] * _inv_1zeta3**2 * (3.0 + n[4] * _inv_1zeta3))
        f_hc = -(self.parameters.m - 1.0) * \
            n[0] * (jnp.log(y_dd * n[3]) - 1.0)

        # Dispersion
        eta = 4.0 / 3.0 * jnp.pi * n[5] * radius**3
        _m1 = ((self.parameters.m - 1.0) / self.parameters.m)
        _m2 = ((self.parameters.m - 2.0) / self.parameters.m)
        i1 = jnp.zeros(self.grid.n_grid)
        c1i2 = jnp.zeros(self.grid.n_grid)

        for i in range(0, 7):
            i1 += (self.A0[i] + _m1 * self.A1[i]
                   + _m1 * _m2 * self.A2[i]) * eta**i
            c1i2 += (self.B0[i] + _m1 * self.B1[i] +
                     _m1 * _m2 * self.B2[i]) * eta**i
        c1i2 *= (1.0 / (1.0 + self.parameters.m * (8.0*eta - 2.0*eta**2) / (1.0 - eta)**4
                        + (1.0 - self.parameters.m) *
                        (20.0*eta - 27.0*eta**2 + 12.0*eta**3 - 2.0*eta**4)
                        / ((1.0 - eta) * (2.0 - eta))**2))

        f_disp = (- 2.0 * jnp.pi * i1 - jnp.pi * self.parameters.m *
                  c1i2 * self.parameters.epsilon_k / temperature) * n[5]**2 * self.parameters.epsilon_k / temperature * self.parameters.sigma**3

        return f_fmt + f_hc + f_disp

    @staticmethod
    def _phi_fmt_vec(n2, nv2nv2):
        """Produces the original White Bear FMT functional."""
        return n2**3 - 3 * n2 * nv2nv2


class PurePcSaftAntiSym(PurePcSaft):
    __doc__ = 'Antisymmetrized ' + PurePcSaft.__doc__

    @ staticmethod
    def _phi_fmt_vec(n2, nv2nv2):
        """Produces the antisymmetrized White Bear FMT functional."""
        return (n2 - nv2nv2 / n2)**3
