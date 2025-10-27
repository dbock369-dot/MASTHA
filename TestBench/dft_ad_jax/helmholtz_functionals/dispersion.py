import jax.numpy as jnp
from jax import Array
from scipy.special import spherical_jn

from . import HelmholtzFunctional
from ..parameters import PcSaftParameters
from ..grid import Grid


class Dispersion(HelmholtzFunctional):
    """PC-SAFT dispersion Helmholtz energy functional. 

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
    helmholtz_energy_density(grid, weighted_density, temperature):
        Calculates the Helmholtz energy density from weighted densities.
    """

    def __init__(self, parameters: PcSaftParameters, grid: Grid):
        """Dispersion contribution of PC-SAFT functional. 

        Parameters
        ----------
        parameters : PcSaftParameters
            Parameters for PC-SAFT functionals.
        grid : Grid
            Grid information.

        Returns
        -------
        Dispersion contribution of PC-SAFT functional. 
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

    def _weight_functions(self, temperature: float) -> Array:
        """Weight functions of PC-SAFT dispersion contribution.

        Parameters
        ----------
        temperature : float 
            Temperature in Kelvin at which the DFT calculation if performed. 

        Returns
        -------
        Weight functions of PC-SAFT dispersion contribution.
        """
        radius = self.parameters._hard_sphere_radius(
            temperature)[(...,) + (None,)*self.grid.dim]

        # Could be calculated on GPU (preventing .cpu() by coding `jn()` by hand)
        omega = jnp.array(spherical_jn(0, 2.0 * self.PSI * radius * self.grid.k_abs) +
                          spherical_jn(2, 2.0 * self.PSI * radius * self.grid.k_abs))

        if self.grid.lanczos_sigma:
            omega *= self.grid.sigma

        return self.parameters.m[(...,) + (None,)*self.grid.dim] * omega

    def weighted_density(self, density: Array, weight_functions: Array) -> Array:
        """Weighted density of PC-SAFT dispersion contribution.

        Parameters
        ----------
        density : Tensor
            Density profile. 
        weight_functions : 
            Weight functions of PC-SAFT dispersion contribution.

        Returns
        -------
        Weighted densities of PC-SAFT dispersion contribution.
        """
        rho = jnp.fft.rfftn(density, s=self.grid.n_grid)
        wd = jnp.fft.irfftn(rho * weight_functions, s=self.grid.n_grid)
        return wd

    def helmholtz_energy_density(self, weighted_density: Array, temperature: float) -> Array:
        """Helmholtz energy density of PC-SAFT dispersion contribution.

        Parameters
        ----------
        weighted_density : Tensor
            Weighted densities of dispersion contribution. 
        temperature : float 
            Temperature in Kelvin at which the DFT calculation if performed. 

        Returns
        -------
        Helmholtz energy density of PC-SAFT dispersion contribution.
        """
        n = self.non_negative_weighted_density(weighted_density)
        radius = self.parameters._hard_sphere_radius(
            temperature)[(...,) + (None,)*self.grid.dim]

        # Configuration integral representation
        eta = (4.0 / 3.0 * jnp.pi * n * radius**3).sum(axis=0)

        # All of these have dimensions of grid
        m_hat = n.sum(axis=0) / \
            (n / self.parameters.m[(...,) + (None,)*self.grid.dim]).sum(axis=0)
        _m1 = ((m_hat - 1.0) / m_hat)
        _m2 = ((m_hat - 2.0) / m_hat)
        i1 = jnp.zeros(self.grid.n_grid)
        c1i2 = jnp.zeros(self.grid.n_grid)

        for i in range(0, 7):
            i1 += (self.A0[i] + _m1 * self.A1[i]
                   + _m1 * _m2 * self.A2[i]) * eta**i
            c1i2 += (self.B0[i] + _m1 * self.B1[i] +
                     _m1 * _m2 * self.B2[i]) * eta**i

        # Reuse memory and multiply by `C_1`
        c1i2 *= (1.0 / (1.0 + m_hat * (8.0*eta - 2.0*eta**2) / (1.0 - eta)**4
                        + (1.0 - m_hat) *
                        (20.0*eta - 27.0*eta**2 + 12.0*eta**3 - 2.0*eta**4)
                        / ((1.0 - eta) * (2.0 - eta))**2))

        # Helmholtz energy density
        e1sig3 = self.parameters.eps_ij_k / temperature * self.parameters.sigma_ij**3
        e2sig3 = (self.parameters.eps_ij_k / temperature)**2 * \
            self.parameters.sigma_ij**3

        helmholtz_energy_density = - 2.0 * jnp.pi * i1 * \
            (n[None] * n[:, None] *
             e1sig3[(...,) + (None,)*self.grid.dim]).sum(axis=(0, 1))
        helmholtz_energy_density += -jnp.pi * m_hat * c1i2 * \
            (n[None] * n[:, None] *
             e2sig3[(...,) + (None,)*self.grid.dim]).sum(axis=(0, 1))

        return helmholtz_energy_density


class DispersionPure(Dispersion):
    """PC-SAFT dispersion Helmholtz energy functional. 

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
    helmholtz_energy_density(grid, weighted_density, temperature):
        Calculates the Helmholtz energy density from weighted densities.
    """

    def helmholtz_energy_density(self, weighted_density: Array, temperature: float) -> Array:
        """Helmholtz energy density of PC-SAFT dispersion contribution.

        Parameters
        ----------
        weighted_density : Tensor
            Weighted densities of dispersion contribution. 
        temperature : float 
            Temperature in Kelvin at which the DFT calculation if performed. 

        Returns
        -------
        Helmholtz energy density of PC-SAFT dispersion contribution.
        """
        n = self.non_negative_weighted_density(weighted_density)[0]
        m = self.parameters.m[0]

        # Configuration integral representation
        eta = 4.0 / 3.0 * jnp.pi * n * self.parameters._hard_sphere_radius(
            temperature)[0]**3

        # All of these have dimensions of grid
        i1 = jnp.zeros(self.grid.n_grid)
        c1i2 = jnp.zeros(self.grid.n_grid)

        for i in range(0, 7):
            i1 += (self.A0[i] + (m - 1.0) / m * self.A1[i]
                   + (m - 1.0) / m * (m - 2.0) / m * self.A2[i]) * eta**i
            c1i2 += (self.B0[i] + ((m - 1.0) / m) * self.B1[i] +
                     (m - 1.0) / m * (m - 2.0) / m * self.B2[i]) * eta**i

        # Reuse memory and multiply by `C_1`
        c1i2 *= (1.0 / (1.0 + m * (8.0*eta - 2.0*eta**2) / (1.0 - eta)**4
                        + (1.0 - m) * (20.0*eta - 27.0*eta **
                                       2 + 12.0*eta**3 - 2.0*eta**4)
                        / ((1.0 - eta) * (2.0 - eta))**2))

        # Helmholtz energy density
        return (- 2.0 * jnp.pi * i1 - jnp.pi * m *
                c1i2 * self.parameters.epsilon_k / temperature) *\
            (n**2 * self.parameters.epsilon_k /
             temperature * self.parameters.sigma**3)
