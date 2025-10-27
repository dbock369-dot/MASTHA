"""Abstract base class for Helmholtz energy functionals. 

The module defines the abstract class for 
* contribtutions to the Helmholtz energy functionals
* weighted densities
* Helmholtz energy density & the Helmholtz energy
"""

import abc
import jax.numpy as jnp
from jax import Array


class HelmholtzFunctional(abc.ABC):
    """Abstract base class for Helmholtz energy contributions."""

    @abc.abstractmethod
    def _weight_functions(self, temperature: float) -> Array:
        """Weight functions."""

    @abc.abstractmethod
    def weighted_density(self, density: Array) -> Array:
        """Weighted densities for the Helmholtz energy functional contribution.

        .. math::

            \\text{weighted densities: }\\bar{rho}_i(\\mathbf{r})

        Args:
            rho: partial number density profiles of all components in Angstrom^-3

        Returns:
            Weighted densities for the Helmholtz energy functional.
        """

    @abc.abstractmethod
    def helmholtz_energy_density(self, density: Array) -> Array:
        """Helmholtz energy density.

        .. math::

            \\text{reduced helmholtz energy density: }\\beta f = \\frac{\\beta F}{V k_B T}

        Args:
            rho: partial number density profiles of all components in Angstrom^-3

        Returns:
            Helmholtz energy density of the density profile. 
        """

    @staticmethod
    def non_negative_weighted_density(weighted_density: Array) -> Array:
        """Clean negative or exact zero weighted densities. 

        Parameters
        ----------
        weighted_density : array_like
            Weighted densities occurring in Helmholtz energy densities. 

        Returns
        -------
        Non-negative and non-zero weighted densities.
        """
        tiny = jnp.finfo(weighted_density.dtype).smallest_normal
        return jnp.abs(weighted_density) + tiny
        # return jnp.where(
        #     # check if positive
        #     ~jnp.signbit(weighted_density),
        #     # if positive: just add tiny number to prevent exact zero
        #     weighted_density,
        #     # if negative: switch sign to positive and add tiny number to prevent exact zero
        #     -weighted_density
        # ) + tiny
