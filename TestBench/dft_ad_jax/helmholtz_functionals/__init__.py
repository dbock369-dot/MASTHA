"""Helmholtz energy functional contributions to the PC-SAFT equation of state
functionals (:mod:`dft_ad.helmholtz_functionals`).

.. currentmodule:: dft_ad.helmholtz_functionals

PyDFTSAFT ``helmholtz_functionals`` provides classes for the Helmholtz energy
contributions to the PC-SAFT equation of state functionals.

.. math::
    F[\\rho(\\mathbf{r})] = F^\mathrm{ig}[\\rho(\\mathbf{r})] + F^\mathrm{hs}[\\rho(\\mathbf{r})] + F^\mathrm{hc}[\\rho(\\mathbf{r})] + F^\mathrm{disp}[\\rho(\\mathbf{r})]


Derived from the abstract base class HelmholtzFunctional, the contributions to
the PC-SAFT equation of state functionals share the following functions and
objects:

.. autosummary::
    :toctree: generated/

    weight_functions - calculates weight functions
    weighted_density - calculate weighted densities
    helmholtz_energy_density - Helmholtz energy density contribution
"""

from .helmholtzfunctional import HelmholtzFunctional
from .fmt import FMT, FMTAntiSym, FMTPure, FMTAntiSymPure
from .hardchain import HardChain, HardChainPure
from .dispersion import Dispersion, DispersionPure
from .pure_pcsaft import PurePcSaft, PurePcSaftAntiSym

__all__ = ['HelmholtzFunctional', 'FMT', 'HardChain',
           'Dispersion', 'FMTAntiSym', 'PurePcSaft', 'PurePcSaftAntiSym',
           'FMTPure', 'FMTAntiSymPure', 'HardChainPure', 'DispersionPure']
