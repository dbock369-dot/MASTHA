from abc import ABC, abstractmethod  # noqa: D100
from operator import itemgetter
from enum import Enum

from jax.typing import ArrayLike
import jax.numpy as jnp

# Physical constants (Boltzmann, Avogadro, atomic mass unit)
KB = 1.380649e-23
NA = 6.02214076e23
AMU = 1.0 / (1.0e3 * NA)  # u in [kg]
HBAR = 6.62607015e-34 / (2.0 * jnp.pi)


def arithmetic_mean_sigma(
    fluid: dict[str, ArrayLike],
    solid: dict[str, ArrayLike],
    binary: dict[str, ArrayLike],
) -> float:
    r"""Calculates the arithmetic mean (Lorentz rule).

    This rule applies a modified arithmetic mean to determine the
    cross-interaction collision diameter (sigma) between fluid and solid
    particles, including a binary correction factor (l_si).
    \[
    \sigma_{is} = \frac{\sigma_{ii} + \sigma_{ss}}{2} (1 - l_{is})
    \]

    Args:
        fluid: A dictionary of fluid parameter arrays. Must contain 'sigma'.
        solid: A dictionary of solid parameter arrays. Must contain 'sigma'.
        binary: A dictionary of binary parameter arrays. Must contain 'l_si'.

    Returns:
        A JAX array of the combined sigma values for each fluid-solid pair.
    """
    return (
        (fluid["sigma"][:, None] + solid["sigma"][None])
        / 2.0
        * (1.0 - binary["l_si"])
    )


def arithmetic_mean_inverse_mass(
    fluid: dict[str, ArrayLike],
    solid: dict[str, ArrayLike],
    binary: dict[str, ArrayLike],
) -> float:
    r"""Calculates the arithmetic mean (Lorentz rule) of the inverse parameters: reduced mass.

    This is required for quantum corrections like the Feynman-Hibbs model.
    The formula used is the harmonic mean of the two molar masses.
    \[
    M_{is} = \frac{1 M_i M_s}{M_i + M_s}
    \]

    Args:
        fluid: A dict of fluid parameter arrays. Must contain 'molarweight'.
        solid: A dict of solid parameter arrays. Must contain 'molarweight'.
        binary: A dict of binary parameter arrays. Not used in this rule.

    Returns:
        A JAX array of the reduced molar mass for each fluid-solid pair.
    """
    return (
        2.0
        * fluid["molarweight"][:, None]
        * solid["molarweight"][None]
        / (fluid["molarweight"][:, None] + solid["molarweight"][None])
    )


def geometric_mean_epsilon(
    fluid: dict[str, ArrayLike],
    solid: dict[str, ArrayLike],
    binary: dict[str, ArrayLike],
) -> float:
    r"""Calculates the geometric mean (Berthelot rule).

    This rule applies a modified geometric mean to determine the
    cross-interaction energy well depth (epsilon) between fluid and solid
    particles, including a binary correction factor (k_si).
    \[
    \varepsilon_{is} = \sqrt{\varepsilon_{ii} \varepsilon_{ss}} (1 - k_{is})
    \]

    Args:
        fluid: A dict of fluid parameter arrays. Must contain 'epsilon_k'.
        solid: A dict of solid parameter arrays. Must contain 'epsilon_k'.
        binary: A dict of binary parameter arrays. Must contain 'k_si'.

    Returns:
        A JAX array of the combined epsilon values for each fluid-solid pair.
    """
    return jnp.sqrt(fluid["epsilon_k"][:, None] * solid["epsilon_k"][None]) * (
        1.0 - binary["k_si"]
    )


def geometric_mean_size_modified_epsilon(
    fluid: dict[str, ArrayLike],
    solid: dict[str, ArrayLike],
    binary: dict[str, ArrayLike],
) -> float:
    r"""Calculates the size-modified geometric mean.

    \[
    \sigma_{is} = \frac{\sigma_{ii} + \sigma_{ss}}{2} (1 - l_{is})
    \]

    \[
    \varepsilon_{is} = \frac{\sqrt{\sigma_{ii}^3 \sigma_{ss}^3}}{\sigma_{is}^3} \sqrt{\varepsilon_{ii} \varepsilon_{ss}} (1 - k_{is})
    \]
    """
    sigma_fluid_solid = arithmetic_mean_sigma(fluid, solid, binary)
    return (
        jnp.sqrt(fluid["sigma"][:, None] ** 3 * solid["sigma"][None] ** 3)
        / sigma_fluid_solid**3
        * jnp.sqrt(fluid["epsilon_k"][:, None] * solid["epsilon_k"][None])
        * (1.0 - binary["k_si"])
    )


def geometric_mean_3_exponents(
    fluid: dict[str, float],
    solid: dict[str, float],
    binary: dict[str, ArrayLike],
):
    r"""Calculates the geometric mean of the exponents minus 3.

    This combining rule adjusts the standard geometric mean (Berthelot)
    for epsilon by a factor related to the individual and combined
    collision diameters (sigma).
    \[
    \lambda_{ij} = 3 + \sqrt{(\lambda_{ii} - 3) (\lambda_{ss} - 3)}
    \]

    fluid: A dict of fluid parameter arrays. Must contain 'sigma' and
            'epsilon_k'.
        solid: A dict of solid parameter arrays. Must contain 'sigma' and
            'epsilon_k'.
        binary: A dict of binary parameter arrays. Must contain 'k_si' and
            'l_si', used by the internal `arithmetic_mean_sigma` call.

    Returns:
        A JAX array of the combined, size-modified epsilon values for each
        fluid-solid pair.
    """
    return 3.0 + jnp.sqrt(
        (fluid["lambda_r"][:, None] - 3.0) * (solid["lambda_r"][None] - 3.0)
    )


class InteractionPotential(ABC):
    """Abstract base class of an interaction potential.

    This class establishes a common interface for all intermolecular potential
    models, such as Lennard-Jones or Mie potentials. It uses `abc.ABC` to
    ensure that any concrete subclass provides a consistent structure.

    It should not be instantiated directly. Instead, it should be subclassed
    to create a specific potential model.

    Class Attributes:
        combining_rules (dict[str, callable]): A dictionary that maps a
            parameter name (e.g., 'sigma') to the function used to calculate
            its value for cross-interactions between different particle types.
        temperature_dependent (bool): A flag that must be set to `True` if
            the potential has an explicit linear dependence on temperature,
            as in the case of Feynman-Hibbs quantum corrections.

    Raises:
        TypeError: If a subclass fails to define the required class
            attributes (`combining_rules`, `temperature_dependent`) or, if
            `temperature_dependent` is True, fails to implement the
            `potential_linear_temperature_dependence` method.
    """

    def __init_subclass__(cls, **kwargs):
        """Additional subclass initialization: Checks if sublcasses have required class attributes.

        This special method is automatically called when `InteractionPotential`
        is subclassed. It ensures that any new potential model adheres to the
        required interface by checking for mandatory class attributes and
        methods.

        Args:
            **kwargs: Catches any keyword arguments passed during subclassing.

        Raises:
            TypeError: If the subclass does not define the `combining_rules` or
                `temperature_dependent` class attributes, or if it sets
                `temperature_dependent` to True without implementing the
                `potential_linear_temperature_dependence` method.
        """
        super().__init_subclass__(**kwargs)
        # Check if the new subclass has defined the required class attributes:
        if not hasattr(cls, "combining_rules"):
            raise TypeError(
                f"Class {cls.__name__} must define a 'combining_rules: callable' class attribute."
            )
        if not hasattr(cls, "temperature_dependent"):
            raise TypeError(
                f"Class {cls.__name__} must define a 'temperature_dependent: bool' class attribute."
            )
        if getattr(cls, "temperature_dependent", False):
            if not hasattr(
                cls, "potential_linear_temperature_dependence"
            ) or not callable(
                getattr(cls, "potential_linear_temperature_dependence")
            ):
                raise TypeError(
                    f"Class {cls.__name__} has 'temperature_dependent=True' and so must implement "
                    f"the 'potential_linear_temperature_dependence' method."
                )

    def __repr__(self) -> str:
        """Returns an official string representation of the potential model."""
        return f"{self.__class__.__name__}({', '.join([f'{key}' for key in self.combining_rules.keys()])})"

    @abstractmethod
    def potential(
        self,
        r: ArrayLike,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        """Calculates the potential energy at a given distance.

        This abstract method must be implemented by all concrete subclasses.
        It defines the core function of the potential model, which computes the
        interaction energy between particles as a function of their separation.

        Args:
            r: A JAX array of distances at which to calculate the potential.
            combined_params: A dictionary containing the combined interaction
                parameters (e.g., 'sigma', 'epsilon_k') necessary for the
                potential calculation.

        Returns:
            A JAX array of the calculated potential energy values.
        """
        raise NotImplementedError

    @abstractmethod
    def tail_correction_raspa(
        self,
        cutoff_radius: float,
        volume_unitcell: float,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        """Calculates the tail-correction assuming a homogeneous systems (approximation).

        This abstract method must be implemented by all concrete subclasses.
        It defines the tail-correction to the interaction energy between
        particles as a function of their separation, assuming a hommogeneous
        system (approximation).

        Not perfekt but makes results less dependent on cutoff radius, see:
        https://doi.org/10.1021/acs.jctc.9b00586

        Args:
            cutoff_radius: Cutoff radius for which the tail-correction is calculated.
            volume_unitcell: Volume of the unitcell.
            combined_params: A dictionary containing the combined interaction
                parameters (e.g., 'sigma', 'epsilon_k') necessary for the
                potential calculation.

        Returns:
            A JAX array of the calculated tail-correction.
        """
        raise NotImplementedError


class LennardJones(InteractionPotential):
    r"""Implements the classic 12-6 Lennard-Jones potential.

    This class provides a concrete implementation of the `InteractionPotential`
    interface for the Lennard-Jones (LJ) potential, which models the
    interaction between two neutral particles.

    \[
    \frac{\phi_{is}}{k_\mathrm{B}} = 4 \frac{\varepsilon_{is}}{k_\mathrm{B}} \left( \left( \frac{\sigma_{is}}{r} \right)^12 -  \left( \frac{\sigma_{is}}{r} \right)^6 \right)
    \]

    Class Attributes:
        combining_rules: Defines the standard Lorentz-Berthelot combining
            rules. `sigma` is combined with an arithmetic mean and
            `epsilon_k` with a geometric mean.
        temperature_dependent: Set to `False` as the classic LJ potential
            does not have an explicit temperature dependence.
    """

    combining_rules = {
        "sigma": arithmetic_mean_sigma,
        "epsilon_k": geometric_mean_epsilon,
    }
    temperature_dependent = False

    def potential(
        self,
        r: ArrayLike,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        r"""Calculates the Lennard-Jones potential energy.

        This method computes the interaction energy at given separation
        distances using the standard 12-6 Lennard-Jones formula:
        \[
        \frac{\phi_{is}}{k_\mathrm{B}} = 4 \frac{\varepsilon_{is}}{k_\mathrm{B}} \left( \left( \frac{\sigma_{is}}{r} \right)^12 -  \left( \frac{\sigma_{is}}{r} \right)^6 \right)
        \]

        Args:
            r: A JAX array of distances (in Angstrom) at which to calculate the potential.
            combined_params: A dictionary containing the combined interaction
                parameters 'sigma' and 'epsilon_k'.

        Returns:
            A JAX array of the calculated potential energy values.
        """
        sigma, epsilon_k = itemgetter(*self.combining_rules.keys())(
            combined_params
        )
        return 4 * epsilon_k * ((sigma / r) ** 12 - (sigma / r) ** 6)

    def tail_correction_raspa(
        self,
        cutoff_radius: float,
        volume_unitcell: float,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        r"""Calculates the Lennard-Jones tail-correction to the potential energy.

        This method computes the interaction energy at given separation
        distances using the 12-6 Lennard-Jones potential:
        \[
        \sum_s\int_{r=r_\mathrm{c}}^\infty\frac{\phi_{is}}{k_\mathrm{B}} \mathrm{d}r
        = \frac{4\pi}{V} \sum_s \frac{4 \varepsilon_{is}}{k_\mathrm{B}} \frac{\sigma_{is}^3}{3}\left( \frac{1}{3}\left( \frac{\sigma_{is}}{r} \right)^9 -  \left( \frac{\sigma_{is}}{r} \right)^3 \right)
        \]

        Args:
            cutoff_radius: Cutoff radius for which the tail-correction is calculated.
            volume_unitcell: Volume of the unitcell.
            combined_params: A dictionary containing the combined interaction
                parameters 'sigma' and 'epsilon_k'.

        Returns:
            A JAX array of the calculated tail-correction.
        """
        sigma, epsilon_k = map(
            jnp.squeeze,
            itemgetter(*self.combining_rules.keys())(combined_params),
        )

        return (
            (4 * jnp.pi / volume_unitcell)
            * (4 * epsilon_k * sigma**3 / 3)
            * ((sigma / cutoff_radius) ** 9 - (sigma / cutoff_radius) ** 3)
        ).sum(axis=-1)


class Mie6(InteractionPotential):
    r"""Implements the Mie (λ-6) potential.

    This class provides a concrete implementation of the `InteractionPotential`
    interface for the Mie potential. This model generalizes the Lennard-Jones
    potential by allowing the repulsive exponent, `lambda_r`, to be a
    variable parameter, offering greater flexibility.
    \[
    \frac{\phi_{is}}{k_\mathrm{B}} = \left( \frac{\lambda_\mathrm{r}}{\lambda_\mathrm{r} - 6} \right) \left( \frac{\lambda_\mathrm{r}}{6} \right)^{\frac{6}{\lambda_\mathrm{r} - 6}} \frac{\varepsilon_{is}}{k_\mathrm{B}} \left( \left( \frac{\sigma_{is}}{r} \right)^{\lambda_\mathrm{r}} -  \left( \frac{\sigma_{is}}{r} \right)^6 \right)
    \]

    The attractive exponent is fixed at 6. The potential is defined by
    three parameters: `sigma`, `epsilon_k`, and the repulsive exponent
    `lambda_r`.

    Class Attributes:
        combining_rules: Defines how to combine parameters for
            cross-interactions. Uses standard Lorentz-Berthelot for 'sigma'
            and 'epsilon_k', and a geometric mean for 'lambda_r - 3'.
        temperature_dependent: Set to `False`.
    """

    combining_rules = {
        "sigma": arithmetic_mean_sigma,
        "epsilon_k": geometric_mean_epsilon,
        "lambda_r": geometric_mean_3_exponents,
    }
    temperature_dependent = False

    def potential(
        self,
        r: ArrayLike,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        r"""Calculates the Mie potential energy.

        This method computes the interaction energy at given separation
        distances using the Mie potential formula:
        \[
        \frac{\phi_{is}}{k_\mathrm{B}} = \left( \frac{\lambda_\mathrm{r}}{\lambda_\mathrm{r} - 6} \right) \left( \frac{\lambda_\mathrm{r}}{6} \right)^{\frac{6}{\lambda_\mathrm{r} - 6}} \frac{\varepsilon_{is}}{k_\mathrm{B}} \left( \left( \frac{\sigma_{is}}{r} \right)^{\lambda_\mathrm{r}} -  \left( \frac{\sigma_{is}}{r} \right)^6 \right)
        \]

        Args:
            r: A JAX array of distances (in Angstrom) at which to calculate the potential.
            combined_params: A dictionary containing the combined interaction
                parameters 'sigma', 'epsilon_k', and 'lambda_r'.

        Returns:
            A JAX array of the calculated potential energy values.
        """
        sigma, epsilon_k, lambda_r = itemgetter(*self.combining_rules.keys())(
            combined_params
        )
        c = (
            lambda_r
            / (lambda_r - 6.0)
            * (lambda_r / 6.0) ** (6.0 / (lambda_r - 6.0))
        )
        return c * epsilon_k * ((sigma / r) ** lambda_r - (sigma / r) ** 6)

    def tail_correction_raspa(
        self,
        cutoff_radius: float,
        volume_unitcell: float,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        r"""Calculates the Mie-6 tail-correction to the potential energy.

        This method computes the interaction energy at given separation
        distances using the Mie-6 potential:
        \[
        \sum_s\int_{r=r_\mathrm{c}}^\infty\frac{\phi_{is}}{k_\mathrm{B}} \mathrm{d}r
        = \frac{4\pi}{V} \sum_s \frac{c \varepsilon_{is}}{k_\mathrm{B}} sigma_{is}^3 \left( \frac{1}{\lambda-3}\left( \frac{\sigma_{is}}{r} \right)^9{\lambda-3} - \left( \frac{\sigma_{is}}{r} \right)^3 \right)
        \]

        Args:
            cutoff_radius: Cutoff radius for which the tail-correction is calculated.
            volume_unitcell: Volume of the unitcell.
            combined_params: A dictionary containing the combined interaction
                parameters 'sigma' and 'epsilon_k'.

        Returns:
            A JAX array of the calculated tail-correction.
        """
        sigma, epsilon_k, lambda_r = map(
            jnp.squeeze,
            itemgetter(*self.combining_rules.keys())(combined_params),
        )
        c = (
            lambda_r
            / (lambda_r - 6.0)
            * (lambda_r / 6.0) ** (6.0 / (lambda_r - 6.0))
        )
        return (
            (4 * jnp.pi / volume_unitcell)
            * (c * epsilon_k * sigma**3)
            * (
                (sigma / cutoff_radius) ** (lambda_r - 3) / (lambda_r - 3)
                - (sigma / cutoff_radius) ** 3 / 3
            )
        ).sum(axis=-1)


class Mie6FH1(InteractionPotential):
    r"""Implements a Mie (λ-6) potential with a quantum correction.

    This class extends the Mie potential by incorporating a first-order
    Feynman-Hibbs (FH) quantum correction. This correction introduces a
    temperature-dependent term to the potential, making it more accurate
    for light particles (e.g., hydrogen or helium) where quantum effects are
    significant.

    The total potential is expressed as $\phi(r, T) = \phi_\mathrm{Mie}(r) + \frac{\phi_\mathrm{FH1}(r)}{T}$.

    Class Attributes:
        combining_rules: Extends the Mie potential's rules by adding one
            for 'molarweight' to calculate the reduced mass needed for the
            Feynman-Hibbs term.
        temperature_dependent: Set to `True` due to the inclusion of the
            quantum correction term.
    """

    combining_rules = {
        "sigma": arithmetic_mean_sigma,
        "epsilon_k": geometric_mean_size_modified_epsilon,
        "lambda_r": geometric_mean_3_exponents,
        "molarweight": arithmetic_mean_inverse_mass,
    }
    temperature_dependent = True

    def potential(
        self,
        r: ArrayLike,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        r"""Calculates the Mie potential energy.

        This method computes the interaction energy at given separation
        distances using the Mie potential formula:
        \[
        \frac{\phi_{is}}{k_\mathrm{B}} = \left( \frac{\lambda_\mathrm{r}}{\lambda_\mathrm{r} - 6} \right) \left( \frac{\lambda_\mathrm{r}}{6} \right)^{\frac{6}{\lambda_\mathrm{r} - 6}} \frac{\varepsilon_{is}}{k_\mathrm{B}} \left( \left( \frac{\sigma_{is}}{r} \right)^{\lambda_\mathrm{r}} -  \left( \frac{\sigma_{is}}{r} \right)^6 \right)
        \]

        Args:
            r: A JAX array of distances (in Angstrom) at which to calculate the potential.
            combined_params: A dictionary containing the combined interaction
                parameters 'sigma', 'epsilon_k', and 'lambda_r'.

        Returns:
            A JAX array of the calculated potential energy values.
        """
        sigma, epsilon_k, lambda_r, _ = itemgetter(
            *self.combining_rules.keys()
        )(combined_params)
        c = (
            lambda_r
            / (lambda_r - 6.0)
            * (lambda_r / 6.0) ** (6.0 / (lambda_r - 6.0))
        )
        return c * epsilon_k * ((sigma / r) ** lambda_r - (sigma / r) ** 6)

    def potential_linear_temperature_dependence(
        self,
        r: ArrayLike,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        r"""Calculates the Feynman-Hibbs quantum correction term.

        This method computes the first-order Feynman-Hibbs (FH) correction
        to the Mie potential. This term, which is proportional to the second
        derivative of the classical potential, accounts for quantum effects like
        zero-point energy.

        The value returned is the numerator of the temperature-dependent part
        of the total potential ($\phi_\mathrm{FH1}$ in $\phi_{is} = \phi_\mathrm{Mie} + \frac{\phi_\mathrm{FH1}}{T}$).

        Args:
            r: A JAX array of distances (in Angstrom) at which to calculate the correction.
            combined_params: A dictionary containing the combined interaction
                parameters 'sigma', 'epsilon_k', 'lambda_r', and
                'reduced_molarweight'.

        Returns:
            A JAX array of the Feynman-Hibbs correction term values.
        """
        # TODO: Docstring richtige Gleichung!!
        sigma, epsilon_k, lambda_r, reduced_molarweight = itemgetter(
            *self.combining_rules.keys()
        )(combined_params)

        c = (
            lambda_r
            / (lambda_r - 6.0)
            * (lambda_r / 6.0) ** (6.0 / (lambda_r - 6.0))
        )
        q1 = lambda_r * (lambda_r - 1.0)
        d = HBAR**2 / (KB * AMU * 1e-20) / reduced_molarweight / 12

        return (
            (c * epsilon_k * d)
            * (q1 * (sigma / r) ** lambda_r - 30 * (sigma / r) ** 6)
            / r**2
        )

    def tail_correction_raspa(
        self,
        cutoff_radius: float,
        volume_unitcell: float,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        r"""Calculates the Mie-6 tail-correction to the potential energy.

        This method computes the interaction energy at given separation
        distances using the Mie-6 potential:
        \[
        \sum_s\int_{r=r_\mathrm{c}}^\infty\frac{\phi_{is}}{k_\mathrm{B}} \mathrm{d}r
        = \frac{4\pi}{V} \sum_s \frac{c \varepsilon_{is}}{k_\mathrm{B}} sigma_{is}^3 \left( \frac{1}{\lambda-3}\left( \frac{\sigma_{is}}{r} \right)^9{\lambda-3} - \left( \frac{\sigma_{is}}{r} \right)^3 \right)
        \]

        Args:
            cutoff_radius: Cutoff radius for which the tail-correction is calculated.
            volume_unitcell: Volume of the unitcell.
            combined_params: A dictionary containing the combined interaction
                parameters 'sigma' and 'epsilon_k'.

        Returns:
            A JAX array of the calculated tail-correction.
        """
        sigma, epsilon_k, lambda_r, reduced_molarweight = map(
            jnp.squeeze,
            itemgetter(*self.combining_rules.keys())(combined_params),
        )

        c = (
            lambda_r
            / (lambda_r - 6.0)
            * (lambda_r / 6.0) ** (6.0 / (lambda_r - 6.0))
        )
        q1 = lambda_r * (lambda_r - 1.0)
        d = HBAR**2 / (KB * AMU * 1e-20) / reduced_molarweight / 12.0

        return (
            (4 * jnp.pi / volume_unitcell)
            * (c * epsilon_k * sigma**3)
            * (
                (sigma / cutoff_radius) ** (lambda_r - 3) / (lambda_r - 3)
                - (sigma / cutoff_radius) ** 3 / 3
            )
            + (4 * jnp.pi / volume_unitcell)
            * (c * epsilon_k * sigma * d)
            * (
                q1 / (lambda_r - 1) * (sigma / cutoff_radius) ** (lambda_r - 1)
                - 30 / 5 * (sigma / cutoff_radius) ** 5
            )
        ).sum(axis=-1)


class HardSphere(InteractionPotential):
    """Implements the classic hard-sphere potential.

    This class provides a concrete implementation of the `InteractionPotential`
    interface for the hard-sphere model. This is one of the simplest
    intermolecular models, representing particles as impenetrable spheres.

    The potential is infinite if the distance between particle centers is
    less than their combined collision diameter (`sigma`), and zero
    otherwise. It is a purely repulsive potential.

    Class Attributes:
        combining_rules: Defines an arithmetic mean for combining `sigma`.
        temperature_dependent: Set to `False`.
    """

    # TODO: Equation of HS-potential
    combining_rules = {"sigma": arithmetic_mean_sigma}
    temperature_dependent = False

    def potential(
        self,
        r: ArrayLike,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        """Calculates the hard-sphere potential energy.

        This method returns `jnp.inf` if the separation distance `r` is
        less than the collision diameter `sigma`, and `0.0` otherwise. This
        reflects the impenetrable nature of the particles.

        Args:
            r: A JAX array of distances (in Angstrom) at which to calculate the potential.
            combined_params: A dictionary containing the combined interaction
                parameter 'sigma'.

        Returns:
            A JAX array of the potential energy values (0.0 or inf).
        """
        sigma = itemgetter(*self.combining_rules.keys())(combined_params)
        return jnp.where(r < sigma, jnp.inf, 0.0)

    def tail_correction_raspa(
        self,
        cutoff_radius: float,
        volume_unitcell: float,
        combined_params: dict[str, ArrayLike],
    ) -> float:
        """TODO."""
        # Remove unnecessary dimensions of individual parameter arrays for each array directly:
        # [n_comp, 1, 1, n_solid_atoms] -> [n_comp, n_solid_atoms]
        sigma = map(
            jnp.squeeze,
            itemgetter(*self.combining_rules.keys())(combined_params),
        )
        raise jnp.zeros(sigma.shape[0])


class PotentialType(Enum):
    """Enumeration of the available interaction potential models.

    This enum provides a convenient and type-safe way to select a specific
    intermolecular potential model for use in calculations. Each member of the
    enum corresponds to a concrete implementation of the
    `InteractionPotential` base class.

    Access the instantiated potential object via the `.value` attribute of a
    member (e.g., `PotentialType.LennardJones.value`).

    Attributes:
        LennardJones: The classic 12-6 Lennard-Jones potential.
        Mie6: The generalized Mie (λ-6) potential with a variable repulsive
            exponent.
        Mie6FH1: The Mie potential including a first-order Feynman-Hibbs
            quantum correction, making it temperature-dependent.
        HardSphere: The purely repulsive hard-sphere model.
    """

    LennardJones = LennardJones()
    Mie6 = Mie6()
    Mie6FH1 = Mie6FH1()
    HardSphere = HardSphere()
