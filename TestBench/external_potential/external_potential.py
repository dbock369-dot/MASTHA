"""TODO. Visualization functionality should be here!!"""

from dataclasses import dataclass, field
import json

import jax.numpy as jnp
from jax import Array


from .interaction_potentials import InteractionPotential


@dataclass(frozen=True)
class ExternalPotential:
    r"""A data container for a calculated external potential energy field.

    This class stores the 3D potential energy grid for a given framework,
    along with all the metadata required to reproduce or interpret the
    calculation. It acts as a comprehensive record of a simulation setup
    and its primary result.

    Attributes:
        formula (str): The chemical formula of the unit cell.
        cif (str): The file name of the framework's CIF file.
        block_files (list | None): File names of the block files.
        fluid_parameters (dict): The fluid parameters used in the calculation.
        solid_parameters (dict): The solid parameters used in the calculation.
        binary_interaction_parameters (dict): The binary correction
            parameters used.
        lengths (tuple): The unit cell lattice lengths (a, b, c).
        skew_angles (tuple): The unit cell lattice angles (alpha, beta, gamma).
        ngrid (tuple): The resolution of the potential grid (nx, ny, nz).
        interaction_potential (InteractionPotential): The potential model used.
        cutoff_radius (float): The cutoff radius used in the calculation.
        tail_correction_type (str | None): Type of tail-correction used.
    """

    formula: str
    cif: str
    block_files: list[str] | None

    ngrid: tuple[int, int, int] = field(init=False, repr=True)
    ncomp: int = field(init=False, repr=True)
    lengths: tuple[float, float, float]
    skew_angles: tuple[float, float, float]

    interaction_potential: InteractionPotential
    cutoff_radius: float
    tail_correction_type: str | None
    fluid_parameters: dict[str, dict[str, float]]
    solid_parameters: dict[str, dict[str, float]]
    binary_interaction_parameters: dict[tuple[str, str], float]

    _external_potential_k: Array = field(init=True, repr=False)
    _external_potential_k_lin_t: Array = field(init=True, repr=False)
    _external_potential_k_blocked: Array = field(init=True, repr=False)
    _tail_correction_k: Array = field(init=True, repr=False)

    def __post_init__(self):
        """Initializes `ngrid` and `ncomp` from initialization data."""
        # self.ngrid = self._external_potential_k.shape[1:]
        # self.ncomp = self._external_potential_k.shape[0]
        object.__setattr__(self, "ngrid", self._external_potential_k.shape[1:])
        object.__setattr__(self, "ncomp", self._external_potential_k.shape[0])

    @property
    def temperature_dependent(self) -> bool:
        r"""Indicator if external potential depends on temperature."""
        self.temperature_dependent = (
            self.interaction_potential.temperature_dependent
        )

    @property
    def lattice_matrix(self) -> Array:
        r"""Calculates and returns the 3x3 lattice matrix (vectors as rows).

        The matrix is derived from the unit cell's lengths and angles and
        is used to convert fractional coordinates to Cartesian coordinates.

        Returns:
            A 3x3 JAX array representing the lattice matrix.
        """
        # Convert angles from degrees to radians for trigonometric functions.
        skew_angles_rad = jnp.deg2rad(jnp.asarray(self.skew_angles))

        # Calculate help variables
        cos_alpha, cos_beta, cos_gamma = jnp.cos(skew_angles_rad)
        sin_gamma = jnp.sin(skew_angles_rad[2])
        # Prevent degenerate lattice
        zeta = (cos_alpha - cos_gamma * cos_beta) / sin_gamma
        # Prevent negative arguments
        nu = jnp.sqrt(jnp.maximum(0.0, 1 - cos_beta**2 - zeta**2))

        # Angle matrix
        angle_matrix = jnp.array(
            [[1, cos_gamma, cos_beta], [0, sin_gamma, zeta], [0, 0, nu]]
        )

        return (angle_matrix * jnp.asarray(self.lengths)).transpose()

    @property
    def external_potential_k(self) -> Array:
        r"""The temperature-independent external potential grid in units of K.

        Raises:
            ValueError: If the potential model is temperature-dependent, as
                the potential is not fully defined without a temperature.
                Use `reduced_potential` instead.
        """
        if self.interaction_potential.temperature_dependent:
            raise ValueError(
                f"The used {self.interaction_potential} potential is temperature-dependent. Only `reduced_potential` can be calculated."
            )
        else:
            return (
                self._external_potential_k
                + self._external_potential_k_blocked
                + self._tail_correction_k[:, None, None, None]
            )

    @property
    def external_potential_k_linear_temperature_dependence(self) -> Array:
        r"""The linear temperature-dependent component of the potential.

        This represents the numerator of the term that is divided by
        temperature (e.g., the Feynman-Hibbs correction term, $U_{FH}$).

        Returns:
            A JAX array of the temperature-dependent component.
        """
        return self._external_potential_k_lin_t

    @property
    def tail_correction_k(self) -> Array:
        r"""Value of the tail correction per component.

        Componentwise value of the tail correction of the specified type,
        e.g. `RASPA`, `Ewald`, or None.

        Returns:
            A JAX array of the componentwise tail-correction.
        """
        return self._tail_correction_k

    @property
    def external_potential_k_blocked(self) -> Array:
        r"""The external potential from blocked pores.

        If pores are blocked using block files, this represents these blocked
        pores by an external potential being `jnp.inf` or zero.

        Returns:
            A JAX array of the blocked-pore external potential.
        """
        return self._external_potential_k_blocked

    def reduced_potential(
        self, temperature: float, maximum_reduced_energy: float
    ) -> Array:
        r"""Calculates the reduced external potential (\phi/kT) at a temperature.

        This method computes the dimensionless potential `U/kT` that is
        required for statistical mechanics calculations. It correctly handles
        both temperature-independent and dependent models.

        Args:
            temperature: The system temperature in Kelvin.
            maximum_reduced_energy: A ceiling value to prevent the reduced
                potential from becoming excessively large or infinite, which
                can cause numerical instability.

        Returns:
            The dimensionless (`\phi/kT`) potential grid as a JAX array.
        """
        # Limit maximum value of external potential to `maximum_reduced_energy`.
        external_potential_kt = (
            self._external_potential_k
            + self._external_potential_k_lin_t / temperature
            + self._external_potential_k_blocked
            + self._tail_correction_k[:, None, None, None]
        ) / temperature
        return jnp.where(
            external_potential_kt < maximum_reduced_energy,
            external_potential_kt,
            maximum_reduced_energy,
        )

    def save(self, filepath: str):
        """Saves the potential to a JSON metadata file and an NPZ array file.

        This method serializes the dataclass instance into two parts:
        1. A JSON file containing all serializable metadata.
        2. A compressed NPZ file for the large JAX numerical arrays.

        The name of the NPZ file is included in the JSON metadata for easy association.

        Args:
            filepath: The base path and name for the output files,
                      e.g., 'data/my_potential'. The method will append
                      '.json' and '.npz' automatically.
        """
        # Define paths for the JSON and NPZ files.
        json_path = f"{filepath}.json"
        npz_path = f"{filepath}.npz"

        # Save the large JAX arrays to a compressed NPZ file.
        jnp.savez(
            npz_path,
            external_potential_k=self._external_potential_k,
            external_potential_k_lin_t=self._external_potential_k_lin_t,
            external_potential_k_blocked=self._external_potential_k_blocked,
            tail_correction_k=self._tail_correction_k,
        )

        # Create the metadata dictionary for JSON serialization.
        metadata = {
            # Add class references for easier deserialization later.
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            # Add the filename of the corresponding array data.
            "npz_filename": npz_path,
            # Include all other serializable attributes.
            "formula": self.formula,
            "cif": self.cif,
            "block_files": self.block_files,
            "ngrid": self.ngrid,
            "ncomp": self.ncomp,
            "lengths": self.lengths,
            "skew_angles": self.skew_angles,
            "cutoff_radius": self.cutoff_radius,
            "tail_correction_type": self.tail_correction_type,
            # For the InteractionPotential object, save its class name.
            # This allows you to reconstruct it during loading.
            "interaction_potential_name": self.interaction_potential.__class__.__name__,
            "fluid_parameters": self.fluid_parameters,
            "solid_parameters": self.solid_parameters,
            "binary_interaction_parameters": [
                {"pair": key, "parameters": value}
                for key, value in self.binary_interaction_parameters.items()
            ],
        }

        # Save the metadata dictionary to a JSON file.
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "ExternalPotential":
        """Loads an ExternalPotential instance from its JSON and NPZ files.

        This factory method reads the metadata and array data from the
        specified file paths and reconstructs a complete ExternalPotential object.

        Args:
            filepath: The base path and name of the files to load (without extension).

        Returns:
            A new instance of the ExternalPotential class.

        Raises:
            FileNotFoundError: If the JSON or NPZ file cannot be found.
            TypeError: If the InteractionPotential subclass cannot be found.
        """
        # Define path to JSON file and load the metadata.
        json_path = f"{filepath}.json"
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Load the array data from the corresponding NPZ file.
        npz_path = metadata["npz_filename"]
        with jnp.load(npz_path) as data:
            external_potential_k = jnp.asarray(data["external_potential_k"])
            external_potential_k_lin_t = jnp.asarray(
                data["external_potential_k_lin_t"]
            )
            external_potential_k_blocked = jnp.asarray(
                data["external_potential_k_blocked"]
            )
            tail_correction_k = jnp.asarray(data["tail_correction_k"])

        # Dynamically find and instantiate the InteractionPotential subclass.
        potential_class_name = metadata["interaction_potential_name"]
        interaction_potential_instance = None

        # This search requires the InteractionPotential subclasses to be imported.
        # into the scope where .load() is called.
        for subclass in InteractionPotential.__subclasses__():
            if subclass.__name__ == potential_class_name:
                # Assuming the potential class has a no-argument constructor.
                interaction_potential_instance = subclass()
                break
        if interaction_potential_instance is None:
            raise TypeError(
                f"Could not find InteractionPotential subclass named '{potential_class_name}'."
                "Ensure it is imported correctly."
            )

        # Reconstruct the dictionary from the list of objects.
        reconstructed_binary_params = {
            tuple(item["pair"]): item["parameters"]
            for item in metadata["binary_interaction_parameters"]
        }

        # Instantiate the main class with all the reconstructed data.
        return cls(
            formula=metadata["formula"],
            cif=metadata["cif"],
            block_files=metadata["block_files"],
            lengths=tuple(metadata["lengths"]),  # Ensure it's a tuple
            skew_angles=tuple(metadata["skew_angles"]),  # Ensure it's a tuple
            interaction_potential=interaction_potential_instance,
            cutoff_radius=metadata["cutoff_radius"],
            tail_correction_type=metadata["tail_correction_type"],
            fluid_parameters=metadata["fluid_parameters"],
            solid_parameters=metadata["solid_parameters"],
            binary_interaction_parameters=reconstructed_binary_params,
            _external_potential_k=external_potential_k,
            _external_potential_k_lin_t=external_potential_k_lin_t,
            _external_potential_k_blocked=external_potential_k_blocked,
            _tail_correction_k=tail_correction_k,
        )


#     # def boltzmann_factor(self, temperature: float) -> Array:
#     #     """Calculates the Boltzmann factor, exp(-V/kT).

#     #     This factor is fundamental for calculating properties like
#     #     adsorption isotherms, representing the relative probability of a
#     #     particle occupying a position on the grid.

#     #     Args:
#     #         temperature: The system temperature in Kelvin.
#     #         maximum_reduced_energy: A ceiling value for the reduced potential,
#     #             used to prevent numerical underflow in the exponential.

#     #     Returns:
#     #         A JAX array of the Boltzmann factor at each grid point.
#     #     """
#     #     reduced_potential = self.get_reduced_potential(temperature)
#     #     return jnp.exp(-reduced_potential)

#     # def get_weighted_potential(self, weights: ArrayLike) -> Array:
#     #     """Returns the potential field weighted by a given weight function via convolution.

#     #     This is useful for smoothing or applying filter functions to the potential field.

#     #     Args:
#     #         weights: An array representing the kernel or weight function for the convolution.

#     #     Returns:
#     #         A JAX array of the convoluted (weighted) potential field.
#     #     """
#     #     # We apply the convolution to each component's potential field separately
#     #     # return jax.vmap(lambda p: convolve(p, weights, mode="same"))(
#     #     #     self.potential_grid
#     #     # )

#     # def get_energy_histogram(self, bins: int = 100) -> tuple[Array, Array]:
#     #     """Calculates a histogram of the potential energy values across the grid.

#     #     Args:
#     #         bins: The number of bins to use for the histogram.

#     #     Returns:
#     #         A tuple containing:
#     #             - The histogram counts for each bin.
#     #             - The bin edges.
#     #     """
#     #     # # Flatten the grid to get a 1D array of all potential values
#     #     # all_potential_values = self.potential_grid.flatten()

#     #     # # Use jax.numpy.histogram to calculate the distribution
#     #     # counts, bin_edges = jnp.histogram(all_potential_values, bins=bins)
#     #     # return counts, bin_edges

#     # def as_dict(self) -> dict[str, Any]:
#     #     """TODO."""
#     #     return {
#     #         "@module": self.__class__.__module__,
#     #         "@class": self.__class__.__name__,
#     #         "formula": self.formula,
#     #         "cif": self.cif,
#     #         "lengths": self.lengths,
#     #         "angles": self.angles,
#     #         "ngrid": self.ngrid,
#     #         "cutoff_radius": self.cutoff_radius,
#     #         "maximum_reduced_energy": self.maximum_reduced_energy,
#     #         "temperature_dependent": self.potential_type.temperature_dependent,
#     #     }
