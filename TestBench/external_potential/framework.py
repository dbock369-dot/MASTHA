"""TODO."""

import jax.numpy as jnp
from pymatgen.core import Structure
from os import PathLike
import jax
from jax import Array, lax
from jax.typing import ArrayLike
import itertools
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import functools
import numpy as np

from .interaction_potentials import (
    PotentialType,
    InteractionPotential,
)
from .external_potential import ExternalPotential
from .utils import (
    _get_binary_parameters,
    _build_calculation_grid,
    _extract_binary_interaction_parameters,
    _create_supercell,
)


def _calculate_unit_cell_heights(cell_matrix: ArrayLike) -> Array:
    """Calculates the perpendicular distances between parallel faces for a unit cell.

    This function computes the perpendicular distances between the three pairs of
    parallel faces of a unit cell. These heights are essential for determining
    the minimum supercell size required for a given interaction cutoff radius.
    The calculation is based on the formula: height = volume / area, where
    the area is that of the face perpendicular to the height.

    Args:
        cell_matrix: A 3x3 JAX array-like object where rows represent the lattice
            vectors (a, b, c) in Cartesian coordinates (transforms from skewed
            to Cartesian coordinates).

    Returns:
        A JAX array containing the three calculated heights [h_a, h_b, h_c],
        corresponding to the directions perpendicular to the b-c, c-a,
        and a-b planes.

    Raises:
         ValueError: If the input `cell_matrix` is not of shape (3, 3).
    """
    if not isinstance(cell_matrix, jnp.ndarray):
        # Ensure input is a JAX NumPy array.
        cell_matrix = jnp.asarray(cell_matrix)

    if cell_matrix.shape != (3, 3):
        raise ValueError(
            f"Input cell matrix must be of shape (3, 3), got {cell_matrix.shape}"
        )

    # Axis vectors in Cartesian coordinates.
    a, b, c = cell_matrix[0, :], cell_matrix[1, :], cell_matrix[2, :]

    # Calculate volume of unit cell.
    volume = jnp.abs(jnp.linalg.det(cell_matrix))

    # Handle degenerate case (zero volume).
    zero_precision = 1e-12
    if volume < zero_precision:
        return jnp.array([jnp.inf, jnp.inf, jnp.inf])

    # Calculate face areas using the cross product.
    area_bc = jnp.linalg.norm(jnp.cross(b, c))
    area_ca = jnp.linalg.norm(jnp.cross(c, a))
    area_ab = jnp.linalg.norm(jnp.cross(a, b))

    # Calculate heights, handling potential zero areas explicitly.
    h_a = jnp.inf if area_bc < zero_precision else volume / area_bc
    h_b = jnp.inf if area_ca < zero_precision else volume / area_ca
    h_c = jnp.inf if area_ab < zero_precision else volume / area_ab

    heights = jnp.array([h_a, h_b, h_c])
    return heights


def _find_minimal_supercell(
    cell_matrix: Array, min_required_radius: float, max_cells: int = 100
) -> Array | None:
    """Determines the minimal supercell scaling factors `[nx, ny, nz]`.

    Determines the smallest integer multipliers `[nx, ny, nz]` needed to
    construct a supercell from a unit cell. The method ensures that the
    supercell is large enough to contain a sphere of a given radius, which
    is critical for calculations involving a cutoff distance.

    The required scaling is found by comparing the unit cell's perpendicular
    heights to the target diameter (2 * min_required_radius).

    Args:
        cell_matrix: A 3x3 JAX array with unit cell lattice vectors `a, b, c` as
            rows (transforms from skewed to Cartesian coordinates).
        min_required_radius: The minimum radius (in Angstrom) of a sphere that
            the supercell must be able to fully contain.
        max_cells:
            Maximum allowed total number of unit cells (nx  *ny * nz) in the
            resulting supercell. Defaults to 100.

    Returns:
        A JAX array of the integer scaling factors `[nx, ny, nz]`, or
        `None` if the required supercell exceeds `max_cells` or the unit
        cell is degenerate (has zero volume).

    Raises:
        ValueError: If `min_required_radius` is non-positive.
    """
    if min_required_radius <= 0:
        raise ValueError(
            f"`min_required_radius` must be positive, is {min_required_radius}."
        )

    # Calculate heights in each direction of the unit cell
    unit_heights = _calculate_unit_cell_heights(cell_matrix)

    # Ratio of the diameter of the inscribed sphere and the unit cell heigth.
    # Here still of type `float`
    target_diameter = 2.0 * min_required_radius
    ratio = target_diameter / unit_heights

    # Round up to nearest integer value and save as integer (`int32').
    # Infinite values are set to unity.
    n = jnp.ceil(ratio)
    # TODO: Check error handling!
    n = jnp.where(jnp.isinf(n), 1.0, n).astype(jnp.int32)

    if jnp.isnan(n).any():
        raise ValueError(
            f"Error: Unit cell dimension potentially zero ({unit_heights}), cannot achieve target radius {min_required_radius:.4f}."
        )
        return None

    if jnp.prod(n) > max_cells:
        raise ValueError(
            f"Required supercell ({n[0]}x{n[1]}x{n[2]}={jnp.prod(n)} cells) exceeds limit of {max_cells} cells."
        )
        return None

    return n


def _check_and_filter_required_keys(
    parameters: dict[str, dict[str, float]],
    required_keys: list[str],
    phase: str,
) -> dict[str, dict[str, float]]:
    """Validates that all required parameter keys are present.

    This function iterates through a dictionary of components (e.g., fluids or
    solid atoms) and checks if each component's parameter dictionary
    contains a set of required keys.

    It is used to ensure that all necessary parameters for a potential model
    are defined before calculations begin.

    Args:
        parameters: A dictionary where keys are component identifiers (str) and
            values are dictionaries of their parameters.
        required_keys: A list of strings representing the keys that must be
            present in each component's parameter dictionary.
        phase: A string (e.g., 'fluid', 'solid') used to construct a
            descriptive error message.

    Raises:
        KeyError: If any component in `parameters` is missing one or more of
            the keys specified in `required_keys`. The error message
            details which keys are missing for each component.
    """
    # Transform to set to check for multiples
    required_set = set(required_keys)

    # Initialize empty dictionary
    missing_keys = dict()

    # Empty `dict` for all parameters that are actually required
    parameters_required = dict()
    # Go through all given parameters
    for name, params in parameters.items():
        # Check if the `required_set` is a subset of `params` and save missing `params` in `missing_keys`
        if not required_set.issubset(params.keys()):
            missing = required_set - set(params.keys())
            missing_keys.update({name: list(missing)})

        # Add only required parameters to `dict` (filters out all non-necessary interaction parameters)
        parameters_required[name] = {
            key: value for key, value in params.items() if key in required_keys
        }

    if missing_keys:
        raise KeyError(
            *(
                f"Missing {phase} parameters: {key}: {value}."
                for key, value in missing_keys.items()
            )
        )

    return parameters_required


def _check_keys_binary_params(data: dict):
    """Checks the structure of a dictionary of binary interaction parameters.

    This function validates that every value in the input dictionary is another
    dictionary containing exactly the keys 'k_si' and 'l_si'.

    Args:
        data: The dictionary to validate.

    Raises:
        KeyError: If an inner dictionary contains unexpected keys or is
                  missing required keys.
        TypeError: If a value in the main dictionary is not a dictionary itself.
    """
    required_keys = set({"k_si", "l_si"})

    # Using .items() gives us the top-level key for better error messages.
    for top_level_key, inner_dict in data.items():
        if not isinstance(inner_dict, dict):
            raise TypeError(
                f"Value for key '{top_level_key}' must be a dict, but got {type(inner_dict)}."
            )

        inner_keys = set(inner_dict.keys())

        # Check if keys are valid (subset of required 'k_si' and 'l_si').
        # If yes, jump out of check
        if inner_keys.issubset(required_keys):
            continue

        # Are there unexpected keys? (Not a subset)
        unexpected_keys = inner_keys - required_keys
        if unexpected_keys:
            raise KeyError(
                f"For key '{top_level_key}', found unexpected sub-key(s): {unexpected_keys}. Valid names are `k_si` and `l_si`."
            )


def _build_combined_parameters(
    interaction_potential: InteractionPotential,
    solid_site_names: list[str],
    fluid_parameters: dict[str, dict[str, float]],
    solid_parameters: dict[str, dict[str, float]],
    binary_interaction_parameters: dict[tuple[str, str], float] | None = None,
) -> dict[str, Array]:
    """Builds binary fluid-solid interaction parameters form combining rules.

    This function orchestrates the calculation of cross-interaction parameters
    between each fluid component and each solid site. It uses the specific
    combining rules (e.g., Lorentz-Berthelot) defined by the given
    interaction potential model.

    Binary interaction parameters (`k_si`, `l_si`) that modify the standard
    combining rules are also incorporated (adjusting the energy and size
    parameter, respectively). The resulting parameter arrays are reshaped to
    facilitate broadcasting in downstream calculations.

    Args:
        interaction_potential: An `InteractionPotential` object that provides
            the combining rule functions for each parameter.
        solid_site_names: A list of unique string identifiers for the solid
            atom sites used in the calculation.
        fluid_parameters: A dictionary mapping fluid names to their individual
            parameter dictionaries (e.g., {'sigma': 3.7, 'epsilon_k': 120.0}).
        solid_parameters: A dictionary mapping solid site names to their
            individual parameter dictionaries (e.g., {'sigma': 3.7,
            'epsilon_k': 120.0}).
        binary_interaction_parameters: An optional dictionary mapping a
            (fluid, solid) tuple to a dictionary of correction factors
            (e.g., {'k_si': 0.1, 'l_si': 0.0}).

    Returns:
        A dictionary where keys are parameter names (e.g., 'sigma') and
        values are JAX arrays of the combined parameters. Each array has a
        shape of (n_fluids, 1, 1) for broadcasting purposes.

    Raises:
        KeyError: If `fluid_parameters` or `solid_parameters` are missing
            keys that are required by the `interaction_potential`.
    """
    # Remove all unnecessary solid interaction site parameters.
    solid_parameters_filtered = {
        key: solid_parameters[key]
        for key in set(solid_site_names)
        if key in solid_parameters
    }

    # Check if all required parameters are available & raise errors if not.
    required_params = interaction_potential.combining_rules.keys()
    fluid_parameters_requierd = _check_and_filter_required_keys(
        fluid_parameters, ["m"] + list(required_params), "fluid"
    )
    solid_parameters_requierd = _check_and_filter_required_keys(
        solid_parameters_filtered, required_params, "solid"
    )

    # Saves all fluid parameters from array of dicts to a dict of arrays, e.g., {'sigma': [...], 'epsilon_K': [...], 'm': [...]}
    fluid_params_individual = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *list(fluid_parameters_requierd.values()),
    )

    #  Save solid parameters for each interaction site to a dict of arrays, e.g., {'sigma': [...], 'epsilon_K': [...]}
    solid_params_individual = {
        param: jnp.array(
            [
                solid_parameters_requierd[site][param]
                for site in solid_site_names
            ]
        )
        for param in required_params
    }

    # # NEW solid_params_unique <--- CHANGE
    # solid_params_individual = jax.tree.map(
    #     lambda *leaves: jnp.stack(leaves),
    #     *list(solid_parameters_requierd.values()),
    # )

    fluid_names = fluid_parameters.keys()
    if binary_interaction_parameters is not None:
        _check_keys_binary_params(binary_interaction_parameters)
        # Loop over all possible (fluid, solid) pairs and get `k_si`
        # and reshape array for convenient binary interaction array
        list_binary_parameters = [
            _get_binary_parameters(fluid_solid, binary_interaction_parameters)
            for fluid_solid in itertools.product(
                fluid_names, solid_site_names
            )  # len(set(solid_site_names))  <--- CHANGE
        ]
        k_si, l_si = zip(*list_binary_parameters)
        binary_params_individual = {
            "k_si": jnp.array(k_si).reshape(
                len(fluid_names),
                len(
                    solid_site_names
                ),  # len(set(solid_site_names))  <--- CHANGE
            ),
            "l_si": jnp.array(l_si).reshape(
                len(fluid_names),
                len(
                    solid_site_names
                ),  # len(set(solid_site_names))  <--- CHANGE
            ),
        }
    else:
        binary_params_individual = {
            "k_si": jnp.zeros(
                [len(fluid_names), len(solid_site_names)]
            ),  # len(set(solid_site_names))  <--- CHANGE
            "l_si": jnp.zeros(
                [len(fluid_names), len(solid_site_names)]
            ),  # len(set(solid_site_names))  <--- CHANGE
        }

    # Calculate and Reshape Combined Parameters
    combined_params = dict()
    for param, rule in interaction_potential.combining_rules.items():
        # Note: This assumes the rule functions can handle the binary_params dict.
        # The `rule` function calls if they expect needs adjustment if the
        # individual k_si, l_si values are required.
        # Additional array dimensions added for broadcasting for each item of the
        # dict to prevent loop over all items later.
        combined_params[param] = rule(
            fluid_params_individual,
            solid_params_individual,
            binary_params_individual,
        )[:, None, None]  # comment [:, None, None]  <--- CHANGE

    # <--- CHANGE
    # type_to_idx_map = {name: i for i, name in enumerate(set(solid_site_names))}

    # <--- CHANGE
    # For each site in the full supercell list, find its corresponding index
    # site_type_indices = jnp.array(
    #     [type_to_idx_map[name] for name in solid_site_names]
    # )

    # <--- CHANGE
    # Use advanced JAX indexing to "expand" the unique parameter arrays
    # to the full size of the supercell.
    # combined_params2 = {
    #     param: value[:, None, None, site_type_indices]
    #     for param, value in combined_params.items()
    # }

    return combined_params, fluid_parameters_requierd, solid_parameters_requierd


def _read_block_files(block_files: list[str]) -> tuple[Array, dict[str, Array]]:
    """Parses a list of Zeo++ block files into coordinates, radii, and site names.

    This function iterates through a list of file paths pointing to Zeo++ block
    files. For each file, it extracts the Cartesian coordinates (x, y, z)
    and the radius of every blocking sphere. It also generates a site name for
    each blocking site indicating for which component the block site is
    relevant.

    Args:
        block_files: A list of string paths of Zeo++ block files. Each file
            corresponds to a different component.

    Returns:
        A tuple containing:
            - coords_cart (Array): A JAX array of shape `(n_total_spheres, 3)`
              with the Cartesian coordinates of all blocking spheres.
            - radii (dict): A dictionary with a single key 'sigma' mapping to
              a JAX array of shape `(n_total_spheres,)` with the radii.
            - site_names (list[str]): A list of strings identifying the
              component of each blocking sphere.
    """
    # Empty list for block information for each component.
    data = list()
    block_site_comp = list()

    # Loop over all block files, open and read content.
    for comp_index, file in enumerate(block_files):
        # try:
        #     # Use np.loadtxt for a massive speedup.
        #     parsed_data = np.loadtxt(
        #         file,
        #         skiprows=1,
        #         usecols=(1, 2, 3, 4),  # Extracts x, y, z, radius
        #     )
        #     # Ensure 2D array for single-line files
        #     if parsed_data.ndim == 1:
        #         parsed_data = parsed_data.reshape(1, -1)
        # except (StopIteration, IOError):  # Handle empty or missing files
        #     continue

        # if parsed_data.size > 0:
        #     data.append(parsed_data)
        #     block_site_comp.append(
        #         np.full(parsed_data.shape[0], comp_index, dtype=np.int32)
        #     )
        # Read block file.
        with open(file, "r") as f:
            # Skip header and read all lines
            lines = f.readlines()[1:]
            # Split the line into its components based on whitespace and extract
            # coordinates (columns 2, 3, 4) and radii (column 5).
            parsed_data = jnp.array(
                [line.split()[1:5] for line in lines if line.strip()],
                dtype=float,
            )

            if parsed_data.size > 0:
                # Save component identifier.
                data.append(parsed_data)
                block_site_comp.append(
                    jnp.full(parsed_data.shape[0], comp_index, dtype=jnp.int32)
                )

    # If no data was read, return empty structures.
    if not data:
        return jnp.array([]).reshape(0, 3), {
            "sigma": jnp.array([]).reshape(0, 0)
        }

    # Save all appearing block sites from all block files into one array and
    # save the information which component the site belongs to in a second array.
    data_final = jnp.concatenate(data)
    comp_indices = jnp.concatenate(block_site_comp, dtype=jnp.int32)
    nsites = data_final.shape[0]

    # Separate coordinates and block radii with shpes [nsites, 3] and [nsites].
    coords_cart = data_final[:, :3]
    radii = data_final[:, 3]

    # Ascending column index [0, ..., (nsites-1)].
    site_indices = jnp.arange(nsites)

    # Create a zero matrix of the desired shape for the parameter output.
    ncomp = len(block_files)
    parameter_matrix = jnp.zeros((ncomp, nsites))

    # Write the information from the 1D array `radii` into rows determined by
    # `com_indices` and in columns `site_indices`. The remaining arrays is
    # composed of zeros.
    parameter_matrix = parameter_matrix.at[comp_indices, site_indices].set(
        radii
    )

    # Radii are used as `sigma` here as the Lorentz combining rule
    # `sigma_si = 1/2 * (sigma_ss + sigma_ii)` is already applied with `sigma_ii = 0`.
    return coords_cart, {"sigma": parameter_matrix[:, None, None]}


class Framework:
    """Represents a porous framework from a Crystallographic Information File (CIF).

    This class serves as a primary interface for defining a crystalline
    structure, such as a metal-organic framework (MOF) or a zeolite. It
    loads structural data from a Crystallographic Information File (CIF) and
    provides methods to compute properties based on that structure.

    The main functionality is to calculate the external potential energy field
    that a guest (fluid) molecule would experience within the framework's
    pores, which is a key step in simulating adsorption processes.

    Args:
        cif: The file path to the CIF file describing the crystal structure.

    Attributes:
        cif: Path to the source CIF file.
        unitcell: A `pymatgen.core.Structure` object representing the
            framework's unit cell, loaded from the CIF file.
    """

    def __init__(
        self,
        cif: PathLike,
    ):
        """Initializes the Framework instance.

        This constructor loads the crystal structure from a specified
        Crystallographic Information File (CIF) and prepares the object for
        further calculations. It uses the `pymatgen` library to parse the
        file and create a `Structure` object.

        Args:
            cif: The file path to the CIF file that defines the framework's
                unit cell structure.
        """
        # Save fluid, solid, and binary interaction parameters parameters
        self.cif = cif

        # Build unitcell.
        self.unitcell = Structure.from_file(cif)

    def __repr__(self):
        """Returns an official string representation of the Framework.

        This representation provides a concise summary of the framework's unit
        cell, including its chemical formula and the number of atomic sites.

        Returns:
            A string with key details of the unit cell.
        """
        unitcell = f"Unitcell(formula: {self.unitcell.formula}, sites: {len(self.unitcell)})"
        # supercell = f"Supercell(formula: {self.supercell.formula}, sites: {len(self.supercell)})"
        return f"{unitcell}"  # \n{supercell}"

    @property
    def lengths(self) -> tuple[float, float, float]:
        """The lengths (a, b, c) of the unit cell lattice vectors in Angstrom."""
        return self.unitcell.lattice.lengths

    @property
    def skew_angles(self) -> tuple[float, float, float]:
        """The angles (alpha, beta, gamma) of the unit cell in degrees."""
        return self.unitcell.lattice.angles

    @property
    def lattice_matrix(self) -> Array:
        """The 3x3 lattice matrix for coordinate transformations.

        Returns:
            A 3x3 JAX array where rows represent the Cartesian lattice
            vectors [a, b, c]. This matrix converts fractional
            coordinates to Cartesian coordinates.
        """
        return jnp.array(self.unitcell.lattice.matrix)

    @property
    def solid_coords_frac(self) -> dict[str, Array]:
        """Fractional coordinates of solid atoms grouped by species.

        Returns:
            A dictionary where keys are atom species strings (e.g., 'O' or 'Si')
            and values are JAX arrays, with each row being the fractional
            coordinates of an atom of that species.
        """
        # Extract fractional & Cartesian coordinates for each solid site
        frac_coords_lists = dict()
        for site in self.unitcell:
            species = site.species_string
            # If species `key` does not exist yet, `.setdefault(key, [])` creates empty list
            frac_coords_lists.setdefault(species, []).append(site.frac_coords)

        # Re-structure to `Array`
        solid_coords_frac = {
            species: jnp.array(coords)
            for species, coords in frac_coords_lists.items()
        }
        return solid_coords_frac

    @property
    def solid_coords_cart(self) -> dict[str, Array]:
        """Cartesian coordinates of solid atoms grouped by species.

        The coordinates are calculated by transforming the fractional
        coordinates using the lattice matrix.

        Returns:
            A dictionary where keys are atom species strings (e.g., 'O' or 'Si')
            and values are JAX arrays, with each row being the Cartesian
            coordinates in Angstrom of an atom of that species.
        """
        return {
            key: coords @ self.lattice_matrix
            for key, coords in self.solid_coords_frac.items()
        }

    def calculate_external_potential(
        self,
        ngrid: tuple[int, int, int],
        interaction_potential_type: PotentialType,
        cutoff_radius: float,
        fluid_parameters: dict[str, dict[str, float]],
        solid_parameters: dict[str, dict[str, float]],
        tail_correction_type: str | None = None,
        binary_interaction_parameters: dict[tuple[str, str], float]
        | None = None,
        block_files: list[PathLike] | None = None,
    ) -> ExternalPotential:
        """Calculates the external potential on the unit cell grid points (in KELVIN).

        Calculated intermolecular interactions between each fluid component
        and all solid atoms within the unitcell subject to periodic boundary
        conditions and cutoff radius to generate a 3D grid of the potential energy.

        Args:
            ngrid: A tuple of three integers (nx, ny, nz) defining the
                resolution of the calculation grid in each dimension.
            interaction_potential_type: An enum from `PotentialType` that
                specifies the interaction model to use (e.g., LennardJones).
            cutoff_radius: The cutoff radius in Angstroms. This is used to
                determine the minimum supercell size and restricts the distance
                for energy calculations.
            fluid_parameters: A dictionary mapping fluid names to their
                parameter dictionaries (e.g., {'methane': {'m': 1.0,
                'sigma': 3.70051, 'epsilon_k': 150.07147}}). Keys are fluid
                identifyers.
            solid_parameters: A dictionary mapping solid atom names to their
                parameter dictionaries (e.g., {'O': {'sigma': 3.21,
                'epsilon_k': 76.3}}). Keys are site identifiers.
            tail_correction: String for the type of tail-correction used
                ('RASPA' or 'Ewald').
            binary_interaction_parameters: An optional dictionary for custom
            combining rule corrections between specific fluid-solid pairs.
            block_files: An optional list of file paths for Zeo++ block files
                (each component needs a separate block file: *.res).

        Returns:
            An `ExternalPotential` object containing the calculated potential (V/kB) for each
            fluid component at each grid point and all metadata related to the calculation
            (e.g., parameters, structure info, grid size).

        Raises:
            KeyError: If the provided parameter dictionaries are missing keys
                required by the specified `interaction_potential_type`.
        """
        # Extract `InteractionPotential` from enum `PotentialType`.
        interaction_potential = interaction_potential_type.value

        # Calculate number of unit cells in each direction to generate minimal supercell.
        multiples = _find_minimal_supercell(
            cell_matrix=self.unitcell.lattice.matrix,
            min_required_radius=cutoff_radius,
        )

        # Build binary interaction parameters for the specified `InteractionPotential`.
        (
            combined_params_unitcell,
            fluid_parameters_required,
            solid_parameters_required,
        ) = _build_combined_parameters(
            interaction_potential=interaction_potential,
            solid_site_names=[site.species_string for site in self.unitcell],
            fluid_parameters=fluid_parameters,
            solid_parameters=solid_parameters,
            binary_interaction_parameters=binary_interaction_parameters,
        )

        # Extract number of components `ncomp` from `fluid_parameters_required` to check if
        # correct number of block files are provided.
        ncomp = len(fluid_parameters_required)

        # Create supercell with `multiples` in each direction due to cutoff radius.
        supercell, combined_params = _create_supercell(
            self.unitcell.frac_coords,
            multiples,
            combined_params_unitcell,
        )

        # Extract the chain-length parameter `m` for multiplicity of interactions.
        if fluid_parameters and "m" in next(iter(fluid_parameters.values())):
            m = jnp.array(
                [
                    fluid_parameters[name]["m"]
                    for name in fluid_parameters.keys()
                ]
            )
        else:
            m = jnp.ones([len(fluid_parameters.keys())])

        # Defince grid in fractional of unitcell for which the external potential will be calculated.
        grid = _build_calculation_grid(ngrid, (1.0, 1.0, 1.0))

        # Check if correct number of block files are provided.
        if block_files is not None:
            if len(block_files) != ncomp:
                raise ValueError(
                    f"Number of `block_files` {len(block_files)} does not match the number of provided fluid parameter components {ncomp}."
                )

            # Read all Zeo++ block files to extract:
            #  - Cartesian coordinates [x, y, z] of the centers of blocked sphere sites.
            #  - Block-radius for each block site.
            # Transform for fractional coordinats of the unitcell (along cell array & 0<x<1).
            unitcell_block_cart, combined_params_unitcell_block = (
                _read_block_files(block_files)
            )
            unitcell_block = unitcell_block_cart @ jnp.linalg.inv(
                self.lattice_matrix
            )

            # Caluc late how many multiples are required for the supercell
            # (only largest cutoff is taken for all components) & create supercell
            # coordinates (coordinates an be largen than 1, reference is the unitcell).
            cutoff_radius_block = jnp.max(
                combined_params_unitcell_block["sigma"]
            )
            multiples_block = _find_minimal_supercell(
                cell_matrix=self.lattice_matrix,
                min_required_radius=cutoff_radius_block,
            )
            supercell_block, combined_params_block = _create_supercell(
                frac_coords=unitcell_block,
                multiples=multiples_block,
                parameters=combined_params_unitcell_block,
            )

        def _scan_ext_pot(
            _: Array,
            grid_slice: Array,
        ):
            # Compute distance vectors: subtract atom coordinates from each grid point.
            # Broadcasting: (ny, nz, 1, 3) - (1, 1, n_atoms/n_block_sites, 3) results in shape (ny, nz, n_atoms, 3).
            distance_vectors = grid_slice[:, :, None] - supercell[None, None]

            # 1. Apply the minimum image convention in skewed coordinates:
            #    For each component, adjust the distance vector to the nearest image.
            #    This works element-wise since lengths has shape (3,) and distance_vectors shape is
            #    (ny, nz, n_atoms, 3).
            # 2. Transform skewed distance vectors to cartesian distance using the supercell matrix.
            distance_vectors = (
                distance_vectors
                - multiples * jnp.round(distance_vectors / multiples)
            ) @ self.lattice_matrix

            # Compute the scalar squared distances at this slice: shape (1, ny, nz, n_atoms).
            # 1st dimension added for broadcasting with `combined_params` in `interaction_potential.potential()`.
            distances = jnp.linalg.norm(distance_vectors, axis=-1)[None]

            # Compute interaction energy between each grid point in slice and each solid atom for
            # all species at once.
            # TODO: Check here for overflow if distance to atom is to small???
            reduced_energy = interaction_potential.potential(
                distances, combined_params
            )

            # Apply the cutoff.
            reduced_energy = jnp.where(
                distances > cutoff_radius,
                0.0,
                reduced_energy,
            )

            # Sum the contributions over all atoms to get the potential for this slice and apply
            # chain-length parameter `m` for multiplicity of interactions.
            potential_slice = m[:, None, None] * jnp.sum(
                reduced_energy, axis=-1
            )

            # Treat temperature-dependence for `T = 1`
            if interaction_potential.temperature_dependent:
                reduced_energy_t = interaction_potential.potential_linear_temperature_dependence(
                    distances, combined_params
                )
                # TODO: Nochmal überprüfe
                reduced_energy_t = jnp.where(
                    distances > cutoff_radius,
                    0.0,
                    reduced_energy_t,
                )
                potential_slice_t = m[:, None, None] * jnp.sum(
                    reduced_energy_t, axis=-1
                )
            else:
                potential_slice_t = jnp.zeros_like(potential_slice)

            # Treat pore blocking
            if block_files is not None:
                distance_vectors_block = (
                    grid_slice[:, :, None] - supercell_block[None, None]
                )
                distance_vectors_block = (
                    distance_vectors_block
                    - multiples_block
                    * jnp.round(distance_vectors_block / multiples_block)
                ) @ self.lattice_matrix
                distances_block = jnp.linalg.norm(
                    distance_vectors_block, axis=-1
                )[None]
                reduced_energy_block = PotentialType.HardSphere.value.potential(
                    distances_block, combined_params_block
                )
                # TODO: Nochmal überprüfen
                reduced_energy_block = jnp.where(
                    distances_block > cutoff_radius_block,
                    0.0,
                    reduced_energy_block,
                )
                potential_slice_block = jnp.sum(reduced_energy_block, axis=-1)
            else:
                potential_slice_block = jnp.zeros_like(potential_slice)

            return _, (
                potential_slice,
                potential_slice_t,
                potential_slice_block,
            )

        (
            _,
            (
                potential_slices_stacked,
                potential_slices_t_stacked,
                potential_slices_block_stacked,
            ),
        ) = lax.scan(f=_scan_ext_pot, init=None, xs=grid)

        external_potential_k = jnp.swapaxes(potential_slices_stacked, 0, 1)
        external_potential_k_lin_t = jnp.swapaxes(
            potential_slices_t_stacked, 0, 1
        )
        external_potential_k_block = jnp.swapaxes(
            potential_slices_block_stacked, 0, 1
        )

        # Handling of tail-correction.
        match (
            tail_correction_type.lower()
            if isinstance(tail_correction_type, str)
            else tail_correction_type
        ):
            case "raspa":
                volume_unitcell = jnp.abs(jnp.linalg.det(self.lattice_matrix))
                tail_correction_k = (
                    m
                    * interaction_potential.tail_correction_raspa(
                        cutoff_radius, volume_unitcell, combined_params_unitcell
                    )
                )
            case "ewald":
                raise NotImplementedError
            case None:
                tail_correction_k = jnp.zeros(ncomp)
            case _:
                raise ValueError(
                    f"Invalid value for `tail_correction_type`: '{tail_correction_type}'. Must be one of 'RASPA', 'Ewald', or None."
                )

        external_potential = {
            "formula": self.unitcell.formula,
            "cif": self.cif,
            "block_files": block_files,
            "fluid_parameters": fluid_parameters_required,
            "solid_parameters": solid_parameters_required,
            "binary_interaction_parameters": _extract_binary_interaction_parameters(
                fluid_parameters_required,
                solid_parameters_required,
                binary_interaction_parameters,
            ),
            "lengths": self.lengths,
            "skew_angles": self.skew_angles,
            "interaction_potential": interaction_potential,
            "cutoff_radius": cutoff_radius,
            "tail_correction_type": tail_correction_type,
            "_external_potential_k": external_potential_k,
            "_external_potential_k_lin_t": external_potential_k_lin_t,
            "_external_potential_k_blocked": external_potential_k_block,
            "_tail_correction_k": tail_correction_k,
        }

        return ExternalPotential(**external_potential)

    def visualize_external_potential(
        self,
        ax: Axes,
        external_potential: Array,
        fluid_index: int = 0,
        z_index: int = 0,
        show_contour_lines: bool = False,
    ):
        """Visualizes a 2D slice of the calculated external potential.

        Displays a heatmap of the potential energy between a specific
        fluid component in an X-Y plane at a given Z-index within the unit cell.
        Overlays the positions of unit cell atoms projected onto that plane.

        Args:
            ax: A matplotlib Axes object to plot on.
            external_potential: The 4D array of external potential
                (n_comp, nx, ny, nz) as returned by `calculate_external_potential`.
            fluid_index: The index of the fluid component to visualize (default: 0).
            z_index: The index of the grid slice along the Z-axis to display
                (default: 0).
            show_contour_lines: Overlay plot with energy value contour lines.
                (default: False)

        Returns:
            The matplotlib Axes object with the plot added.
        """
        grid_frac = _build_calculation_grid(
            external_potential.shape[1:], (1.0, 1.0, 1.0)
        )
        grid_cartesian = grid_frac @ self.lattice_matrix
        X = grid_cartesian[:, :, z_index, 0]
        Y = grid_cartesian[:, :, z_index, 1]

        slice_2d = external_potential[fluid_index, :, :, z_index]

        _ = ax.pcolormesh(X, Y, slice_2d, cmap="viridis", shading="auto")
        if show_contour_lines:
            contour_levels = jnp.linspace(
                slice_2d.min(), slice_2d.max(), 10
            )  # Example levels
            contours = ax.contour(
                X,
                Y,
                slice_2d,
                levels=contour_levels,
                colors="white",
                linewidths=0.5,
            )
            ax.clabel(
                contours, inline=True, fontsize=8, fmt="%.1f"
            )  # Optional: label contours
        ax.plot(
            self.unitcell.cart_coords.T[0],
            self.unitcell.cart_coords.T[1],
            "or",
            alpha=0.8,
        )
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        return ax

    def visualize_cells(self, external_potential: Array, ax=None):
        """Visualizes projections of the unit cell and supercell atoms.

        Creates three plots showing XY, XZ, and YZ projections of the supercell
        atom positions. Overlays the unit cell grid points and a
        circle representing the cutoff radius.

        Args:
            ax: An optional array-like object containing three matplotlib Axes objects
                (ax[0] for XY, ax[1] for XZ, ax[2] for YZ). If None, a new figure
                and axes are created.
        """
        cm = 1 / 2.54
        if ax is None:
            _, ax = plt.subplots(1, 3, figsize=(40 * cm, 12 * cm), sharey=False)

        ax[0].plot(
            self.supercell.cart_coords.T[0],
            self.supercell.cart_coords.T[1],
            "o",
        )
        ax[1].plot(
            self.supercell.cart_coords.T[0],
            self.supercell.cart_coords.T[2],
            "o",
        )
        ax[2].plot(
            self.supercell.cart_coords.T[1],
            self.supercell.cart_coords.T[2],
            "o",
        )

        grid_frac = _build_calculation_grid(
            external_potential.shape[1:], (1.0, 1.0, 1.0)
        )
        grid_plot = grid_frac @ self.lattice_matrix
        x_grid = grid_plot.T[0].flatten()
        y_grid = grid_plot.T[1].flatten()
        z_grid = grid_plot.T[2].flatten()

        cirle_center = (
            jnp.array([0.5, 0.5, 0.5]) @ self.supercell.lattice.matrix
        )
        circle1 = plt.Circle(
            (cirle_center[0], cirle_center[1]),
            self.cutoff_radius,
            color="grey",
            alpha=0.4,
        )
        ax[0].axis("equal")
        ax[0].add_patch(circle1)
        ax[0].plot(x_grid, y_grid, ".", ms=1, alpha=0.1)
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Y")

        circle2 = plt.Circle(
            (cirle_center[0], cirle_center[2]),
            self.cutoff_radius,
            color="grey",
            alpha=0.4,
        )
        ax[1].axis("equal")
        ax[1].add_patch(circle2)
        ax[1].plot(x_grid, z_grid, ".", ms=1, alpha=0.1)
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Z")

        circle3 = plt.Circle(
            (cirle_center[1], cirle_center[2]),
            self.cutoff_radius,
            color="grey",
            alpha=0.4,
        )
        ax[2].axis("equal")
        ax[2].add_patch(circle3)
        ax[2].plot(y_grid, z_grid, ".", ms=1, alpha=0.1)
        ax[2].set_xlabel("Y")
        ax[2].set_ylabel("Z")

    def visualize_external_potential_contour(
        self,
        ax: plt.Axes,
        external_potential: Array,
        fluid_index: int = 0,
        z_index: int = 0,
        num_min_contours: int = 4,  # How many contours around the minimum
        min_contour_offset: float = 0.5,  # Base offset (e.g., in kT units if potential is U/kT)
        contour_labels: bool = True,
    ) -> plt.Axes:
        """Visualizes a 2D slice of the calculated external potential.

        Displays a heatmap of the potential energy, overlays unit cell atoms,
        and adds contour lines specifically highlighting the minimum potential
        energy region in the slice.

        Args:
            ax: A matplotlib Axes object to plot on.
            external_potential: The 4D array of external potential
                (n_comp, nx, ny, nz) as returned by `calculate_external_potential`.
            fluid_index: The index of the fluid component to visualize (default: 0).
            z_index: The index of the grid slice along the Z-axis to display
                (default: 0).
            num_min_contours: The number of contour levels to draw around the minimum
                (default: 4). The levels will be min, min+offset, min+2*offset, ...
            min_contour_offset: The spacing between contour levels drawn around the minimum.
                If the potential is U/kT, this offset is in units of kT (default: 0.5).
            contour_labels: Add labels for energy values to contour lines (default: True).

        Returns:
            The matplotlib Axes object with the plot added.
        """
        # Transform unit cell grid points from skewed to Cartesian coordinates
        grid_frac = _build_calculation_grid(
            external_potential.shape[1:], (1.0, 1.0, 1.0)
        )
        grid_cartesian = grid_frac @ self.lattice_matrix

        # Extract X and Y coordinates for the chosen Z-slice
        X = grid_cartesian[:, :, z_index, 0]
        Y = grid_cartesian[:, :, z_index, 1]

        # Extract the 2D potential slice
        slice_2d = external_potential[fluid_index, :, :, z_index]

        # --- Visualization ---
        # Create the pseudocolor plot (heatmap)
        mesh = ax.pcolormesh(
            X,
            Y,
            slice_2d,
            cmap="viridis",
            shading="auto",
            rasterized=True,
        )
        plt.colorbar(mesh, ax=ax, label="Reduced Potential (U/kT)")

        # --- Highlight Minimum Potential Region ---
        # Find the minimum potential value and its index
        min_potential = jnp.min(slice_2d)
        # Convert flat index to 2D index (row, col) corresponding to (ny, nz) axes

        # Define contour levels around the minimum
        # Start at the minimum, then add steps of the offset
        min_contour_levels = jnp.array(
            [
                min_potential + i * min_contour_offset
                for i in range(num_min_contours)
            ]
        )

        # Ensure levels are within the data range to avoid plotting issues
        max_potential = jnp.max(slice_2d)
        valid_levels = min_contour_levels[min_contour_levels < max_potential]

        if len(valid_levels) > 0:
            # Plot contours specifically highlighting the minimum region
            min_contours = ax.contour(
                X,
                Y,
                slice_2d,
                levels=valid_levels,
                colors="white",  # Choose a standout color (white, magenta, yellow...)
                linewidths=0.8,
                alpha=0.9,
            )
            # Label these contours (optional, can get crowded)
            if contour_labels:
                ax.clabel(min_contours, inline=True, fontsize=8, fmt="%.1f")

        # --- Plot Atoms and Finalize ---
        # Overlay projected unit cell atom positions
        ax.plot(
            self.unitcell.cart_coords[:, 0],
            self.unitcell.cart_coords[:, 1],
            "o",
            color="black",
            markeredgecolor="white",
            markersize=10,
            alpha=0.6,  # Adjusted style
            label="Unit Cell Atoms",
        )

        ax.set_xlabel("X-axis (Å)")
        ax.set_ylabel("Y-axis (Å)")
        ax.set_title(
            f"Potential Slice (Fluid {fluid_index}, Z-index {z_index})"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False)

        return ax
