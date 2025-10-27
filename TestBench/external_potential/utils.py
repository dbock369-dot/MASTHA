"""TODO."""

import itertools

import jax.numpy as jnp
from jax import Array


def _get_binary_parameters(
    ids: tuple[str, str],
    dict_binary: dict[tuple[str, str], dict[str, float]] | None = None,
) -> float:
    """Gets the binary interaction parameter `k_si` for a given pair of components.

    This function safely looks up the `k_si` and `l_si` correction
    factors for a given pair of component identifiers from a dictionary.
    The lookup is order-insensitive (e.g., ('A', 'B') is the same as
    ('B', 'A')).

    Args:
        ids: A tuple of two strings representing the component names (e.g.,
            ('fluid_name', 'solid_name')).
        dict_binary: A dictionary where keys are component pair tuples and
            values are dicts containing the 'k_si' and/or 'l_si' values.
            Defaults to None.

    Returns:
        A tuple containing the (k_si, l_si) values. Defaults to
        (0.0, 0.0) if the pair is not found or `dict_binary` is None.

    Raises:
        KeyError: If `dict_binary` contains duplicate pairs.
    """
    # If no binary parameters are given, return array of zeros
    if dict_binary is None:
        return (0.0, 0.0)

    # Go through list of dicts and check which entries match the `unique_id`
    seen_keys = set()
    for pair, params in dict_binary.items():
        # Check for duplicates (save seen sets and compare)
        pair_set = frozenset(pair)
        if pair_set in seen_keys:
            raise KeyError(f"Duplicate binary interaction pair found: {pair}")
        seen_keys.add(pair_set)

        # Write parameter or 0.0 to dict
        if pair_set == frozenset(ids):
            k_si = 0.0
            l_si = 0.0
            if "k_si" in params:
                k_si = params["k_si"]
            if "l_si" in params:
                l_si = params["l_si"]
            return (k_si, l_si)

    # Return zero if no match is found
    return (0.0, 0.0)


def _extract_binary_interaction_parameters(
    fluid_parameters: dict[str, dict[str, float]],
    solid_parameters: dict[str, dict[str, float]],
    binary_interaction_parameters: dict[tuple[str, str], dict[str, float]]
    | None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Creates a complete dictionary of all fluid-solid binary parameters.

    This function generates an exhaustive dictionary containing the binary
    interaction parameters for every possible fluid-solid pair. It
    iterates through the Cartesian product of fluid and solid names.

    Args:
        fluid_parameters: A dictionary of fluid parameters.
        solid_parameters: A dictionary of solid parameters.
        binary_interaction_parameters: The dictionary of user-provided,
            sparse binary parameters.

    Returns:
        A dictionary where keys are (fluid, solid) tuples and values are
        dicts containing the 'k_si' and 'l_si' parameters for that pair.
    """
    full = {}
    for fluid_solid in itertools.product(
        fluid_parameters.keys(), solid_parameters.keys()
    ):
        k_si, l_si = _get_binary_parameters(
            fluid_solid, binary_interaction_parameters
        )
        full[fluid_solid] = {"k_si": k_si, "l_si": l_si}
    return full


def _build_calculation_grid(
    ngrid: tuple[int, int, int], lengths: tuple[float, float, float]
) -> Array:
    """Builds a 3D grid of points within an orthorhombic cell.

    This function generates a grid of uniformly spaced, cell-centered
    points. It is used to define the locations at which properties like
    the external potential are calculated.

    Args:
        ngrid: A tuple of three integers (nx, ny, nz) specifying the
            number of grid points in each dimension.
        lengths: A tuple of three floats (la, lb, lc) representing the
            dimensions of the grid's bounding box in Angstrom.

    Returns:
        A JAX array of shape (nx, ny, nz, 3) containing the Cartesian
        coordinates of each grid point.
    """
    la, lb, lc = lengths
    x, y, z = jnp.meshgrid(
        jnp.linspace(
            0.5 * la / ngrid[0],
            la - 0.5 * la / ngrid[0],
            ngrid[0],
        ),
        jnp.linspace(
            0.5 * lb / ngrid[1],
            lb - 0.5 * lb / ngrid[1],
            ngrid[1],
        ),
        jnp.linspace(
            0.5 * lc / ngrid[2],
            lc - 0.5 * lc / ngrid[2],
            ngrid[2],
        ),
        indexing="ij",
    )
    return jnp.stack((x, y, z), axis=-1)


def _create_supercell(
    frac_coords: Array, multiples: Array, parameters: dict[str, Array]
) -> tuple[Array, dict[str, Array]]:
    """Creates coordinates & interaction parameters of a supercell by duplicating atoms from fractional coordinates.

    Args:
        frac_coords: A JAX array of shape `[n_atoms, 3]` containing the
            initial fractional coordinates of atoms in the unit cell.
        multiples: A tuple of three integers (nx, ny, nz) specifying the
            number of repetitions in each direction (a, b, c).
        parameters: Dictionary of each of the interaction potential's
            parameters. `Keys` are the names of the parameters (e.g. sigma,
            epsilon_k), `values` are Array with paramters ordered w.r.t. the
            sites coordinates.

    Returns:
        A JAX array of shape `[n_atoms * nx * ny * nz, 3]` containing the
        fractional coordinates of atoms in the new supercell.
    """
    # Immediate return if no supercell needed.
    if jnp.all(multiples == jnp.array([1, 1, 1], dtype=jnp.int32)):
        return frac_coords, parameters

    # Grid of coordinates (shape [3, nx, ny, nz]) with all integer combinations
    # from [0, 0, 0] to [nx-1, ny-1, nz-1].
    tiled_indices = jnp.indices(multiples)

    # Reshape the indices into a list of [x, y, z] translation vectors.
    # The shape becomes [nx*ny*nz, 3]. Effectively gives a list of all possible
    # translations in each direction.
    translation_vectors = tiled_indices.reshape(3, -1).T

    # Use broadcasting to add translation vectors to original coordinates.
    # frac_coords shape:  [n_atoms, 1, 3]
    # translation_vectors shape: [1, n_translations, 3]
    # supercell_coords shape: [n_atoms, n_translations, 3]
    supercell_coords = frac_coords[:, None, :] + translation_vectors[None, :, :]

    # Reshape and rescale the coordinates..
    # First, flatten the array to a list of coordinates of shape [n_total_atoms, 3].
    n_atoms = frac_coords.shape[0]
    n_translations = translation_vectors.shape[0]
    frac_coords_supercell = supercell_coords.reshape(
        n_atoms * n_translations, 3
    )

    # Repeat each radius `n_translations` times. This matches the memory layout of the reshaped supercell_coords.
    supercell_parameters = {
        key: jnp.repeat(value, repeats=n_translations, axis=-1)
        for key, value in parameters.items()
    }

    return frac_coords_supercell, supercell_parameters
