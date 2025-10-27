"""Todo"""

import jax.numpy as jnp
from pymatgen.core import Structure
from pymatgen.core.sites import PeriodicSite
from os import PathLike
import numpy as np
import jax
import matplotlib.pyplot as plt
from typing import Any


def calculate_unit_cell_heights(cell: jnp.ndarray) -> jnp.ndarray:
    """Calculates the perpendicular distances between parallel faces for a unit cell.

    Args:
        cell: A 3x3 array where rows represent the lattice vectors a, b, c.

    Returns:
        An array containing the three heights [h_a, h_b, h_c].

    Raises:
         ValueError: If the input cell matrix is not 3x3.
    """
    if not isinstance(cell, jnp.ndarray):
        cell = jnp.array(cell)  # Ensure input is a NumPy array

    if cell.shape != (3, 3):
        raise ValueError(
            f"Input cell matrix must be of shape (3, 3), got {cell.shape}"
        )

    a, b, c = cell[0, :], cell[1, :], cell[2, :]

    # Calculate volume
    b_cross_c = jnp.cross(b, c)
    volume = jnp.abs(jnp.dot(a, b_cross_c))

    # Handle degenerate case (zero volume)
    if volume < 1e-12:
        return jnp.array([jnp.inf, jnp.inf, jnp.inf])

    # Calculate face areas
    area_bc = jnp.linalg.norm(b_cross_c)
    area_ca = jnp.linalg.norm(jnp.cross(c, a))
    area_ab = jnp.linalg.norm(jnp.cross(a, b))

    # Calculate heights, handling potential zero areas explicitly
    h_a = jnp.inf if area_bc < 1e-12 else volume / area_bc
    h_b = jnp.inf if area_ca < 1e-12 else volume / area_ca
    h_c = jnp.inf if area_ab < 1e-12 else volume / area_ab

    heights = jnp.array([h_a, h_b, h_c])
    return heights


def find_minimal_supercell(
    cell_matrix: jnp.ndarray, min_required_radius: float, max_cells: int = 100
) -> jnp.ndarray:
    """Determines the smallest supercell scaling factors (nx, ny, nz).

    This ensures the supercell's largest inscribed sphere radius meets provided
    minimum requirement. Calculation is based on the unit cell heights.

    Args:
        cell_matrix: A 3x3 array where rows represent the lattice vectors a, b, c.
        min_required_radius:
            The minimum radius in Angstrom of the inscribed sphere of the supercell.
        max_cells:
            Maximum allowed total number of unit cells (nx*ny*nz) in the
            resulting supercell.

    Returns:
        scaling_factors (jnp.ndarray): The [nx, ny, nz] multipliers.

    Raises:
        ValueError: If min_required_radius is non-positive.
    """
    if min_required_radius <= 0:
        raise ValueError("min_required_radius must be positive.")

    unit_heights = calculate_unit_cell_heights(cell_matrix)
    target_diameter = 2.0 * min_required_radius
    ratio = target_diameter / unit_heights

    n = jnp.ceil(ratio)
    n = jnp.where(jnp.isinf(n), 1.0, n).astype(jnp.int32)

    if jnp.isnan(n).any():
        print(
            f"Error: Unit cell dimension potentially zero ({unit_heights}), cannot achieve target radius {min_required_radius:.4f}."
        )
        return None

    if jnp.prod(n) > max_cells:
        print(
            f"Required supercell ({n[0]}x{n[1]}x{n[2]}={jnp.prod(n)} cells) exceeds limit of {max_cells} cells."
        )
        return None

    return n


def build_interactions(
    supercell: Structure,
    unitcell: Structure,
    solid_parameters: dict[str, dict[str, float]],
    fluid_parameters: dict[str, dict[str, float]],
) -> tuple[list[float], np.ndarray[float], np.ndarray[float]]:
    """Calculates LJ interaction parameters between fluid species and solid sites.

    Computes the Lorentz-Berthelot mixing rules for sigma and epsilon
    for every pair of fluid species and solid atom site in the supercell.
    It pre-calculates sigma^6 and 4*epsilon for efficiency.

    Args:
        supercell: A pymatgen Structure object representing the supercell
            containing all solid atom sites.
        solid_parameters: A dictionary where keys are solid site identifiers
            (e.g., atom types like 'Si', 'O') and values are dictionaries
            containing 'sigma' (Angstrom) and 'epsilon_k' (Kelvin) LJ parameters.
        fluid_parameters: A dictionary where keys are fluid species names
            (e.g., 'methane', 'co2') and values are dictionaries containing
            'm' (PC-SAFT segment number), 'sigma' (Angstrom), and
            'epsilon_k' (Kelvin) parameters.

    Returns:
        A tuple containing:
            - m (List[float]): List of PC-SAFT segment numbers for each fluid species.
            - sigma_pow6_si (np.ndarray): Array of shape (n_fluids, n_solid_sites).
              Each element (i, j) contains (sigma_ij)^6 where i is the fluid index
              and j is the solid site index, calculated using Lorentz-Berthelot
              mixing rules: sigma_ij = 0.5 * (sigma_solid_j + sigma_fluid_i).
            - epsilon_k_4_si (np.ndarray): Array of shape (n_fluids, n_solid_sites).
              Each element (i, j) contains 4.0 * epsilon_k_ij where i is the
              fluid index and j is the solid site index, calculated using
              Lorentz-Berthelot mixing rules:
              epsilon_k_ij = sqrt(epsilon_k_solid_j * epsilon_k_fluid_i).
            - sigma_pow6_uc (np.ndarray): Array of shape (n_fluid, n_solid_sites_uc).
              Same as sigma_pow6_si, but for the single unitcell, not the supercell.
            - epsilon_k_4_uc (np.ndarray): Array of shape (n_fluid, n_solid_sites_uc).
              Same as epsilon_k_4_si, but for the single unitcell, not the supercell.
    """
    site_names = set(site.species_string for site in supercell)
    print(site_names)
    parameter_names = solid_parameters.keys()

    # Make sure that there are parameters for all interaction sites of solid.
    assert site_names.issubset(parameter_names)

    # Calculate effective interaction parameters.
    # Todo: remove outer loop and use arrays for fluid parameters
    sigma_pow6 = np.zeros((len(fluid_parameters), supercell.num_sites))
    epsilon_k_4 = np.zeros((len(fluid_parameters), supercell.num_sites))
    sigma_pow6_uc = np.zeros((len(fluid_parameters), unitcell.num_sites))
    epsilon_k_4_uc = np.zeros((len(fluid_parameters), unitcell.num_sites))
    m = [fluid["m"] for fluid in fluid_parameters.values()]
    for i, fluid in enumerate(fluid_parameters.values()):
        for j, site in enumerate(supercell):
            solid = solid_parameters[site.species_string]
            sigma_pow6[i, j] = (0.5 * (solid["sigma"] + fluid["sigma"])) ** 6
            epsilon_k_4[i, j] = 4.0 * np.sqrt(
                solid["epsilon_k"] * fluid["epsilon_k"]
            )
        for j, site in enumerate(unitcell):
            solid = solid_parameters[site.species_string]
            sigma_pow6_uc[i, j] = (0.5 * (solid["sigma"] + fluid["sigma"])) ** 6
            epsilon_k_4_uc[i, j] = 4.0 * np.sqrt(
                solid["epsilon_k"] * fluid["epsilon_k"]
            )
    return m, sigma_pow6, epsilon_k_4, sigma_pow6_uc, epsilon_k_4_uc


def build_interactions_block(
    supercell: Structure,
    block_parameters: dict[str, dict[str, float]],
    fluid_parameters: dict[str, dict[str, float]],
) -> tuple[list[float], np.ndarray[float], np.ndarray[float]]:
    """Calculates LJ interaction parameters between fluid species and blocking spheres.

    Computes the Lorentz-Berthelot mixing rules for sigma and epsilon
    for every pair of fluid species and solid atom site in the supercell.
    It pre-calculates sigma^6 and 4*epsilon for efficiency.

    Args:
        supercell: A pymatgen Structure object representing the supercell
            containing all solid atom sites.
        block_parameters: A dictionary where keys are blocking sphere label identifiers
            (e.g., label types like 'B1', 'B2') and values are dictionaries
            containing 'sigma' (Angstrom) and 'epsilon_k' (Kelvin) LJ parameters.
        fluid_parameters: A dictionary where keys are fluid species names
            (e.g., 'methane', 'co2') and values are dictionaries containing
            'm' (PC-SAFT segment number), 'sigma' (Angstrom), and
            'epsilon_k' (Kelvin) parameters.

    Returns:
        A tuple containing:
            - m (List[float]): List of PC-SAFT segment numbers for each fluid species.
            - sigma_pow6_si (np.ndarray): Array of shape (n_fluids, n_solid_sites).
              Each element (i, j) contains (sigma_ij)^6 where i is the fluid index
              and j is the blocking sphere index, calculated using Lorentz-Berthelot
              mixing rules: sigma_ij = 0.5 * (sigma_solid_j + sigma_fluid_i).
            - epsilon_k_4_si (np.ndarray): Array of shape (n_fluids, n_solid_sites).
              Each element (i, j) contains 4.0 * epsilon_k_ij where i is the
              fluid index and j is the blocking sphere index, calculated using
              Lorentz-Berthelot mixing rules:
              epsilon_k_ij = sqrt(epsilon_k_solid_j * epsilon_k_fluid_i).
    """
    label_names = set(site.label for site in supercell)
    print(label_names)
    parameter_names = block_parameters.keys()

    # Make sure that there are parameters for all interaction sites of blocking spheres.
    assert label_names.issubset(parameter_names)

    # Calculate effective interaction parameters.
    # Todo: remove outer loop and use arrays for fluid parameters
    sigma_pow6 = np.zeros((len(fluid_parameters), supercell.num_sites))
    epsilon_k_4 = np.zeros((len(fluid_parameters), supercell.num_sites))
    for i, fluid in enumerate(fluid_parameters.values()):
        for j, site in enumerate(supercell):
            block_sphere = block_parameters[site.label]
            sigma_pow6[i, j] = (
                0.5 * (block_sphere["sigma"] + fluid["sigma"])
            ) ** 6
            epsilon_k_4[i, j] = 4.0 * np.sqrt(
                block_sphere["epsilon_k"] * fluid["epsilon_k"]
            )
    return sigma_pow6, epsilon_k_4


def generate_block_parameters(
    cif_file: PathLike,
    block_file: PathLike,
) -> dict:
    """Creates a blocking sphere parameter dictionary.

    Reads a CIF file and a BLOCK file, generated from Zeo++ (according to the RASPA2.0 format), and
    generate a dictionary of "force field" parameters for the external potential.

    Args:
        cif_file: Path to the CIF file describing the framework's unit cell.
        block_file: Path to the BLOCK file describing the coordinates and size of blocking sphere.

    Returns:
        A dictionary containing:
            - site labels
            - sigma, equivalent to diameter of blocking sphere
            - epsilon/kB = 15 K = constant
    """
    structure = Structure.from_file(cif_file)
    structure_block = Structure(structure.lattice, [], [])
    with open(block_file, "r") as tmp_file:
        num_block_sites = int(tmp_file.readline().strip())

        block_parameters = {}
        blocking_sites = []
        updated_forcefield = []

        for _ in range(num_block_sites):
            xb, yb, zb, rb = map(float, tmp_file.readline().split())
            blocking_sites.append([xb, yb, zb, rb])

        for i, site in enumerate(blocking_sites):
            label = f"B{i}"
            new_site = PeriodicSite(
                "B", site[:3], structure.lattice, label=label
            )
            structure_block.sites.append(new_site)
            block_size = site[3]
            updated_forcefield.append(f"{label} {block_size:>8.4f}  15.0  0.0")

        for entry in updated_forcefield:
            parts = entry.split()
            label = parts[0]
            sigma = float(parts[1])
            epsilon_k = float(parts[2])
            block_parameters[label] = {"sigma": sigma, "epsilon_k": epsilon_k}

    return structure_block, block_parameters


class OldFramework:
    """Represents a porous framework material.

    This class handles the setup of the framework structure, including
    building an appropriate supercell based on a cutoff radius, defining
    interaction parameters between the solid and fluid components, and
    setting up a grid for calculating properties like the external potential.
    """

    def __init__(
        self,
        cif_file: PathLike,
        temperature: float,
        cutoff_radius: float,
        ngrid: int | list,
        solid_parameters: dict[str, dict[str, float]],
        fluid_parameters: dict[str, dict[str, float]],
        maximum_reduced_energy: float = 10,
        jax_enable_x64: bool = True,
        block_file: PathLike = None,
    ):
        """Initializes the Framework object.

        Reads a CIF file, determines the minimal supercell size required for
        the given cutoff radius, calculates interaction parameters, and sets up
        the simulation grid.

        Args:
            cif_file: Path to the CIF file describing the framework's unit cell.
            temperature: The simulation temperature in Kelvin. Used for calculating
                reduced potential energy (U/kT).
            cutoff_radius: The cutoff radius in Angstroms for Lennard-Jones
                interactions. This is used to determine the minimum supercell size and
                restricts the distance for energy calculations.
            ngrid: The number of grid points along each dimension (x, y, z)
                of the unit cell grid.
            solid_parameters: Dictionary defining LJ parameters ('sigma', 'epsilon_k')
                for each type of solid atom site. Keys are site identifiers.
            fluid_parameters: Dictionary defining LJ parameters ('sigma', 'epsilon_k')
                and PC-SAFT segment number ('m') for each fluid component. Keys
                are fluid identifiers.
            maximum_reduced_energy: A ceiling value for the calculated reduced
                external potential (U/kT) to prevent numerical issues for small
                distances. Defaults to 10.0.
            jax_enable_x64: If True, enables 64-bit precision in JAX calculations.
                Defaults to True.
        """
        # Use double precission with JAX.
        jax.config.update("jax_enable_x64", jax_enable_x64)

        self.temperature = temperature
        self.ncomp = len(fluid_parameters)
        self.cutoff_radius = cutoff_radius
        self.maximum_reduced_energy = maximum_reduced_energy

        # Build unitcell.
        self.unitcell = Structure.from_file(cif_file)
        # Calculate number of unit cells in each direction to generate minimal supercell.
        multiples = find_minimal_supercell(
            cell_matrix=self.unitcell.lattice.matrix,
            min_required_radius=cutoff_radius,
        )
        # Create supercell.
        self.supercell = self.unitcell.make_supercell(multiples, in_place=False)

        # Create solid positions in skewed coordinates.
        # Skewed coordinates are fractional coordiantes scaled with lengths.
        self.solid_positions_skewed = (
            self.supercell.frac_coords * self.supercell.lattice.lengths
        )

        # Build sigma and epsilon arrays at for each substance and solid position of supercell.
        (
            self.m,
            self.sigma_pow6_si,
            self.epsilon_k_4_si,
            self.sigma_pow6_si_uc,
            self.epsilon_k_4_si_uc,
        ) = build_interactions(
            self.supercell, self.unitcell, solid_parameters, fluid_parameters
        )

        # Build grid in skewed coordinates of unitcell.
        if isinstance(ngrid, int):
            self.ngrid = [ngrid] * 3
        elif isinstance(ngrid, list):
            self.ngrid = ngrid
        la, lb, lc = self.unitcell.lattice.lengths
        x, y, z = jnp.meshgrid(
            jnp.linspace(
                0.5 * la / self.ngrid[0],
                la - 0.5 * la / self.ngrid[0],
                self.ngrid[0],
            ),
            jnp.linspace(
                0.5 * lb / self.ngrid[1],
                lb - 0.5 * lb / self.ngrid[1],
                self.ngrid[1],
            ),
            jnp.linspace(
                0.5 * lc / self.ngrid[2],
                lc - 0.5 * lc / self.ngrid[2],
                self.ngrid[2],
            ),
            indexing="ij",
        )
        self.grid = jnp.stack((x, y, z), axis=-1)

        # Transformation matrix from skewed to cartesian distance.
        self.skewed2cartesian = jnp.array(
            (self.supercell.lattice.matrix.T / self.supercell.lattice.lengths)
        ).T

        if block_file is not None:
            # Build unitcell.
            self.unitcell_block, block_parameters = generate_block_parameters(
                cif_file, block_file
            )
            # Create supercell.
            self.supercell_block = self.unitcell_block.make_supercell(
                multiples, in_place=False
            )
            # Create solid positions in skewed coordinates.
            # Skewed coordinates are fractional coordiantes scaled with lengths.
            self.solid_positions_skewed_block = (
                self.supercell_block.frac_coords
                * self.supercell_block.lattice.lengths
            )
            # Build sigma and epsilon arrays at for each substance and blocking sphere position of supercell.
            self.sigma_pow6_si_block, self.epsilon_k_4_si_block = (
                build_interactions_block(
                    self.supercell_block, block_parameters, fluid_parameters
                )
            )

    def __repr__(self):
        """Docstring for Framework."""
        unitcell = f"Unitcell(formula: {self.unitcell.formula}, sites: {len(self.unitcell)})"
        supercell = f"Supercell(formula: {self.supercell.formula}, sites: {len(self.supercell)})"
        return f"{unitcell}\n{supercell}"

    def calculate_external_potential(
        self, tail_correction_bool: bool = False
    ) -> jnp.ndarray:
        """Calculates the external potential on the unit cell grid points.

        Calculated as Lennard-Jones interactions between each fluid component
        and all solid atoms within the supercell subject to periodic boundary
        conditions and cutoff radius.

        Returns:
            A jax.numpy array representing the reduced external potential (U/kT)
            for each fluid component at each grid point. The shape is
            (n_components, ngrid, ngrid, ngrid).
        """
        external_potential = jnp.empty([self.ncomp, *self.grid.shape[:-1]])
        lengths = jnp.array(self.supercell.lattice.lengths)
        inverse_temperature = 1.0 / self.temperature

        if tail_correction_bool:
            tail_correction = self.calculate_tail_correction()

        # Loop over slices of the grid to keep dimensionality of matrix
        # of dinstances between grid points and solid atom positions in check.
        for k in range(self.grid.shape[0]):
            # Extract the slab at x-index k. This slice has shape (ny, nz, 3)
            slice_grid = self.grid[k, :, :, :]

            # Compute distance vectors: subtract atom coordinates from each grid point.
            # Broadcasting: (ny, nz, 1, 3) - (1, 1, n_atoms, 3) results in shape (ny, nz, n_atoms, 3)
            distance_vectors = (
                slice_grid[..., jnp.newaxis, :]
                - self.solid_positions_skewed[jnp.newaxis, jnp.newaxis, :, :]
            )

            # 1. Apply the minimum image convention in skewed coordinates:
            #   For each component, adjust the distance vector to the nearest image.
            #   This works element-wise since lengths has shape (3,) and distance_vectors shape is (ny, nz, n_atoms, 3).
            # 2. Transform skewed distance vectors to cartesian distance using the supercell matrix.
            distance_vectors = (
                distance_vectors
                - lengths * jnp.round(distance_vectors / lengths)
            ) @ self.skewed2cartesian

            # Compute the scalar squared distances at this slice: shape (ny, nz, n_atoms)
            distances = jnp.linalg.norm(distance_vectors, axis=-1)

            # Calculate energy for each fluid species
            for i in range(self.ncomp):
                sigma6 = self.sigma_pow6_si[i]  # sigma**6
                eps4 = self.epsilon_k_4_si[i]  # 4 * epsilon_k

                # Compute interaction energy between each grid point in slice and each solid atom
                attraction = sigma6[jnp.newaxis, jnp.newaxis, :] / distances**6
                reduced_energy = (
                    eps4[jnp.newaxis, jnp.newaxis, :]
                    * attraction
                    * (attraction - 1.0)
                    * inverse_temperature
                )

                # Apply the cutoff
                reduced_energy = jnp.where(
                    distances > self.cutoff_radius,
                    0.0,
                    reduced_energy,
                )

                # Sum the contributions over all atoms to get the potential for this slice.
                potential_slice = self.m[i] * jnp.sum(reduced_energy, axis=-1)

                # Apply tail correction.
                if tail_correction_bool:
                    potential_slice += tail_correction[i]

                # Limit maximum reduced energy.
                potential_slice = jnp.where(
                    potential_slice < self.maximum_reduced_energy,
                    potential_slice,
                    self.maximum_reduced_energy,
                )

                # Store the potential in the corresponding slice of the external_potential array.
                external_potential = external_potential.at[i, k, :, :].set(
                    potential_slice
                )
        return external_potential

    def calculate_external_potential_blocking_sphere(self) -> jnp.ndarray:
        """Calculates the external potential of blocking spheres on the unit cell grid points.

        Calculated as Lennard-Jones interactions between each fluid component
        and all blocking spheres within the supercell subject to periodic boundary
        conditions and cutoff radius.

        Returns:
            A jax.numpy array representing the reduced external potential (U/kT)
            for each fluid component at each grid point. The shape is
            (n_components, ngrid, ngrid, ngrid).
        """
        external_potential = jnp.empty([self.ncomp, *self.grid.shape[:-1]])
        lengths = jnp.array(self.supercell_block.lattice.lengths)
        inverse_temperature = 1.0 / self.temperature

        # Loop over slices of the grid to keep dimensionality of matrix
        # of dinstances between grid points and solid atom positions in check.
        for k in range(self.grid.shape[0]):
            # Extract the slab at x-index k. This slice has shape (ny, nz, 3)
            slice_grid = self.grid[k, :, :, :]

            # Compute distance vectors: subtract atom coordinates from each grid point.
            # Broadcasting: (ny, nz, 1, 3) - (1, 1, n_atoms, 3) results in shape (ny, nz, n_atoms, 3)
            distance_vectors = (
                slice_grid[..., jnp.newaxis, :]
                - self.solid_positions_skewed_block[
                    jnp.newaxis, jnp.newaxis, :, :
                ]
            )

            # 1. Apply the minimum image convention in skewed coordinates:
            #   For each component, adjust the distance vector to the nearest image.
            #   This works element-wise since lengths has shape (3,) and distance_vectors shape is (ny, nz, n_atoms, 3).
            # 2. Transform skewed distance vectors to cartesian distance using the supercell matrix.
            distance_vectors = (
                distance_vectors
                - lengths * jnp.round(distance_vectors / lengths)
            ) @ self.skewed2cartesian

            # Compute the scalar squared distances at this slice: shape (ny, nz, n_atoms)
            distances = jnp.linalg.norm(distance_vectors, axis=-1)

            # Calculate energy for each fluid species
            for i in range(self.ncomp):
                sigma6 = self.sigma_pow6_si_block[i]  # sigma**6
                eps4 = self.epsilon_k_4_si_block[i]  # 4 * epsilon_k

                # Compute interaction energy between each grid point in slice and each solid atom
                attraction = sigma6[jnp.newaxis, jnp.newaxis, :] / distances**6
                reduced_energy = (
                    eps4[jnp.newaxis, jnp.newaxis, :]
                    * attraction
                    * (attraction - 1.0)
                    * inverse_temperature
                )

                # Apply the cutoff
                reduced_energy = jnp.where(
                    distances > self.cutoff_radius,
                    0.0,
                    reduced_energy,
                )

                # Sum the contributions over all atoms to get the potential for this slice.
                potential_slice = self.m[i] * jnp.sum(reduced_energy, axis=-1)
                # Limit maximum reduced energy.
                # potential_slice = jnp.where(
                #     potential_slice < self.maximum_reduced_energy,
                #     potential_slice,
                #     self.maximum_reduced_energy,
                # )

                # Store the potential in the corresponding slice of the external_potential array.
                external_potential = external_potential.at[i, k, :, :].set(
                    potential_slice
                )
        return external_potential

    def calculate_external_potential_total(
        self, tail_correction_bool: bool = False
    ):
        external_potential = (
            self.calculate_external_potential(
                tail_correction_bool=tail_correction_bool
            )
            + self.calculate_external_potential_blocking_sphere()
        )
        return jnp.where(
            external_potential < self.maximum_reduced_energy,
            external_potential,
            self.maximum_reduced_energy,
        )

    def calculate_tail_correction(self):
        """Calculates the tail corrections for the external potential.

        Calculates the tail corrections between each fluid component and all solid sites
        within the unitcell. The formula is similar to the one that is used in RASPA.

        Returns:
            A jax.numpy array representing the reduced tail correction for the
            external potential (U_tail/kT) for each fluid component. The shape
            is (n_components).
        """
        tail_corrections_individual = (
            4
            * jnp.pi
            / (3 * self.unitcell.volume)
            * self.epsilon_k_4_si_uc
            * self.sigma_pow6_si_uc
            * (
                self.sigma_pow6_si_uc / (3 * self.cutoff_radius**9)
                - 1 / self.cutoff_radius**3
            )
        )
        tail_corrections = jnp.array(self.m) * jnp.sum(
            tail_corrections_individual, axis=-1
        )
        inverse_temperature = 1.0 / self.temperature
        return tail_corrections * inverse_temperature

    def visualize_external_potential(
        self,
        ax: Any,
        external_potential: jnp.ndarray,
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
        grid_cartesian = self.grid @ (
            self.unitcell.lattice.matrix / self.unitcell.lattice.lengths
        )
        X = grid_cartesian[:, :, z_index, 0]
        Y = grid_cartesian[:, :, z_index, 1]

        slice_2d = external_potential[fluid_index, :, :, z_index]

        _ = ax.pcolormesh(X, Y, slice_2d, cmap="viridis", shading="auto")
        if show_contour_lines:
            contour_levels = np.linspace(
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

    def visualize_cells(self, ax=None):
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

        grid_plot = self.grid @ (
            self.unitcell.lattice.matrix / self.unitcell.lattice.lengths
        )
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
        external_potential: jnp.ndarray,
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
        skewed2cart_unit = jnp.array(
            self.unitcell.lattice.matrix
            / jnp.array(self.unitcell.lattice.lengths)[:, None]
        )
        grid_cartesian = self.grid @ skewed2cart_unit

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
        min_contour_levels = np.array(
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
