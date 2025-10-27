from typing import List, Tuple, Optional
from jax.typing import ArrayLike
from jax import Array

import jax.numpy as jnp


class Grid:
    """Representation of 3-D grid of cDFT domain.

    Attributes
    ----------
    dim : int
        Number of dimensions of the problem (can only equal `{1, 2, 3}`).
    n_grid : Array
        Number of grid points in each of the 3 directions (can only be a power of 2 on GPU).
    length : Array
        Length of each spatial grid direction in Angstrom.
    skew_angles : [float]
        Angles `[alpha, beta, gamma]` of the skew coordinate system the grid is defined on. 
    z : [Array]
        Grid in each spatial direction in Angstrom.
    k : Array
        Fourier cooredinates `[k_x, k_y, k_z]` in each grid cell.
    k_abs : Array
        Absolute value of the Fourier coordinates `sqrt(k_x² + k_y² + k_z²)` in each grid cell.
    coordinate_transform : Array
        Coordinate transform matrix `H` to transform skew `s = [u, v, w]` to Cartesian `r = [x, y, z]` coordinates: `r = H s`.
    jdet : float
        Jacobi determinant `det(H) = ` of the skew coordinate system.
    dv : float
        Volume of one grid cell in Angstrom cubed: `dv = det(H) \prod_i \Delta z_i`.
    lanczos_sigma : bool
        Switch if Lanczos sigma factor should be applied or not.
    sigma : Array
        Lanczos sigma factor for each grid cell.
    """

    def __init__(self, n_grid: List[int], length: List[float], skew_angles: Optional[List[float]] = None, lanczos_sigma: bool = True):
        """Grid information.

        Parameters
        ----------
        n_grid : [int]
            Number of grid points of each axis.
        length : [float]
            Length of each axis in Angstrom.
        skew_angles : [float], optional
            Angles `[alpha, beta, gamma]` of skewed coordinate system in `degree`. Default: ``None`` meaning ``[90.0, 90.0, 90.0]``.
            For 2-D systems only ``gamma`` deviates from ``90.0``.
        lanczos : bool, optional
            Apply Lanczos sigma smoothing. Default: ``True``.
        """
        # Grid information
        self.dim = len(n_grid)
        self.n_grid = n_grid
        self.length = length
        self.skew_angles = skew_angles

        # Input checks
        if self.dim not in {1, 2, 3}:
            raise InvalidDimension('n_grid', self.dim)
        if len(self.length) not in {1, 2, 3}:
            raise InvalidDimension('length', len(length))
        if self.dim != len(self.length):
            raise InvalidDimension('length', len(length))
        # if not all([_is_power_of_two(n) for n in n_grid]):
        #     raise InvalidGridPoints(n_grid)
        if not all([l > 0 for l in length]):
            raise InvalidGridLength(length)
        if self.skew_angles != None:
            if self.dim == 3 and self.dim != len(self.skew_angles):
                raise InvalidSkewedAngles(self.skew_angles, self.dim)
            elif self.dim == 2 and 1 != len(self.skew_angles):
                raise InvalidSkewedAngles(self.skew_angles, self.dim)
            elif self.dim == 1 and 0 != len(self.skew_angles):
                raise InvalidSkewedAngles(self.skew_angles, self.dim)
        # if any(ang <= 0 for ang in skew_angles) or any(ang > jnp.pi/2 for ang in skew_angles):
        #     raise InvalidSkewedAngles(skew_angles)

        # Write zeros into skew_angle list for 2-D system
        if skew_angles is not None:
            if len(skew_angles) == 1 and self.dim == 2:
                self.skew_angles = [0.0, 0.0, skew_angles[0]]

        # Help variables
        dl = [li / ni for li, ni in zip(length, n_grid)]
        self.z = [jnp.linspace(dli / 2.0, li - dli / 2.0, ni)
                  for li, dli, ni in zip(length, dl, n_grid)]

        # Building the vectors for the Fourier-grid for Cartesian coordinates
        # only the last/3rd dimension is treated by a real-valued FFT
        ki = [jnp.fft.fftfreq(ni, dli)
              for ni, dli in zip(self.n_grid, dl)]
        ki[-1] = jnp.fft.rfftfreq(self.n_grid[-1], dl[-1])

        # Fourier component meshgrid
        k_meshgrid = jnp.meshgrid(*ki, indexing='ij')

        if skew_angles == None:
            self.jdet = 1.0
            self.dv = jnp.prod(jnp.array(dl))
            self.transformation_matrix = jnp.identity(self.dim)
        # Building the Fourier-grid (k-vector with skew angles != 90° -> in skew coordinates)
        else:
            k_meshgrid, self.dv, self.jdet, self.transformation_matrix = self._fourier_grid_cartesian(
                k_meshgrid=k_meshgrid, grid_spacing=dl)

        # Fourier meshgrid & absolute value of the Fourier grid;
        self.k = 2 * jnp.pi * jnp.stack(k_meshgrid)
        self.k_abs = jnp.sqrt((self.k**2).sum(axis=0))

        # Calculation of Lanczos sigma factor;
        self.lanczos_sigma = lanczos_sigma
        sigmai = [jnp.sinc(kii * li / (ni // 2 + 1))
                  for kii, li, ni in zip(ki, self.length, self.n_grid)]
        self.sigma = jnp.stack(jnp.meshgrid(
            *sigmai, indexing='ij')).prod(axis=0)

    def __repr__(self):
        return f'DFT grid with `dim`: {self.dim}, `n_grid`: {self.n_grid}, `length`: {self.length}, and `skew_angles`: {self.skew_angles}.'

    def _fourier_grid_cartesian(self, k_meshgrid: ArrayLike, grid_spacing: List[float]) -> Tuple[Array, float, float]:
        """Fourier grid vectors in Cartesian coordintes `[x, y, z]` where the FFT is done in skew
        coordinates `[u, v, w]` (depending on skew angles `[alpha, beta, gamma]`).

        Parameters
        ----------
        k_meshgrid : ArrayLike
            Fourier variable as meshhgrid.
        grid_spacing : [float]
            Grid spacing `dz` in each spatial direction.

        Returns
        -------
        k_meshgrid_cart : [Array]
            Fourier variable of skew Fourier coordinates in Cartesian coordinates as meshgrid `[k_x, k_y, k_z]`.
        dv : float
            Volume of skew unit cell.
        det : float
            Determinant of coordinate transform to  Cartesian coordinates.
        """
        # Skew angles in `degree` and help variable `zeta``
        alpha, beta, gamma = jnp.deg2rad(jnp.array(self.skew_angles))
        zeta = (jnp.cos(alpha) - jnp.cos(beta) *
                jnp.cos(gamma)) / jnp.sin(gamma)

        # Jacobi determinant det(H); H: r -> s; 1/det(H): H^-1: s -> r
        nu = jnp.sqrt(1 - jnp.cos(beta)**2 - zeta**2)
        jdet = jnp.sin(gamma) * nu
        dv = jnp.prod(jnp.array(grid_spacing)) * jdet

        # Transforming the skew Fourier grid back to Cartesian coordinates (for 2-D & 3-D);
        # k[0] stays the same
        if self.dim == 3:
            k_meshgrid[2] = (k_meshgrid[0] * (zeta * jnp.cos(gamma) - jnp.cos(beta) *
                                              jnp.sin(gamma)) - k_meshgrid[1] * zeta + k_meshgrid[2] * jnp.sin(gamma)) / jdet
        if self.dim >= 2:
            k_meshgrid[1] = -k_meshgrid[0] / \
                jnp.tan(gamma) + k_meshgrid[1] / jnp.sin(gamma)

        #
        transformation_matrix = jnp.array([[1, jnp.cos(gamma), jnp.cos(beta)], [
                                          0, jnp.sin(gamma), zeta], [0, 0, nu]])[:self.dim, :self.dim]

        return k_meshgrid, dv, jdet, transformation_matrix


def _is_power_of_two(n: int) -> bool:
    """Checks if input is a power of 2.

    Parameters
    ----------
    n : int
        Number to be checked if power of 2.

    Returns
    -------
    Boolean (`True` if input is a power of 2).
    """
    return (n != 0) and (n & (n-1) == 0)


class InvalidDimension(Exception):
    """Raised when other than 3-D system is tried to be initialized."""

    def __init__(self, variable: List, dim: int):
        self.message = f"Grid dimension different from 1, 2, or 3. Dimension of `{variable} = {dim}`."
        super().__init__(self.message)


class InvalidGridLength(Exception):
    """Raised when grid length is non-negative."""

    def __init__(self, length: List[float]):
        self.message = f"Non-negative `length = {length}`"
        super().__init__(self.message)


class InvalidGridPoints(Exception):
    """Raised when number of grid points in each dimension is not a power of 2."""

    def __init__(self, n_grid: List[int]):
        self.message = f"Invalid number of grid points defined `{n_grid}`. Needs to be power of 2."
        super().__init__(self.message)


class InvalidSkewedAngles(Exception):
    """Raised when skewed angles are invalid."""

    def __init__(self, skew_angles: List[float], dim: int):
        self.message = f"Invalid number of skewed angles `{len(skew_angles)}` for dimension `{dim}`. For 3-D ``3 angles, for 2-D `1` angle, for 1-D `0` angles need to be provided."
        super().__init__(self.message)
