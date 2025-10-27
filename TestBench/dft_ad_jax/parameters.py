from typing import List, Dict

import json
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


class PcSaftParameters:
    """Representation of PC-SAFT parameters.

    Attributes 
    ----------
    n_comp : int
        Number of components. 
    m : Array
        Chain length parameter of PC-SAFT.
    sigma : Tensor
        Segment size parameter of PC-SAFT.
    epsilon_k : Array
        Energy parameter of dispersive interactions of PC-SAFT.
    k_ij : Array
        Binary interaction parameter matrix. 
    sigma_ij : Array
        Binary segment size parameter matrix. 
    eps_ij_k : Array
        Binary energy parameter matrix. 
    c1 : float
        First parameter for equivalent hard-sphere diameter. 
    c2 : float
        Second parameter for equivalent hard-sphere diameter. 

    Methods
    -------
    _hard_sphere_radius(temperature):
        Calculates the equivalent hard-sphere radius from the given
        parameters and temperature.
    """

    def __init__(self, m: ArrayLike, sigma: ArrayLike, epsilon_k: ArrayLike, k_ij: ArrayLike = None):
        """Parameters for PC-SAFT functionals.

        Parameters
        ----------
        m : ArrayLike
            Chain length parameter.
        sigma : ArrayLike
            Segment size parmeter in Angstrom.
        epsilon_k : ArrayLike
            Energy parameter in Kelvin.
        k_ij : ArrayLike
            Binary interaction parameters.

        Methods
        -------
        _hard_sphere_radius(temperature):
            Calculates the equivalent hard-sphere radius from the given
            parameters and temperature.
        """
        self.n_comp = len(m)
        self.m = jnp.array(m)
        self.sigma = jnp.array(sigma)
        self.epsilon_k = jnp.array(epsilon_k)

        if jnp.array_equal(self.m.shape, self.sigma.shape) and jnp.array_equal(self.m.shape, self.epsilon_k.shape):
            pass
        else:
            raise InvalidParameterDimension(
                [self.m, self.sigma, self.epsilon_k])

        if k_ij is not None:
            self.k_ij = jnp.array(k_ij)
        else:
            self.k_ij = jnp.zeros([self.n_comp]*2)
        self.sigma_ij = 0.5 * (self.sigma[None] + self.sigma[:, None])
        self.eps_ij_k = jnp.sqrt(
            self.epsilon_k[None] * self.epsilon_k[:, None]) * (1 - self.k_ij)
        self.c1 = 0.12
        self.c2 = 3.0

    def _hard_sphere_radius(self, temperature: float) -> Array:
        """Hard-sphere diameter for PC-SAFT hard-sphere Helmholtz energy functional.

        Parameters
        ----------
        temperature : float 
            Temperature in Kelvin at which the DFT calculation if performed. 

        Returns
        -------
        Hard-sphere diamter in Angstrom.
        """
        return 0.5 * self.sigma * (1 - self.c1 * jnp.exp(-self.c2 * self.epsilon_k / temperature))

    def __repr__(self):
        return f'Parameter set with: `m`: {self.m}, `sigma`: {self.sigma}, `epsilon_k`: {self.epsilon_k}, `k_ij`: {self.k_ij}.'


def pcsaft_from_file(components, pure_parameters: str, binary_parameters: str = None) -> PcSaftParameters:
    """PC-SAFT parameter from JSON input file. 

    Parameters
    ----------
    components : [str] | str
        Name of components to be simulated. 
    pure_paramters : str
        Pure parameter file name. 
    binary_parameters : str
        Binary interaction parameter file name. 

    Returns
    -------
    PcSaftParameters class which is used for DFT computations.
    """
    with open(pure_parameters) as json_file:
        parameters_pure = json.load(json_file)
    if type(components) == str:
        components = [components]
    model_records = _search_by_name_pure(components, parameters_pure)

    m = [mr['m'] for mr in model_records]
    sigma = [mr['sigma'] for mr in model_records]
    epsilon_k = [mr['epsilon_k'] for mr in model_records]

    k_ij = 0.0
    if len(components) > 1:
        if binary_parameters is not None:
            with open(binary_parameters) as json_file:
                parameters_binary = json.load(json_file)
            k_ij = _search_by_name_binary(components, parameters_binary)
        else:
            k_ij = 0.0
        k_ij = [[0.0, k_ij], [k_ij, 0.0]]

    return PcSaftParameters(jnp.array(m), jnp.array(sigma), jnp.array(epsilon_k), jnp.array(k_ij))


class MultipleParametersComponent(Exception):
    "More than 1 parameter set defined."

    def __init__(self, name):
        self.message = f"More than 1 parameter set defined for '{name}'"
        super().__init__(self.message)


class TooManyComponents(Exception):
    "More than 2 components defined for binary record."

    def __init__(self, name):
        self.message = f"More than 2 components defined for binary record"
        super().__init__(self.message)


class InvalidParameterDimension(Exception):
    """Raised when number of pure component parameters per component do not match."""

    def __init__(self, params: List[Array]):
        self.message = f"Non-identical number of components for each parameter: (m, sigma, epsilon_k) `({len(params[0]), len(params[1]), len(params[2])})`."
        super().__init__(self.message)


def _search_by_name_pure(names: List[str], list_of_dicts: List[Dict]) -> List[Dict]:
    """Extract equation of state parameters from a List of Dictionaries.

    Parameters
    ----------
    names : [str]
        List of component names. 
    list_of_dicts : [Dict]
        Equation of state parameters from JSON file.

    Returns
    -------
    Equation of state parameters.
    """
    model_records = []

    for name in names:
        counter = 0
        for component in list_of_dicts:
            if component['identifier']['name'] == name.lower():
                counter += 1
                model_records.append(component['model_record'])
                if counter > 1:
                    print(component['identifier']['name'])
                    raise MultipleParametersComponent(
                        component['identifier']['name'])
    return model_records


def _search_by_name_binary(names: List[str], list_of_dicts: List[Dict]) -> float:
    """Extract binary interaction parameters from a List of Dictionaries.

    Parameters
    ----------
    names : [str]
        List of component names. 
    list_of_dicts : [Dict]
        binary interaction parameter from JSON file.

    Returns
    -------
    Binary interaction parameter.
    """
    if len(names) > 2:
        raise TooManyComponents(names)
    k_ij = None

    counter = 0
    for pair in list_of_dicts:
        if sorted([pair['id1']['name'], pair['id2']['name']]) == sorted([name.lower() for name in names]):
            counter += 1
            k_ij = pair['model_record']['k_ij']
            if counter > 1:
                raise MultipleParametersComponent(pair['model_record']['k_ij'])
    return k_ij
