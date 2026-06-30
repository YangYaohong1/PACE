import numpy as np
from typing import List, Tuple, Union, Callable

class SigmoidParameterHandler:
    """
    A utility class for handling different sigmoid function parameter formats.
    This makes the code more general and flexible for different sigmoid functions.
    """
    
    @staticmethod
    def extract_parameters(grid_sigmoid_params: np.ndarray, exclude_sigma: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract parameters from the grid_sigmoid_params array.
        
        Args:
            grid_sigmoid_params: Array of shape (n_particles, n_params) where n_params includes sigma
            exclude_sigma: If True, treat the last column as sigma and exclude it from function parameters
            
        Returns:
            Tuple of (param_samples, sigma_samples) where:
            - param_samples: List of arrays, one for each function parameter
            - sigma_samples: Array of sigma values
        """
        if exclude_sigma:
            num_params = grid_sigmoid_params.shape[1] - 1
            param_samples = [grid_sigmoid_params[:, i] for i in range(num_params)]
            sigma_samples = grid_sigmoid_params[:, -1]
        else:
            num_params = grid_sigmoid_params.shape[1]
            param_samples = [grid_sigmoid_params[:, i] for i in range(num_params)]
            sigma_samples = None
            
        return param_samples, sigma_samples
    
    @staticmethod
    def get_particle_parameters(grid_sigmoid_params: np.ndarray, particle_idx: int, exclude_sigma: bool = True) -> Tuple[List[float], float]:
        """
        Get parameters for a specific particle.
        
        Args:
            grid_sigmoid_params: Array of shape (n_particles, n_params)
            particle_idx: Index of the particle
            exclude_sigma: If True, treat the last column as sigma
            
        Returns:
            Tuple of (function_params, sigma) where:
            - function_params: List of parameter values for the sigmoid function
            - sigma: Sigma value (None if exclude_sigma=False)
        """
        if exclude_sigma:
            num_params = grid_sigmoid_params.shape[1] - 1
            function_params = [grid_sigmoid_params[particle_idx, i] for i in range(num_params)]
            sigma = grid_sigmoid_params[particle_idx, -1]
        else:
            num_params = grid_sigmoid_params.shape[1]
            function_params = [grid_sigmoid_params[particle_idx, i] for i in range(num_params)]
            sigma = None
            
        return function_params, sigma
    
    @staticmethod
    def compute_weighted_parameters(grid_sigmoid_params: np.ndarray, weights: np.ndarray, exclude_sigma: bool = True) -> Tuple[List[float], float]:
        """
        Compute weighted average of parameters across all particles.
        
        Args:
            grid_sigmoid_params: Array of shape (n_particles, n_params)
            weights: Array of weights for each particle
            exclude_sigma: If True, treat the last column as sigma
            
        Returns:
            Tuple of (weighted_function_params, weighted_sigma) where:
            - weighted_function_params: List of weighted parameter values
            - weighted_sigma: Weighted sigma value (None if exclude_sigma=False)
        """
        if exclude_sigma:
            num_params = grid_sigmoid_params.shape[1] - 1
            weighted_function_params = []
            for i in range(num_params):
                weighted_param = np.sum(weights * grid_sigmoid_params[:, i])
                weighted_function_params.append(weighted_param)
            weighted_sigma = np.sum(weights * grid_sigmoid_params[:, -1])
        else:
            num_params = grid_sigmoid_params.shape[1]
            weighted_function_params = []
            for i in range(num_params):
                weighted_param = np.sum(weights * grid_sigmoid_params[:, i])
                weighted_function_params.append(weighted_param)
            weighted_sigma = None
            
        return weighted_function_params, weighted_sigma
    
    @staticmethod
    def call_sigmoid_function(sigmoid_func: Callable, curve: str, epsilon: Union[float, np.ndarray], 
                            params: List[float], param_names: List[str] = None, **kwargs) -> Union[float, np.ndarray]:
        """
        Call a sigmoid function with unpacked parameters.
        
        Args:
            sigmoid_func: The sigmoid function to call
            curve: The curve type (e.g., 'sigmoid', 'gompertz')
            epsilon: Input value(s) for the sigmoid function
            params: List of parameters to pass to the sigmoid function
            param_names: List of parameter names to use as keyword arguments
            **kwargs: Additional keyword arguments for the sigmoid function
            
        Returns:
            Output of the sigmoid function
        """
        if param_names is not None:
            # Pass parameters as keyword arguments
            param_dict = dict(zip(param_names, params))
            return sigmoid_func(curve, epsilon, **param_dict, **kwargs)
        else:
            # Fallback to positional arguments (for backward compatibility)
            return sigmoid_func(curve, epsilon, *params, **kwargs)
    
    @staticmethod
    def compute_particles_accuracies(grid_sigmoid_params: np.ndarray, epsilon_range: np.ndarray, 
                                   sigmoid_func: Callable, curve: str, exclude_sigma: bool = True) -> List[np.ndarray]:
        """
        Compute accuracy values for all particles using the sigmoid function.
        
        Args:
            grid_sigmoid_params: Array of shape (n_particles, n_params)
            epsilon_range: Array of epsilon values
            sigmoid_func: The sigmoid function to use
            curve: The curve type (e.g., 'sigmoid', 'gompertz')
            exclude_sigma: If True, treat the last column as sigma
            
        Returns:
            List of accuracy arrays, one for each particle
        """
        particles_accuracies = []
        num_particles = grid_sigmoid_params.shape[0]
        
        # Define parameter names and expected number of parameters based on curve type
        if curve == 'sigmoid':
            param_names = ['L', 'k', 'c']
            expected_params = 3
        elif curve == 'gompertz':
            param_names = ['a', 'b', 'c', 'd']
            expected_params = 4
        else:
            param_names = None  # Fallback to positional arguments
            expected_params = None
        
        for i in range(num_particles):
            function_params, _ = SigmoidParameterHandler.get_particle_parameters(
                grid_sigmoid_params, i, exclude_sigma=exclude_sigma)
            
            # Ensure we have the correct number of parameters for the curve type
            if expected_params is not None and len(function_params) != expected_params:
                print(f"Warning: Expected {expected_params} parameters for {curve} curve, but got {len(function_params)}")
                print(f"grid_sigmoid_params shape: {grid_sigmoid_params.shape}")
                # Take only the first expected_params parameters
                function_params = function_params[:expected_params]
            
            accuracy = SigmoidParameterHandler.call_sigmoid_function(
                sigmoid_func, curve, epsilon_range, function_params, param_names)
            particles_accuracies.append(accuracy)
            
        return particles_accuracies 