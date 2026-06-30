import copy
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from .sigmoid_utils import SigmoidParameterHandler

try:
    import stan
except ImportError:  # pragma: no cover - optional dependency
    stan = None


def _require_stan():
    if stan is None:
        raise ImportError(
            "The optional 'stan' package is required for MCMC preference updates."
        )

class SigmoidFunctions:
    """
    Generate sigmoid functions.
    """
    @staticmethod
    def sigmoid(curve, epsilon, **params):
        """
        Compute discrete sigmoid/gompertz function values for the given parameters.
        """
        if curve == 'sigmoid':
            L = params['L']
            k = params['k']
            c = params['c']
            exp_value = np.exp(k * (epsilon - c))

            return L / (1 + exp_value)

        elif curve == 'gompertz':
            a = params['a']
            b = params['b']
            c = params['c']
            d = params['d']
            epsilon = np.asarray(epsilon)
            a, b, c, d = np.asarray(a), np.asarray(b), np.asarray(c), np.asarray(d)
            
            return -a * np.exp(-b * np.exp(-c * epsilon)) + d
    
   
    @staticmethod
    def generate_random_parameters(curve, num_functions):
        """
        Generate random parameters for sigmoid/gompertz functions.
        """
        if curve == 'sigmoid':
            params = []
            for _ in range(num_functions):
                L = np.random.uniform(0.5, 1.0)
                k = np.random.uniform(0.5, 50.0)
                c = np.random.uniform(0, 1.0)
                params.append((L, k, c))
            return params
        elif curve == 'gompertz':
            params = []
            for _ in range(num_functions):
                a = np.random.uniform(1, 2)
                b = np.random.uniform(0.1, 5)
                c = np.random.uniform(0.1, 5)
                d = np.random.uniform(1, 1.5)
                params.append((a, b, c, d))
            return params


    @staticmethod
    def generate_sigmoid_functions(curve, epsilon_range, params):
        """
        Generate discrete sigmoid/gompertz values for all given parameters.
        """
        sigmoid_functions = []
        if curve == 'sigmoid':
            for L, k, c in params:
                sigmoid_functions.append(SigmoidFunctions.sigmoid(curve, epsilon_range, L=L, k=k, c=c))
        elif curve == 'gompertz':
            for a, b, c, d in params:
                sigmoid_functions.append(SigmoidFunctions.sigmoid(curve, epsilon_range, a=a, b=b, c=c, d=d))
        return sigmoid_functions


class UtilityFunctions:
    """
    Utility functions for simulating users actions and inferring users preference.
    """

    @staticmethod
    def utility_min(epsilon, alpha, w):
        """
        Calculate the minimum utility for the given parameters.
        """
        w = np.where(w == 0, 1e-10, w)
        return np.minimum(epsilon / w[0], alpha / w[1])

    @staticmethod
    def utility_max(epsilon, alpha, w):
        """
        Calculate the minimum utility for the given parameters.
        """
        w = np.where(w == 0, 1e-10, w)
        return np.maximum(epsilon / w[0], alpha / w[1])

    @staticmethod
    def utility_linear(epsilon, alpha, w):
        """
        Calculate the linear utility for the given parameters.
        """
        w = np.where(w == 0, 1e-10, w)
        epsilon = np.array(epsilon)
        alpha = np.array(alpha)
        return (epsilon * w[0] + alpha * w[1])


    @staticmethod
    def utiliy_product(epsilon, alpha, w):
        """
        Calculate the product utility of individual utilities.
        """
        pass

    @staticmethod
    def utility_GP(epsilon, alpha, w):
        """
        Simulate the utility distribution using Gaussian Process.
        """
        pass

class UserActionSimulator:
    """
    Simulate user actions.
    """
    @staticmethod
    def boltzmann_probability(utilities, T):
        """
        Calculate the Boltzmann probability distribution for the given utilities and temperature.
        """
        # Add numerical stability to prevent overflow and NaN
        utilities = np.asarray(utilities)
        
        # Clip utilities to prevent extreme values
        utilities = np.clip(utilities, -100, 100)
        
        # Use log-sum-exp trick for numerical stability
        log_utilities = utilities / T
        # Clip log utilities to prevent overflow
        log_utilities = np.clip(log_utilities, -100, 100)
        
        # Subtract max for numerical stability
        max_log = np.max(log_utilities)
        exp_utilities = np.exp(log_utilities - max_log)
        
        # Normalize
        sum_exp = np.sum(exp_utilities)
        if sum_exp == 0:
            # If all utilities are extremely negative, use uniform distribution
            return np.ones_like(utilities) / len(utilities)
        
        probabilities = exp_utilities / sum_exp
        
        # Ensure no NaN values
        if np.any(np.isnan(probabilities)):
            return np.ones_like(utilities) / len(utilities)
        
        return probabilities

    @staticmethod
    def simulate_user_action(epsilons, sigmoids, w, T, utility_function):
        """
        Simulate the user's action based on the given parameters.
        """
        actions = []
        for alpha in sigmoids:
            utilities = utility_function(epsilons, alpha, w)
            probabilities = UserActionSimulator.boltzmann_probability(utilities, T)
            action = np.random.choice(epsilons, p=probabilities)
            actions.append(action)
        return actions


class UpdatePreference:
    """
    Update the inference based on the new data using importance sampling.
    """

    @staticmethod
    def importance_sampling(
            curve,
            epsilon,
            epsilon_range,
            probs_weights,
            grid_weights,
            T,
            utility_function):
        """
        Update the preference inference for a new observation

        Args:
            curve: the curve shown to the user
            epsilon: the user's choice on the given curve, which is the optimal epsilon
            epsilon_range: discretize all epsilons
            probs_weights: the probability of each weight, which will be updated
            grid_weights: all the possible weights, which remain fixed during updating
            T: temperature in Boltzmann probability

        """

        # Determine curve type based on number of parameters
        if len(curve) == 3:
            curve_type = 'sigmoid'
            param_names = ['L', 'k', 'c']
        elif len(curve) == 4:
            curve_type = 'gompertz'
            param_names = ['a', 'b', 'c', 'd']
        else:
            raise ValueError(f"Unknown curve type with {len(curve)} parameters")

        epsilon = np.asarray(epsilon)
        if epsilon.size != 1:
            raise ValueError(
                f"importance_sampling expected a scalar epsilon, got shape {epsilon.shape}"
            )
        epsilon = float(epsilon.reshape(-1)[0])
        
        # Create parameter dictionary
        param_dict = dict(zip(param_names, curve))
        
        alphas = SigmoidFunctions.sigmoid(curve_type, epsilon_range, **param_dict)  
        alpha = SigmoidFunctions.sigmoid(curve_type, epsilon, **param_dict) 

        # Create a copy to avoid modifying the original array
        probs_weights_copy = probs_weights.copy()

        for i, w in enumerate(grid_weights):
            utilities = utility_function(epsilon_range, alphas, w)
            utility = utility_function(epsilon, alpha, w)
            
            # Add numerical stability to prevent overflow
            utilities = np.clip(utilities, -100, 100)
            utility = np.clip(utility, -100, 100)
            
            # Use log-sum-exp trick for numerical stability
            log_utilities = utilities / T
            log_utility = utility / T
            
            # Clip to prevent overflow
            log_utilities = np.clip(log_utilities, -100, 100)
            log_utility = np.clip(log_utility, -100, 100)
            
            # Subtract max for numerical stability
            max_log = np.max(log_utilities)
            exp_utilities = np.exp(log_utilities - max_log)
            exp_utility = np.exp(log_utility - max_log)
            
            sum_exp = np.sum(exp_utilities)
            if sum_exp == 0:
                likelihood = 1.0 / len(utilities)  # Uniform distribution
            else:
                likelihood = exp_utility / sum_exp
            
            # Ensure likelihood is finite
            if not np.isfinite(likelihood):
                likelihood = 1.0 / len(utilities)

            likelihood = np.asarray(likelihood)
            if likelihood.size != 1:
                raise ValueError(
                    f"importance_sampling expected a scalar likelihood, got shape {likelihood.shape}"
                )
            likelihood = float(likelihood.reshape(-1)[0])
            
            probs_weights_copy[i] *= likelihood

        # compute the effective sample size
        ess = AcquisitionFunctionsPL.compute_ess(probs_weights_copy)
        if ess < len(grid_weights) / 5:  # threshold can be adjusted
            print(f"Warning: Low ESS ({ess}) for h={curve}")

        probs_weights_copy = probs_weights_copy / np.sum(probs_weights_copy)

        return probs_weights_copy

    @staticmethod
    def preference_update(
            curve,
            epsilon,
            epsilon_range,
            probs_weights,
            grid_weights,
            true_w,
            T,
            utility_function):
        """
        Update the inferred preference based on the Importance Sampling.

        Returns:
            probs_weights: updated probability
            error_ozaki: the expected weighted error with updated probability
        """

        probs_weights = UpdatePreference.importance_sampling(
            curve,
            epsilon,
            epsilon_range,
            probs_weights,
            grid_weights,
            T,
            utility_function)

        # weights_inferred = np.sum (probs_weights * grid_weights)
        # # there might be the computation issues
        # error = np.linalg.norm(weights_inferred - true_w)

        error_ozaki = np.sum(probs_weights * np.linalg.norm(grid_weights - true_w, axis=1))

        return probs_weights, error_ozaki


class UpdatePreferenceMCMC:
    """
    Update the preference inference based on MCMC samples.
    """

    @staticmethod
    def build_stan_model_PL(
            epsilons,
            actions,
            sigmoids,
            T,
            num_samples=5000,
            num_chains=2):
        """
        Build a Stan model for the given preference-learning data.
        """
        _require_stan()
        model_code = """
                  data {
                    int<lower=1> N;
                    int<lower=1> M;
                    vector[M] epsilons;
                    real T;
                    vector[N] actions;
                    array[N] vector[M] sigmoids;
                  }

                  parameters {
                    real<lower=0, upper=1> w1;
                  }

                  transformed parameters {
                    real w2;
                    w2 = 1 - w1;
                  }

                  model {
                    w1 ~ uniform(0, 1);

                    for (i in 1:N) {
                      vector[M] utilities;

                      for (j in 1:M) {
                        utilities[j] = fmin(epsilons[j] / w1, sigmoids[i][j] / w2);
                      }

                      int action_index = 1;
                      for (k in 1:M) {
                        if (epsilons[k] == actions[i]) {
                          action_index = k;
                        }
                      }

                      action_index ~ categorical_logit(utilities / T);
                    }
                  }
                  """
        data = {
            'N': len(actions),
            'M': len(epsilons),
            'epsilons': epsilons,
            'actions': actions,
            'T': T,
            'sigmoids': sigmoids,
        }
        model = stan.build(model_code, data=data)

        num_warmup = num_samples // 2
        fit = model.sample(
            num_chains=num_chains,
            num_samples=num_samples,
            num_warmup=num_warmup,
        )

        return fit

    @staticmethod
    def weights_update_mcmc(fit, true_w, posterior_size=1000):
        """
        Summarize the posterior weights inferred by MCMC.
        """
        chains = np.column_stack((fit['w1'], fit['w2']))
        weights_inferred = np.mean(chains, axis=0)
        posterior_weights = chains[-posterior_size:]

        error = np.linalg.norm(weights_inferred - true_w)
        error_ozaki = np.mean(np.linalg.norm(chains - true_w, axis=1))

        return weights_inferred, posterior_weights, error, error_ozaki


class AcquisitionFunctionsPL:
    """
    This is for all the acquisition functions used in the Preference Learning.
    """

    @staticmethod
    def calculate_entropy(posterior_probs):
        """
        Calculate the entropy of the posterior distribution.
        """
        return -np.sum(posterior_probs * np.log(posterior_probs + 1e-10))

    @staticmethod
    def kg_IS(particles_h,
              probs_weights,
              grid_weights,
              probs_sigmoid_params,
              grid_sigmoid_params,
              epsilon_range,
              T,
              utility_function,
              j,
              curve,
              N_actions=10,
              n_jobs=-1):
        """
        Parallelly Calculate the Knowledge Gradient (KG) acquisition function using the Importance Sampling for multiple particle curves
        """
        current_utility = AcquisitionFunctionsPL.utility_for_2particles(
            probs_weights,
            grid_weights,
            probs_sigmoid_params,
            grid_sigmoid_params,
            epsilon_range,
            utility_function,
            curve)
        current_utility = np.array(current_utility)

        # Store original probs_sigmoid_params for verification
        original_probs = probs_weights.copy()
        results = Parallel(n_jobs=n_jobs)(
            delayed(AcquisitionFunctionsPL.compute_optimal_utility_for_single_action)(
                h,
                probs_weights.copy(),
                grid_weights.copy(),
                probs_sigmoid_params.copy(),
                grid_sigmoid_params.copy(),
                epsilon_range,
                T,
                utility_function,
                N_actions,
                current_utility,
                j,
                curve
            )
            for h in particles_h
        )

        # Verify probs_sigmoid_params hasn't changed
        assert np.allclose(original_probs,
                           probs_weights), "probs_weights was modified during parallel execution"

        kg_values = results

        # best_idx = AcquisitionFunctionsPL.random_argmax(kg_values)
        best_idx = np.argmax(kg_values)
        best_h = particles_h[best_idx]
        best_acq = kg_values[best_idx]

        return best_h, best_acq


    @staticmethod
    def compute_optimal_utility_for_single_action(
            h,
            probs_weights,
            grid_weights,
            probs_sigmoid_params,
            grid_sigmoid_params,
            epsilon_range,
            T,
            utility_function,
            N_simulations,
            current_utility,
            j,
            curve):
        """
        Compute optimal posterior utility for a new observation (a single simulated action on a single curve)
        Args:
            h: the target curve for which the acquisition value needs to be computed
            probs_weights: the probability of each weight, which will be updated
            grid_weights: all the possible weights, which remain fixed during updating
            probs_sigmoid_params: the probability of each pair of parameters in Sigmoid function, which will be updated
            grid_sigmoid_params: all the possible parameters in Sigmoid function, which remain fixed during updating
            epsilon_range: discretize all epsilons
            T: temperature in Boltzmann probability

        Returns:
            max_value: the maximum value of posterior mean
        """

        probs_weights_copy = copy.deepcopy(probs_weights)
        grid_weights_copy = copy.deepcopy(grid_weights)
    
        # assert np.allclose(probs_weights_copy, probs_weights), "probs_weights was modified during parallel execution"
        # assert np.allclose(grid_weights_copy, grid_weights), "grid_weights was modified during parallel execution"
        
        # Simulate a possible action, sample a pair of weights first
        rand_num = np.random.choice(len(probs_weights), p=probs_weights_copy/np.sum(probs_weights_copy))
        # rand_num = np.random.choice(len(probs_weights))
        w_sample = grid_weights[rand_num]
        sigmoids = SigmoidFunctions.generate_sigmoid_functions(curve, epsilon_range, [h])
        epsilon = UserActionSimulator.simulate_user_action(
            epsilon_range,
            sigmoids,
            w_sample,
            T,
            utility_function,
        )[0]

        updated_probs_weights = UpdatePreference.importance_sampling(
            h,
            epsilon,
            epsilon_range,
            probs_weights_copy,
            grid_weights_copy,
            T,
            utility_function)
    
        # Compute the updated optimal utility
        updated_utility = AcquisitionFunctionsPL.utility_for_2particles(
            updated_probs_weights,
            grid_weights,
            probs_sigmoid_params,
            grid_sigmoid_params,
            epsilon_range,
            utility_function,
            curve)

        updated_utility = np.array(updated_utility)
        max_value = np.max(updated_utility) - np.max(current_utility)

        # idx = np.argmax(updated_probs_weights)
        # print('------------------')
        # print('preference learning - j:', j)
        # print('max prob_of_inferred_params:', max(updated_probs_weights))
        # print('current max utility:', np.argmax(current_utility), max(current_utility))
        # print('updated max utility:', np.argmax(updated_utility), max(updated_utility))
        # print('------------------')

        return max_value

    @staticmethod
    def utility_for_2particles(
            probs_weights,
            grid_weights,
            probs_sigmoid_params,
            grid_sigmoid_params,
            epsilon_range,
            utility_function,
            curve):
        """
        Compute the expected weighted utilities for two particles
        """

        # estimated_accuracies = [SigmoidFunctions.sigmoid(epsilon_range, L, k, c) for L, k, c in
        #                         grid_sigmoid_params[:, 0:3]]
        estimated_accuracies = SigmoidParameterHandler.compute_particles_accuracies(
            grid_sigmoid_params, epsilon_range, SigmoidFunctions.sigmoid, curve)
        estimated_accuracies = np.sum(estimated_accuracies * probs_sigmoid_params[:, np.newaxis], axis=0)

        all_utility = [utility_function(epsilon_range, estimated_accuracies, w) for w in grid_weights]
        all_utility = np.sum(all_utility * probs_weights[:, np.newaxis], axis=0)

        return all_utility

    @staticmethod
    def compute_ess(weights):
        """
        Compute Effective Sample Size
        """
        return 1 / np.sum(weights ** 2)

    @staticmethod
    def random_argmax(arr):
        """
        Returns a random index of the maximum value in the array.
        If there are multiple occurrences of the maximum value,
        randomly selects one of those indices.
        """
        max_value = np.max(arr)
        max_indices = np.where(arr == max_value)[0]
        if len(max_indices) > 1:
            return np.random.choice(max_indices)
        else:
            return np.argmax(arr)

    @staticmethod
    def random(curve):
        """
        Randomly pick a curve.
        """
        best_params = SigmoidFunctions.generate_random_parameters(curve, 1)
        return best_params, 0
