import copy
from scipy.stats import norm
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from .preference_learning import *

try:
    import stan
except ImportError:  # pragma: no cover - optional dependency
    stan = None


def _require_stan():
    if stan is None:
        raise ImportError(
            "The optional 'stan' package is required for MCMC Pareto-front updates."
        )

class UpdateParetoFront:
    """
    Update the inference based on the new data.
    """
    @staticmethod
    def importance_sampling(
            best_epsilon,
            best_accuracy,
            probs_sigmoid_params,
            grid_sigmoid_params):
        """
           Update the sigmoid parameters inference for a new observation

           Args:
               epsilon: a new observed point
               accuracy: the observed accuracy
               probs_sigmoid_params: the probability of each pair of parameters, which will be updated
               grid_sigmoid_params: all the possible parameters, which remain fixed during updating

        """

        best_epsilon = np.asarray(best_epsilon)
        if best_epsilon.size != 1:
            raise ValueError(
                f"importance_sampling expected a scalar epsilon, got shape {best_epsilon.shape}"
            )
        best_epsilon = float(best_epsilon.reshape(-1)[0])

        best_accuracy = np.asarray(best_accuracy)
        if best_accuracy.size != 1:
            raise ValueError(
                f"importance_sampling expected a scalar accuracy, got shape {best_accuracy.shape}"
            )
        best_accuracy = float(best_accuracy.reshape(-1)[0])

        # Get the number of parameters (excluding sigma which is always the last parameter)
        num_params = grid_sigmoid_params.shape[1] - 1  # -1 because last column is sigma
        
        # Extract parameters using indexing instead of hardcoded names
        param_samples = [grid_sigmoid_params[:, i] for i in range(num_params)]
        sigma_samples = abs(grid_sigmoid_params[:, -1])  # Last column is always sigma
        
        N_samples = len(param_samples[0])
        interval_width = 0.002
        new_probs = probs_sigmoid_params.copy()

        for i in range(N_samples):
            # alpha_sample = SigmoidFunctions.sigmoid(best_epsilon, L_samples[i], k_samples[i], c_samples[i])
            # Unpack parameters for the current sample
            current_params = [param_samples[j][i] for j in range(num_params)]
            
            # Call sigmoid function with unpacked parameters
            # Determine curve type based on number of parameters
            if len(current_params) == 3:
                curve_type = 'sigmoid'
                param_names = ['L', 'k', 'c']
            elif len(current_params) == 4:
                curve_type = 'gompertz'
                param_names = ['a', 'b', 'c', 'd']
            else:
                raise ValueError(f"Unknown curve type with {len(current_params)} parameters")
            
            # Create parameter dictionary
            param_dict = dict(zip(param_names, current_params))
            alpha_sample = SigmoidFunctions.sigmoid(curve_type, best_epsilon, **param_dict)
            likelihood = norm.pdf(best_accuracy, loc=alpha_sample, scale=sigma_samples[i]) * interval_width
            likelihood = np.asarray(likelihood)
            if likelihood.size != 1:
                raise ValueError(
                    f"importance_sampling expected a scalar likelihood, got shape {likelihood.shape}"
                )
            new_probs[i] *= float(likelihood.reshape(-1)[0])

        new_probs = new_probs / np.sum(new_probs)
        if np.isnan(new_probs).any():
            new_probs = np.ones(N_samples)/N_samples

        # ess = AcquisitionFunctionsPL.compute_ess(new_probs)
        # if ess < N_samples / 10:
        #     print(f"Warning: Low ESS ({ess}) for epsilon={best_epsilon}")

     
        return new_probs, grid_sigmoid_params

    @staticmethod
    def params_update(
            epsilon,
            accuracy,
            probs_sigmoid_params,
            grid_sigmoid_params
            ):
        """
        Update the parameters in Sigmoid function based on the importance sampling.
        """
        probs_sigmoid_params, grid_sigmoid_params = UpdateParetoFront.importance_sampling(
            epsilon,
            accuracy,
            probs_sigmoid_params,
            grid_sigmoid_params)

        return probs_sigmoid_params, grid_sigmoid_params


class UpdateParetoFront_MCMC:
    """
    Update the Pareto-front inference based on MCMC samples.
    """

    @staticmethod
    def build_stan_model_PF(
            epsilon,
            accuracy,
            seed=None,
            num_samples=20000,
            num_chains=4):
        """
        Perform Bayesian inference on the sigmoid function parameters.
        """
        _require_stan()
        sigmoid_model_code = """
                data {
                    int<lower=0> N;
                    array[N] real epsilon;
                    array[N] real accuracy;
                }
                parameters {
                    real<lower=0, upper=1> L;
                    real<lower=0> k;
                    real<lower=0, upper=1> c;
                    real<lower=0> sigma;
                }
                model {
                    array[N] real mu;
                    for (i in 1:N) {
                        mu[i] = L / (1 + exp(k * (epsilon[i] - c)));
                    }

                    L ~ beta(40, 2);
                    k ~ lognormal(log(5), 1);
                    c ~ beta(2, 2);
                    sigma ~ normal(0, 1);

                    accuracy ~ normal(mu, sigma);
                }
                """

        sigmoid_data = {
            'N': len(epsilon),
            'epsilon': epsilon,
            'accuracy': accuracy,
        }

        build_kwargs = {'data': sigmoid_data}
        if seed is not None:
            build_kwargs['random_seed'] = seed

        model = stan.build(sigmoid_model_code, **build_kwargs)

        num_warmup = num_samples // 2
        fit = model.sample(
            num_chains=num_chains,
            num_samples=num_samples,
            num_warmup=num_warmup,
        )

        return fit

    @staticmethod
    def params_update_mcmc(fit, particles):
        """
        Summarize the posterior sigmoid parameters inferred by MCMC.
        """
        chains = np.column_stack((fit['L'], fit['k'], fit['c'], fit['sigma']))
        params_inferred = np.mean(chains, axis=0)
        posterior_params = chains[-particles:]

        return params_inferred, posterior_params


class AcquisitionFunctionsBO:
    """
    This is for all acquisition functions used in Bayesian Optimization.
    """

    @staticmethod
    def sampled_utility(
            epsilon,
            posterior_params,
            posterior_weights,
            n_samples,
            utility_function):
        """
        This is to calculate the utility at the given points of all parameters.
        """
        all_utility = []
        L_samples = posterior_params[-n_samples:,0]
        k_samples = posterior_params[-n_samples:,1]
        c_samples = posterior_params[-n_samples:,2]
        posterior_ws_samples = posterior_weights[-n_samples:]

        for _ in range(n_samples):
            L = L_samples[_]
            k = k_samples[_]
            c = c_samples[_]
            weights = posterior_ws_samples[_]
            alpha = SigmoidFunctions.sigmoid('sigmoid', epsilon, L=L, k=k, c=c)
            utility = utility_function(epsilon, alpha, weights)
            all_utility.append(utility)

        return all_utility

    @staticmethod
    def kg_IS(
            particles_epsilons,
            probs_weights,
            grid_weights,
            probs_sigmoid_params,
            grid_sigmoid_params,
            epsilon_range,
            utility_function,
            dic,
            j,
            evaluated_epsilons,  # for mcmc
            evaluated_accuracies,  # mcmc
            curve='gompertz',  # default to gompertz for backward compatibility
            N_actions=10,
            n_jobs=-1):
        """
        Parallelly Calculate the Knowledge Gradient (KG) acquisition function using the Importance Sampling across candidate points
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
        original_probs = probs_sigmoid_params.copy()
        results = Parallel(n_jobs=n_jobs)(
            delayed(AcquisitionFunctionsBO.compute_optimal_utility_for_single_action_bo)(
                            epsilon,
                            probs_weights.copy(),
                            grid_weights.copy(),
                            probs_sigmoid_params.copy(),
                            grid_sigmoid_params.copy(),
                            epsilon_range,
                            utility_function,
                            N_actions,
                            current_utility,
                            j,
                            dic,
                            curve,
                            evaluated_epsilons,  # for mcmc
                            evaluated_accuracies  # mcmc
                        )
                        for epsilon in particles_epsilons
        )

        # Verify probs_sigmoid_params hasn't changed
        assert np.allclose(original_probs,
                           probs_sigmoid_params), "probs_sigmoid_params was modified during parallel execution"

        kg_values = results

        best_idx = AcquisitionFunctionsBO.random_argmax(kg_values)
        # best_idx = np.argmax(kg_values)
        best_epsilon = particles_epsilons[best_idx]
        best_acq = kg_values[best_idx]

        plt.plot(particles_epsilons, kg_values, label='kg-IS')
        plt.xlabel('Epsilons')
        plt.ylabel('Acquisition function values')
        plt.title(f'Distribution of Acq with IS: {best_epsilon}')
        plt.grid(True)
        plt.savefig(f'{dic}/Acq_{j}.png')
        plt.clf()

        return best_epsilon, best_acq


    @staticmethod
    def compute_optimal_utility_for_single_action_bo(
            epsilon,
            probs_weights,
            grid_weights,
            probs_sigmoid_params,
            grid_sigmoid_params,
            epsilon_range,
            utility_function,
            N_simulations,
            current_utility,
            j,
            dic,
            curve,
            evaluated_epsilons, # for mcmc
            evaluated_accuracies, # mcmc
            sample='IS' # for mcmc
    ):
        """
        Compute optimal posterior utility for a new observation (a single simulated accuracy on a single epsilon)
        Args:
            epsilon: the target epsilon for which the acquisition value needs to be computed
            probs_weights: the probability of each weight, which will be updated
            grid_weights: all the possible weights, which remain fixed during updating
            probs_sigmoid_params: the probability of each pair of parameters in Sigmoid function, which will be updated
            grid_sigmoid_params: all the possible parameters in Sigmoid function, which remain fixed during updating
            epsilon_range: discretize all epsilons

        Returns:
            max_value: the maximum value of posterior mean
        """

        # Work with copies for the importance sampling
        probs_sigmoid_params_copy = copy.deepcopy(probs_sigmoid_params)
        grid_sigmoid_params_copy = copy.deepcopy(grid_sigmoid_params)

        assert np.allclose(probs_sigmoid_params_copy,
                           probs_sigmoid_params), "probs_sigmoid_params was modified during parallel execution"

        # Get the number of parameters (excluding sigma which is always the last parameter)
        num_params = grid_sigmoid_params_copy.shape[1] - 1  # -1 because last column is sigma
        
        # Extract parameters using indexing instead of hardcoded names
        param_samples = [grid_sigmoid_params_copy[:, i] for i in range(num_params)]
        sigma_samples = grid_sigmoid_params_copy[:, -1]  # Last column is always sigma
        N_samples = len(param_samples[0])
        max_utility = []

        for i in range(N_simulations):
            # 1. Sample parameters for generating simulated accuracy
            idx = np.random.choice(N_samples, p=probs_sigmoid_params_copy/np.sum(probs_sigmoid_params_copy))
            
            # Extract current parameter values
            current_params = [param_samples[j][idx] for j in range(num_params)]
            sigma = abs(sigma_samples[idx])

            assert np.allclose(probs_sigmoid_params_copy, probs_sigmoid_params), "probs_sigmoid_params was modified during parallel execution"

            # 2. Generate observation
            # Determine curve type based on number of parameters
            if len(current_params) == 3:
                curve_type = 'sigmoid'
                param_names = ['L', 'k', 'c']
            elif len(current_params) == 4:
                curve_type = 'gompertz'
                param_names = ['a', 'b', 'c', 'd']
            else:
                raise ValueError(f"Unknown curve type with {len(current_params)} parameters")
            
            # Create parameter dictionary
            param_dict = dict(zip(param_names, current_params))
            accuracy = SigmoidFunctions.sigmoid(curve_type, epsilon, **param_dict)
            # Sample from Gaussian likelihood
            accuracy_obs = min(np.random.normal(accuracy, sigma), 1)
            accuracy_obs = max(accuracy_obs, 0)

            # true_accuracy = SigmoidFunctions.sigmoid(epsilon_range, L, k, c)
            # plt.figure(figsize=(8, 6))
            # plt.plot(epsilon_range, true_accuracy, color='black')
            # plt.scatter(epsilon, accuracy_obs, color='red')
            # plt.xlabel('Epsilons')
            # plt.ylabel('Accuracies')
            # plt.title(f'Simulated points_{j}_{epsilon}')
            # plt.grid(True)
            # plt.savefig(f'{dic}/Acq_simulated_points_{j}_{epsilon}.png')
            # plt.clf()

            if sample == 'IS':
                updated_probs, _ = UpdateParetoFront.importance_sampling(
                    epsilon,
                    accuracy_obs,
                    probs_sigmoid_params_copy,
                    grid_sigmoid_params_copy
                )


                # 4. Compute weighted utility
                updated_utility = AcquisitionFunctionsPL.utility_for_2particles(
                    probs_weights,
                    grid_weights,
                    updated_probs,
                    grid_sigmoid_params_copy,
                    epsilon_range,
                    utility_function,
                    curve)

            elif sample == 'MCMC':
                simulated_epsilons = np.append(evaluated_epsilons, epsilon)
                simulated_accuracies = np.append(evaluated_accuracies, accuracy_obs)
                fit = UpdateParetoFront_MCMC.build_stan_model_PF(
                    simulated_epsilons,
                    simulated_accuracies,
                    num_samples=10000,
                    num_chains=2)

                params_inferred, posterior_params = UpdateParetoFront_MCMC.params_update_mcmc(fit, 1000)

                # 4. Compute weighted utility
                updated_utility = AcquisitionFunctionsPL.utility_for_2particles(
                    probs_weights,
                    grid_weights,
                    probs_sigmoid_params_copy,
                    posterior_params,
                    epsilon_range,
                    utility_function,
                    curve)


            max_value = np.max(updated_utility) - np.max(current_utility)
            max_utility.append(max_value)

        expected_max_utility = np.mean(max_utility)
        return expected_max_utility

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
    def random(self):
        """
        Randomly select a point.
        """
        best_epsilon = np.random.uniform(0, 1)
        return best_epsilon, 0
