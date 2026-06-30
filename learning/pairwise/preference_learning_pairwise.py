import sys

import numpy as np

from ..preference_learning import *
import os
import random
print("Current working directory:", os.getcwd())
sys.path.append(os.path.abspath('.'))
from scipy.stats import norm

class UpdatePreference_pair:
    """
    Update the inference based on the new data using importance sampling.
    """
    def simulate_pair_action(
            T,
            utility_function,
            true_w,
            epsilon1,
            accuracy1,
            epsilon2,
            accuracy2):
        u1 = utility_function(epsilon1, accuracy1, true_w)
        u2 = utility_function(epsilon2, accuracy2, true_w)
        # probability of choosing the first point
        prob = np.exp(u1 / T) / (np.exp(u1 / T) + np.exp(u2 / T))
        threshold = np.random.uniform(0, 1)
        if threshold < prob:
            return epsilon1, accuracy1, epsilon2, accuracy2
        else:
            return epsilon2, accuracy2, epsilon1, accuracy1

    @staticmethod
    def IS_pair(
            point1,
            point2,
            probs_weights,
            grid_weights,
            T,
            utility_function):
        """
        Update the preference inference for a new observation

        Args:
            point1, point2: two points shown to the user, and the choice is point1, which means the first one is better
            probs_weights: the probability of each weight, which will be updated
            grid_weights: all the possible weights, which remain fixed during updating
            T: temperature in Boltzmann probability

        """

        for i, w in enumerate(grid_weights):
            u1 = utility_function(point1[0], point1[1], w)
            u2 = utility_function(point2[0], point2[1], w)
            likelihood = np.exp(u1 /T) / (np.exp(u1 /T) + np.exp(u2 /T))
            probs_weights[i] *= likelihood

        probs_weights = probs_weights / np.sum(probs_weights)

        # compute the effective sample size
        ess = AcquisitionFunctionsPL.compute_ess(probs_weights)
        if ess < len(grid_weights) / 5:  # threshold can be adjusted
            print(f"Warning: Low ESS ({ess}) for point1 {point1}, point2 {point2}")

        return probs_weights

    @staticmethod
    def preference_update_pair(
            point1,
            point2,
            probs_weights,
            grid_weights,
            true_w,
            T,
            utility_function):
        """
        Update the inferred preference based on the Importance Sampling.

        Returns:
            probs_we    sights: updated probability
            error_ozaki: the expected weighted error with updated probability
        """

        probs_weights = UpdatePreference_pair.IS_pair(
            point1,
            point2,
            probs_weights,
            grid_weights,
            T,
            utility_function)

        error_ozaki = np.sum(probs_weights * np.linalg.norm(grid_weights - true_w, axis=1))

        return probs_weights, error_ozaki

    @staticmethod
    def pair_KG_IS(particles_pair,
              probs_weights,
              grid_weights,
              probs_sigmoid_params,
              grid_sigmoid_params,
              epsilon_range,
              T,
              utility_function,
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
            utility_function)
        current_utility = np.array(current_utility)

        # Store original probs_sigmoid_params for verification
        original_probs = probs_weights.copy()
        results = Parallel(n_jobs=n_jobs)(
            delayed(UpdatePreference_pair.compute_optimal_utility_for_single_action)(
                pair,
                probs_weights.copy(),
                grid_weights.copy(),
                probs_sigmoid_params.copy(),
                grid_sigmoid_params.copy(),
                epsilon_range,
                T,
                utility_function,
                current_utility
            )
            for pair in particles_pair
        )

        # Verify probs_sigmoid_params hasn't changed
        assert np.allclose(original_probs,
                           probs_weights), "probs_weights was modified during parallel execution"

        kg_values = results

        # best_idx = AcquisitionFunctionsPL.random_argmax(kg_values)
        best_idx = np.argmax(kg_values)
        best_pair = particles_pair[best_idx]
        best_acq = kg_values[best_idx]

        return best_pair, best_acq

    @staticmethod
    def compute_optimal_utility_for_single_action(
            pair,
            probs_weights,
            grid_weights,
            probs_sigmoid_params,
            grid_sigmoid_params,
            epsilon_range,
            T,
            utility_function,
            current_utility):
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

        assert np.allclose(probs_weights_copy, probs_weights), "probs_weights was modified during parallel execution"

        # Simulate a possible action, sample a pair of weights first
        rand_num = np.random.choice(len(probs_weights), p=probs_weights_copy / np.sum(probs_weights_copy))
        # rand_num = np.random.choice(len(probs_weights))
        w_sample = grid_weights[rand_num]
        epsilon1, accuracy1, epsilon2, accuracy2 = UpdatePreference_pair.simulate_pair_action(
            T,
            utility_function,
            w_sample,
            epsilon1=pair[0],
            accuracy1=pair[1],
            epsilon2=pair[2],
            accuracy2=pair[3]
        )
        point1 = np.array([epsilon1, accuracy1])
        point2 = np.array([epsilon2, accuracy2])

        updated_probs_weights = UpdatePreference_pair.IS_pair(
            point1,
            point2,
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
            utility_function)
        updated_utility = np.array(updated_utility)
        max_value = np.max(updated_utility) - np.max(current_utility)

        return max_value
    @staticmethod
    def bald(
            pairs,
            probs_weights,
            grid_weights,
            T,
            utility_function
            ):
        best_acquisition_value = -np.inf

        for i in range(20):
            print('len(pairs):', len(pairs))
            values = random.sample(range(len(pairs)), 2)
            pair = pairs[values, 0:2]
            grid_probs_actions = [UpdatePreference_pair.action_probs(pair[0, 0], pair[0, 1], pair[1, 0], pair[1, 1], w, T, utility_function) for w in grid_weights]
            margi_prob_action = np.sum(grid_probs_actions * probs_weights[:, np.newaxis], axis=0)

            pred_entropy = UpdatePreference_pair.calculate_entropy(margi_prob_action)

            cond_entropy = [UpdatePreference_pair.calculate_entropy(UpdatePreference_pair.action_probs(pair[0, 0], pair[0, 1], pair[1, 0], pair[1, 1], w, T, utility_function)) for w in grid_weights]
            exp_entropy = np.sum(cond_entropy * probs_weights, axis=0)

            acquisition_value = pred_entropy - exp_entropy

            if acquisition_value > best_acquisition_value:
                best_acquisition_value = acquisition_value
                best_pair = pair

        return best_pair

    @staticmethod
    def epig(
            pairs,
            probs_weights,
            grid_weights,
            T,
            utility_function
    ):
        """
        Calculates EPIG for the given pairs, weights, and utility function.

        Args:
            pairs: A numpy array of pairs of solutions.
            probs_weights: A numpy array of probabilities for each weight in grid_weights.
            grid_weights: A numpy array of weights.
            T: Temperature parameter.
            utility_function: The utility function.

        Returns:
            The pair with the highest EPIG value.
        """

        best_epig_value = -np.inf
        best_pair = None

        for i in range(20):
            print('len(pairs):', len(pairs))
            values = random.sample(range(len(pairs)), 2)
            pair = pairs[values, 0:2]

            # Calculate predictive distributions (action probabilities) for each weight
            grid_probs_actions = [np.array(UpdatePreference_pair.action_probs(pair[0, 0], pair[0, 1], pair[1, 0], pair[1, 1], w, T, utility_function)) for w in grid_weights] # convert to numpy array.


            # Calculate the entropy of the average predictive distribution
            entropy_before = UpdatePreference_pair.calculate_entropy(
                np.sum(grid_probs_actions * probs_weights[:, np.newaxis], axis=0))

            # Calculate the expected entropy after adding the candidate pair (simplified average)
            expected_entropy_after = np.mean(
                [UpdatePreference_pair.calculate_entropy((grid_probs_actions[j] + grid_probs_actions[k]) / 2) for j in
                 range(len(grid_probs_actions)) for k in range(len(grid_probs_actions))], axis=0)

            # Calculate EPIG
            epig_value = entropy_before - expected_entropy_after

            if epig_value > best_epig_value:
                best_epig_value = epig_value
                best_pair = pair

        return best_pair

    @staticmethod
    def qeubo(
            pairs,
            probs_weights,
            grid_weights,
            T,
            utility_function,
            q=4
    ):
        best_acquisition_value = -np.inf

        num = 50 # number of random q points
        for i in range(num):
            values = random.sample(range(len(pairs)), q)
            pair = pairs[values, 0:2]
            acq = np.zeros(num)

            for j in range(q):
                grid_u = [utility_function(pair[j, 0], pair[j, 1], w) for w in grid_weights]
                mean_u = np.sum(grid_u * probs_weights)

                if mean_u > acq[i]:
                    acq[i] = mean_u

            # print('acquisition value:', acquisition_value)
            if acq[i] > best_acquisition_value:
                best_acquisition_value = acq[i]
                best_pair = pair

        return best_pair

    @staticmethod
    def calculate_entropy(posterior_probs):
        posterior_probs = np.array(posterior_probs)
        return -np.sum(posterior_probs * np.log(posterior_probs + 1e-10))

    @staticmethod
    def action_probs(e1, a1, e2, a2, w, T, utility_function):
        u1 = utility_function(e1, a1, w)
        u2 = utility_function(e2, a2, w)
        # print('e1,a1:', [e1,a1])
        # print('u1:', u1)
        # print('u2:', u2)
        prob1 = np.exp(u1 / T) / (np.exp(u1 / T) + np.exp(u2 / T) + 1e-5)
        prob2 = 1 - prob1
        return prob1, prob2



