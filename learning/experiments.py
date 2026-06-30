import os
import sys
import itertools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from .pareto_front_learning import *
from .sigmoid_utils import SigmoidParameterHandler
import pickle
import datetime
import numpy as np


class Experiments:
    def __init__(self, T, num_iteration, num_repetition, mode, simulations, particles, noise, cost, curve):
        """Initialize experiment parameters"""
        self.T = T
        self.num_iteration = num_iteration
        self.num_repetition = num_repetition
        # self.acquisition = acquisition
        self.mode = mode
        self.noise = noise
        self.acquisition = 'kg-IS'
        self.sample = 'IS'
        self.num_range  = 100
        self.particles = particles # particles of sigmoid parameters. We always set the particles of weights as 100.
        self.simulations = simulations
        self.cost_pl = cost # cost of querying the user, set the default cost of evaluating the pareto front as 1.
        self.task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
        self.utility_function = UtilityFunctions.utility_min
        self.curve = curve
        print(f'Task ID: {self.task_id}')

        # Set random seed
        np.random.seed(int(self.task_id))
        true_w1 = np.random.uniform(0, 1)
        self.true_w = [true_w1, 1 - true_w1]

        grid_w1 = np.linspace(0, 1, 100)
        self.grid_weights = np.column_stack((grid_w1, 1-grid_w1))
        self.probs_weights = np.ones(100)/100

        # Generate true parameters for gompertz function
        self.a = np.random.uniform(1, 2)
        self.b = np.random.uniform(0.1, 5)
        self.c = np.random.uniform(0.1, 5)
        self.d = np.random.uniform(1, 1.5)
        self.sigma = self.noise

        # sample from prior distribution for gompertz parameters
        grid_a = np.random.uniform(1, 2, self.particles)
        grid_b = np.random.uniform(0.1, 5, self.particles)
        grid_c = np.random.uniform(0.1, 5, self.particles)
        grid_d = np.random.uniform(1, 1.5, self.particles)
        grid_sigma = np.random.uniform(0.1, 0.5, self.particles)
        self.grid_sigmoid_params = np.column_stack((grid_a, grid_b, grid_c, grid_d, grid_sigma))
        self.probs_sigmoid_params = np.ones(self.particles) / self.particles

        # Setup directories
        self.setup_directories()
        self.initialize_storage()

        # Initialize experiment data
        self.epsilon_range = np.linspace(0, 1, self.num_range)
        self.true_accuracy = SigmoidFunctions.sigmoid(self.curve, self.epsilon_range, a=self.a, b=self.b, c=self.c, d=self.d)
        Utility = self.utility_function(self.epsilon_range, self.true_accuracy, self.true_w)
        self.optimal_utility = np.max(Utility)
        self.optimal_epsilon = self.epsilon_range[np.argmax(Utility)]
        self.optimal_accuracy = self.true_accuracy[np.argmax(Utility)]

        # Initialize and run experiment


    def setup_directories(self):
        """Setup directories for storing experiment data"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = f'experiments_{self.mode}/num_iteration_{self.num_iteration}_num_repetition_{self.num_repetition}_particles_{self.particles}_simulations_{self.simulations}/{self.task_id}_{timestamp}'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    def initialize_storage(self):
        """Initialize storage for experiment data"""
        self.evaluated_epsilons = []
        self.evaluated_accuracies = []
        self.regrets = []
        self.errors_ozaki = []
        self.flags = []
        self.sigmoid_errors = []

    def run_experiment(self):
        """Run experiment"""
        for i in range(self.num_iteration+1):
            self.base_seed = int(self.task_id) * 1000 + i
            np.random.seed(self.base_seed)
            print('Iteration:', i)
            if i == 0:
                flag_pl = 0
                num_pl = 0
                num_bo = 0
                best_acq_value_pl = 0
                best_acq_value_bo = 0
                self.flags.append(flag_pl)  # Store the flag
                error_ozaki = np.sum(self.probs_weights * np.linalg.norm(self.grid_weights - self.true_w, axis=1)) # error of prior
                regret = Acq_update.calculate_regret(self)
                self.errors_ozaki.append(error_ozaki)
                self.regrets.append(regret)

            else:
                if self.mode ==  'sequential':
                    if i%2 == 1:
                        print('This step we choose Preference Learning.')
                        flag_pl = 1
                        num_pl += 1
                        self.flags.append(flag_pl)  # Store the flag
                        # pick the next curve to interact and update the weights
                        best_params, best_acq_value_pl = Acq_update.calculate_acq_pl(self, i)
                        if self.acquisition == 'kg-IS':
                            best_params = np.array([best_params])
                        print('best_params:', best_params)
                        self.probs_weights, error_ozaki = Acq_update.update_pl(self, best_params)
                    elif i%2 == 0:
                        print('This step we choose Pareto Front Learning.')
                        flag_pl = 0
                        num_bo += 1
                        self.flags.append(flag_pl)  # Store the flag
                        # pick the next point to evaluate and update the Pareto front
                        best_epsilon, best_acq_value_bo = Acq_update.calculate_acq_pfl(self, i)
                        self.probs_sigmoid_params, self.grid_sigmoid_params = Acq_update.update_pfl(self, best_epsilon)

                        Acq_update.plot_evaluated_points(self, i)

                    regret = Acq_update.calculate_regret(self)
                    self.errors_ozaki.append(error_ozaki)
                    self.regrets.append(regret)
                elif self.mode == 'interleaved':
                    # pick the next curve to interact and update the weights
                    best_params, best_acq_value_pl = Acq_update.calculate_acq_pl(self, i)
                    if self.acquisition == 'kg-IS':
                        best_params = np.array([best_params])

                    # pick the next point to evaluate and update the Pareto front
                    best_epsilon, best_acq_value_bo = Acq_update.calculate_acq_pfl(self, i)
                    if best_acq_value_pl/self.cost_pl > best_acq_value_bo:
                        print('This step we choose Preference Learning.')
                        flag_pl = 1
                        num_pl += 1
                        self.flags.append(flag_pl)  # Store the flag
                        self.probs_weights, error_ozaki = Acq_update.update_pl(self, best_params)
                    else:
                        print('This step we choose Pareto Front Learning.')
                        flag_pl = 0
                        num_bo += 1
                        self.flags.append(flag_pl)  # Store the flag

                        self.probs_sigmoid_params, self.grid_sigmoid_params = Acq_update.update_pfl(self, best_epsilon)

                        Acq_update.plot_evaluated_points(self, i)

                        labels = ('a', 'b', 'c', 'd')
                        # plot distribution of particles
                        for j in range(4):
                            sorted_indices = np.argsort(self.grid_sigmoid_params[:,j])
                            sorted_particles = self.grid_sigmoid_params[sorted_indices,j]
                            sorted_weights = self.probs_sigmoid_params[sorted_indices]

                            max_particle = sorted_particles[np.argmax(sorted_weights)]

                            # Plot
                            plt.figure(figsize=(8, 6))
                            # Create vertical lines from x-axis to each point
                            for x, y in zip(sorted_particles, sorted_weights):
                                plt.vlines(x=x, ymin=0, ymax=y, color='blue', linewidth=1)

                            # Add points on top of lines
                            plt.plot(sorted_particles, sorted_weights, 'o', color='blue', markersize=4)
                            plt.xlabel('Particles')
                            plt.ylabel('Weights')
                            plt.title(f'Distribution of Particle Weights_{labels[j]}_{i}_{max_particle}')
                            plt.grid(True)
                            plt.savefig(f'{self.data_dir}/Distribution of Particle Weights_{labels[j]}_{i}')
                            plt.clf()

                    regret = Acq_update.calculate_regret(self)
                    self.errors_ozaki.append(error_ozaki)
                    self.regrets.append(regret)

            # check for duplicates
            unique_values, counts = np.unique(self.evaluated_epsilons, return_counts=True)

            # check how many values that have a count greater than 1 (i.e., duplicates)
            duplicates = len(unique_values[counts > 1])
            duplicates_sum = sum(counts) - len(unique_values)

            print('------------------------------------')
            print(f'Evaluated epsilons: {self.evaluated_epsilons}')
            print(f'duplicates_sum:', duplicates_sum)
            print(f'prop_pl:', num_pl / max(num_pl + num_bo, 1))
            print('preference error', self.errors_ozaki)
            print('regret:', self.regrets)
            print('------------------------------------')

        # Save the data
        output_data = {
            'true_w': self.true_w,
            'errors_PL': self.errors_ozaki,
            'regrets': self.regrets,
            'flag_user': self.flags
        }

        with open(f'{self.data_dir}/output_data_{self.task_id}.pkl', 'wb') as f:
            pickle.dump(output_data, f)

        # Acq_update.plot_metric(self, self.errors_ozaki, 'preference inference error')
        # Acq_update.plot_metric(self, self.flags, 'flag of querying users')
        # Acq_update.plot_metric(self, self.regrets, 'regrets')


class Acq_update:
    def calculate_acq_pl(self, i):
        """
        pick a new curve based on different acquisition functions
        """
        np.random.seed(self.base_seed)
        n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))
        probs_w = np.array(self.probs_weights, copy=True)  # Deep copy
        grid_w = np.array(self.grid_weights, copy=True)
        probs_params = np.array(self.probs_sigmoid_params, copy=True)
        grid_params = np.array(self.grid_sigmoid_params, copy=True)

        if self.acquisition == 'random':
            best_params, best_acquisition_value = AcquisitionFunctionsPL.random(self)

        elif self.acquisition == 'kg-IS':
            # particles_h = self.grid_sigmoid_params[:,0:3]
            num_params = self.grid_sigmoid_params.shape[1] - 1  # -1 because last column is sigma
            param_samples = [self.grid_sigmoid_params[:, i] for i in range(num_params)]
            particles_h = np.column_stack(param_samples)

            best_params, best_acquisition_value = AcquisitionFunctionsPL.kg_IS(
                particles_h,
                probs_w,
                grid_w,
                probs_params,
                grid_params,
                self.epsilon_range,
                self.T,
                self.utility_function,
                i,
                self.curve,
                N_actions=1,
                n_jobs=max(1, n_cores - 1)
            )

        return best_params, best_acquisition_value

    def update_pl(self, best_params):
        """
        Collect the user action,
        update the weights
        """
        sigmoid_functions = SigmoidFunctions.generate_sigmoid_functions(self.curve, self.epsilon_range, best_params)
        action = UserActionSimulator.simulate_user_action(
            self.epsilon_range,
            sigmoid_functions,
            self.true_w,
            0.2, # this is only used in sensitivity analysis
            self.utility_function)

        best_params = best_params[0]
        action = action[0]
        print('action:', action)
        probs_weights, error_ozaki = UpdatePreference.preference_update(
            best_params,
            action,
            self.epsilon_range,
            self.probs_weights,
            self.grid_weights,
            self.true_w,
            self.T,
            self.utility_function)


        return probs_weights, error_ozaki

    def calculate_acq_pfl(self, i):
        """
        Pick a new point to evaluate based on different acquisition functions
        """
        if self.acquisition == 'random':
            best_epsilon, best_acq_value = AcquisitionFunctionsBO.random(self)

        elif self.acquisition == 'kg-IS':
            probs_w = np.array(self.probs_weights, copy=True) # Deep copy
            grid_w = np.array(self.grid_weights, copy=True)
            probs_params = np.array(self.probs_sigmoid_params, copy=True)
            grid_params = np.array(self.grid_sigmoid_params, copy=True)
            n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))
            best_epsilon, best_acq_value = AcquisitionFunctionsBO.kg_IS(
                self.epsilon_range,
                probs_w,
                grid_w,
                probs_params,
                grid_params,
                self.epsilon_range,
                self.utility_function,
                self.data_dir,
                i,
                self.evaluated_epsilons,
                self.evaluated_accuracies,
                self.curve,
                N_actions=self.simulations,
                n_jobs=max(1, n_cores - 1)
            )

        return best_epsilon, best_acq_value

    def update_pfl(self, best_epsilon):
        """
        Collect evaluated accuracy,
        update the inference of Pareto front
        """
        np.random.seed(self.base_seed)
        next_epsilon = np.array([best_epsilon])
        next_accuracy = SigmoidFunctions.sigmoid('gompertz', next_epsilon, a=self.a, b=self.b, c=self.c, d=self.d) + np.random.normal(0, self.noise,
                                                                                                          1)
        next_accuracy = np.clip(next_accuracy, 0, 1)
        self.evaluated_epsilons.append(next_epsilon[0])
        self.evaluated_accuracies.append(next_accuracy[0])

        if self.sample == 'IS':
            probs_sigmoid_params, grid_sigmoid_params = UpdateParetoFront.params_update(
                next_epsilon,
                next_accuracy,
                self.probs_sigmoid_params,
                self.grid_sigmoid_params)

            return probs_sigmoid_params, grid_sigmoid_params

        elif self.sample == 'MCMC':
            fit = UpdateParetoFront_MCMC.build_stan_model_PF(
                self.evaluated_epsilons,
                self.evaluated_accuracies,
                seed=self.base_seed,
                num_samples=self.bo_mcmc_samples,
                num_chains=2)

            params_inferred, posterior_params = UpdateParetoFront_MCMC.params_update_mcmc(fit, self.particles)

            return params_inferred, posterior_params

    def calculate_regret(self):
        """
        Calculate the regret
        """
        # choose from all points including non-observed points
        particles_accuracies = SigmoidParameterHandler.compute_particles_accuracies(
            self.grid_sigmoid_params, self.epsilon_range, SigmoidFunctions.sigmoid, self.curve)
        self.estimated_accuracies = np.sum(particles_accuracies * self.probs_sigmoid_params[:, np.newaxis], axis=0)

        self.particles_utilities = [self.utility_function(self.epsilon_range, self.estimated_accuracies, w) for w in self.grid_weights]
        self.estimated_utilities = np.sum(self.particles_utilities * self. probs_weights[:, np.newaxis], axis=0)
        self.epsilon_max = self.epsilon_range[np.argmax(self.estimated_utilities)]
        accuracy_max = SigmoidFunctions.sigmoid('gompertz', self.epsilon_max, a=self.a, b=self.b, c=self.c, d=self.d)

        regret = self.optimal_utility - self.utility_function(self.epsilon_max, accuracy_max, self.true_w)

        return regret

    def count_switches(flags):
        switches = 0
        # Start from second element since we need to compare with previous
        for i in range(1, len(flags)):
            if flags[i] != flags[i - 1]:
                switches += 1
        return switches

    def plot_weights_particles(self, i):
        # plot the probability distribution of particles
        true_params = (self.a, self.b, self.c, self.d)
        labels = ('a', 'b', 'c', 'd')
        # plot distribution of particles
        for j in range(4):
            sorted_indices = np.argsort(self.grid_sigmoid_params[:, j])
            sorted_particles = self.grid_sigmoid_params[sorted_indices, j]
            sorted_weights = self.probs_sigmoid_params[sorted_indices]

            max_particle = sorted_particles[np.argmax(sorted_weights)]

            # Plot
            plt.figure(figsize=(8, 6))
            # Create vertical lines from x-axis to each point
            for x, y in zip(sorted_particles, sorted_weights):
                plt.vlines(x=x, ymin=0, ymax=y, color='blue', linewidth=1)

            # Add points on top of lines
            plt.plot(sorted_particles, sorted_weights, 'o', color='blue', markersize=4)
            plt.xlabel('Particles')
            plt.ylabel('Weights')
            plt.title(f'Distribution of Particle Weights_{labels[j]}_{i}_{max_particle}: {true_params[j]}')
            plt.grid(True)
            plt.savefig(f'{self.data_dir}/Distribution of Particle Weights_{labels[j]}_{i}')
            plt.clf()

    def plot_evaluated_points(self, i):
        plt.figure(figsize=(8, 6))
        plt.plot(self.epsilon_range, self.true_accuracy, color='black')
        plt.scatter(self.evaluated_epsilons, self.evaluated_accuracies, color='red')
        plt.scatter(self.optimal_epsilon, SigmoidFunctions.sigmoid('gompertz', self.optimal_epsilon, a=self.a, b=self.b, c=self.c, d=self.d),
                    marker='*')
        plt.xlabel('Epsilons')
        plt.ylabel('Accuracies')
        plt.title(f'Evaluated points_{i}')
        plt.grid(True)
        plt.savefig(f'{self.data_dir}/Evaluated points_{i}')
        plt.clf()


def main():
    """Main execution function"""
    # Parse command line arguments
    T = float(sys.argv[1])  # temperature
    num_iteration = int(sys.argv[2])  # number of iterations in each run
    num_repetition = int(sys.argv[3])  # number of repeated runs, set the same as array
    mode = str(sys.argv[4])  # 'simultaneous' or 'interleaved'
    simulations = int(sys.argv[5]) # number of simulations
    particles = int(sys.argv[6]) # particles in sigmoid update
    noise = float(sys.argv[7]) # noise level
    cost = float(sys.argv[8]) # cost of querying the user

    experiment = Experiments(T, num_iteration, num_repetition, mode, simulations, particles, noise, cost)
    experiment.run_experiment()


if __name__ == "__main__":
    main() 
