import os
import sys
from pathlib import Path
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch

PACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACE_ROOT))
from learning.pareto_front_learning import *
from learning.experiments import Acq_update
from learning.pairwise.separate_preference import pair_pl_update
from learning.sigmoid_utils import SigmoidParameterHandler
import wandb
import pickle
import datetime
import numpy as np
from tueplots import axes,bundles,figsizes, fontsizes
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class dp_exp:
    def __init__(self, T, num_iteration, num_repetition, mode, particles, cost, dataset, interaction, curve):
        """Initialize experiment parameters"""
        self.T = T
        self.num_iteration = num_iteration # total interation steps for interaction and evluation
        self.num_repetition = num_repetition # run the experiment in parallel, every experiment runs with a different preference
        self.mode = mode
        self.acquisition = 'kg-IS'
        self.sample = 'IS'
        self.particles = particles #
        self.simulations = 1
        self.dataset = dataset
        self.interaction = interaction
        self.curve = curve
        self.cost_pl = cost # cost of querying the user, set the default cost of evaluating the pareto front as 1.
        self.task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
        self.utility_function = UtilityFunctions.utility_min
        print(f'Task ID: {self.task_id}')
        print(f'Dataset: {self.dataset}')
        print(f'Interaction: {self.interaction}')
        print(f'Mode: {self.mode}')
        print(f'Particles: {self.particles}')
        print(f'Cost: {self.cost_pl}')
        print(f'T: {self.T}')
        print(f'Curve: {self.curve}')
        print(f'Num Iteration: {self.num_iteration}')

        num_q = 0

        # Set random seeds for reproducibility across different environments
        seed = int(self.task_id)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        

        # Sample from the Dirichlet distribution
        alpha = [2, 2]
        sample = np.random.dirichlet(alpha)
        # Extract w_1 and w_2
        self.true_w = sample
        print('true_w:', self.true_w)
        w_1, w_2 = sample

        grid_w1 = np.linspace(0.01, 0.99, 100)
        self.grid_weights = np.column_stack((grid_w1, 1-grid_w1))
        self.probs_weights = np.ones(100)/100

        if self.curve == 'sigmoid':
            # sample from prior distribution for sigmoid parameters
            grid_L = np.random.beta(40, 2, self.particles)
            grid_k = np.random.lognormal(np.log(10), 0.2, self.particles)
            grid_c = np.random.beta(2, 2, self.particles)
            grid_sigma = np.random.gamma(0.5, 0.1, self.particles)
            self.grid_sigmoid_params = np.column_stack((grid_L, grid_k, grid_c, grid_sigma))
        elif self.curve == 'gompertz':
            # # sample from prior distribution for gompertz parameters
            grid_a = np.random.uniform(0.8, 4, self.particles)
            grid_b = np.random.uniform(10, 100, self.particles)
            grid_c = np.random.uniform(1, 10, self.particles)
            grid_d = np.random.uniform(0.8, 1.1, self.particles)
            grid_sigma = np.random.gamma(0.5, 0.1, self.particles)
            self.grid_sigmoid_params = np.column_stack((grid_a, grid_b, grid_c, grid_d, grid_sigma))

        self.probs_sigmoid_params = np.ones(self.particles) / self.particles

        # Setup directories
        self.setup_directories()
        self.initialize_storage()


        # set epsilon range and transformation to [0,1]
        if self.dataset == 'adult':
            df = pd.read_csv(PACE_ROOT / 'data' / 'raw' / 'optimal_results_adult_N_TRIALS_20.csv')
            log_epsilons = np.log(df['epsilon'])
            num_q = len(log_epsilons)
            log_epsilons = np.array(log_epsilons)
            range = (log_epsilons - min(log_epsilons)) / (max(log_epsilons) - min(log_epsilons))
            self.epsilon_range = 1- range
            normalized_accuracy = (df['best_test_accuracy'] - min(df['best_test_accuracy'])) / (max(df['best_test_accuracy']) - min(df['best_test_accuracy']))
            # fit smooth GP model 
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=10)
            # Convert pandas Series to numpy arrays
            epsilon_range_np = np.array(self.epsilon_range)
            accuracy_range_np = np.array(normalized_accuracy)
            gp.fit(epsilon_range_np.reshape(-1, 1), accuracy_range_np)
            self.gp_accuracy = gp.predict(self.epsilon_range.reshape(-1, 1))

            self.true_accuracy = np.array(normalized_accuracy)
            self.All_true_Utility = self.utility_function(self.epsilon_range, self.gp_accuracy, self.true_w)
            self.optimal_utility = np.max(self.All_true_Utility)
            self.optimal_epsilon = self.epsilon_range[np.argmax(self.All_true_Utility)]
            self.optimal_accuracy = self.gp_accuracy[np.argmax(self.All_true_Utility)]


        elif self.dataset == 'dutch':
            df = pd.read_csv(PACE_ROOT / 'data' / 'raw' / 'optimal_results_dutch_N_TRIALS_20.csv')
            log_epsilons = np.log(df['epsilon'])
            num_q = len(log_epsilons)
            log_epsilons = np.array(log_epsilons)
            range = (log_epsilons - min(log_epsilons)) / (max(log_epsilons) - min(log_epsilons))
            self.epsilon_range = 1- range
            normalized_accuracy = (df['best_test_accuracy'] - min(df['best_test_accuracy'])) / (max(df['best_test_accuracy']) - min(df['best_test_accuracy']))
            # fit smooth GP model
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=10)
            # Convert pandas Series to numpy arrays
            epsilon_range_np = np.array(self.epsilon_range)
            accuracy_range_np = np.array(normalized_accuracy)
            gp.fit(epsilon_range_np.reshape(-1, 1), accuracy_range_np)
            self.gp_accuracy = gp.predict(self.epsilon_range.reshape(-1, 1))

            self.true_accuracy = np.array(normalized_accuracy)
            self.All_true_Utility = self.utility_function(self.epsilon_range, self.gp_accuracy, self.true_w)
            self.optimal_utility = np.max(self.All_true_Utility)
            self.optimal_epsilon = self.epsilon_range[np.argmax(self.All_true_Utility)]
            self.optimal_accuracy = self.gp_accuracy[np.argmax(self.All_true_Utility)]
           
        else:
            df = pd.read_csv(PACE_ROOT / 'data' / 'raw' / 'epsilon-accuracy.csv')
            data = df[df['dataset_name'] == self.dataset]
            # data = df[df['dataset_name'] == 'cifar100']
            self.num_range = data.shape[0]  # how many epsilons are used to discretize sigmoid
            num_q = self.num_range

            epsilon_min = min(data['target_epsilon'])
            epsilon_max = max(data['target_epsilon'])
            epsilons = data['target_epsilon']
            self.epsilons = np.array(epsilons)
            log_epsilons = np.log(self.epsilons)
            # transformation
            range = (log_epsilons - min(log_epsilons)) / (max(log_epsilons) - min(log_epsilons))
            self.epsilon_range = 1 - range
            true_accuracy = data['accuracy']

            normalized_accuracy = (true_accuracy - min(true_accuracy)) / (max(true_accuracy) - min(true_accuracy))
            if self.dataset == 'dpdl-benchmark/patch_camelyon':
                # fit smooth GP model
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=10)
                # Convert pandas Series to numpy arrays
                epsilon_range_np = np.array(self.epsilon_range)
                accuracy_range_np = np.array(normalized_accuracy)
                gp.fit(epsilon_range_np.reshape(-1, 1), accuracy_range_np)
                self.gp_accuracy = gp.predict(self.epsilon_range.reshape(-1, 1))

                self.true_accuracy = np.array(normalized_accuracy)
                self.All_true_Utility = self.utility_function(self.epsilon_range, self.gp_accuracy, self.true_w)
                self.optimal_utility = np.max(self.All_true_Utility)
                self.optimal_epsilon = self.epsilon_range[np.argmax(self.All_true_Utility)]
                self.optimal_accuracy = self.gp_accuracy[np.argmax(self.All_true_Utility)]
        
            else:
                self.gp_accuracy = np.array(normalized_accuracy)
                self.true_accuracy = np.array(normalized_accuracy)
                self.All_true_Utility = self.utility_function(self.epsilon_range, self.true_accuracy, self.true_w)
                self.optimal_utility = np.max(self.All_true_Utility)
                self.optimal_epsilon = self.epsilon_range[np.argmax(self.All_true_Utility)]
                self.optimal_accuracy = self.true_accuracy[np.argmax(self.All_true_Utility)]
        # self.optimal_epsilon_orig = self.epsilons[np.argmax(Utility)]

        # Initialize and run experiment
        if os.getenv('WANDB_MODE') not in {'disabled', 'offline'}:
            wandb.login()
        run = wandb.init(
            project='DP_ablation_q',
            entity='yaohong-yang-aalto-university',
            # name=f"experiment_{task_id}",
            config={
                'mode': self.mode,
                'flag': 'IS',
                'particles': self.particles,
                'simulations': self.simulations,
                'cost_pl': self.cost_pl,
                'T': self.T,
                'task_id': self.task_id,
                'utility': self.utility_function,
                'acquisition': self.acquisition,
                # 'epsilon_min': epsilon_min,
                # 'epsilon_max': epsilon_max,
                'dataset': self.dataset,
                'interaction': self.interaction,
                'true_w': w_1,
                'sigmoid_type':self.curve,
                'num_q': num_q


            },
            resume="never")

    def setup_directories(self):
        """Setup directories for storing experiment data"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = f'results_DP/{self.dataset}/{self.mode}_num_iteration_{self.num_iteration}_num_repetition_{self.num_repetition}_particles_{self.particles}_curve_{self.curve}/{self.task_id}/{timestamp}'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        print(f'Results directory: {Path(self.data_dir).resolve()}')


    def initialize_storage(self):
        """Initialize storage for experiment data"""
        self.evaluated_epsilons = []
        self.evaluated_accuracies = []
        self.evaluated_utilities = []
        self.utilities = []
        self.regrets = []
        self.regrets_evaluated = []
        self.errors_ozaki = []
        self.flags = []
        self.all_util = []


    def run_experiment(self):
        """Run experiment"""
        for i in range(self.num_iteration+1):
            self.base_seed = int(self.task_id) * 1000 + i
            # Set seeds for each iteration to ensure reproducibility
            np.random.seed(self.base_seed)
            random.seed(self.base_seed)
            torch.manual_seed(self.base_seed)
            print('Iteration:', i)
            if i == 0:
                flag_pl = 0
                num_pl = 0
                num_bo = 0
                best_acq_value_pl = 0
                best_acq_value_bo = 0
                self.flags.append(flag_pl)  # Store the flag
                error_ozaki = np.sum(self.probs_weights * np.linalg.norm(self.grid_weights - self.true_w, axis=1)) # error of prior
                utility, regret, regret_evaluated = dp_update.calculate_utility(self, i)
                self.errors_ozaki.append(error_ozaki)
                self.utilities.append(utility)
                regret = self.optimal_utility
                regret_evaluated = self.optimal_utility
                self.regrets.append(regret)
                self.regrets_evaluated.append(regret_evaluated)

            else:
                if self.mode == 'sequential':
                    if i%2 == 1:
                        print('This step we choose Preference Learning.')
                        flag_pl = 1
                        num_pl += 1
                        self.flags.append(flag_pl)  
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
                        self.probs_sigmoid_params, self.grid_sigmoid_params = dp_update.update_pfl(self, best_epsilon)
                        error_ozaki = self.errors_ozaki[-1]

                        if self.curve == 'gompertz':
                            labels = ('a', 'b', 'c', 'd')
                            # plot distribution of particles
                            for j in range(4):
                                sorted_indices = np.argsort(self.grid_sigmoid_params[:, j])
                                sorted_particles = self.grid_sigmoid_params[sorted_indices, j]
                                sorted_weights = self.probs_sigmoid_params[sorted_indices]

                                max_particle = sorted_particles[np.argmax(sorted_weights)]

                                plt.figure(figsize=(8, 6))
                                for x, y in zip(sorted_particles, sorted_weights):
                                    plt.vlines(x=x, ymin=0, ymax=y, color='blue', linewidth=1)

                                plt.plot(sorted_particles, sorted_weights, 'o', color='blue', markersize=4)
                                plt.xlabel('Particles')
                                plt.ylabel('Weights')
                                plt.title(f'Distribution of Particle Weights_{labels[j]}_{i}_{max_particle}')
                                plt.grid(True)
                                plt.savefig(f'{self.data_dir}/Distribution of Particle Weights_{labels[j]}_{i}')
                                plt.clf()

                    utility, regret, regret_evaluated = dp_update.calculate_utility(self, i)
                    self.errors_ozaki.append(error_ozaki)
                    self.utilities.append(utility)
                    self.regrets.append(regret)
                    self.regrets_evaluated.append(regret_evaluated)

                    if i % 2 == 0:
                        dp_update.plot_evaluated_points(self, i)

                elif self.mode == 'interleaved':
                    # pick the next curve to interact and update the weights
                    if self.interaction == 'curve':
                        best_params, best_acq_value_pl = Acq_update.calculate_acq_pl(self, i)
                        best_params = np.array([best_params])

                    elif self.interaction == 'pair':
                        best_pair, best_acq_value_pl = pair_pl_update.cal_pair_acq(self)

                    elif self.interaction == 'true':
                        self.grid_weights = [self.true_w]
                        self.probs_weights = np.ones(1)
                        best_acq_value_pl = -np.inf

                    # pick the next point to evaluate and update the Pareto front
                    best_epsilon, best_acq_value_bo = Acq_update.calculate_acq_pfl(self, i)
                    if best_acq_value_pl/self.cost_pl > best_acq_value_bo:
                        print('This step we choose Preference Learning.')
                        flag_pl = 1
                        num_pl += 1
                        self.flags.append(flag_pl) 
                        if self.interaction == 'curve':
                            self.probs_weights, error_ozaki = Acq_update.update_pl(self, best_params)
                        elif self.interaction == 'pair':
                            error_ozaki = pair_pl_update.pair_update_error(self, best_pair[0], best_pair[1],
                                                                           best_pair[2], best_pair[3])
                    else:
                        print('This step we choose Pareto Front Learning.')
                        flag_pl = 0
                        num_bo += 1
                        self.flags.append(flag_pl)  # Store the flag

                        self.probs_sigmoid_params, self.grid_sigmoid_params = dp_update.update_pfl(self, best_epsilon)

                        dp_update.plot_evaluated_points(self, i)

                        labels = ('L', 'k', 'c')
                        # plot distribution of particles
                        for j in range(3):
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

                    utility, regret, regret_evaluated = dp_update.calculate_utility(self, i)
                    self.errors_ozaki.append(error_ozaki)
                    self.utilities.append(utility)
                    self.regrets.append(regret)
                    self.regrets_evaluated.append(regret_evaluated)

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
            print('regret_evaluated:', self.regrets_evaluated)
            print('utility:',self.utilities)
            print('------------------------------------')

            wandb.log({
                f'error_PL': error_ozaki,
                f'utility': utility,
                f'regret': regret,
                f'regret_observed': regret_evaluated,
                f'acq_value_pl': best_acq_value_pl,
                f'acq_value_bo': best_acq_value_bo,
                # f'flag_pl': flag_pl,
                # f'prop_pl': num_pl / max(num_pl + num_bo, 1),
                # f'switch': Acq_update.count_switches(self.flags),
                # f'duplicates': duplicates,
                # f'duplicates_sum': duplicates_sum,
            })

        # Save the data
        output_data = {
            'true_w': self.true_w,
            'errors_PL': self.errors_ozaki,
            'utilities': self.utilities,
            'regrets': self.regrets,
            'flag_user': self.flags
        }

        with open(f'{self.data_dir}/output_data_{self.task_id}.pkl', 'wb') as f:
            pickle.dump(output_data, f)

class dp_update:
    def _nearest_epsilon_index(self, epsilon):
        epsilon = float(np.asarray(epsilon).reshape(-1)[0])
        return int(np.argmin(np.abs(self.epsilon_range - epsilon)))

    def _scalarize(self, value):
        return float(np.asarray(value).reshape(-1)[0])

    def update_pfl(self, best_epsilon):
        """
        Collect evaluated accuracy,
        update the inference of Pareto front
        """
        # Set seeds for reproducibility
        np.random.seed(self.base_seed)
        random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)
        next_epsilon = float(np.asarray(best_epsilon).reshape(-1)[0])

        # recover the epsilon value
        idx = dp_update._nearest_epsilon_index(self, next_epsilon)
        # recover_epsilon = self.epsilons[self.num_range-1-idx]
        # recover_epsilon = self.epsilons[idx]
        # chage this sigmoid function to DP model
        next_accuracy = dp_update._scalarize(self, self.true_accuracy[idx])
        self.evaluated_epsilons.append(next_epsilon)
        self.evaluated_accuracies.append(next_accuracy)

        probs_sigmoid_params, grid_sigmoid_params = UpdateParetoFront.params_update(
            next_epsilon,
            next_accuracy,
            self.probs_sigmoid_params,
            self.grid_sigmoid_params)

        return probs_sigmoid_params, grid_sigmoid_params

    def calculate_utility(self, i):
        """
        Calculate the optimal utility so far
        """
        # choose from all points including non-observed points
        # particles_accuracies = [SigmoidFunctions.sigmoid(self.epsilon_range, L, k, c) for L, k, c in
        #                         self.grid_sigmoid_params[:, 0:3]]
        particles_accuracies = SigmoidParameterHandler.compute_particles_accuracies(
            self.grid_sigmoid_params, self.epsilon_range, SigmoidFunctions.sigmoid, self.curve)
        self.estimated_accuracies = np.sum(particles_accuracies * self.probs_sigmoid_params[:, np.newaxis], axis=0)
        self.std_accuracies = np.sum((particles_accuracies - self.estimated_accuracies)**2 * self.probs_sigmoid_params[:, np.newaxis], axis=0)

        self.particles_utilities = [self.utility_function(self.epsilon_range, self.estimated_accuracies, w) for w in self.grid_weights]
        self.estimated_utilities = np.sum(self.particles_utilities * self. probs_weights[:, np.newaxis], axis=0)
        self.epsilon_max = self.epsilon_range[np.argmax(self.estimated_utilities)]
        # recover epsilon
        idx = dp_update._nearest_epsilon_index(self, self.epsilon_max)
        # self.accuracy_max = self.true_accuracy[idx]
        self.accuracy_max = dp_update._scalarize(self, self.gp_accuracy[idx])

        utility = dp_update._scalarize(
            self,
            self.utility_function(self.epsilon_max, self.accuracy_max, self.true_w),
        )
        
        if i != 0 and i !=1:
            # test the observed utility so far
            self.estimated_w = np.sum(self.grid_weights * self.probs_weights[:, np.newaxis], axis=0)
            self.evaluated_utilities = self.utility_function(self.evaluated_epsilons, self.evaluated_accuracies, self.estimated_w)
            self.evaluated_utility_max = np.max(self.evaluated_utilities)
            self.evaluated_epsilon_max = self.evaluated_epsilons[np.argmax(self.evaluated_utilities)]
            # where self.epsilon_range == self.evaluated_epsilon_max and take that as the max estimated accuracy
            evaluated_idx = dp_update._nearest_epsilon_index(self, self.evaluated_epsilon_max)
            self.evaluated_accuracy_max = dp_update._scalarize(self, self.true_accuracy[evaluated_idx])
            evaluated_utility = dp_update._scalarize(
                self,
                self.utility_function(
                    self.evaluated_epsilon_max,
                    self.evaluated_accuracy_max,
                    self.true_w,
                )
            )
        else: 
            evaluated_utility = 0

 

        regret = float(self.optimal_utility - utility)
        # report the evaluted regret 
        regret_evaluated = float(self.optimal_utility - evaluated_utility)

        if i != 0 and i !=1:
            print('---------')
            # print('grid_weights:', self.grid_weights)
            # print('ture chosen utility:', utility)
            print('max estimated utility:', np.max(self.estimated_utilities))
            # print('ture chosen accuracy:', accuracy_max)
            print('optimal utility:', self.optimal_utility)
            print('chosed epsilon-kg:', self.epsilon_max)
            print('chosed epsilon-evaluated:', self.evaluated_epsilon_max)
            print('optimal epsilon:', self.optimal_epsilon)
            print('----------')

        return utility, regret, regret_evaluated

    def plot_metric(self, metric, title):
        """
        plot evaluation metric
        """
        plt.figure(figsize=(8,6))
        plt.plot(range(0, self.num_iteration+1), metric, marker = 'o', linestyle='-', color='blue')
        plt.xlabel('Steps')
        plt.title(f'{title}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.data_dir}/{title}.png')

  

    def plot_evaluated_points(self, i):
        if i > 2 and hasattr(self, 'evaluated_epsilon_max') and hasattr(self, 'evaluated_accuracy_max'):

          
            weighted_params, weighted_sigma = SigmoidParameterHandler.compute_weighted_parameters(
                self.grid_sigmoid_params, self.probs_sigmoid_params)

            # compute the sigmoid function
            # weighted_sigmoid = SigmoidFunctions.sigmoid(self.epsilon_range, weighted_L, weighted_k, weighted_c)
            # Determine parameter names based on curve type
            if self.curve == 'sigmoid':
                param_names = ['L', 'k', 'c']
            elif self.curve == 'gompertz':
                param_names = ['a', 'b', 'c', 'd']
            else:
                raise ValueError(f"Unknown curve type: {self.curve}")
            
            weighted_sigmoid = SigmoidParameterHandler.call_sigmoid_function(
                SigmoidFunctions.sigmoid, self.curve, self.epsilon_range, weighted_params, param_names)

            plt.rcParams.update(figsizes.icml2024_half(height_to_width_ratio=1))
            # plt.rcParams.update(fontsizes.icml2024())
            plt.rcParams['text.usetex'] = False
            plt.figure()
            plt.scatter(self.epsilon_range, self.true_accuracy, color='black', label='True Pareto Front', s=60, zorder=1)
            plt.plot(self.epsilon_range, self.estimated_accuracies, color="blue", label="Weighted Accuracy", linewidth=3,zorder=2)
            plt.plot(self.epsilon_range, weighted_sigmoid, color="green", label="Weighted Parameters", linewidth=3,zorder=3)
            plt.fill_between(self.epsilon_range, self.estimated_accuracies - self.std_accuracies, self.estimated_accuracies + self.std_accuracies, color='blue', alpha=0.3, label='95% CI')

            plt.scatter((self.evaluated_epsilons), self.evaluated_accuracies, color="red", label="Evaluated Data", s=100, zorder=3)
            plt.scatter((self.optimal_epsilon), self.optimal_accuracy, marker='*', label='Optimal Point', s=100, zorder=4)
            plt.scatter((self.epsilon_max), self.accuracy_max, marker='>', label='Optimal-KG', s=100, zorder=5)
            plt.scatter((self.evaluated_epsilon_max), self.evaluated_accuracy_max, marker='<', label='Evaluated Optimal', s=100, zorder=6)
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.legend(fontsize=8, handletextpad=0.5, labelspacing=0.3, borderpad=0.3)
            plt.title(f'{self.dataset} - step {i}')
            plt.savefig(f'{self.data_dir}/Estimated_Pareto_Front_sigmoid_{i}', dpi=900)
            plt.clf()

def main():
    """Main execution function"""
    # Parse command line arguments
    T = float(sys.argv[1])  # stochasticity
    num_iteration = int(sys.argv[2])  # number of iterations in each run
    num_repetition = int(sys.argv[3])  # number of repeated runs, set the same as array
    mode = str(sys.argv[4])  # 'sequential' or 'interleaved'
    particles = int(sys.argv[5]) # particles in sigmoid update
    cost = float(sys.argv[6]) # cost of querying the user
    dataset = str(sys.argv[7]) # dataset
    interaction = str(sys.argv[8]) # true / pair / curve
    curve = str(sys.argv[9]) # sigmoid / gompertz

    experiment = dp_exp(T, num_iteration, num_repetition, mode, particles, cost, dataset, interaction, curve)
    experiment.run_experiment()


if __name__ == "__main__":
    main()


