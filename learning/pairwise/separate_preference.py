import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
from ..preference_learning import AcquisitionFunctionsPL, UtilityFunctions, SigmoidFunctions, UserActionSimulator, UpdatePreference
from .preference_learning_pairwise import *
from ..experiments import Acq_update
import pickle
import numpy as np


class Experiments:
    def __init__(self, T, num_iteration, num_repetition, acquisition, interaction):
        """Initialize experiment parameters"""
        self.T = T
        self.trueT = 0.2
        self.num_iteration = num_iteration
        self.num_repetition = num_repetition
        self.acquisition = acquisition
        self.interaction = interaction
        self.num_range  = 100
        self.cost_pl = 1
        self.task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
        self.utility_function = UtilityFunctions.utility_min
        print(f'Task ID: {self.task_id}')

        # Set random seed
        np.random.seed(int(self.task_id))
        true_w1 = np.random.uniform(0, 1)
        self.true_w = [true_w1, 1 - true_w1]

        grid_w1 = np.linspace(0, 1, 100)
        self.grid_weights = np.column_stack((grid_w1, 1-grid_w1))
        self.probs_weights = np.ones(100)/100

        # Generate true parameters
        self.L = np.random.beta(40, 2)
        self.k = np.random.lognormal(np.log(15), 0.5)
        self.c = np.random.beta(2, 2)
        self.grid_sigmoid_params = np.column_stack((self.L, self.k, self.c))
        self.probs_sigmoid_params = np.ones(1)

        # generate particles
        grid_L = np.random.beta(40, 2, 1000)
        grid_k = np.random.lognormal(np.log(15), 0.5, 1000)
        grid_c = np.random.beta(2, 2, 1000)
        self.curve_particles = np.column_stack((grid_L, grid_k, grid_c))
        self.probs_particles = np.ones(1000) / 1000

        # generate points particles
        x1 = np.random.uniform(0,1,1000)
        y1 = np.random.uniform(0, 1, 1000)
        # y1 = SigmoidFunctions.sigmoid(x1, self.L, self.k, self.c)
        x2 = np.random.uniform(0, 1, 1000)
        y2 = np.random.uniform(0, 1, 1000)
        # y2 = SigmoidFunctions.sigmoid(x2, self.L, self.k, self.c)
        self.pair_particles = np.column_stack((x1, y1, x2, y2))

        # Setup directories
        self.setup_directories()
        self.initialize_storage()

        # Initialize experiment data
        self.epsilon_range = np.linspace(0, 1, self.num_range)
        true_accuracy = SigmoidFunctions.sigmoid(self.epsilon_range, self.L, self.k, self.c)
        Utility = self.utility_function(self.epsilon_range, true_accuracy, self.true_w)
        self.optimal_utility = np.max(Utility)
        self.optimal_epsilon = self.epsilon_range[np.argmax(Utility)]



    def setup_directories(self):
        """Setup directories for storing experiment data"""
        self.data_dir = f'experiments_{self.interaction}/{self.acquisition}/T_{self.T}_num_iteration_{self.num_iteration}_num_repetition_{self.num_repetition}'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    def initialize_storage(self):
        """Initialize storage for experiment data"""
        self.posterior_weights = []
        self.posterior_params = []
        self.errors_ozaki = []

    def run_experiment(self):
        """Run experiment"""
        for i in range(self.num_iteration+1):
            print('Iteration:', i)
            self.base_seed = int(self.task_id) * 1000 + i
            np.random.seed(self.base_seed)
            if i == 0:
                error_ozaki = np.sum(
                    self.probs_weights * np.linalg.norm(self.grid_weights - self.true_w, axis=1))  # error of prior
                self.errors_ozaki.append(error_ozaki)
            else:
              if self.interaction == 'curve':
                if self.acquisition == 'random':
                    best_params, best_acq_value_pl = AcquisitionFunctionsPL.random(self)
                    self.probs_weights, error_ozaki = Acq_update.update_pl(self, best_params)

                else:
                    # pick the next curve to interact and update the weights
                    best_params, best_acq_value_pl = pair_pl_update.calculate_acq_pl(self, i)
                    best_params = np.array([best_params])
                    self.probs_weights, error_ozaki = Acq_update.update_pl(self, best_params)

              elif self.interaction == 'points':
                if self.acquisition == 'random':
                    epsilon1 = np.random.choice(self.epsilon_range)
                    accuracy1 = np.random.uniform(0,1)
                    # accuracy1 = SigmoidFunctions.sigmoid(epsilon1, self.L, self.k, self.c)
                    # accuracy1 = np.random.uniform(0, true_accuracy1)

                    epsilon2 = np.random.choice(self.epsilon_range)
                    accuracy2 = np.random.uniform(0,1)
                    # accuracy2 = SigmoidFunctions.sigmoid(epsilon2, self.L, self.k, self.c)
                    # accuracy2 = np.random.uniform(0, true_accuracy2)

                    error_ozaki = pair_pl_update.pair_update_error(self, epsilon1, accuracy1, epsilon2, accuracy2)

                elif self.acquisition == 'kg-IS':
                    best_pair, best_acq_value_pl = pair_pl_update.cal_pair_acq(self)
                    error_ozaki = pair_pl_update.pair_update_error(self, best_pair[0], best_pair[1], best_pair[2], best_pair[3])

              self.errors_ozaki.append(error_ozaki)


            print(f'true w: {self.true_w}')
            # print('acq_value_pl:', best_acq_value_pl)
            print(f'Error in PL: {self.errors_ozaki}')

        # Save the data
        output_data = {
            'true_w': self.true_w,
            'errors_PL': self.errors_ozaki,
        }

        with open(f'{self.data_dir}/output_data_{self.task_id}.pkl', 'wb') as f:
            pickle.dump(output_data, f)

class pair_pl_update:
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


        particles_h = self.curve_particles[:,0:3]

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
            N_actions=1,
            n_jobs=max(1, n_cores - 1)
        )

        return best_params, best_acquisition_value

    def cal_pair_acq(self):
        np.random.seed(self.base_seed)
        n_cores = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))
        probs_w = np.array(self.probs_weights, copy=True)  # Deep copy
        grid_w = np.array(self.grid_weights, copy=True)
        probs_params = np.array(self.probs_sigmoid_params, copy=True)
        grid_params = np.array(self.grid_sigmoid_params, copy=True)

        particles_pair = self.pair_particles
        best_pair, best_acquisition_value = UpdatePreference_pair.pair_KG_IS(
            particles_pair,
            probs_w,
            grid_w,
            probs_params,
            grid_params,
            self.epsilon_range,
            self.T,
            self.utility_function,
            n_jobs=max(1, n_cores - 1)
        )
        return best_pair, best_acquisition_value
    
    def pair_update_error(self, epsilon1, accuracy1, epsilon2, accuracy2):
        e1, a1, e2, a2 = UpdatePreference_pair.simulate_pair_action(
            self.T,
            self.utility_function,
            self.true_w,
            epsilon1,
            accuracy1,
            epsilon2,
            accuracy2
        )
        point1 = np.array([e1, a1])
        point2 = np.array([e2, a2])

        self.probs_weights, error_ozaki = UpdatePreference_pair.preference_update_pair(
            point1,
            point2,
            self.probs_weights,
            self.grid_weights,
            self.true_w,
            self.T,
            self.utility_function
        )

        return error_ozaki


def main():
    """Main execution function"""
    # Parse command line arguments
    T = float(sys.argv[1])  # stochasticity
    num_iteration = int(sys.argv[2])  # number of iterations in each run
    num_repetition = int(sys.argv[3])  # number of repeated runs, set the same as array
    acquisition = str(sys.argv[4])  # 'random', or 'kg-IS'
    interaction = str(sys.argv[5]) # 'curve' or 'points'

    experiment = Experiments(T, num_iteration, num_repetition, acquisition, interaction)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
