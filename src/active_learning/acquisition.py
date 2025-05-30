from typing import List

import pandas as pd
import torch
import numpy as np
from botorch.acquisition.analytic import PosteriorStandardDeviation, UpperConfidenceBound
# from botorch.acquisition import PosteriorStandardDeviation, UpperConfidenceBound
# from botorch.optim import optimize_acqf, optimize_acqf_mixed

from src.active_learning.gaussian_process import GaussianProcess

class DiscreteGrid:
    """
    A class to construct a discrete grid of input features and perform acquisition functions
    for active learning with Gaussian Processes.

    Attributes:
        x_range_min (list): Minimum values for each input feature.
        x_range_max (list): Maximum values for each input feature.
        x_step (list): Step sizes for each input feature.
        GP (GaussianProcess): An instance of the GaussianProcess class.
        list_grids (list): List of grids for each input feature.
        n_grid_total (int): Total number of combinations in the constructed grid.
        X_discrete_wi (pd.DataFrame): Unlabeled data set for WI method.
        X_discrete_np (pd.DataFrame): Unlabeled data set for NP method.

    Methods:
        construct_grid():
            Constructs a grid of all possible combinations of input features.

        uncertainty_sampling_discrete(GP, transformer_X, synth_method, n_candidates=5, columns=None):
            Suggests samples with the highest uncertainty on the discrete grid.

        upper_confidence_bound_discrete(GP, n_candidates):
            TBD

        expected_improvement_discrete(GP, n_candidates):
            TBD
    """
    def __init__(
            self, x_range_min: list, x_range_max: list, x_step: list,
            GP: GaussianProcess = None
    ):
        self.x_range_min = x_range_min
        self.x_range_max = x_range_max
        self.x_step = x_step
        self.GP = GP
        self.list_grids = None
        self.n_grid_total = None
        self.X_discrete_wi = None
        self.X_discrete_np = None

    def construct_grid(self, columns: List[str] = None):
        if columns is None:
            columns = ['reaction_temp', 'Rh_weight_loading', 'Rh_total_mass', 'synth_method']

        # Unlabeled data set for all the possible combinations of input features
        list_grids = []
        # Create a grid for each input feature
        for i in range(len(self.x_step)):
            n_grid = int((self.x_range_max[i] - self.x_range_min[i]) / self.x_step[i] + 1)
            grid = np.linspace(self.x_range_min[i], self.x_range_max[i], n_grid)
            list_grids.append(grid)
        self.list_grids = list_grids

        # Calculate the total number of combinations
        self.n_grid_total = np.prod([len(grid) for grid in self.list_grids])
        print(f'{self.n_grid_total} combinations are possible in the constructed grid.')

        # Create a meshgrid of all combinations
        grid_combinations = np.meshgrid(*self.list_grids, indexing='ij')
        # Reshape and stack to get all combinations as rows
        X_discrete = np.column_stack([g.flatten() for g in grid_combinations])
        X_discrete = pd.DataFrame(X_discrete)

        # Unlabeled data set for WI and NP
        self.X_discrete_wi = X_discrete[X_discrete.iloc[:, -1] == 0].reset_index(drop=True)
        self.X_discrete_np = X_discrete[X_discrete.iloc[:, -1] == 1].reset_index(drop=True)

        # Adding column information for preprocessor
        self.X_discrete_wi.columns = columns
        self.X_discrete_np.columns = columns

        # Round all the rows of each column
        self.X_discrete_wi['reaction_temp'] = self.X_discrete_wi['reaction_temp'].round().astype(int)
        self.X_discrete_wi['Rh_weight_loading'] = self.X_discrete_wi['Rh_weight_loading'].round(1)
        self.X_discrete_wi['Rh_total_mass'] = self.X_discrete_wi['Rh_total_mass'].round(4)
        self.X_discrete_wi['synth_method'] = self.X_discrete_wi['synth_method'].round().astype(int)
        self.X_discrete_np['reaction_temp'] = self.X_discrete_np['reaction_temp'].round().astype(int)
        self.X_discrete_np['Rh_weight_loading'] = self.X_discrete_np['Rh_weight_loading'].round(1)
        self.X_discrete_np['Rh_total_mass'] = self.X_discrete_np['Rh_total_mass'].round(4)
        self.X_discrete_np['synth_method'] = self.X_discrete_np['synth_method'].round().astype(int)

    def uncertainty_sampling_discrete(
            self,
            # gp: GaussianProcess = None,
            synth_method: str = 'WI',
            n_candidates: int = 5,
    ):
        # suggest n_candidates samples with the highest uncertainty on discrete grid

        # Instantiate a acquisition function
        US = PosteriorStandardDeviation(self.GP.gp)

        if synth_method == 'WI':
            # scaling and making tensor
            X_discrete_wi_trans = torch.tensor(
                self.GP.transformer_X.transform(self.X_discrete_wi)
            )
            # calculate posterior standard deviation for all the possible feature vectors
            std = US.forward(
                X_discrete_wi_trans.reshape(
                    len(X_discrete_wi_trans), 1, X_discrete_wi_trans.shape[1])
            ).detach().numpy()

            top_ids = np.argsort(-std)[:n_candidates]  # negativity: sort in reverse order

            # show top 'n_candidates' uncertain conditions
            print(
                self.X_discrete_wi.join(
                    pd.DataFrame(std, columns=['std. dev.'])  # append uncertainty info.
                ).iloc[top_ids, :]
            )
        if synth_method == 'NP':
            # scaling and making tensor
            X_discrete_np_trans = torch.tensor(
                self.GP.transformer_X.transform(self.X_discrete_np)
            )
            # calculate posterior standard deviation for all the possible feature vectors
            std = US.forward(
                X_discrete_np_trans.reshape(
                    len(X_discrete_np_trans), 1, X_discrete_np_trans.shape[1])
            ).detach().numpy()

            top_ids = np.argsort(-std)[:n_candidates]  # negativity: sort in reverse order

            # show top 'n_candidates' uncertain conditions
            print(
                self.X_discrete_np.join(
                    pd.DataFrame(std, columns=['std. dev.'])  # append uncertainty info.
                ).iloc[top_ids, :]
            )

    def upper_confidence_bound_discrete(self, GP, n_candidates):
        # suggest n_candidates samples with the highest upper confidence bound on discrete grid

        UCB = UpperConfidenceBound(GP, beta=0.1)
        pass

    def expected_improvement_discrete(self, GP, n_candidates):
        # suggest n_candidates samples with the highest expected improvement on discrete grid
        pass