from typing import List

import pandas as pd
import torch
import numpy as np
from botorch.acquisition.analytic import PosteriorStandardDeviation, UpperConfidenceBound, PosteriorMean
# from botorch.acquisition import PosteriorStandardDeviation, UpperConfidenceBound
# from botorch.optim import optimize_acqf, optimize_acqf_mixed

from src.active_learning.gaussian_process import GaussianProcess

class DiscreteGrid:
    """
    A class to construct a discrete grid of input features and perform acquisition functions
    for active learning with Gaussian Processes.

    Attributes:
        GP (GaussianProcess): An instance of the GaussianProcess class used for predictions.
        x_range_min (list): The minimum values for each input feature in the grid.
        x_range_max (list): The maximum values for each input feature in the grid.
        x_step (list): The step sizes for each input feature in the grid.
        list_grids (list): A list containing the grid values for each input feature.
        n_grid_total (int): The total number of combinations in the constructed grid.
        X_discrete_wi (pd.DataFrame): The subset of the grid corresponding to the WI method.
        X_discrete_np (pd.DataFrame): The subset of the grid corresponding to the NP method.
        acq_function: The acquisition function instance used for optimization, e.g., PosteriorStandardDeviation or UpperConfidenceBound.

    Methods:
        construct_grid():
            Generates a grid of all possible combinations of input features.

        optimize_uncertainty_sampling_discrete(synth_method, n_candidates=5):
            Selects the top samples with the highest uncertainty from the discrete grid.

        optimize_upper_confidence_bound_discrete(synth_method, n_candidates):
            Placeholder for selecting samples based on the upper confidence bound.

        optimize_expected_improvement_discrete(synth_method, n_candidates):
            Placeholder for selecting samples based on expected improvement.

    """
    def __init__(
            self, x_range_min: list, x_range_max: list, x_step: list,
            GP: GaussianProcess
    ):
        """
        Initializes the DiscreteGrid with specified ranges and step sizes for input features.

        Args:
            x_range_min (list): Minimum values for each input feature in the grid.
            x_range_max (list): Maximum values for each input feature in the grid.
            x_step (list): Step sizes for each input feature in the grid.
            GP (GaussianProcess): An instance of the GaussianProcess class used for predictions.

        Returns:
            None
        """
        self.x_range_min = x_range_min
        self.x_range_max = x_range_max
        self.x_step = x_step
        if GP is None:
            raise ValueError("A GaussianProcess instance must be provided.")
        self.GP = GP
        self.list_grids = None
        self.n_grid_total = None
        self.X_discrete_wi = None
        self.X_discrete_np = None
        self.acq_function = None
        self.acq_label = None  # Placeholder for acquisition function label

    def construct_grid(self, columns: List[str] = None):
        """
        Constructs a discrete grid of all possible combinations of input features
        based on the specified ranges and step sizes. The grid is divided into two
        subsets based on the 'synth_method' column: WI (wet impregnation) and NP
        (colloidal nanoparticle).

        Args:
            columns (List[str], optional): A list of column names for the grid.
                Defaults to ['reaction_temp', 'Rh_weight_loading', 'Rh_total_mass', 'synth_method'].

        Returns:
            None
        """
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

    def optimize_posterior_std_dev_discrete(
            self,
            synth_method: str = 'WI',
            n_candidates: int = 5,
    ):
        """
        Suggests n_candidates samples with the highest uncertainty on the discrete grid. Not a batch sampling method.

        Args:
        synth_method (str): The synthesis method to consider, either 'WI' for wet impregnation or 'NP' for colloidal nanoparticle. Default is 'WI'.
        n_candidates (int): The number of candidates to suggest. Default is 5.

        Returns:
            None, Prints the top n_candidates uncertain conditions along with their standard deviation.
        """

        # TODO: implement batch sampling method such as q-batch, local penalization, and thompson sampling, etc.
        # Instantiate a acquisition function
        US = PosteriorStandardDeviation(self.GP.gp)
        self.acq_function = US
        self.acq_label = 'Posterior Standard Deviation'

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
                    pd.DataFrame(std, columns=[self.acq_label])  # append uncertainty info.
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
                    pd.DataFrame(std, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]
            )

    def optimize_posterior_mean_discrete(
            self,
            synth_method: str = 'WI',
            n_candidates: int = 5,
    ):
        """
        Suggests n_candidates samples with the highest posterior mean on the discrete grid. Not a batch sampling method.

        Args:
            synth_method (str): The synthesis method to consider, either 'WI' for wet impregnation or 'NP' for colloidal nanoparticle. Default is 'WI'.
            n_candidates (int): The number of candidates to suggest. Default is 5.
        Returns:
            None, Prints the top n_candidates conditions along with their posterior mean values.
        """
        PM = PosteriorMean(self.GP.gp)
        self.acq_function = PM
        self.acq_label = 'Posterior Mean'

        if synth_method == 'WI':
            # scaling and making tensor
            X_discrete_wi_trans = torch.tensor(
                self.GP.transformer_X.transform(self.X_discrete_wi)
            )
            # calculate posterior standard deviation for all the possible feature vectors
            acq = PM.forward(
                X_discrete_wi_trans.reshape(
                    len(X_discrete_wi_trans), 1, X_discrete_wi_trans.shape[1])
            ).detach().numpy()

            top_ids = np.argsort(-acq)[:n_candidates]  # negativity: sort in reverse order

            # show top 'n_candidates' uncertain conditions
            print(
                self.X_discrete_wi.join(
                    pd.DataFrame(acq, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]
            )

        if synth_method == 'NP':
            # scaling and making tensor
            X_discrete_np_trans = torch.tensor(
                self.GP.transformer_X.transform(self.X_discrete_np)
            )
            # calculate posterior standard deviation for all the possible feature vectors
            acq = PM.forward(
                X_discrete_np_trans.reshape(
                    len(X_discrete_np_trans), 1, X_discrete_np_trans.shape[1])
            ).detach().numpy()

            top_ids = np.argsort(-acq)[:n_candidates]  # negativity: sort in reverse order

            # show top 'n_candidates' conditions with high posterior mean
            print(
                self.X_discrete_np.join(
                    pd.DataFrame(acq, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]
            )

    def optimize_upper_confidence_bound_discrete(
            self,
            synth_method: str = 'WI',
            n_candidates: int = 5,
    ):
        """
        Suggests n_candidates samples with the highest upper confidence bound on the discrete grid. Not a batch sampling method.

        Args:
            synth_method (str): The synthesis method to consider, either 'WI' for wet impregnation or 'NP' for colloidal nanoparticle. Default is 'WI'.
            n_candidates (int): The number of candidates to suggest. Default is 5.
        Returns:
            None, Prints the top n_candidates conditions along with their upper confidence bound values.
        """
        # instantiate an acquisition function
        UCB = UpperConfidenceBound(self.GP.gp, beta=0.1)
        self.acq_function = UCB
        self.acq_label = 'Acquisition_UCB'

        if synth_method == 'WI':
            # scaling and making tensor
            X_discrete_wi_trans = torch.tensor(
                self.GP.transformer_X.transform(self.X_discrete_wi)
            )
            # calculate posterior standard deviation for all the possible feature vectors
            acq = UCB.forward(
                X_discrete_wi_trans.reshape(
                    len(X_discrete_wi_trans), 1, X_discrete_wi_trans.shape[1])
            ).detach().numpy()

            top_ids = np.argsort(-acq)[:n_candidates]  # negativity: sort in reverse order

            # show top 'n_candidates' uncertain conditions
            print(
                self.X_discrete_wi.join(
                    pd.DataFrame(acq, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]
            )

        if synth_method == 'NP':
            # scaling and making tensor
            X_discrete_np_trans = torch.tensor(
                self.GP.transformer_X.transform(self.X_discrete_np)
            )
            # calculate posterior standard deviation for all the possible feature vectors
            acq = UCB.forward(
                X_discrete_np_trans.reshape(
                    len(X_discrete_np_trans), 1, X_discrete_np_trans.shape[1])
            ).detach().numpy()

            top_ids = np.argsort(-acq)[:n_candidates]  # negativity: sort in reverse order

            # show top 'n_candidates' conditions with high upper confidence bound value
            print(
                self.X_discrete_np.join(
                    pd.DataFrame(acq, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]
            )

    def optimize_expected_improvement_discrete(
            self,
            synth_method: str = 'WI',
            n_candidates: int = 5,
    ):
        # suggest n_candidates samples with the highest expected improvement on discrete grid
        pass