from typing import List

import pandas as pd
import torch
import numpy as np
from botorch.acquisition.analytic import PosteriorStandardDeviation, UpperConfidenceBound, PosteriorMean
from botorch.acquisition.monte_carlo import qPosteriorStandardDeviation
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from sklearn.metrics import pairwise_distances_argmin_min

# from botorch.acquisition import PosteriorStandardDeviation, UpperConfidenceBound
# from botorch.optim import optimize_acqf, optimize_acqf_mixed

from .gaussian_process import GaussianProcess

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
        self.maximizer = None  # Placeholder for the maximizer condition
        self.qmaximizer = None  # Placeholder for the batch maximizer conditions

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

    def select_uncertain_temperatures(
            self,
            synth_method: str = 'WI',
            n_temperatures: int = 1,
    ) -> List[float]:
        """
        Selects the top n_temperatures with the highest average uncertainty (posterior standard deviation)
        across all other features in the discrete grid.

        Args:
            synth_method (str): The synthesis method to consider, either 'WI' for wet impregnation or 'NP' for colloidal nanoparticle. Default is 'WI'.
            n_temperatures (int): The number of temperatures to select based on uncertainty. Default is 1.

        Returns:
            top_temps (List[float]): A list of the top n_temperatures with the highest average uncertainty.
        """

        if n_temperatures > len(self.list_grids[0]):
            raise ValueError(f"n_temperatures ({n_temperatures}) cannot be greater than the total unique temperatures ({len(self.list_grids[0])}).")

        # Instantiate a acquisition function
        PSTD = PosteriorStandardDeviation(self.GP.gp)
        self.acq_function = PSTD
        self.acq_label = 'Posterior Standard Deviation'

        avg_std_dict = {}

        # Loop over all unique temperatures in the discrete grid
        # assuming index 0 is temperature.
        if synth_method == 'WI':
            for temperature in self.list_grids[0].tolist():
                # selecting data at a specific temperature
                X_discrete_temp = self.X_discrete_wi[self.X_discrete_wi['reaction_temp'] == temperature]

                # scaling and making tensor
                X_discrete_temp_trans = torch.tensor(self.GP.transformer_X.transform(X_discrete_temp))

                std = PSTD.forward(X_discrete_temp_trans.unsqueeze(1)).detach().numpy()
                # adding a dummy dimension for batch size

                avg_std_dict[temperature] = float(std.mean())

        if synth_method == 'NP':
            for temperature in self.list_grids[0].tolist():
                # selecting data at a specific temperature
                X_discrete_temp = self.X_discrete_np[self.X_discrete_np['reaction_temp'] == temperature]

                # scaling and making tensor
                X_discrete_temp_trans = torch.tensor(self.GP.transformer_X.transform(X_discrete_temp))

                std = PSTD.forward(X_discrete_temp_trans.unsqueeze(1)).detach().numpy()
                # adding a dummy dimension for batch size

                avg_std_dict[temperature] = float(std.mean())

        # print the avg_std_dict
        print('Average Std. Dev. for each temperature:')
        for key in avg_std_dict:
            print(f'Temperature: {key} C, Average Std. Dev.: {avg_std_dict[key]}')

        # select top n_temperatures uncertain temperatures
        top_temps = sorted(avg_std_dict, key=avg_std_dict.get, reverse=True)[:n_temperatures]

        return top_temps

    def optimize_posterior_std_dev_discrete(
            self,
            synth_method: str = 'WI',
            n_candidates: int = 5,
            verbose: bool = False
    ) -> pd.DataFrame:
        """
        Suggests n_candidates samples with the highest uncertainty on the discrete grid. Not a batch sampling method.

        Args:
        synth_method (str): The synthesis method to consider, either 'WI' for wet impregnation or 'NP' for colloidal nanoparticle. Default is 'WI'.
        n_candidates (int): The number of candidates to suggest. Default is 5.
        verbose (bool): If True, prints suggested condition. Default is False.

        Returns:
            result (pd.DataFrame): A DataFrame containing the top n_candidates conditions along with their posterior standard deviation values.
        """

        # TODO: implement batch sampling method such as q-batch, local penalization, and thompson sampling, etc.
        # TODO: integrate all the acquisition functions in one method
        # Instantiate a acquisition function
        PSTD = PosteriorStandardDeviation(self.GP.gp)
        self.acq_function = PSTD
        self.acq_label = 'Posterior Standard Deviation'

        if synth_method == 'WI':
            # scaling and making tensor
            X_discrete_trans = torch.tensor(self.GP.transformer_X.transform(self.X_discrete_wi))
            # calculate posterior standard deviation for all the possible feature vectors
            std = PSTD.forward(X_discrete_trans.unsqueeze(1)).detach().numpy()

            # select top n_candidates uncertain conditions
            top_ids = np.argsort(-std)[:n_candidates]  # negativity: sort in reverse order

            # save top n_candidates uncertain conditions
            result = self.X_discrete_wi.join(
                pd.DataFrame(std, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]

        if synth_method == 'NP':
            # scaling and making tensor
            X_discrete_trans = torch.tensor(self.GP.transformer_X.transform(self.X_discrete_np))
            # calculate posterior standard deviation for all the possible feature vectors
            std = PSTD.forward(X_discrete_trans.unsqueeze(1)).detach().numpy()

            # select top n_candidates uncertain conditions
            top_ids = np.argsort(-std)[:n_candidates]  # negativity: sort in reverse order

            # save top n_candidates uncertain conditions
            result = self.X_discrete_np.join(
                pd.DataFrame(std, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]

        self.maximizer = result.iloc[0, :-1].to_dict()  # Save the maximizer condition

        if verbose:
            print(result)

        return result

    def optimize_posterior_std_dev_discrete_batch(
            self,
            synth_method: str = 'WI',
            temperature: float = None,
            n_candidates: int = 5,
            verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Suggests n_candidates samples with the highest uncertainty on the discrete grid using batch sampling.

        Args:
        synth_method (str): The synthesis method to consider, either 'WI' for wet impregnation or 'NP' for colloidal nanoparticle. Default is 'WI'.
        temperature (float): The reaction temperature to fix during optimization. Required.
        n_candidates (int): The number of candidates to suggest using q-batch sampling. Default is 5.
        verbose (bool): If True, prints suggested condition. Default is False.

        Returns:
            result (pd.DataFrame): A DataFrame containing the top n_candidates conditions.
        """

        if temperature is None:
            raise ValueError("Please specify a temperature value to fix during optimization.")

        # Transform the fixed temperature value to the scaled space
        temperature_trans = self.GP.transformer_X.transform(
            pd.DataFrame(
                [[temperature, 0.1, 0.005, 0]], # dummy values for other features
                columns=self.X_discrete_wi.columns if synth_method == 'WI' else self.X_discrete_np.columns
            )
        )[0][0]
        print(f'Temperature {temperature} C is transformed to {temperature_trans}.')

        # Instantiate a acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512,1])) # MC sampling used to approximately calculate acquisition function
        qPSTD = qPosteriorStandardDeviation(model=self.GP.gp, sampler=sampler)
        self.acq_function = qPSTD
        self.acq_label = 'q-Posterior Standard Deviation'

        # Define equality constraints on the input features
        temperature_constraint = (torch.tensor([0]), torch.tensor([1.0]), temperature_trans)
        # temperature constraint: 'x0 * 1.0 = temperature_trans'

        if synth_method == 'WI':
            synth_method_constraint = (torch.tensor([3]), torch.tensor([1.0]), 0)
            # synth_method constraint: 'x3 * 1.0 = 0' (WI)
        elif synth_method == 'NP':
            synth_method_constraint = (torch.tensor([3]), torch.tensor([1.0]), 1)
            # synth_method constraint: 'x3 * 1.0 = 1' (NP)
        else:
            raise ValueError("synth_method must be either 'WI' or 'NP'.")

        equality_constraints = [
                temperature_constraint,
                synth_method_constraint
        ]

        # Optimize the acquisition function
        batch_candidates, acq_value = optimize_acqf(
            acq_function=qPSTD,
            bounds=torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
            # constraint can also be implicitly applied here.
            q=n_candidates,
            num_restarts=5,
            raw_samples=20,
            equality_constraints=equality_constraints,
        )

        print(f"\nBatch candidates shape: {batch_candidates.shape}")
        print(f"Acquisition values shape: {acq_value.shape}")
        print(f"Acquisition values: {acq_value}")
        print(f"\nBatch candidates:\n{batch_candidates}")

        # Choose the closest points from the discrete grid to the optimized candidates
        closest_points = []
        idx_list = []
        if synth_method == 'WI':
            X_discrete_trans = self.GP.transformer_X.transform(self.X_discrete_wi)
        elif synth_method == 'NP':
            X_discrete_trans = self.GP.transformer_X.transform(self.X_discrete_np)

        for batch in batch_candidates:
            idx = pairwise_distances_argmin_min(batch.unsqueeze(0), X_discrete_trans)[0]
            closest_points.append(X_discrete_trans[idx][0])
            idx_list.append(idx)
        closest_points = torch.tensor(closest_points)
        idx_list = np.array(idx_list).reshape(-1)
        print(f"\nBatch candidates (closest grid points):\n{closest_points}")

        if synth_method == 'WI':
            result = self.X_discrete_wi.iloc[idx_list, :]
        elif synth_method == 'NP':
            result = self.X_discrete_np.iloc[idx_list, :] # DataFrame

        self.qmaximizer = result

        if verbose:
            print(result)

        return result

    def optimize_posterior_mean_discrete(
            self,
            synth_method: str = 'WI',
            n_candidates: int = 5,
            verbose: bool = False
    ):
        """
        Suggests n_candidates samples with the highest posterior mean on the discrete grid. Not a batch sampling method.

        Args:
            synth_method (str): The synthesis method to consider, either 'WI' for wet impregnation or 'NP' for colloidal nanoparticle. Default is 'WI'.
            n_candidates (int): The number of candidates to suggest. Default is 5.
            verbose (bool): If True, prints suggested condition. Default is False.
        Returns:
            result (pd.DataFrame): A DataFrame containing the top n_candidates conditions along with their posterior mean values.
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

            # save top 'n_candidates' uncertain conditions
            result = self.X_discrete_wi.join(
                pd.DataFrame(acq, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]

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

            # save top 'n_candidates' conditions with high posterior mean
            result = self.X_discrete_np.join(
                pd.DataFrame(acq, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]

        self.maximizer = result.iloc[0, :-1].to_dict()  # Save the maximizer condition

        if verbose:
            print(result)

        return result

    def optimize_upper_confidence_bound_discrete(
            self,
            synth_method: str = 'WI',
            n_candidates: int = 5,
            verbose: bool = False
    ):
        """
        Suggests n_candidates samples with the highest upper confidence bound on the discrete grid. Not a batch sampling method.

        Args:
            synth_method (str): The synthesis method to consider, either 'WI' for wet impregnation or 'NP' for colloidal nanoparticle. Default is 'WI'.
            n_candidates (int): The number of candidates to suggest. Default is 5.
            verbose (bool): If True, prints suggested condition. Default is False.
        Returns:
            result (pd.DataFrame): A DataFrame containing the top n_candidates conditions along with their upper confidence bound values.
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

            # save top 'n_candidates' uncertain conditions
            result = self.X_discrete_wi.join(
                pd.DataFrame(acq, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]

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

            # save top 'n_candidates' conditions with high upper confidence bound value
            result = self.X_discrete_np.join(
                pd.DataFrame(acq, columns=[self.acq_label])  # append uncertainty info.
                ).iloc[top_ids, :]

        self.maximizer = result.iloc[0, :-1].to_dict()  # Save the maximizer condition

        if verbose:
            print(result)

        return result

    def optimize_expected_improvement_discrete(
            self,
            synth_method: str = 'WI',
            n_candidates: int = 5,
    ):
        # suggest n_candidates samples with the highest expected improvement on discrete grid
        pass