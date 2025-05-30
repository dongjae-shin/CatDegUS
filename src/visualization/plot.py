from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.active_learning.acquisition import DiscreteGrid
from src.active_learning.gaussian_process import GaussianProcess

class Plotter:
    def __init__(self,
                 GP: GaussianProcess,
                 Grid: DiscreteGrid
                 ):
        """
        Initializes the Plotter with a model and an acquisition function.

        Args:
            GP (GaussianProcess):
            Grid (DiscreteGrid): Contains the grid information and acquisition function.
        """
        self.GP = GP
        self.Grid = Grid
        self.synth_method = None  # Default to None, can be set later
        self.allowed_meshgrid1 = None  # Placeholder for allowed meshgrid1
        self.allowed_meshgrid2 = None  # Placeholder for allowed meshgrid2
        self.meshgrid1 = None  # Placeholder for meshgrid1
        self.meshgrid2 = None  # Placeholder for meshgrid2
        self.meshgrid3 = None  # Placeholder for meshgrid3

    def plot_2d_acquisition_function(self,
                                     synth_method: str = 'NP',
                                     acq_max: float = 1.1,
                                     n_levels: int = 32,
                                     temperature_list: List[int] = None,
                                     mode: str = 'boundary',
                                     custom_range: tuple = None,
                                     contour_resolution: int = 50):
        """
        Plots a 2D acquisition function contour plot per temperature

        Args:
            synth_method (str): Synthesis method, either 'WI' or 'NP'.
            acq_max (float): Maximum value for the acquisition function in colorbar of plot.
            n_levels (int): Number of contour levels.
            temperature_list (List[int]): List of temperatures to plot.
            mode (str): Mode for contour plotting, either 'boundary' or 'custom'.
            custom_range (tuple): Custom range for contour plotting in the form (start_w_rh, stop_w_rh, start_m_rh, stop_m_rh). Only used if mode is 'custom'.
            contour_resolution (int): Resolution for the contour grid. Number of points in each dimension for contour plot.

        """

        # Convert synth_method to numerical value
        if synth_method == 'WI':
            synth_method = 0
        elif synth_method == 'NP':
            synth_method = 1
        else:
            raise ValueError("synth_method must be either 'WI' or 'NP'")
        self.synth_method = synth_method

        # 1. Grid for plotting allowed points
        W_rh_allowed, M_rh_allowed = np.meshgrid(self.Grid.list_grids[1], self.Grid.list_grids[2])
        self.allowed_meshgrid1 = W_rh_allowed
        self.allowed_meshgrid2 = M_rh_allowed
        #self.Grid.list_grids[1]  # Rh_weight_loading
        #self.Grid.list_grids[2]  # Rh_total_mass

        # 2. Grid for contour plot
        if mode == 'boundary':
            start_w_rh = self.Grid.x_range_min[1]
            stop_w_rh = self.Grid.x_range_max[1]
            start_m_rh = self.Grid.x_range_min[2]
            stop_m_rh = self.Grid.x_range_max[2]
        elif mode == 'custom':
            if custom_range is None or len(custom_range) != 4:
                raise ValueError("custom_range must be a tuple of four values: (start_w_rh, stop_w_rh, start_m_rh, stop_m_rh)")
            start_w_rh = custom_range[0]
            stop_w_rh = custom_range[1]
            start_m_rh = custom_range[2]
            stop_m_rh = custom_range[3]
        else:
            raise ValueError("mode must be either 'boundary' or 'custom'")

        w_rh_contour = np.linspace(
            start_w_rh,
            stop_w_rh,
            num=contour_resolution
        )
        m_rh_contour = np.linspace(
            start_m_rh,
            stop_m_rh,
            num=contour_resolution
        )
        W_rh_contour, M_rh_contour = np.meshgrid(w_rh_contour, m_rh_contour)
        self.meshgrid1 = W_rh_contour
        self.meshgrid2 = M_rh_contour

        # 3. Contour plot of the acquisition function
        self._plot_2d_contour(
            acq_max=acq_max, n_levels=n_levels, temperature_list=temperature_list,
            xmin=start_w_rh, xmax=stop_w_rh, ymin=start_m_rh, ymax=stop_m_rh
        )

    def _plot_2d_contour(self,
                      acq_max: float = 1.1, n_levels: int = 32, temperature_list: List[int] = None,
                      xmin: float = 0.0, xmax: float = 6.0,
                      ymin: float = 0.0, ymax: float = 0.05
                      ):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

        levels = np.linspace(0, acq_max, n_levels)  # for std

        if temperature_list is None:
            temperature_list = self.GP.list_grids[0]

        # Contour plot for each temperature
        for temperature in temperature_list:
            cmap = plt.contourf(
                self.meshgrid1,
                self.meshgrid2,
                wrapper_acq_value(
                    self.meshgrid1, self.meshgrid2, temperature=temperature,
                    acq_func=self.Grid.acq_function,
                    synth_method=self.synth_method,
                    GP=self.GP
                    ),
                extend='max',
                levels=levels
            )

            cbar = plt.colorbar(
                cmap,
                label=self.Grid.acq_label,
                ticks=np.linspace(0, acq_max, int(acq_max / 0.1 + 1))

            )

            plot_grid(self.allowed_meshgrid1, self.allowed_meshgrid2)
            plot_train_data(
                self.GP.df_Xtrain[
                    (self.GP.df_Xtrain['synth_method'] == self.synth_method) & (self.GP.df_Xtrain['reaction_temp'] == temperature)
                ]
            )

            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            # plt.tight_layout()
            plt.xlabel('Rh weight loading (wt%)')
            plt.ylabel('Rh mass in reactor (mg)')
            plt.title(f'temperature: {temperature} ℃')
            plt.legend(
                # bbox_to_anchor=(1.02, -0.12)
            )
            plt.show()

    def plot_3d_contour(self):
        pass


@np.vectorize
def wrapper_acq_value(
        wrh, mrh, temperature=500,
        acq_func=None, synth_method: int = 0, GP: GaussianProcess = None
        ):
    """
    Wrapper function to compute the acquisition function value for given parameters.

    Args:
        wrh (float): Weight loading of Rh in wt%.
        mrh (float): Total mass of Rh in mg.
        temperature (int): Temperature in Celsius.
        acq_func: Acquisition function to be used, e.g., PosteriorStandardDeviation, UpperConfidenceBound.
        synth_method (int): Synthesis method, 0 for 'WI' and 1 for 'NP'.
    """

    # Ensure acquisition function is provided
    if acq_func is None:
        raise ValueError("acquisition function must be provided, e.g., PosteriorStandardDeviation, UpperConfidenceBound")

    # Dataframing for preprocessor
    X = pd.DataFrame(
        np.array([[temperature, wrh, mrh, synth_method]])
    )

    # Adding column information for preprocessor
    X.columns = GP.columns

    # Scaling and tensorizing
    X_tensor = torch.tensor(
        GP.transformer_X.transform(X)
    )
    return acq_func.forward(
        X_tensor.reshape(len(X_tensor), 1, len(X.columns))  # Reshape to match model input
    ).detach().numpy()

def plot_grid(x_points, y_points):
    X, Y = np.meshgrid(x_points, y_points)
    plt.scatter(
        X, Y,
        s=0.1, c='k', label='grid'
    )

def plot_train_data(df_Xtrain):
    plt.scatter(
        df_Xtrain.loc[:, 'Rh_weight_loading'], df_Xtrain.loc[:, 'Rh_total_mass'],
        s=5.0, c='r', marker='D', label='train data'
    )