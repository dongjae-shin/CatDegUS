from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
import json

from ..active_learning.acquisition import DiscreteGrid
from ..active_learning.gaussian_process import GaussianProcess

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
        self.synth_method_num = None  # Default to None, can be set later
        self.synth_method_label = None
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
                                     contour_resolution: int = 50,
                                     plot_allowed_grid: bool = True,
                                     plot_train: bool = True,
                                     show: bool = True
    ):
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
            plot_allowed_grid (bool): Whether to plot the allowed grid points.
            plot_train (bool): Whether to plot the training data points.
            show (bool): Whether to show the plot immediately. If False, the plot will saved as an image file.

        """
        self.synth_method_label = synth_method
        # Convert synth_method to numerical value
        if synth_method == 'WI':
            synth_method = 0
        elif synth_method == 'NP':
            synth_method = 1
        else:
            raise ValueError("synth_method must be either 'WI' or 'NP'")
        self.synth_method_num = synth_method

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

        w_rh_contour = np.linspace(start_w_rh, stop_w_rh, num=contour_resolution)
        m_rh_contour = np.linspace(start_m_rh, stop_m_rh, num=contour_resolution)

        W_rh_contour, M_rh_contour = np.meshgrid(w_rh_contour, m_rh_contour)
        self.meshgrid1 = W_rh_contour
        self.meshgrid2 = M_rh_contour

        # 3. Contour plot of the acquisition function
        self._plot_2d_contour(
            acq_max=acq_max, n_levels=n_levels, temperature_list=temperature_list,
            xmin=start_w_rh, xmax=stop_w_rh, ymin=start_m_rh, ymax=stop_m_rh,
            plot_allowed_grid=plot_allowed_grid, plot_train=plot_train, show=show
        )

    def _plot_2d_contour(self,
                      acq_max: float = 1.1, n_levels: int = 32, temperature_list: List[int] = None,
                      xmin: float = 0.0, xmax: float = 6.0,
                      ymin: float = 0.0, ymax: float = 0.05,
                      plot_allowed_grid: bool = True, plot_train: bool = True, show: bool = True
                      ):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 5))

        levels = np.linspace(0, acq_max, n_levels)  # for std

        if temperature_list is None:
            temperature_list = self.GP.list_grids[0]

        # Contour plot for each temperature
        for temperature in temperature_list:
            contour = plt.contourf(
                self.meshgrid1,
                self.meshgrid2,
                wrapper_acq_value(
                    self.meshgrid1, self.meshgrid2, temperature=temperature,
                    acq_func=self.Grid.acq_function,
                    synth_method=self.synth_method_num,
                    GP=self.GP
                    ),
                extend='max',
                levels=levels
            )

            cbar = plt.colorbar(
                contour,
                label=self.Grid.acq_label,
                ticks=np.linspace(0, acq_max, int(acq_max / 0.1 + 1))
            )

            # Plot the maximizer
            if self.Grid.maximizer['synth_method'] == self.synth_method_num and self.Grid.maximizer['reaction_temp'] == temperature:
                print(f"Maximizer for synthesis method {self.synth_method_label} at temperature {temperature}: {self.Grid.maximizer}")
                plt.scatter(
                    self.Grid.maximizer['Rh_weight_loading'],
                    self.Grid.maximizer['Rh_total_mass'],
                    c='white', edgecolor='black', linewidth=1.5,
                    s=200, marker='*', alpha=0.5,
                    label='maximizer'
                )
            else:
                print(f"No maximizer found for synthesis method {self.synth_method_label} at temperature {temperature}")

            if plot_allowed_grid:
                plot_grid(self.allowed_meshgrid1, self.allowed_meshgrid2)
            if plot_train:
                plot_train_data(
                    self.GP.df_Xtrain[
                        (self.GP.df_Xtrain['synth_method'] == self.synth_method_num) &
                        (self.GP.df_Xtrain['reaction_temp'] == temperature) &
                        # Filter out the data outside (xmin, xmax, ymin, ymax)
                        (self.GP.df_Xtrain['Rh_weight_loading'] >= xmin) &
                        (self.GP.df_Xtrain['Rh_weight_loading'] <= xmax) &
                        (self.GP.df_Xtrain['Rh_total_mass'] >= ymin) &
                        (self.GP.df_Xtrain['Rh_total_mass'] <= ymax)
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
            if show:
                plt.show()
            else:
                plt.savefig(f"./acq_distr_2d_synth_{self.synth_method_label}_{temperature}C.png", bbox_inches='tight', dpi=200)
                plt.clf()  # Clear the current figure for the next temperature

            # Output JSON file with raw data for the contour plot for programmatic interaction with an LLM agent
            contour_data = {
                'acq_values': {
                    'x_flattened_meshgrid': self.meshgrid1[0].flatten().tolist(),
                    'y_flattened_meshgrid': self.meshgrid2[0].flatten().tolist(),
                    'z_flattened_meshgrid': wrapper_acq_value(
                        self.meshgrid1, self.meshgrid2, temperature=temperature,
                        acq_func=self.Grid.acq_function,
                        synth_method=self.synth_method_num,
                        GP=self.GP
                    ).flatten().tolist(),
                    'xlabel': 'Rh weight loading (wt%)',
                    'ylabel': 'Rh mass in reactor (mg)',
                    'zlabel': self.Grid.acq_label,
                },
                'global_maximizer_condition': {
                    'Rh_weight_loading': self.Grid.maximizer['Rh_weight_loading'],
                    'Rh_total_mass': self.Grid.maximizer['Rh_total_mass'],
                    'reaction_temp': self.Grid.maximizer['reaction_temp'],
                    'synth_method': self.Grid.maximizer['synth_method'],
                },
                'training_data': {
                    'Rh_weight_loading': self.GP.df_Xtrain[
                        (self.GP.df_Xtrain['synth_method'] == self.synth_method_num) &
                        (self.GP.df_Xtrain['reaction_temp'] == temperature)
                    ]['Rh_weight_loading'].tolist(),
                    'Rh_total_mass': self.GP.df_Xtrain[
                        (self.GP.df_Xtrain['synth_method'] == self.synth_method_num) &
                        (self.GP.df_Xtrain['reaction_temp'] == temperature)
                    ]['Rh_total_mass'].tolist(),
                },
                'allowed_grid': {
                    'x_flattened_meshgrid': self.allowed_meshgrid1.flatten().tolist(),
                    'y_flattened_meshgrid': self.allowed_meshgrid2.flatten().tolist(),
                },
                'acquisition_function': self.Grid.acq_label,
                'synth_method': self.synth_method_label,
                'temperature': temperature
            }
            with open(f"./acq_distr_2d_synth_{self.synth_method_label}_{temperature}C.json", 'w') as f:
                json.dump(contour_data, f, indent=4)

    def plot_3d_acquisition_function(self,
                                     synth_method: str = 'NP',
                                     acq_max: float = 1.1,
                                     mode: str = 'boundary',
                                     custom_range: tuple = None,
                                     contour_resolution: int = 20,
                                     plot_allowed_grid: bool = True,
                                     plot_train: bool = True,
                                     show: bool = True
    ):
        """
        Plots a 3D acquisition function contour plot.

        Args:
            synth_method (str): Synthesis method, either 'WI' or 'NP'.
            acq_max (float): Maximum value for the acquisition function in colorbar of plot.
            mode (str): Mode for contour plotting, either 'boundary' or 'custom'.
            custom_range (tuple): Custom range for contour plotting in the form (start_w_rh, stop_w_rh, start_m_rh, stop_m_rh, start_temp, stop_temp). Only used if mode is 'custom'.
            contour_resolution (int): Resolution for the contour grid. Number of points in each dimension for contour plot.
            plot_allowed_grid (bool): Whether to plot the allowed grid points.
            plot_train (bool): Whether to plot the training data points.
            show (bool): Whether to show the plot immediately. If False, the plot will saved as an image file.
        """
        self.synth_method_label = synth_method
        # Convert synth_method to numerical value
        if synth_method == 'WI':
            synth_method = 0
        elif synth_method == 'NP':
            synth_method = 1
        else:
            raise ValueError("synth_method must be either 'WI' or 'NP'")
        self.synth_method_num = synth_method

        matplotlib.rcParams.update({'font.size': 12})

        # 1. Grid for plotting allowed points
        W_rh_allowed, M_rh_allowed, T_allowed = np.meshgrid(self.Grid.list_grids[1], self.Grid.list_grids[2], self.Grid.list_grids[0])
        self.allowed_meshgrid1 = W_rh_allowed
        self.allowed_meshgrid2 = M_rh_allowed
        self.allowed_meshgrid3 = T_allowed
        # self.Grid.list_grids[1]  # Rh_weight_loading
        # self.Grid.list_grids[2]  # Rh_total_mass
        # self.Grid.list_grids[0]  # reaction_temp

        # 2. Grid for contour plot
        if mode == 'boundary':
            start_w_rh = self.Grid.x_range_min[1]
            stop_w_rh = self.Grid.x_range_max[1]
            start_m_rh = self.Grid.x_range_min[2]
            stop_m_rh = self.Grid.x_range_max[2]
            start_temp = self.Grid.x_range_min[0]
            stop_temp = self.Grid.x_range_max[0]
        elif mode == 'custom':
            if custom_range is None or len(custom_range) != 6:
                raise ValueError(
                    "custom_range must be a tuple of four values: (start_w_rh, stop_w_rh, start_m_rh, stop_m_rh, start_temp, stop_temp)")
            start_w_rh = custom_range[0]
            stop_w_rh = custom_range[1]
            start_m_rh = custom_range[2]
            stop_m_rh = custom_range[3]
            start_temp = custom_range[4]
            stop_temp = custom_range[5]
        else:
            raise ValueError("mode must be either 'boundary' or 'custom'")

        w_rh_contour = np.linspace(start_w_rh, stop_w_rh, num=contour_resolution)
        m_rh_contour = np.linspace(start_m_rh, stop_m_rh, num=contour_resolution)
        t_contour = np.linspace(start_temp, stop_temp, num=contour_resolution)

        W_rh_contour, M_rh_contour, T_contour = np.meshgrid(w_rh_contour, m_rh_contour, t_contour)
        self.meshgrid1 = W_rh_contour
        self.meshgrid2 = M_rh_contour
        self.meshgrid3 = T_contour

        # 3. Contour plot of the acquisition function
        self._plot_3d_contour(
            acq_max=acq_max,
            xmin=start_w_rh, xmax=stop_w_rh, ymin=start_m_rh, ymax=stop_m_rh, zmin=start_temp, zmax=stop_temp,
            plot_allowed_grid=plot_allowed_grid, plot_train=plot_train, show=show
        )

    def _plot_3d_contour(self,
                         acq_max: float = 1.1,
                         xmin: float = 0.0, xmax: float = 6.0,
                         ymin: float = 0.0, ymax: float = 0.05,
                         zmin: float = 300, zmax: float = 550,
                         plot_allowed_grid: bool = True, plot_train: bool = True, show: bool = True
                         ):

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(8, 6.5), constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

        contour = ax.scatter(
            self.meshgrid1, self.meshgrid2, self.meshgrid3,
            vmin=0, vmax=acq_max,
            c=wrapper_acq_value(
                self.meshgrid1, self.meshgrid2, self.meshgrid3,
                acq_func=self.Grid.acq_function,
                synth_method=self.synth_method_num,
                GP=self.GP
                ),
            alpha=0.4, s=20,
            # cmap='coolwarm',
            zorder=2.5
        )

        cbar = plt.colorbar(
            contour,
            label=self.Grid.acq_label,
            ticks=np.linspace(0, acq_max, int(acq_max / 0.1 + 1)),
            pad=0.13
        )

        # Plot the suggestion
        if self.Grid.maximizer['synth_method'] == self.synth_method_num:
            print(f"Maximizer for synthesis method {self.synth_method_label}: {self.Grid.maximizer}")
            ax.scatter(
                self.Grid.maximizer['Rh_weight_loading'],
                self.Grid.maximizer['Rh_total_mass'],
                self.Grid.maximizer['reaction_temp'],
                c='white', edgecolor='black', linewidth=1.5,
                s=200, marker='*', alpha=0.5,
                label='maximizer',
                zorder=2.6
            )
        else:
            print(f"No maximizer has been obtained for synthesis method {self.synth_method_label}")

        if plot_allowed_grid:
            plot_grid_3d(self.allowed_meshgrid1, self.allowed_meshgrid2, self.allowed_meshgrid3, ax=ax)
        if plot_train:
            df_Xtrain = self.GP.df_Xtrain[
                (self.GP.df_Xtrain['synth_method'] == self.synth_method_num) &
                # Filter out the data outside (xmin, xmax, ymin, ymax, zmin, zmax)
                (self.GP.df_Xtrain['reaction_temp'] >= zmin) &
                (self.GP.df_Xtrain['reaction_temp'] <= zmax) &
                (self.GP.df_Xtrain['Rh_weight_loading'] >= xmin) &
                (self.GP.df_Xtrain['Rh_weight_loading'] <= xmax) &
                (self.GP.df_Xtrain['Rh_total_mass'] >= ymin) &
                (self.GP.df_Xtrain['Rh_total_mass'] <= ymax)
            ]
            plot_train_data_3d(df_Xtrain, ax=ax)

        # set limits, ticks, and labels
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.tick_params(axis='x', which='major', pad=-2, labelrotation=0)
        ax.set_xticks(self.Grid.list_grids[1])
        # ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_yticks(self.Grid.list_grids[2])
        # ax.set_yticks(np.linspace(m_rh_min, m_rh_max, n_m_rh_grid))
        ax.tick_params(axis='y', which='major', pad=7, labelrotation=0)
        ax.tick_params(axis='z', which='major', pad=9)
        ax.set_xlabel('Rh weight loading (wt%)', labelpad=7, fontsize=17)
        ax.set_ylabel('Rh mass\nin reactor (mg)', labelpad=30, fontsize=17)
        ax.set_zlabel('Reaction\ntemperature ($^o$C)', labelpad=25,  fontsize=17)
        ax.set_title(f'Synthesis method: {self.synth_method_label}', fontsize=17)
        ax.set_box_aspect(None, zoom=0.85)  # to prevent z label cut off
        plt.legend(bbox_to_anchor=(1.02, 0.0), ncol=3)

        if show:
            plt.show()
        else:
            plt.savefig(f"./acq_distr_3d_synth_{self.synth_method_label}.png", bbox_inches='tight', dpi=200)
            plt.clf()  # Clear the current figure for the next temperature

        # Output JSON file with raw data for the contour plot for programmatic interaction with an LLM agent
        contour_data = {
            'acq_values': {
                'x_flattened_meshgrid': self.meshgrid1[0].flatten().tolist(),
                'y_flattened_meshgrid': self.meshgrid2[0].flatten().tolist(),
                'z_flattened_meshgrid': self.meshgrid3[0].flatten().tolist(),
                'c_flattened_meshgrid': wrapper_acq_value(
                    self.meshgrid1, self.meshgrid2, self.meshgrid3,
                    acq_func=self.Grid.acq_function,
                    synth_method=self.synth_method_num,
                    GP=self.GP
                ).flatten().tolist(),
                'xlabel': 'Rh weight loading (wt%)',
                'ylabel': 'Rh mass in reactor (mg)',
                'zlabel': 'Reaction temperature ($^o$C)',
                'clabel': self.Grid.acq_label,
            },
            'global_maximizer_condition': {
                'Rh_weight_loading': self.Grid.maximizer['Rh_weight_loading'],
                'Rh_total_mass': self.Grid.maximizer['Rh_total_mass'],
                'reaction_temp': self.Grid.maximizer['reaction_temp'],
                'synth_method': self.Grid.maximizer['synth_method'],
            },
            'training_data': {
                'Rh_weight_loading': self.GP.df_Xtrain[
                    (self.GP.df_Xtrain['synth_method'] == self.synth_method_num)
                    ]['Rh_weight_loading'].tolist(),
                'Rh_total_mass': self.GP.df_Xtrain[
                    (self.GP.df_Xtrain['synth_method'] == self.synth_method_num)
                    ]['Rh_total_mass'].tolist(),
                'reaction_temp': self.GP.df_Xtrain[
                    (self.GP.df_Xtrain['synth_method'] == self.synth_method_num)
                    ]['reaction_temp'].tolist(),
            },
            'allowed_grid': {
                'x_flattened_meshgrid': self.allowed_meshgrid1.flatten().tolist(),
                'y_flattened_meshgrid': self.allowed_meshgrid2.flatten().tolist(),
                'z_flattened_meshgrid': self.allowed_meshgrid3.flatten().tolist(),
            },
            'acquisition_function': self.Grid.acq_label,
            'synth_method': self.synth_method_label,
        }
        with open(f"./acq_distr_3d_synth_{self.synth_method_label}.json", 'w') as f:
            json.dump(contour_data, f, indent=4)


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
    """
    Plots a 2D grid of points.
    :param x_points: expected to be a numpy array made by np.meshgrid
    :param y_points: expected to be a numpy array made by np.meshgrid
    """
    # X, Y = np.meshgrid(x_points, y_points)
    plt.scatter(
        x_points, y_points,
        s=0.3, c='k', label='allowed grid',
    )

def plot_grid_3d(x_points, y_points, z_points, ax=None):
    """
    Plots a 3D grid of points.
    :param x_points: expected to be a numpy array made by np.meshgrid
    :param y_points: expected to be a numpy array made by np.meshgrid
    :param z_points: expected to be a numpy array made by np.meshgrid
    :param ax: matplotlib 3D axis object
    """
    # X, Y, Z = np.meshgrid(x_points, y_points, z_points)
    ax.scatter(
        x_points, y_points, z_points,
        s=0.3, c='k', label='allowed grid',
        zorder=2.7, alpha=0.5
    )

def plot_train_data(df_Xtrain):
    plt.scatter(
        df_Xtrain.loc[:, 'Rh_weight_loading'], df_Xtrain.loc[:, 'Rh_total_mass'],
        s=5.0, c='r', marker='D', label='train data'
    )

def plot_train_data_3d(df_Xtrain, ax=None):
    scatter = ax.scatter(
        df_Xtrain.loc[:, 'Rh_weight_loading'], df_Xtrain.loc[:, 'Rh_total_mass'], df_Xtrain.loc[:, 'reaction_temp'],
        c='red',
        label='training data',
        alpha=0.6,
        s=30,
    )
    scatter.set(zorder=2.4)