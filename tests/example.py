import catdegus.active_learning.acquisition as aq
import catdegus.active_learning.gaussian_process as gpc
import catdegus.visualization.plot as pl

# Define the path to data
# Target metric: initial CO2 conversion
path = './20250228_sheet_for_ML_unique.xlsx'

# Train the Gaussian Process model
GP = gpc.GaussianProcess()
GP.preprocess_data_at_once(path=path,
                           target='CO2 Conversion (%)_initial value',
                           x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1])
GP.train_gp()

# Construct the discrete grid for optimization
Grid = aq.DiscreteGrid(
    GP=GP,
    x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1], x_step=[50, 0.1, 0.0025, 1]
)
Grid.construct_grid()

Grid.optimize_posterior_std_dev_discrete(synth_method='NP', n_candidates=5)
# Grid.optimize_upper_confidence_bound_discrete(synth_method_num='NP', n_candidates=5)

# Plot the acquisition function
Plot = pl.Plotter(GP=GP, Grid=Grid)

Plot.plot_2d_acquisition_function(
    synth_method='NP',
    acq_max=1.1,
    n_levels=32,
    temperature_list=[300, 350, 400, 450, 500, 550],
    mode='custom', #'boundary',
    custom_range=(0.0, 6.0, 0.0, 0.05),  # Custom range for contour plot
    contour_resolution=50,
    plot_allowed_grid=True,
    plot_train=True
)

Plot.plot_3d_acquisition_function(
    synth_method='NP',
    acq_max=1.1,
    mode='boundary', #'custom',
    custom_range=(0.0, 6.0, 0.0, 0.05, 300, 550),  # Custom range for contour plot
    contour_resolution=20,
    plot_allowed_grid=True,
    plot_train=True
)
