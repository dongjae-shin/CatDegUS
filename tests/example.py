import catdegus.active_learning.acquisition as aq
import catdegus.active_learning.gaussian_process as gpc
import catdegus.visualization.plot as pl

# Define inputs to code --------
path_to_data = './20250228_sheet_for_ML_unique.xlsx'
target_metric = 'CO2 Conversion (%)_initial value'
x_range_min = [300, 0.1, 0.005, 0]                  # Lower boundaries for Temperature, Rh weight loading (wt%), Rh total mass (mg), Synthesis method
x_range_max = [550, 1.0, 0.02, 1]                   # Upper boundaries for Temperature, Rh weight loading (wt%), Rh total mass (mg), Synthesis method
x_step = [50, 0.1, 0.0025, 1]                       # Step sizes for each feature
n_candidates = 5                                    # Code will suggest top n_candidates informative conditions
temps_to_plot_2d = [300, 350, 400, 450, 500, 550]   # Temperatures to plot in the 2d acquisition function
synth_method_to_plot = 'NP'                         # Synthesis method to plot the 2d/3d acquisition function for
# -----------------------------

# Train the Gaussian Process model
GP = gpc.GaussianProcess()
GP.preprocess_data_at_once(path=path_to_data,
                           target=target_metric,
                           x_range_min=x_range_min, x_range_max=x_range_max)
GP.train_gp()

# Construct the discrete grid for optimization
Grid = aq.DiscreteGrid(
    GP=GP,
    x_range_min=x_range_min, x_range_max=x_range_max, x_step=x_step
)
Grid.construct_grid()

Grid.optimize_posterior_std_dev_discrete(synth_method=synth_method_to_plot, n_candidates=5)
# Grid.optimize_upper_confidence_bound_discrete(synth_method_num='NP', n_candidates=5)

# Plot the 2D/3D acquisition function
Plot = pl.Plotter(GP=GP, Grid=Grid)

Plot.plot_2d_acquisition_function(
    synth_method=synth_method_to_plot,
    acq_max=1.1,
    n_levels=32,
    temperature_list=temps_to_plot_2d,
    mode='custom', #'boundary',
    custom_range=(0.0, 6.0, 0.0, 0.05),  # Custom range for contour plot
    contour_resolution=50,
    plot_allowed_grid=True,
    plot_train=True,
    show=False
)

Plot.plot_3d_acquisition_function(
    synth_method=synth_method_to_plot,
    acq_max=1.1,
    mode='boundary', #'custom',
    custom_range=(0.0, 6.0, 0.0, 0.05, 300, 550),  # Custom range for contour plot
    contour_resolution=20,
    plot_allowed_grid=True,
    plot_train=True,
    show=False
)
