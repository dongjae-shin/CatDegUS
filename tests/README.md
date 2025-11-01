### Run example codes

* In the [`tests/`](https://github.com/dongjae-shin/CatDegUS/blob/main/tests/), run as follows:
  ``` bash
  python ./example.py
  ```
* Alternatively, in a Jupyter notebook (`example.ipynb`):
  ``` python
  import catdegus.active_learning.gaussian_process as gpc

  # Define the home directory and path to data
  # Target metric: initial CO2 conversion
  path = "./20250228_sheet_for_ML_unique.xlsx"

  # Train the Gaussian Process model
  GP = gpc.GaussianProcess()
  GP.preprocess_data_at_once(path=path,
                            target='CO2 Conversion (%)_initial value',
                            x_range_min=[300, 0.1, 0.005, 0], 
                            x_range_max=[550, 1.0, 0.02, 1])
  GP.train_gp()
  ```
### Example codes:
* Example python codes to use CatDegUS are in [`tests/`](https://github.com/dongjae-shin/CatDegUS/blob/main/tests/) directory.
* `example.ipynb` (or `example.py`): sequential uncertainty sampling for catalyst testing and the visualization of uncertainty.
* `example_HT_reactor`: batch uncertainty sampling for **high-throughput (HT) reactor** with specific 4×4 reactor architecture.