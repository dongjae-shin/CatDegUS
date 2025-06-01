# CatDegUS
Python module for **Cat**alysts' **Deg**radation navigated by **U**ncertainty **S**ampling

## Getting started
### 1. Make a virtual environment (e.g., when using `conda`):
```
conda create -n catdegus python=3.13
conda activate catdegus
```
### 2. Installation
* **Directly install using pip (the simplest way)**
  ```
  pip install git+https://github.com/dongjae-shin/CatDegUS.git
  ```
* cf. Alternative way: clone repository & install using pip
  ```
  git clone https://github.com/dongjae-shin/CatDegUS.git
  cd CatDegUS
  pip install .
  ```
* cf. You can also install all the requirements by:
  ```
  pip install -r requirements.txt
  ```

### 3. Run the example code
* Example python codes (`example.py`,`example.ipynb`) to use CatDegUS are in [`tests/`](https://github.com/dongjae-shin/CatDegUS/blob/main/tests/) directory.
* In the `tests/`, run as follows:
  ```
  python ./example.py
  ```
* Alternatively, in a Jupyter notebook (`example.ipynb`):
  ```
  import catdegus.active_learning.gaussian_process as gpc

  # Define the home directory and path to data
  # Target metric: initial CO2 conversion
  path = "./20250228_sheet_for_ML_unique.xlsx"

  # Train the Gaussian Process model
  GP = gpc.GaussianProcess()
  GP.preprocess_data_at_once(path=path,
                            target='CO2 Conversion (%)_initial value',
                            x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1])
  GP.train_gp()
  ```

## Requirements
* Required modules: `pandas`, `torch`, `botorch`, `matplotlib`, `openpyxl`
* All specified in `setup.py`

## Supported acquisition functions
* Posterior Standard Deviation: used for uncertainty sampling (US)
* Posterior Mean
* Upper Confidence Bound (UCB)
* Expected Improvement (EI): to be added.

## Input to the code
* A data file (`*.excel`): [example](https://github.com/dongjae-shin/CatDegUS/blob/main/tests/20250228_sheet_for_ML_unique.xlsx)

## Output from the code
* Maximizer condition for posterior standard deviation: US-guided experimental condition
  <div align="center">
    <img src="./imgs/maximizer.png" alt="img" width="500">
  </div>
* Maximizer condition for other supported acquisition functions
* 2D visualization of a selected acquisition function for a selected synthesis method and temperature
  <div align="center">
    <img src="./imgs/2d_plot.png" alt="img" width="500">
  </div>
* 3D visualization of a selected acquisition function for a selected synthesis method
  <div align="center">
    <img src="./imgs/3d_plot.png" alt="img" width="500">
  </div>