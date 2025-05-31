# CatDegUS
Python module for **Cat**alysts' **Deg**radation navigated by **U**ncertainty **S**ampling

## How to use
### 1. Make a virtual environment (e.g., when using `conda`):
```
conda create -n catdegus python=3.13
conda activate catdegus
```
### 2. Clone the repository
```
git clone https://github.com/dongjae-shin/CatDegUS.git
cd CatDegUS/tests/
```

### 3. Install requirements
* Required modules: `pandas`, `torch`, `botorch`, `matplotlib`, `openpyxl` (as of May 30, 2025)
  ```
  pip install pandas torch botorch matplotlib openpyxl
  ```
* Alternatively, using [`requirements.txt`](https://github.com/dongjae-shin/CatDegUS/blob/main/requirements.txt) (to be fixed)
  ```
  pip install -r requirements.txt
  ```
  
### 4. Run the example code
* Example python codes (`*.py`,`*.ipynb`) to use CatDegUS are in [`tests/`](https://github.com/dongjae-shin/CatDegUS/blob/main/tests/) directory.
  ```
  python ./main.py
  ```
* Or, you can run the example Jupyter notebook (`*.ipynb`) in the same directory.

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
    <img src="./imgs/maximizer.png" alt="img">
  </div>
* Maximizer condition for other supported acquisition functions
* 2D visualization of a selected acquisition function for a selected synthesis method and temperature
  <div align="center">
    <img src="./imgs/2d_plot.png" alt="img">
  </div>
* 3D visualization of a selected acquisition function for a selected synthesis method
  <div align="center">
    <img src="./imgs/3d_plot.png" alt="img">
  </div>

  
