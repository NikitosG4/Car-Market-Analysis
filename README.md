# Car Market Analysis

This repository contains a complete workflow for analyzing and modeling used car prices based on a real-world dataset of vehicle listings. The project covers data cleaning, exploratory analysis, feature engineering, and the application of several regression models to predict car prices.

## Project Highlights
- Cleans and preprocesses raw vehicle data
- Visualizes key trends and relationships in the data
- Encodes categorical features for modeling
- Implements and compares multiple regression models:
  - Linear Regression
  - Random Forest
  - XGBoost
  - K-Nearest Neighbors
  - Ridge and Lasso Regression
  - Neural Network (MLPRegressor)
- Evaluates models using R², MAE, and RMSE
- Shows feature importances and coefficients
- Includes clear visualizations for both EDA and model results

## Requirements
- Python 3.11 or newer
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Getting Started
1. Clone this repository and enter the project folder:
   ```sh
   git clone <repo-url>
   cd Car-Market-Analysis
   ```
2. (Recommended) Set up a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
   Or, if you prefer, install them individually:
   ```sh
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
4. If you're on an Apple Silicon Mac and using XGBoost, you may need OpenMP:
   ```sh
   brew install libomp
   ```

## How to Run
1. Make sure `vehicles.csv` is in the project directory.
2. Run the analysis script:
   ```sh
   python vehicles.py
   ```
   Or use the "Run" button in VS Code, ensuring the correct Python interpreter is selected.

## What You'll See
- Printed metrics for each model (R², MAE, RMSE)
- Plots showing data distributions, relationships, and model results
- Feature importance and coefficient summaries

## Tips
- The script is set up to use all available CPU cores for supported models, making it efficient on Apple Silicon (M1/M2) Macs.
- For best performance, consider using a Conda environment with ARM-optimized packages.

## License
MIT License