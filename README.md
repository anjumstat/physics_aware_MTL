# Complete Workflow for Crack Growth Prediction using Physics-Aware Multi-Task Learning
# 📋 Overview
This repository provides a complete end-to-end pipeline for predicting fracture mechanics parameters from experimental crack growth data. The workflow processes raw experimental data, trains a physics-aware multi-task neural network, computes baseline models for comparison, and generates publication-ready visualizations.

# Data Source: Experimental data downloaded from Materials Data Facility
https://www.materialsdatafacility.org/detail/36a0cbae-899b-4adb-84d9-28d72f786659-1.0 

# 🚀 Complete Procedure
Step 1: Download Data:
Download the experimental dataset from the Materials Data Facility website and save the CSV files to your local computer.

Step 2: Install Dependencies
pip install torch pandas numpy scikit-learn matplotlib seaborn scipy
# Step 3: Run the Pipeline
Execute the scripts in the following order:
python 01_data_processing.py      # Process raw experimental data

python 02_comprehensive_analysis.py  # Train physics-aware MTL model

python 03_Baseline_Models.py      # Compute baseline models for comparison

python 04_article_figures.py      # Generate publication-ready figures

Script Descriptions
# 01_data_processing.py - Data Preprocessing
Purpose: Converts raw experimental CSV files into standardized, machine-learning-ready datasets.

# What it does:

Extracts metadata (spacer height, mica thickness) from file headers

Reads multiple CSV files from a directory

Combines all experimental data into a single DataFrame

Handles missing values using forward/backward fill

Standardizes features using StandardScaler

Preserves experiment traceability for cross-validation

Input: Raw CSV files in E:/materials2/data

Output: Processed files saved to E:/materials2/processed_data

processed_features.csv - Standardized feature matrix

processed_targets.csv - Target variables (G and velocity)

original_combined_data.csv - Raw combined data

processing_metadata.json - Complete preprocessing metadata

processing_summary.csv - Quick reference summary

# 02_comprehensive_analysis.py - Physics-Aware Multi-Task Learning Model
Purpose: Trains a novel physics-aware neural network for simultaneous prediction of G (fracture energy) and crack velocity.

Model Architecture:

Shared feature extractor (128 → 64 → 32 dimensions)

G branch with direct crack length connection (physics-aware design)

Velocity branch using only shared features

Batch normalization and dropout for regularization

Key Features:

Direct skip connection from crack length to G prediction branch

Dynamic task weighting based on loss ratios

Variance-normalized loss calculation

Gradient clipping and learning rate scheduling

Early stopping with patience of 30 epochs

Training Configuration:

Data split: 70% training, 15% validation, 15% test

Batch size: 32

Maximum epochs: 150

Initial learning rate: 0.001

Optimizer: Adam with weight decay

Input: Processed data from 01_data_processing.py

Output: Results saved to E:/materials2/physics_aware_mtl_results2

best_model_*.pth - Best model checkpoint

training_history.csv - Loss and R² history

test_predictions.csv - Predictions vs true values

test_metrics.json - Performance metrics

experiment_config.json - Experiment configuration

# 03_Baseline_Models.py - Baseline Model Comparison
Purpose: Computes baseline machine learning models for fair comparison with the physics-aware MTL model.

Implemented Models:

Linear Regression

Ridge Regression (L2 regularization)

Random Forest (ensemble of 100 trees)

Key Features:

Uses EXACT same data splitting as the MTL model (random_split with seed 42)

Proper feature scaling for linear models

Random Forest uses raw features (no scaling needed)

Comprehensive metrics including R², RMSE, MAE, MSE, and Max Error

Input: Processed data from 01_data_processing.py

Output: Results saved to E:/materials2/baseline_results

baseline_results.json - Complete metrics for all models

baseline_results.csv - Tabular results

baseline_comparison_table.tex - LaTeX table for papers

baseline_summary.txt - Text summary

predictions/ - Individual model predictions

# 04_article_figures.py - Publication-Ready Visualization
Purpose: Generates professional, publication-quality figures and statistical tables for reporting results in scientific papers.

Generated Figures:

Figure	Description
Figure 1	Prediction scatter plots (True vs. Predicted) with R² annotations
Figure 2	Comprehensive residual analysis (residuals vs predictions, histograms with normal fits, Q-Q plots)
Figure 3	Training history (loss curves, R² progression, learning rate schedule)
Figure 4	Model comparison bar chart (Linear Regression, Ridge, Random Forest, MTL)
Generated Tables:

Table1_model_comparison.csv - Performance comparison of all models

Table1_latex.txt - LaTeX-ready table for manuscripts

Table2_detailed_results.csv - Detailed metrics for the MTL model

publication_summary.txt - Key statistics for reporting

Publication Style:

Times New Roman fonts

Colorblind-friendly color palette

300 DPI resolution for raster images

Vector PDF output for scalable figures

Input:

MTL results from 02_comprehensive_analysis.py

Baseline results from 03_Baseline_Models.py

Output: Figures and tables saved to E:/materials2/publication_figures1

📁 Repository Structure
text
├── 01_data_processing.py          # Data preprocessing pipeline
├── 02_comprehensive_analysis.py   # Physics-aware MTL model training
├── 03_Baseline_Models.py          # Baseline model comparison
├── 04_article_figures.py          # Publication-ready visualization
└── README.md                      # This file
🔧 Customization
Modify Data Paths
Update the directory paths in each script:

python
DATA_DIR = "your/data/path"
OUTPUT_DIR = "your/output/path"
Adjust Model Architecture
Edit the PhysicsAwareMTL class in 02_comprehensive_analysis.py:

Change hidden layer sizes

Modify dropout rates

Adjust the crack length index if needed

Modify Training Parameters
Update training configuration in 02_comprehensive_analysis.py:

Batch size

Learning rate

Number of epochs

Early stopping patience

# 📦 Dependencies
text
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0


