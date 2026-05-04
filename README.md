# Physics-Aware Multi-Task Learning for Crack Growth Prediction

## Overview

This repository contains complete Python code for predicting fracture mechanics parameters (strain energy release rate G and crack velocity) using a Physics-Aware Multi-Task Learning (PA-MTL) framework. The code processes experimental crack growth data, trains baseline models, implements a novel physics-aware neural network, and generates publication-ready figures and statistical analyses.

## Data Source

The experimental data used in this study is available from the Materials Data Facility:
https://www.materialsdatafacility.org/detail/36a0cbae-899b-4adb-84d9-28d72f786659-1.0 also you can use doi. https://doi.org/10.18126/faxs-ga32 

## Repository Structure

01_data_processing.py          # Preprocesses raw CSV files
02_baseline_models.py          # Linear, Ridge, Random Forest baselines with 5-fold CV
03_comprehensive_analysis.py   # PA-MTL training and ensemble model
04_enhanced_analysis.py        # SHAP analysis, ablation study, CV plots
05_article_figures.py          # Publication-ready figures and tables
06_statistical_tests.py        # Friedman, Nemenyi, Wilcoxon tests

## Script Descriptions

### 01_data_processing.py
Parses experimental CSV files with special header format containing metadata (spacer height, mica thickness). Extracts features, handles missing values, standardizes data, and saves processed files for machine learning.

Input: Raw CSV files in specified directory
Output: processed_features.csv, processed_targets.csv, original_combined_data.csv

### 02_baseline_models.py
Trains Linear Regression, Ridge Regression, and Random Forest models. Performs 5-fold cross-validation and saves results for statistical comparison.

Output: baseline_results/ (CV matrices, predictions, LaTeX tables)

### 03_comprehensive_analysis.py
Implements the Physics-Aware Multi-Task Learning architecture with shared encoder, direct crack length bypass connection to G branch, and dynamic task weighting. Trains ensemble model combining PA-MTL with Random Forest. Saves training history for visualization.

Output: optimized_mtl_results/ (model weights, predictions, training_history.csv)

### 04_enhanced_analysis.py
Generates SHAP feature importance analysis, ablation study validating each architectural component, and cross-validation box plots.

Output: enhanced_analysis_results/ (SHAP plots, ablation bars, CV box plots)

### 05_article_figures.py
Creates publication-ready figures including prediction scatter plots, residual analysis, training history curves, and model comparison bar charts. Generates LaTeX tables for manuscripts.

Output: publication_figures_optimized/ (PNG, PDF, CSV tables)

### 06_statistical_tests.py
Performs Friedman test for overall significance, Nemenyi post-hoc test for pairwise comparisons, and Wilcoxon signed-rank tests. Generates Critical Difference diagrams for visual interpretation.

Output: statistical_test_results/ (CD diagrams, summary tables, report)

## Installation

pip install torch pandas numpy scikit-learn matplotlib seaborn shap scipy

## Execution Order

python 01_data_processing.py
python 02_baseline_models.py
python 03_comprehensive_analysis.py
python 04_enhanced_analysis.py
python 05_article_figures.py
python 06_statistical_tests.py

## Dependencies

torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.40.0
scipy>=1.7.0

## License

MIT License

## Citation

If you use this code, please cite:
https://github.com/anjumstat/physics_aware_MTL

## Contact

For questions or issues, please open an issue on GitHub.
