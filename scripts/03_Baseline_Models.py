# -*- coding: utf-8 -*-
"""
Created on Sat Apr 4 2026
@author: H.A.R

Baseline Models Comparison for Crack Growth Prediction
Computes Linear Regression, Ridge Regression, and Random Forest baselines
Uses EXACT same data splitting as the MTL model for fair comparison
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import os
from datetime import datetime
import torch
from torch.utils.data import random_split
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_DIR = "E:/materials2/processed_data"
OUTPUT_DIR = "E:/materials2/baseline_results"
RANDOM_STATE = 42

# ==================== HELPER FUNCTIONS ====================
def ensure_directory_exists(dir_path):
    """Ensure output directory exists"""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'MSE': float(mean_squared_error(y_true, y_pred)),
        'Max_Error': float(np.max(np.abs(y_true - y_pred)))
    }

def save_results(results_dict, output_dir):
    """Save all baseline results to files"""
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'baseline_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"   ✅ Results saved to: {json_path}")
    
    # Save as CSV for easy viewing
    rows = []
    for model_name, tasks in results_dict.items():
        for task_name, metrics in tasks.items():
            row = {'Model': model_name, 'Task': task_name}
            row.update(metrics)
            rows.append(row)
    
    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'baseline_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"   ✅ CSV saved to: {csv_path}")
    
    return df_results

def create_comparison_table(results_dict):
    """Create a formatted comparison table"""
    
    print("\n" + "="*80)
    print("BASELINE MODELS PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create table data
    table_data = []
    
    for model_name, tasks in results_dict.items():
        for task_name, metrics in tasks.items():
            table_data.append({
                'Model': model_name,
                'Task': task_name,
                'R²': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE']
            })
    
    df_table = pd.DataFrame(table_data)
    
    # Pivot for better display
    pivot_r2 = df_table.pivot(index='Model', columns='Task', values='R²')
    pivot_rmse = df_table.pivot(index='Model', columns='Task', values='RMSE')
    pivot_mae = df_table.pivot(index='Model', columns='Task', values='MAE')
    
    print("\n📊 R² SCORES:")
    print(pivot_r2.round(6))
    
    print("\n📊 RMSE VALUES:")
    print(pivot_rmse.round(4))
    
    print("\n📊 MAE VALUES:")
    print(pivot_mae.round(4))
    
    return pivot_r2, pivot_rmse, pivot_mae

def generate_latex_table(results_dict, output_dir):
    """Generate LaTeX table for paper submission"""
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\caption{Performance comparison of baseline models for crack growth prediction.}")
    latex_lines.append(r"\label{tab:baseline_comparison}")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\begin{tabular}{lcccccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{G (J/m\textsuperscript{2})} & \multicolumn{3}{c}{Crack Velocity ($\mu$m/s)} \\")
    latex_lines.append(r"\cline{2-7}")
    latex_lines.append(r" & R\textsuperscript{2} & RMSE & MAE & R\textsuperscript{2} & RMSE & MAE \\")
    latex_lines.append(r"\hline")
    
    for model_name in ['Linear Regression', 'Ridge Regression', 'Random Forest']:
        if model_name in results_dict:
            g_metrics = results_dict[model_name].get('G (J/m^2)', {})
            v_metrics = results_dict[model_name].get('Crack velocity (um/s)', {})
            
            g_r2 = f"{g_metrics.get('R2', 0):.4f}"
            g_rmse = f"{g_metrics.get('RMSE', 0):.4f}"
            g_mae = f"{g_metrics.get('MAE', 0):.4f}"
            v_r2 = f"{v_metrics.get('R2', 0):.4f}"
            v_rmse = f"{v_metrics.get('RMSE', 0):.2f}"
            v_mae = f"{v_metrics.get('MAE', 0):.2f}"
            
            latex_lines.append(f"{model_name} & {g_r2} & {g_rmse} & {g_mae} & {v_r2} & {v_rmse} & {v_mae} \\\\")
    
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    # Save LaTeX table
    latex_path = os.path.join(output_dir, 'baseline_comparison_table.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    print(f"   ✅ LaTeX table saved to: {latex_path}")
    
    return latex_content

# ==================== MAIN EXECUTION ====================
def main():
    print("="*80)
    print("BASELINE MODELS FOR CRACK GROWTH PREDICTION")
    print("="*80)
    
    # Create output directory
    output_dir = ensure_directory_exists(OUTPUT_DIR)
    
    # Load processed data
    features_path = os.path.join(DATA_DIR, "processed_features.csv")
    targets_path = os.path.join(DATA_DIR, "processed_targets.csv")
    
    print(f"\n📂 Loading data from:")
    print(f"   Features: {features_path}")
    print(f"   Targets: {targets_path}")
    
    # Check if files exist
    if not os.path.exists(features_path):
        print(f"\n❌ Error: Features file not found at {features_path}")
        print("   Please run 01_data_processing.py first.")
        return
    
    if not os.path.exists(targets_path):
        print(f"\n❌ Error: Targets file not found at {targets_path}")
        print("   Please run 01_data_processing.py first.")
        return
    
    # Load data
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    
    # Remove Source_File column if present
    if 'Source_File' in features.columns:
        X = features.drop('Source_File', axis=1).values
        print(f"\n   Removed 'Source_File' column from features")
    else:
        X = features.values
    
    y_G = targets['G (J/m^2)'].values
    y_V = targets['Crack velocity (um/s)'].values
    
    print(f"\n📊 Dataset Information:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target 1 (G): range [{y_G.min():.4f}, {y_G.max():.4f}], mean={y_G.mean():.4f}")
    print(f"   Target 2 (Velocity): range [{y_V.min():.2f}, {y_V.max():.2f}], mean={y_V.mean():.2f}")
    
    # ==================== EXACT SAME SPLIT AS MTL MODEL ====================
    # This matches the random_split approach in your MTL code
    print(f"\n🔪 Splitting data (Training: 70%, Validation: 15%, Test: 15%)...")
    print("   Using EXACT same random seed (42) as MTL model for fair comparison")
    
    # Convert to torch tensor for consistent splitting
    X_tensor = torch.arange(len(X))
    dataset = list(zip(X_tensor, y_G, y_V))
    
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    # Use same random_split as MTL model
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Extract indices
    train_indices = [idx for idx, _, _ in train_dataset]
    val_indices = [idx for idx, _, _ in val_dataset]
    test_indices = [idx for idx, _, _ in test_dataset]
    
    # Create splits
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    
    yG_train = y_G[train_indices]
    yG_val = y_G[val_indices]
    yG_test = y_G[test_indices]
    
    yV_train = y_V[train_indices]
    yV_val = y_V[val_indices]
    yV_test = y_V[test_indices]
    
    print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Standardize features (for linear and ridge regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ==================== TRAIN BASELINE MODELS ====================
    results = {}
    
    # Store trained models in a dictionary for later prediction saving
    trained_models = {}
    
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS")
    print("="*80)
    
    # 1. Linear Regression
    print("\n📈 Training Linear Regression...")
    lr_G = LinearRegression()
    lr_V = LinearRegression()
    
    lr_G.fit(X_train_scaled, yG_train)
    lr_V.fit(X_train_scaled, yV_train)
    
    trained_models['Linear Regression'] = {'G': lr_G, 'V': lr_V, 'scaled': True}
    
    yG_pred = lr_G.predict(X_test_scaled)
    yV_pred = lr_V.predict(X_test_scaled)
    
    results['Linear Regression'] = {
        'G (J/m^2)': calculate_metrics(yG_test, yG_pred),
        'Crack velocity (um/s)': calculate_metrics(yV_test, yV_pred)
    }
    print(f"   G R²: {results['Linear Regression']['G (J/m^2)']['R2']:.6f}")
    print(f"   Velocity R²: {results['Linear Regression']['Crack velocity (um/s)']['R2']:.6f}")
    
    # 2. Ridge Regression
    print("\n📈 Training Ridge Regression...")
    ridge_G = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge_V = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    
    ridge_G.fit(X_train_scaled, yG_train)
    ridge_V.fit(X_train_scaled, yV_train)
    
    trained_models['Ridge Regression'] = {'G': ridge_G, 'V': ridge_V, 'scaled': True}
    
    yG_pred = ridge_G.predict(X_test_scaled)
    yV_pred = ridge_V.predict(X_test_scaled)
    
    results['Ridge Regression'] = {
        'G (J/m^2)': calculate_metrics(yG_test, yG_pred),
        'Crack velocity (um/s)': calculate_metrics(yV_test, yV_pred)
    }
    print(f"   G R²: {results['Ridge Regression']['G (J/m^2)']['R2']:.6f}")
    print(f"   Velocity R²: {results['Ridge Regression']['Crack velocity (um/s)']['R2']:.6f}")
    
    # 3. Random Forest
    print("\n📈 Training Random Forest...")
    rf_G = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf_V = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    
    rf_G.fit(X_train, yG_train)  # Random Forest doesn't need scaling
    rf_V.fit(X_train, yV_train)
    
    trained_models['Random Forest'] = {'G': rf_G, 'V': rf_V, 'scaled': False}
    
    yG_pred = rf_G.predict(X_test)
    yV_pred = rf_V.predict(X_test)
    
    results['Random Forest'] = {
        'G (J/m^2)': calculate_metrics(yG_test, yG_pred),
        'Crack velocity (um/s)': calculate_metrics(yV_test, yV_pred)
    }
    print(f"   G R²: {results['Random Forest']['G (J/m^2)']['R2']:.6f}")
    print(f"   Velocity R²: {results['Random Forest']['Crack velocity (um/s)']['R2']:.6f}")
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save all results
    df_results = save_results(results, output_dir)
    
    # Create comparison table
    pivot_r2, pivot_rmse, pivot_mae = create_comparison_table(results)
    
    # Generate LaTeX table
    generate_latex_table(results, output_dir)
    
    # Save predictions for each model
    print("\n📁 Saving individual model predictions...")
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    for model_name, model_dict in trained_models.items():
        # Get the appropriate test data (scaled or unscaled)
        if model_dict['scaled']:
            X_test_data = X_test_scaled
        else:
            X_test_data = X_test
        
        # Make predictions
        preds_G = model_dict['G'].predict(X_test_data)
        preds_V = model_dict['V'].predict(X_test_data)
        
        # Create DataFrame
        pred_df = pd.DataFrame({
            'True_G': yG_test,
            'Pred_G': preds_G,
            'True_Velocity': yV_test,
            'Pred_Velocity': preds_V
        })
        
        # Save to CSV (replace spaces with underscores)
        safe_filename = model_name.replace(' ', '_')
        pred_path = os.path.join(predictions_dir, f'{safe_filename}_predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"   ✅ Saved {model_name} predictions")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("BASELINE MODELS - SUMMARY STATISTICS")
    print("="*80)
    
    # Find best model for each task
    best_g_model = max(results.keys(), key=lambda x: results[x]['G (J/m^2)']['R2'])
    best_v_model = max(results.keys(), key=lambda x: results[x]['Crack velocity (um/s)']['R2'])
    
    print(f"\n🏆 BEST PERFORMING MODELS:")
    print(f"   For G (J/m²): {best_g_model} (R² = {results[best_g_model]['G (J/m^2)']['R2']:.6f})")
    print(f"   For Velocity: {best_v_model} (R² = {results[best_v_model]['Crack velocity (um/s)']['R2']:.6f})")
    
    print(f"\n📊 PERFORMANCE RANKING FOR G:")
    g_ranking = sorted(results.items(), key=lambda x: x[1]['G (J/m^2)']['R2'], reverse=True)
    for i, (model, metrics) in enumerate(g_ranking, 1):
        print(f"   {i}. {model}: R² = {metrics['G (J/m^2)']['R2']:.6f}, RMSE = {metrics['G (J/m^2)']['RMSE']:.6f}")
    
    print(f"\n📊 PERFORMANCE RANKING FOR VELOCITY:")
    v_ranking = sorted(results.items(), key=lambda x: x[1]['Crack velocity (um/s)']['R2'], reverse=True)
    for i, (model, metrics) in enumerate(v_ranking, 1):
        print(f"   {i}. {model}: R² = {metrics['Crack velocity (um/s)']['R2']:.6f}, RMSE = {metrics['Crack velocity (um/s)']['RMSE']:.2f}")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'baseline_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("BASELINE MODELS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Data split (same as MTL model):\n")
        f.write(f"  Training: {len(X_train)} samples\n")
        f.write(f"  Validation: {len(X_val)} samples\n")
        f.write(f"  Test: {len(X_test)} samples\n\n")
        f.write(f"Best model for G: {best_g_model}\n")
        f.write(f"  R² = {results[best_g_model]['G (J/m^2)']['R2']:.6f}\n")
        f.write(f"  RMSE = {results[best_g_model]['G (J/m^2)']['RMSE']:.6f}\n\n")
        f.write(f"Best model for Velocity: {best_v_model}\n")
        f.write(f"  R² = {results[best_v_model]['Crack velocity (um/s)']['R2']:.6f}\n")
        f.write(f"  RMSE = {results[best_v_model]['Crack velocity (um/s)']['RMSE']:.2f}\n\n")
        f.write("COMPLETE RESULTS:\n")
        f.write(df_results.to_string())
    
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # Print comparison with MTL model (if available)
    mtl_results_path = "E:/materials2/physics_aware_mtl_results2/test_metrics.json"
    if os.path.exists(mtl_results_path):
        print("\n" + "="*80)
        print("COMPARISON WITH PHYSICS-AWARE MTL MODEL")
        print("="*80)
        
        with open(mtl_results_path, 'r') as f:
            mtl_metrics = json.load(f)
        
        print("\n📊 R² COMPARISON:")
        print(f"   {'Model':<25} {'G R²':<12} {'Velocity R²':<12}")
        print(f"   {'-'*50}")
        print(f"   {'Linear Regression':<25} {results['Linear Regression']['G (J/m^2)']['R2']:<12.6f} {results['Linear Regression']['Crack velocity (um/s)']['R2']:<12.6f}")
        print(f"   {'Ridge Regression':<25} {results['Ridge Regression']['G (J/m^2)']['R2']:<12.6f} {results['Ridge Regression']['Crack velocity (um/s)']['R2']:<12.6f}")
        print(f"   {'Random Forest':<25} {results['Random Forest']['G (J/m^2)']['R2']:<12.6f} {results['Random Forest']['Crack velocity (um/s)']['R2']:<12.6f}")
        print(f"   {'Physics-Aware MTL (Ours)':<25} {mtl_metrics['G (J/m^2)']['R2']:<12.6f} {mtl_metrics['Crack velocity (um/s)']['R2']:<12.6f}")
    
    print("\n" + "="*80)
    print("✅ BASELINE MODELS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📁 All results saved to: {output_dir}")
    print("\nFiles created:")
    print("  - baseline_results.json (Complete metrics)")
    print("  - baseline_results.csv (Tabular format)")
    print("  - baseline_comparison_table.tex (LaTeX table)")
    print("  - baseline_summary.txt (Text summary)")
    print("  - predictions/ (Individual model predictions)")
    
    return results, output_dir

if __name__ == "__main__":
    results, output_dir = main()