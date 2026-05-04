# -*- coding: utf-8 -*-
"""
Created on Sat Apr 4 2026
@author: H.A.R

Baseline Models Comparison for Crack Growth Prediction
Computes Linear Regression, Ridge Regression, and Random Forest baselines
WITH 5-FOLD CROSS-VALIDATION FOR STATISTICAL TESTS
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_DIR = "E:/materials2/RAA/processed_data"
OUTPUT_DIR = "E:/materials2/RAA/baseline_results"
RANDOM_STATE = 42
N_FOLDS = 5

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


def run_5fold_cross_validation(X, y, output_dir):
    """Run 5-fold cross-validation for all baseline models"""
    
    print("\n" + "="*80)
    print(f"📊 RUNNING {N_FOLDS}-FOLD CROSS-VALIDATION FOR BASELINE MODELS")
    print("="*80)
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Initialize storage for results
    cv_results = {
        'Linear Regression': {'G_R2': [], 'V_R2': [], 'G_RMSE': [], 'V_RMSE': [], 'G_MAE': [], 'V_MAE': []},
        'Ridge Regression': {'G_R2': [], 'V_R2': [], 'G_RMSE': [], 'V_RMSE': [], 'G_MAE': [], 'V_MAE': []},
        'Random Forest': {'G_R2': [], 'V_R2': [], 'G_RMSE': [], 'V_RMSE': [], 'G_MAE': [], 'V_MAE': []}
    }
    
    fold_details = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}/{N_FOLDS}")
        print(f"{'='*50}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 1. Linear Regression
        print("\n   Training Linear Regression...")
        lr_G = LinearRegression()
        lr_V = LinearRegression()
        lr_G.fit(X_train_scaled, y_train[:, 0])
        lr_V.fit(X_train_scaled, y_train[:, 1])
        
        pred_G = lr_G.predict(X_val_scaled)
        pred_V = lr_V.predict(X_val_scaled)
        
        lr_g_r2 = r2_score(y_val[:, 0], pred_G)
        lr_v_r2 = r2_score(y_val[:, 1], pred_V)
        lr_g_rmse = np.sqrt(mean_squared_error(y_val[:, 0], pred_G))
        lr_v_rmse = np.sqrt(mean_squared_error(y_val[:, 1], pred_V))
        lr_g_mae = mean_absolute_error(y_val[:, 0], pred_G)
        lr_v_mae = mean_absolute_error(y_val[:, 1], pred_V)
        
        cv_results['Linear Regression']['G_R2'].append(lr_g_r2)
        cv_results['Linear Regression']['V_R2'].append(lr_v_r2)
        cv_results['Linear Regression']['G_RMSE'].append(lr_g_rmse)
        cv_results['Linear Regression']['V_RMSE'].append(lr_v_rmse)
        cv_results['Linear Regression']['G_MAE'].append(lr_g_mae)
        cv_results['Linear Regression']['V_MAE'].append(lr_v_mae)
        print(f"      G R² = {lr_g_r2:.4f}, V R² = {lr_v_r2:.4f}")
        
        # 2. Ridge Regression
        print("\n   Training Ridge Regression...")
        ridge_G = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        ridge_V = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        ridge_G.fit(X_train_scaled, y_train[:, 0])
        ridge_V.fit(X_train_scaled, y_train[:, 1])
        
        pred_G = ridge_G.predict(X_val_scaled)
        pred_V = ridge_V.predict(X_val_scaled)
        
        ridge_g_r2 = r2_score(y_val[:, 0], pred_G)
        ridge_v_r2 = r2_score(y_val[:, 1], pred_V)
        ridge_g_rmse = np.sqrt(mean_squared_error(y_val[:, 0], pred_G))
        ridge_v_rmse = np.sqrt(mean_squared_error(y_val[:, 1], pred_V))
        ridge_g_mae = mean_absolute_error(y_val[:, 0], pred_G)
        ridge_v_mae = mean_absolute_error(y_val[:, 1], pred_V)
        
        cv_results['Ridge Regression']['G_R2'].append(ridge_g_r2)
        cv_results['Ridge Regression']['V_R2'].append(ridge_v_r2)
        cv_results['Ridge Regression']['G_RMSE'].append(ridge_g_rmse)
        cv_results['Ridge Regression']['V_RMSE'].append(ridge_v_rmse)
        cv_results['Ridge Regression']['G_MAE'].append(ridge_g_mae)
        cv_results['Ridge Regression']['V_MAE'].append(ridge_v_mae)
        print(f"      G R² = {ridge_g_r2:.4f}, V R² = {ridge_v_r2:.4f}")
        
        # 3. Random Forest
        print("\n   Training Random Forest...")
        rf_G = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
        rf_V = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
        rf_G.fit(X_train, y_train[:, 0])
        rf_V.fit(X_train, y_train[:, 1])
        
        pred_G = rf_G.predict(X_val)
        pred_V = rf_V.predict(X_val)
        
        rf_g_r2 = r2_score(y_val[:, 0], pred_G)
        rf_v_r2 = r2_score(y_val[:, 1], pred_V)
        rf_g_rmse = np.sqrt(mean_squared_error(y_val[:, 0], pred_G))
        rf_v_rmse = np.sqrt(mean_squared_error(y_val[:, 1], pred_V))
        rf_g_mae = mean_absolute_error(y_val[:, 0], pred_G)
        rf_v_mae = mean_absolute_error(y_val[:, 1], pred_V)
        
        cv_results['Random Forest']['G_R2'].append(rf_g_r2)
        cv_results['Random Forest']['V_R2'].append(rf_v_r2)
        cv_results['Random Forest']['G_RMSE'].append(rf_g_rmse)
        cv_results['Random Forest']['V_RMSE'].append(rf_v_rmse)
        cv_results['Random Forest']['G_MAE'].append(rf_g_mae)
        cv_results['Random Forest']['V_MAE'].append(rf_v_mae)
        print(f"      G R² = {rf_g_r2:.4f}, V R² = {rf_v_r2:.4f}")
        
        # Store fold details
        fold_details.append({
            'Fold': fold,
            'Linear_G_R2': lr_g_r2, 'Linear_V_R2': lr_v_r2,
            'Linear_G_RMSE': lr_g_rmse, 'Linear_V_RMSE': lr_v_rmse,
            'Ridge_G_R2': ridge_g_r2, 'Ridge_V_R2': ridge_v_r2,
            'Ridge_G_RMSE': ridge_g_rmse, 'Ridge_V_RMSE': ridge_v_rmse,
            'RF_G_R2': rf_g_r2, 'RF_V_R2': rf_v_r2,
            'RF_G_RMSE': rf_g_rmse, 'RF_V_RMSE': rf_v_rmse
        })
    
    # Save cross-validation results
    cv_results_df = pd.DataFrame(fold_details)
    cv_results_df.to_csv(os.path.join(output_dir, 'baseline_cv_results_all_folds.csv'), index=False)
    
    # Create matrices for statistical tests
    cv_matrix_g = pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5],
        'Linear Regression': cv_results['Linear Regression']['G_R2'],
        'Ridge Regression': cv_results['Ridge Regression']['G_R2'],
        'Random Forest': cv_results['Random Forest']['G_R2']
    })
    cv_matrix_g.to_csv(os.path.join(output_dir, 'baseline_cv_matrix_G_R2.csv'), index=False)
    
    cv_matrix_v = pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5],
        'Linear Regression': cv_results['Linear Regression']['V_R2'],
        'Ridge Regression': cv_results['Ridge Regression']['V_R2'],
        'Random Forest': cv_results['Random Forest']['V_R2']
    })
    cv_matrix_v.to_csv(os.path.join(output_dir, 'baseline_cv_matrix_V_R2.csv'), index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 CROSS-VALIDATION SUMMARY (5-Fold)")
    print("="*80)
    print("\nG R² Results:")
    print(cv_matrix_g.to_string(index=False))
    print("\nV R² Results:")
    print(cv_matrix_v.to_string(index=False))
    
    print("\n📈 Mean ± Std for G R²:")
    for model in cv_results.keys():
        mean_val = np.mean(cv_results[model]['G_R2'])
        std_val = np.std(cv_results[model]['G_R2'])
        print(f"   {model}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n📈 Mean ± Std for V R²:")
    for model in cv_results.keys():
        mean_val = np.mean(cv_results[model]['V_R2'])
        std_val = np.std(cv_results[model]['V_R2'])
        print(f"   {model}: {mean_val:.4f} ± {std_val:.4f}")
    
    return cv_results, cv_matrix_g, cv_matrix_v


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
    y = np.column_stack([y_G, y_V])
    
    print(f"\n📊 Dataset Information:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target 1 (G): range [{y_G.min():.4f}, {y_G.max():.4f}], mean={y_G.mean():.4f}")
    print(f"   Target 2 (Velocity): range [{y_V.min():.2f}, {y_V.max():.2f}], mean={y_V.mean():.2f}")
    
    # ==================== RUN 5-FOLD CROSS-VALIDATION ====================
    cv_results, cv_matrix_g, cv_matrix_v = run_5fold_cross_validation(X, y, output_dir)
    
    # ==================== FINAL TRAINING ON FULL DATA (for test set comparison) ====================
    print("\n" + "="*80)
    print("🚀 FINAL TRAINING ON FULL DATASET (for test set evaluation)")
    print("="*80)
    
    # Split for final test (70/15/15 to match MTL)
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE  # 15% of original
    )
    
    print(f"\n📊 Final Data Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Standardize for final training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train final models
    results = {}
    trained_models = {}
    
    print("\n" + "="*80)
    print("TRAINING FINAL MODELS ON FULL DATASET")
    print("="*80)
    
    # 1. Linear Regression
    print("\n📈 Training Linear Regression...")
    lr_G = LinearRegression()
    lr_V = LinearRegression()
    lr_G.fit(X_train_scaled, y_train[:, 0])
    lr_V.fit(X_train_scaled, y_train[:, 1])
    trained_models['Linear Regression'] = {'G': lr_G, 'V': lr_V, 'scaled': True}
    
    yG_pred = lr_G.predict(X_test_scaled)
    yV_pred = lr_V.predict(X_test_scaled)
    results['Linear Regression'] = {
        'G (J/m^2)': calculate_metrics(y_test[:, 0], yG_pred),
        'Crack velocity (um/s)': calculate_metrics(y_test[:, 1], yV_pred)
    }
    print(f"   G R²: {results['Linear Regression']['G (J/m^2)']['R2']:.6f}")
    print(f"   Velocity R²: {results['Linear Regression']['Crack velocity (um/s)']['R2']:.6f}")
    
    # 2. Ridge Regression
    print("\n📈 Training Ridge Regression...")
    ridge_G = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge_V = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge_G.fit(X_train_scaled, y_train[:, 0])
    ridge_V.fit(X_train_scaled, y_train[:, 1])
    trained_models['Ridge Regression'] = {'G': ridge_G, 'V': ridge_V, 'scaled': True}
    
    yG_pred = ridge_G.predict(X_test_scaled)
    yV_pred = ridge_V.predict(X_test_scaled)
    results['Ridge Regression'] = {
        'G (J/m^2)': calculate_metrics(y_test[:, 0], yG_pred),
        'Crack velocity (um/s)': calculate_metrics(y_test[:, 1], yV_pred)
    }
    print(f"   G R²: {results['Ridge Regression']['G (J/m^2)']['R2']:.6f}")
    print(f"   Velocity R²: {results['Ridge Regression']['Crack velocity (um/s)']['R2']:.6f}")
    
    # 3. Random Forest
    print("\n📈 Training Random Forest...")
    rf_G = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    rf_V = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    rf_G.fit(X_train, y_train[:, 0])
    rf_V.fit(X_train, y_train[:, 1])
    trained_models['Random Forest'] = {'G': rf_G, 'V': rf_V, 'scaled': False}
    
    yG_pred = rf_G.predict(X_test)
    yV_pred = rf_V.predict(X_test)
    results['Random Forest'] = {
        'G (J/m^2)': calculate_metrics(y_test[:, 0], yG_pred),
        'Crack velocity (um/s)': calculate_metrics(y_test[:, 1], yV_pred)
    }
    print(f"   G R²: {results['Random Forest']['G (J/m^2)']['R2']:.6f}")
    print(f"   Velocity R²: {results['Random Forest']['Crack velocity (um/s)']['R2']:.6f}")
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save final test results
    df_results = save_results(results, output_dir)
    
    # Create comparison table
    pivot_r2, pivot_rmse, pivot_mae = create_comparison_table(results)
    
    # Generate LaTeX table
    generate_latex_table(results, output_dir)
    
    # Save predictions
    print("\n📁 Saving individual model predictions...")
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    for model_name, model_dict in trained_models.items():
        if model_dict['scaled']:
            X_test_data = X_test_scaled
        else:
            X_test_data = X_test
        
        preds_G = model_dict['G'].predict(X_test_data)
        preds_V = model_dict['V'].predict(X_test_data)
        
        pred_df = pd.DataFrame({
            'True_G': y_test[:, 0],
            'Pred_G': preds_G,
            'True_Velocity': y_test[:, 1],
            'Pred_Velocity': preds_V
        })
        
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
        f.write("CROSS-VALIDATION RESULTS (5-Fold):\n")
        f.write(f"  Linear Regression G: {np.mean(cv_results['Linear Regression']['G_R2']):.4f} ± {np.std(cv_results['Linear Regression']['G_R2']):.4f}\n")
        f.write(f"  Ridge Regression G: {np.mean(cv_results['Ridge Regression']['G_R2']):.4f} ± {np.std(cv_results['Ridge Regression']['G_R2']):.4f}\n")
        f.write(f"  Random Forest G: {np.mean(cv_results['Random Forest']['G_R2']):.4f} ± {np.std(cv_results['Random Forest']['G_R2']):.4f}\n\n")
        f.write(f"Best model for G: {best_g_model}\n")
        f.write(f"  R² = {results[best_g_model]['G (J/m^2)']['R2']:.6f}\n")
        f.write(f"  RMSE = {results[best_g_model]['G (J/m^2)']['RMSE']:.6f}\n\n")
        f.write(f"Best model for Velocity: {best_v_model}\n")
        f.write(f"  R² = {results[best_v_model]['Crack velocity (um/s)']['R2']:.6f}\n")
        f.write(f"  RMSE = {results[best_v_model]['Crack velocity (um/s)']['RMSE']:.2f}\n\n")
        f.write("COMPLETE RESULTS:\n")
        f.write(df_results.to_string())
    
    print(f"\n✅ Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("✅ BASELINE MODELS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📁 All results saved to: {output_dir}")
    print("\n📊 FILES CREATED FOR STATISTICAL ANALYSIS:")
    print("   - baseline_cv_results_all_folds.csv (Per-fold results)")
    print("   - baseline_cv_matrix_G_R2.csv (Matrix for Friedman test - G)")
    print("   - baseline_cv_matrix_V_R2.csv (Matrix for Friedman test - V)")
    print("   - baseline_results.json (Complete metrics)")
    print("   - baseline_results.csv (Tabular format)")
    print("   - baseline_comparison_table.tex (LaTeX table)")
    print("   - baseline_summary.txt (Text summary)")
    print("   - predictions/ (Individual model predictions)")
    print("="*80)
    
    return results, output_dir

def create_comparison_table(results_dict):
    """Create a formatted comparison table"""
    
    print("\n" + "="*80)
    print("BASELINE MODELS PERFORMANCE COMPARISON")
    print("="*80)
    
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
    
    latex_path = os.path.join(output_dir, 'baseline_comparison_table.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    print(f"   ✅ LaTeX table saved to: {latex_path}")
    
    return latex_content

if __name__ == "__main__":
    results, output_dir = main()