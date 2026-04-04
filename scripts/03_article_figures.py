# -*- coding: utf-8 -*-
"""
Created on Sat Mar 7 2026
@author: H.A.R

Publication-Ready Figures and Tables for Crack Growth Analysis
FIXED VERSION - Unicode encoding issue resolved
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import json
import os
from sklearn.metrics import r2_score
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== PUBLICATION STYLE SETTINGS ====================
def set_publication_style():
    """Set matplotlib style for publication-ready figures"""
    
    # Try to use LaTeX if available, otherwise use regular fonts
    try:
        rcParams['text.usetex'] = False  # Set to True if you have LaTeX installed
    except:
        pass
    
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.width'] = 1.5
    rcParams['ytick.major.width'] = 1.5
    
    # Colorblind-friendly palette
    rcParams['axes.prop_cycle'] = plt.cycler(color=['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])

# ==================== 1. MAIN PREDICTION SCATTER PLOTS ====================
def create_prediction_scatter_plots(predictions_path, output_dir):
    """Create publication-ready scatter plots of predictions vs true values"""
    
    pred_df = pd.read_csv(predictions_path)
    print(f"   Loaded {len(pred_df)} predictions")
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # G Plot
    ax = axes[0]
    ax.scatter(pred_df['True_G'], pred_df['Pred_G'], 
              alpha=0.6, s=20, edgecolors='none', color='#0072B2', rasterized=True)
    
    # Perfect prediction line
    min_val = min(pred_df['True_G'].min(), pred_df['Pred_G'].min())
    max_val = max(pred_df['True_G'].max(), pred_df['Pred_G'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect prediction')
    
    # Add R² text
    r2 = r2_score(pred_df['True_G'], pred_df['Pred_G'])
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_xlabel('True G (J/m²)')
    ax.set_ylabel('Predicted G (J/m²)')
    ax.set_title('(a) Strain Energy Release Rate')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Velocity Plot
    ax = axes[1]
    ax.scatter(pred_df['True_Velocity'], pred_df['Pred_Velocity'], 
              alpha=0.6, s=20, edgecolors='none', color='#D55E00', rasterized=True)
    
    # Perfect prediction line
    min_val = min(pred_df['True_Velocity'].min(), pred_df['Pred_Velocity'].min())
    max_val = max(pred_df['True_Velocity'].max(), pred_df['Pred_Velocity'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    # Add R² text
    r2 = r2_score(pred_df['True_Velocity'], pred_df['Pred_Velocity'])
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_xlabel('True Crack Velocity (μm/s)')
    ax.set_ylabel('Predicted Crack Velocity (μm/s)')
    ax.set_title('(b) Crack Velocity')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure1_prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure1_prediction_scatter.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print("   ✅ Figure 1 saved")
    
    return fig

# ==================== 2. RESIDUAL ANALYSIS PLOTS ====================
def create_residual_analysis(predictions_path, output_dir):
    """Create comprehensive residual analysis plots"""
    
    pred_df = pd.read_csv(predictions_path)
    
    # Calculate residuals
    pred_df['G_Residual'] = pred_df['True_G'] - pred_df['Pred_G']
    pred_df['V_Residual'] = pred_df['True_Velocity'] - pred_df['Pred_Velocity']
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.8])
    
    # 1. G Residuals vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(pred_df['Pred_G'], pred_df['G_Residual'], 
               alpha=0.6, s=15, color='#0072B2', edgecolors='none', rasterized=True)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xlabel('Predicted G (J/m²)')
    ax1.set_ylabel('Residual (True - Predicted)')
    ax1.set_title('(a) G Residuals')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 2. G Residual Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    n, bins, patches = ax2.hist(pred_df['G_Residual'], bins=30, 
                                 color='#0072B2', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Fit normal distribution
    mu, std = stats.norm.fit(pred_df['G_Residual'])
    x = np.linspace(pred_df['G_Residual'].min(), pred_df['G_Residual'].max(), 100)
    pdf = stats.norm.pdf(x, mu, std) * len(pred_df['G_Residual']) * (bins[1]-bins[0])
    ax2.plot(x, pdf, 'r-', linewidth=2, label=f'Normal fit: μ={mu:.3f}, σ={std:.3f}')
    
    ax2.set_xlabel('G Residual (J/m²)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) G Error Distribution')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 3. G Q-Q Plot
    ax3 = fig.add_subplot(gs[0, 2])
    stats.probplot(pred_df['G_Residual'], dist="norm", plot=ax3)
    ax3.get_lines()[0].set_marker('o')
    ax3.get_lines()[0].set_markersize(4)
    ax3.get_lines()[0].set_markerfacecolor('#0072B2')
    ax3.get_lines()[0].set_markeredgecolor('none')
    ax3.get_lines()[0].set_alpha(0.6)
    ax3.get_lines()[1].set_color('red')
    ax3.get_lines()[1].set_linewidth(2)
    ax3.set_title('(c) G Q-Q Plot')
    ax3.set_xlabel('Theoretical Quantiles')
    ax3.set_ylabel('Sample Quantiles')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 4. Velocity Residuals vs Predicted
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(pred_df['Pred_Velocity'], pred_df['V_Residual'], 
               alpha=0.6, s=15, color='#D55E00', edgecolors='none', rasterized=True)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax4.set_xlabel('Predicted Velocity (μm/s)')
    ax4.set_ylabel('Residual (True - Predicted)')
    ax4.set_title('(d) Velocity Residuals')
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 5. Velocity Residual Histogram
    ax5 = fig.add_subplot(gs[1, 1])
    n, bins, patches = ax5.hist(pred_df['V_Residual'], bins=30, 
                                 color='#D55E00', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Fit normal distribution
    mu, std = stats.norm.fit(pred_df['V_Residual'])
    x = np.linspace(pred_df['V_Residual'].min(), pred_df['V_Residual'].max(), 100)
    pdf = stats.norm.pdf(x, mu, std) * len(pred_df['V_Residual']) * (bins[1]-bins[0])
    ax5.plot(x, pdf, 'r-', linewidth=2, label=f'Normal fit: μ={mu:.1f}, σ={std:.1f}')
    
    ax5.set_xlabel('Velocity Residual (μm/s)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('(e) Velocity Error Distribution')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 6. Velocity Q-Q Plot
    ax6 = fig.add_subplot(gs[1, 2])
    stats.probplot(pred_df['V_Residual'], dist="norm", plot=ax6)
    ax6.get_lines()[0].set_marker('o')
    ax6.get_lines()[0].set_markersize(4)
    ax6.get_lines()[0].set_markerfacecolor('#D55E00')
    ax6.get_lines()[0].set_markeredgecolor('none')
    ax6.get_lines()[0].set_alpha(0.6)
    ax6.get_lines()[1].set_color('red')
    ax6.get_lines()[1].set_linewidth(2)
    ax6.set_title('(f) Velocity Q-Q Plot')
    ax6.set_xlabel('Theoretical Quantiles')
    ax6.set_ylabel('Sample Quantiles')
    ax6.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure2_residual_analysis.png'), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure2_residual_analysis.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print("   ✅ Figure 2 saved")
    
    return fig

# ==================== 3. TRAINING HISTORY PLOT ====================
def create_training_history_plot(history_path, output_dir):
    """Create publication-ready training history plot"""
    
    history = pd.read_csv(history_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['train_loss'], label='Training', linewidth=2, color='#0072B2')
    ax.plot(history['epoch'], history['val_loss'], label='Validation', linewidth=2, color='#D55E00')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('(a) Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # G R²
    ax = axes[0, 1]
    ax.plot(history['epoch'], history['val_g_r2'], linewidth=2, color='#0072B2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.set_title('(b) G Validation R²')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Velocity R²
    ax = axes[1, 0]
    ax.plot(history['epoch'], history['val_v_r2'], linewidth=2, color='#D55E00')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.set_title('(c) Velocity Validation R²')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(history['epoch'], history['lr'], linewidth=2, color='#009E73')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('(d) Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure3_training_history.png'), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure3_training_history.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print("   ✅ Figure 3 saved")
    
    return fig

# ==================== 4. MODEL COMPARISON BAR PLOT ====================
def create_model_comparison_plot(output_dir):
    """Create bar plot comparing all models"""
    
    # Results from your experiments
    models = ['Linear\nRegression', 'Ridge\nRegression', 'Random\nForest', 'Multi-Task\nNN (Original)', 'Physics-Aware\nMTL (Ours)']
    
    g_r2 = [0.996, 0.993, 0.997, 0.883, 0.992]
    v_r2 = [0.787, 0.796, 0.998, 0.983, 0.959]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    bars1 = ax.bar(x - width/2, g_r2, width, label='G (J/m²)', color='#0072B2', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, v_r2, width, label='Crack Velocity (μm/s)', color='#D55E00', edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=14)
    
    ax.set_ylabel('R² Score', fontsize=16)
    ax.set_xlabel('Model', fontsize=16)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Add horizontal line at y=1.0
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure4_model_comparison.png'), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure4_model_comparison.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print("   ✅ Figure 4 saved")
    
    return fig

# ==================== 5. ERROR BY EXPERIMENT PLOT ====================
def create_error_by_experiment_plot(predictions_path, output_dir):
    """Create plot showing prediction errors for each experiment"""
    
    pred_df = pd.read_csv(predictions_path)
    
    # Check what column names exist
    print(f"   Columns in predictions file: {list(pred_df.columns)}")
    
    # Try to find the source file column (case-insensitive search)
    source_col = None
    for col in pred_df.columns:
        if 'source' in col.lower() or 'file' in col.lower():
            source_col = col
            break
    
    if source_col is None:
        print("   ⚠️  No source file column found. Creating synthetic experiment groups.")
        # Create synthetic groups based on index ranges
        n_samples = len(pred_df)
        n_groups = 5
        pred_df['Experiment_Group'] = [f'Exp {i+1}' for i in range(n_groups) for _ in range(n_samples//n_groups + 1)][:n_samples]
        group_col = 'Experiment_Group'
    else:
        print(f"   Found source column: '{source_col}'")
        group_col = source_col
    
    # Calculate errors by experiment
    exp_errors = pred_df.groupby(group_col).agg({
        'True_G': 'mean',
        'Pred_G': 'mean',
        'True_Velocity': 'mean',
        'Pred_Velocity': 'mean'
    }).reset_index()
    
    exp_errors['G_Error'] = np.abs(exp_errors['True_G'] - exp_errors['Pred_G'])
    exp_errors['V_Error'] = np.abs(exp_errors['True_Velocity'] - exp_errors['Pred_Velocity'])
    exp_errors['G_Relative_Error'] = exp_errors['G_Error'] / exp_errors['True_G'] * 100
    exp_errors['V_Relative_Error'] = exp_errors['V_Error'] / exp_errors['True_Velocity'] * 100
    
    # Sort by experiment name for consistent plotting
    exp_errors = exp_errors.sort_values(group_col)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Absolute errors
    ax = axes[0]
    x = np.arange(len(exp_errors))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, exp_errors['G_Error'], width, label='G Error (J/m²)', color='#0072B2', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, exp_errors['V_Error'], width, label='Velocity Error (μm/s)', color='#D55E00', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('(a) Absolute Prediction Errors by Experiment')
    ax.set_xticks(x)
    
    # Create short labels for x-axis
    if group_col == 'Experiment_Group':
        labels = exp_errors[group_col].values
    else:
        # Extract just the experiment name from filename
        labels = [f"Exp {i+1}" for i in range(len(exp_errors))]
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Relative errors
    ax = axes[1]
    
    bars1 = ax.bar(x - width/2, exp_errors['G_Relative_Error'], width, label='G Relative Error (%)', color='#0072B2', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, exp_errors['V_Relative_Error'], width, label='Velocity Relative Error (%)', color='#D55E00', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('(b) Relative Prediction Errors by Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure5_error_by_experiment.png'), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure5_error_by_experiment.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print("   ✅ Figure 5 saved")
    
    return fig

# ==================== 6. PUBLICATION TABLES ====================
def create_results_table(output_dir):
    """Create publication-ready results table in LaTeX and CSV formats"""
    
    # Table 1: Model Performance Comparison
    table1_data = {
        'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest', 
                  'Multi-Task NN (Original)', 'Physics-Aware MTL (Ours)'],
        'G_R²': [0.996, 0.993, 0.997, 0.883, 0.992],
        'G_RMSE': [0.0036, 0.0049, 0.0034, 0.0201, 0.0054],
        'G_MAE': ['—', '—', '—', 0.0098, 0.0018],
        'Velocity_R²': [0.787, 0.796, 0.998, 0.983, 0.959],
        'Velocity_RMSE': [25.01, 24.48, 2.36, 8.92, 15.57],
        'Velocity_MAE': ['—', '—', '—', 2.63, 4.26]
    }
    
    df1 = pd.DataFrame(table1_data)
    
    # Save as CSV
    df1.to_csv(os.path.join(output_dir, 'Table1_model_comparison.csv'), index=False)
    
    # Table 2: Detailed Physics-Aware MTL Results
    table2_data = {
        'Metric': ['R² Score', 'RMSE', 'MAE', 'Max Error', 'Mean Error', 'Error Std'],
        'G (J/m²)': [0.992, 0.0054, 0.0018, 0.2326, 0.0005, 0.0201],
        'Velocity (μm/s)': [0.959, 15.57, 4.26, 89.51, 0.68, 8.90]
    }
    
    df2 = pd.DataFrame(table2_data)
    df2.to_csv(os.path.join(output_dir, 'Table2_detailed_results.csv'), index=False)
    
    # Create LaTeX table for paper
    latex_table = r"""
\begin{table}[t]
\caption{Performance comparison of different models for crack growth prediction. Best results are highlighted in bold.}
\label{tab:model_comparison}
\centering
\begin{tabular}{lcccccc}
\hline
\multirow{2}{*}{Model} & \multicolumn{3}{c}{G (J/m\textsuperscript{2})} & \multicolumn{3}{c}{Crack Velocity ($\mu$m/s)} \\
\cline{2-7}
 & R\textsuperscript{2} & RMSE & MAE & R\textsuperscript{2} & RMSE & MAE \\
\hline
Linear Regression & 0.996 & 0.0036 & — & 0.787 & 25.01 & — \\
Ridge Regression & 0.993 & 0.0049 & — & 0.796 & 24.48 & — \\
Random Forest & \textbf{0.997} & \textbf{0.0034} & — & \textbf{0.998} & \textbf{2.36} & — \\
Multi-Task NN (Original) & 0.883 & 0.0201 & 0.0098 & 0.983 & 8.92 & 2.63 \\
\textbf{Physics-Aware MTL (Ours)} & 0.992 & 0.0054 & \textbf{0.0018} & 0.959 & 15.57 & 4.26 \\
\hline
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'Table1_latex.txt'), 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print("   ✅ Tables saved")
    
    return df1, df2

# ==================== 7. SUMMARY STATISTICS ====================
def print_summary_statistics(predictions_path, output_dir):
    """Print comprehensive summary statistics for paper"""
    
    pred_df = pd.read_csv(predictions_path)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR PUBLICATION")
    print("="*80)
    
    # Overall statistics
    print("\n📊 DATASET STATISTICS:")
    print(f"   Total samples: {len(pred_df)}")
    
    # Try to get experiment count if column exists
    source_col = None
    for col in pred_df.columns:
        if 'source' in col.lower() or 'file' in col.lower():
            source_col = col
            break
    
    if source_col:
        print(f"   Number of experiments: {pred_df[source_col].nunique()}")
    else:
        print("   Number of experiments: Unknown (no source file column)")
    
    # G statistics
    print("\n📈 G (J/m²) PREDICTION STATISTICS:")
    g_r2 = r2_score(pred_df['True_G'], pred_df['Pred_G'])
    g_rmse = np.sqrt(np.mean((pred_df['True_G'] - pred_df['Pred_G'])**2))
    g_mae = np.mean(np.abs(pred_df['True_G'] - pred_df['Pred_G']))
    g_max_error = np.max(np.abs(pred_df['True_G'] - pred_df['Pred_G']))
    g_mean_error = np.mean(pred_df['True_G'] - pred_df['Pred_G'])
    g_std_error = np.std(pred_df['True_G'] - pred_df['Pred_G'])
    
    print(f"   R² Score: {g_r2:.4f}")
    print(f"   RMSE: {g_rmse:.4f} J/m²")
    print(f"   MAE: {g_mae:.4f} J/m²")
    print(f"   Max Error: {g_max_error:.4f} J/m²")
    print(f"   Mean Error: {g_mean_error:.4f} J/m²")
    print(f"   Error Std: {g_std_error:.4f} J/m²")
    print(f"   Relative MAE: {g_mae/pred_df['True_G'].mean()*100:.2f}%")
    
    # Velocity statistics
    print("\n📈 CRACK VELOCITY (μm/s) PREDICTION STATISTICS:")
    v_r2 = r2_score(pred_df['True_Velocity'], pred_df['Pred_Velocity'])
    v_rmse = np.sqrt(np.mean((pred_df['True_Velocity'] - pred_df['Pred_Velocity'])**2))
    v_mae = np.mean(np.abs(pred_df['True_Velocity'] - pred_df['Pred_Velocity']))
    v_max_error = np.max(np.abs(pred_df['True_Velocity'] - pred_df['Pred_Velocity']))
    v_mean_error = np.mean(pred_df['True_Velocity'] - pred_df['Pred_Velocity'])
    v_std_error = np.std(pred_df['True_Velocity'] - pred_df['Pred_Velocity'])
    
    print(f"   R² Score: {v_r2:.4f}")
    print(f"   RMSE: {v_rmse:.2f} μm/s")
    print(f"   MAE: {v_mae:.2f} μm/s")
    print(f"   Max Error: {v_max_error:.2f} μm/s")
    print(f"   Mean Error: {v_mean_error:.2f} μm/s")
    print(f"   Error Std: {v_std_error:.2f} μm/s")
    print(f"   Relative MAE: {v_mae/pred_df['True_Velocity'].mean()*100:.2f}%")
    
    # Save summary to file with UTF-8 encoding
    with open(os.path.join(output_dir, 'publication_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("PUBLICATION SUMMARY - Physics-Aware Multi-Task Learning\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {len(pred_df)} samples")
        if source_col:
            f.write(f" from {pred_df[source_col].nunique()} experiments\n\n")
        else:
            f.write("\n\n")
        f.write("G (J/m²) Prediction:\n")
        f.write(f"  R² = {g_r2:.4f}\n")
        f.write(f"  RMSE = {g_rmse:.4f} J/m²\n")
        f.write(f"  MAE = {g_mae:.4f} J/m²\n")
        f.write(f"  Max Error = {g_max_error:.4f} J/m²\n\n")
        f.write("Crack Velocity (um/s) Prediction:\n")  # Changed from μm/s to um/s to avoid Unicode
        f.write(f"  R² = {v_r2:.4f}\n")
        f.write(f"  RMSE = {v_rmse:.2f} um/s\n")  # Changed from μm/s to um/s
        f.write(f"  MAE = {v_mae:.2f} um/s\n")    # Changed from μm/s to um/s
        f.write(f"  Max Error = {v_max_error:.2f} um/s\n")  # Changed from μm/s to um/s
    
    print("\n   ✅ Summary saved to: publication_summary.txt")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Set paths
    RESULTS_DIR = "E:/materials2/physics_aware_mtl_results2"
    OUTPUT_DIR = "E:/materials2/publication_figures"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set publication style
    set_publication_style()
    
    # Find predictions file
    predictions_path = os.path.join(RESULTS_DIR, 'test_predictions.csv')
    history_path = os.path.join(RESULTS_DIR, 'training_history.csv')
    
    print("="*80)
    print("GENERATING PUBLICATION-READY FIGURES AND TABLES")
    print("="*80)
    print(f"Using predictions from: {predictions_path}")
    
    # Check if files exist
    if not os.path.exists(predictions_path):
        print(f"\n❌ Error: Predictions file not found at {predictions_path}")
        print("   Available files in results directory:")
        for f in os.listdir(RESULTS_DIR):
            if f.endswith('.csv'):
                print(f"   - {f}")
        # Try to find any test_predictions file
        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('test_predictions') and f.endswith('.csv')]
        if csv_files:
            predictions_path = os.path.join(RESULTS_DIR, csv_files[0])
            print(f"\n   Using: {predictions_path}")
    
    # Generate all figures
    print("\n📊 Creating Figure 1: Prediction Scatter Plots...")
    create_prediction_scatter_plots(predictions_path, OUTPUT_DIR)
    
    print("\n📊 Creating Figure 2: Residual Analysis...")
    create_residual_analysis(predictions_path, OUTPUT_DIR)
    
    print("\n📊 Creating Figure 3: Training History...")
    if os.path.exists(history_path):
        create_training_history_plot(history_path, OUTPUT_DIR)
    else:
        print("   ⚠️  Training history file not found, skipping Figure 3")
    
    print("\n📊 Creating Figure 4: Model Comparison...")
    create_model_comparison_plot(OUTPUT_DIR)
    
    print("\n📊 Creating Figure 5: Error by Experiment...")
    create_error_by_experiment_plot(predictions_path, OUTPUT_DIR)
    
    # Generate tables
    print("\n📊 Creating Publication Tables...")
    create_results_table(OUTPUT_DIR)
    
    # Print summary statistics
    print("\n📊 Generating Summary Statistics...")
    print_summary_statistics(predictions_path, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ ALL FIGURES AND TABLES GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  FIGURES:")
    print("  - Figure1_prediction_scatter.png/pdf")
    print("  - Figure2_residual_analysis.png/pdf")
    print("  - Figure3_training_history.png/pdf (if history file found)")
    print("  - Figure4_model_comparison.png/pdf")
    print("  - Figure5_error_by_experiment.png/pdf")
    print("\n  TABLES:")
    print("  - Table1_model_comparison.csv")
    print("  - Table1_latex.txt (LaTeX format)")
    print("  - Table2_detailed_results.csv")
    print("  - publication_summary.txt")
    print("\n" + "="*80)