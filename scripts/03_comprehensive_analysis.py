# -*- coding: utf-8 -*-
"""
Created on April 2026
@author: H.A.R

SIMPLIFIED OPTIMIZED Physics-Aware Multi-Task Learning
WITH COMPLETE CROSS-VALIDATION RESULTS FOR STATISTICAL TESTS
AND TRAINING HISTORY SAVING FOR FIGURE 3 - FIXED
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== IMPROVED MODEL ARCHITECTURE ====================
class OptimizedPhysicsAwareMTL(nn.Module):
    """Enhanced multi-task model with better architecture"""
    def __init__(self, input_size, crack_length_idx=2):
        super().__init__()
        self.crack_length_idx = crack_length_idx
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.g_branch = nn.Sequential(
            nn.Linear(32 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.v_branch = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        crack_length = x[:, self.crack_length_idx:self.crack_length_idx+1]
        shared_features = self.shared(x)
        g_features = torch.cat([shared_features, crack_length], dim=1)
        g_pred = self.g_branch(g_features)
        v_pred = self.v_branch(shared_features)
        return torch.cat([g_pred, v_pred], dim=1)


class CrackDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_mtl_on_fold(X_train, y_train, X_val, y_val, input_size, device, batch_size=64, lr=0.001, epochs=100):
    """Train MTL model on a specific fold and return validation predictions"""
    
    train_dataset = CrackDataset(X_train, y_train)
    val_dataset = CrackDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = OptimizedPhysicsAwareMTL(input_size=input_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    g_stats = {'std': 0.0600}
    v_stats = {'std': 67.5846}
    
    # Training
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = 0.6 * criterion(predictions[:, 0], y_batch[:, 0]) / (g_stats['std']**2) + \
                   0.4 * criterion(predictions[:, 1], y_batch[:, 1]) / (v_stats['std']**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    
    # Validation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    g_r2 = r2_score(all_targets[:, 0], all_preds[:, 0])
    v_r2 = r2_score(all_targets[:, 1], all_preds[:, 1])
    
    return g_r2, v_r2, all_preds, all_targets


def run_5fold_cross_validation(X, y, output_dir):
    """Run 5-fold cross-validation for all models and save results"""
    
    print("\n" + "="*70)
    print("📊 RUNNING 5-FOLD CROSS-VALIDATION FOR ALL MODELS")
    print("="*70)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize storage for results
    cv_results = {
        'Linear Regression': {'G_R2': [], 'V_R2': [], 'G_RMSE': [], 'V_RMSE': []},
        'Ridge Regression': {'G_R2': [], 'V_R2': [], 'G_RMSE': [], 'V_RMSE': []},
        'Random Forest': {'G_R2': [], 'V_R2': [], 'G_RMSE': [], 'V_RMSE': []},
        'Optimized MTL': {'G_R2': [], 'V_R2': [], 'G_RMSE': [], 'V_RMSE': []}
    }
    
    fold_details = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}/5")
        print(f"{'='*50}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize for linear models
        from sklearn.preprocessing import StandardScaler
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
        
        cv_results['Linear Regression']['G_R2'].append(lr_g_r2)
        cv_results['Linear Regression']['V_R2'].append(lr_v_r2)
        cv_results['Linear Regression']['G_RMSE'].append(lr_g_rmse)
        cv_results['Linear Regression']['V_RMSE'].append(lr_v_rmse)
        print(f"      G R² = {lr_g_r2:.4f}, V R² = {lr_v_r2:.4f}")
        
        # 2. Ridge Regression
        print("\n   Training Ridge Regression...")
        ridge_G = Ridge(alpha=1.0, random_state=42)
        ridge_V = Ridge(alpha=1.0, random_state=42)
        ridge_G.fit(X_train_scaled, y_train[:, 0])
        ridge_V.fit(X_train_scaled, y_train[:, 1])
        pred_G = ridge_G.predict(X_val_scaled)
        pred_V = ridge_V.predict(X_val_scaled)
        
        ridge_g_r2 = r2_score(y_val[:, 0], pred_G)
        ridge_v_r2 = r2_score(y_val[:, 1], pred_V)
        ridge_g_rmse = np.sqrt(mean_squared_error(y_val[:, 0], pred_G))
        ridge_v_rmse = np.sqrt(mean_squared_error(y_val[:, 1], pred_V))
        
        cv_results['Ridge Regression']['G_R2'].append(ridge_g_r2)
        cv_results['Ridge Regression']['V_R2'].append(ridge_v_r2)
        cv_results['Ridge Regression']['G_RMSE'].append(ridge_g_rmse)
        cv_results['Ridge Regression']['V_RMSE'].append(ridge_v_rmse)
        print(f"      G R² = {ridge_g_r2:.4f}, V R² = {ridge_v_r2:.4f}")
        
        # 3. Random Forest
        print("\n   Training Random Forest...")
        rf_G = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf_V = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf_G.fit(X_train, y_train[:, 0])
        rf_V.fit(X_train, y_train[:, 1])
        pred_G = rf_G.predict(X_val)
        pred_V = rf_V.predict(X_val)
        
        rf_g_r2 = r2_score(y_val[:, 0], pred_G)
        rf_v_r2 = r2_score(y_val[:, 1], pred_V)
        rf_g_rmse = np.sqrt(mean_squared_error(y_val[:, 0], pred_G))
        rf_v_rmse = np.sqrt(mean_squared_error(y_val[:, 1], pred_V))
        
        cv_results['Random Forest']['G_R2'].append(rf_g_r2)
        cv_results['Random Forest']['V_R2'].append(rf_v_r2)
        cv_results['Random Forest']['G_RMSE'].append(rf_g_rmse)
        cv_results['Random Forest']['V_RMSE'].append(rf_v_rmse)
        print(f"      G R² = {rf_g_r2:.4f}, V R² = {rf_v_r2:.4f}")
        
        # 4. Optimized MTL
        print("\n   Training Optimized MTL...")
        mtl_g_r2, mtl_v_r2, _, _ = train_mtl_on_fold(
            X_train, y_train, X_val, y_val, 
            input_size=X.shape[1], device=device, 
            batch_size=64, lr=0.001, epochs=100
        )
        
        cv_results['Optimized MTL']['G_R2'].append(mtl_g_r2)
        cv_results['Optimized MTL']['V_R2'].append(mtl_v_r2)
        cv_results['Optimized MTL']['G_RMSE'].append(np.nan)
        cv_results['Optimized MTL']['V_RMSE'].append(np.nan)
        print(f"      G R² = {mtl_g_r2:.4f}, V R² = {mtl_v_r2:.4f}")
        
        # Store fold details
        fold_details.append({
            'Fold': fold,
            'Linear_G_R2': lr_g_r2, 'Linear_V_R2': lr_v_r2,
            'Ridge_G_R2': ridge_g_r2, 'Ridge_V_R2': ridge_v_r2,
            'RF_G_R2': rf_g_r2, 'RF_V_R2': rf_v_r2,
            'MTL_G_R2': mtl_g_r2, 'MTL_V_R2': mtl_v_r2
        })
    
    # Compute summary statistics
    summary = []
    for model in cv_results.keys():
        summary.append({
            'Model': model,
            'G_R2_Mean': np.mean(cv_results[model]['G_R2']),
            'G_R2_Std': np.std(cv_results[model]['G_R2']),
            'V_R2_Mean': np.mean(cv_results[model]['V_R2']),
            'V_R2_Std': np.std(cv_results[model]['V_R2']),
            'G_RMSE_Mean': np.mean(cv_results[model]['G_RMSE']) if cv_results[model]['G_RMSE'][0] is not np.nan else np.nan,
            'V_RMSE_Mean': np.mean(cv_results[model]['V_RMSE']) if cv_results[model]['V_RMSE'][0] is not np.nan else np.nan
        })
    
    # Save all results
    cv_results_df = pd.DataFrame(fold_details)
    cv_results_df.to_csv(os.path.join(output_dir, 'cv_results_all_folds.csv'), index=False)
    
    cv_summary_df = pd.DataFrame(summary)
    cv_summary_df.to_csv(os.path.join(output_dir, 'cv_results_summary.csv'), index=False)
    
    # Save detailed per-fold results in format ready for statistical tests
    cv_matrix_g = pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5],
        'Linear Regression': cv_results['Linear Regression']['G_R2'],
        'Ridge Regression': cv_results['Ridge Regression']['G_R2'],
        'Random Forest': cv_results['Random Forest']['G_R2'],
        'Optimized MTL': cv_results['Optimized MTL']['G_R2']
    })
    cv_matrix_g.to_csv(os.path.join(output_dir, 'cv_matrix_G_R2.csv'), index=False)
    
    cv_matrix_v = pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5],
        'Linear Regression': cv_results['Linear Regression']['V_R2'],
        'Ridge Regression': cv_results['Ridge Regression']['V_R2'],
        'Random Forest': cv_results['Random Forest']['V_R2'],
        'Optimized MTL': cv_results['Optimized MTL']['V_R2']
    })
    cv_matrix_v.to_csv(os.path.join(output_dir, 'cv_matrix_V_R2.csv'), index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 CROSS-VALIDATION SUMMARY (5-Fold)")
    print("="*70)
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


# ==================== MAIN TRAINING ====================
def train_optimized_mtl(features_path, targets_path, output_dir):
    """Complete training pipeline with cross-validation"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("OPTIMIZED PHYSICS-AWARE MULTI-TASK LEARNING")
    print("="*70)
    
    # Load data
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    
    if 'Source_File' in features.columns:
        X = features.drop('Source_File', axis=1).values
        feature_names = [col for col in features.columns if col != 'Source_File']
    else:
        X = features.values
        feature_names = features.columns.tolist()
    
    y = targets.values
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # ==================== RUN CROSS-VALIDATION ====================
    cv_results, cv_matrix_g, cv_matrix_v = run_5fold_cross_validation(X, y, output_dir)
    
    # ==================== FINAL TRAINING ON FULL DATA ====================
    print("\n" + "="*70)
    print("🚀 FINAL TRAINING ON FULL DATASET")
    print("="*70)
    
    # Split for final test
    dataset = list(zip(range(len(X)), y[:, 0], y[:, 1]))
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_indices = [idx for idx, _, _ in train_dataset]
    val_indices = [idx for idx, _, _ in val_dataset]
    test_indices = [idx for idx, _, _ in test_dataset]
    
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    
    print(f"\n📊 Final Data Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Train Random Forest on full training data
    print("\n📊 Training Random Forest on full data...")
    rf_G = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_V = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_G.fit(X_train, y_train[:, 0])
    rf_V.fit(X_train, y_train[:, 1])
    
    rf_pred_G = rf_G.predict(X_test)
    rf_pred_V = rf_V.predict(X_test)
    
    rf_g_r2 = r2_score(y_test[:, 0], rf_pred_G)
    rf_v_r2 = r2_score(y_test[:, 1], rf_pred_V)
    
    print(f"   Random Forest: G R² = {rf_g_r2:.4f}, V R² = {rf_v_r2:.4f}")
    
    # Train MTL on full data
    print("\n📊 Training Optimized MTL on full data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Quick hyperparameter selection using validation set
    best_score = -np.inf
    best_params = {'learning_rate': 0.001, 'batch_size': 64}
    
    for lr in [0.001, 0.0005]:
        for bs in [32, 64]:
            mtl_g_r2_val, mtl_v_r2_val, _, _ = train_mtl_on_fold(
                X_train, y_train, X_val, y_val,
                input_size=X.shape[1], device=device,
                batch_size=bs, lr=lr, epochs=50
            )
            avg_score = (mtl_g_r2_val + mtl_v_r2_val) / 2
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'learning_rate': lr, 'batch_size': bs}
    
    print(f"   Best params: LR={best_params['learning_rate']}, BS={best_params['batch_size']}")
    
    # Final MTL training (store history for plotting)
    print("\n   Training MTL with history tracking...")
    
    # Create datasets for final training
    train_final_dataset = CrackDataset(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]))
    test_dataset = CrackDataset(X_test, y_test)
    
    train_loader = DataLoader(train_final_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
    
    # Initialize model
    model = OptimizedPhysicsAwareMTL(input_size=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    g_stats = {'std': 0.0600}
    v_stats = {'std': 67.5846}
    
    # Training history storage
    history_epochs = []
    history_train_loss = []
    history_val_loss = []
    history_val_g_r2 = []
    history_val_v_r2 = []
    
    # For validation during training, we need a validation loader
    # Create validation split from training data (10% of combined train+val)
    val_split_size = int(0.1 * len(train_final_dataset))
    
    # Ensure validation split size is at least batch_size
    if val_split_size < best_params['batch_size']:
        val_split_size = best_params['batch_size']
    
    train_subset, val_subset = random_split(
        train_final_dataset, 
        [len(train_final_dataset) - val_split_size, val_split_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Use drop_last=True to avoid batch size 1 issue
    train_loader_subset = DataLoader(train_subset, batch_size=best_params['batch_size'], shuffle=True, drop_last=True)
    val_loader_subset = DataLoader(val_subset, batch_size=best_params['batch_size'], drop_last=True)
    
    print("   Starting training loop...")
    
    for epoch in range(150):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader_subset:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = 0.6 * criterion(predictions[:, 0], y_batch[:, 0]) / (g_stats['std']**2) + \
                   0.4 * criterion(predictions[:, 1], y_batch[:, 1]) / (v_stats['std']**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader_subset)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader_subset:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = 0.6 * criterion(predictions[:, 0], y_batch[:, 0]) / (g_stats['std']**2) + \
                       0.4 * criterion(predictions[:, 1], y_batch[:, 1]) / (v_stats['std']**2)
                val_loss += loss.item()
                val_preds.append(predictions.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())
        val_loss /= len(val_loader_subset)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        g_r2 = r2_score(val_targets[:, 0], val_preds[:, 0])
        v_r2 = r2_score(val_targets[:, 1], val_preds[:, 1])
        
        # Store history
        history_epochs.append(epoch + 1)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_g_r2.append(g_r2)
        history_val_v_r2.append(v_r2)
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"G R²: {g_r2:.4f}, V R²: {v_r2:.4f}")
    
    # Final evaluation on test set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    mtl_preds = np.vstack(all_preds)
    mtl_targets = np.vstack(all_targets)
    
    mtl_g_r2 = r2_score(mtl_targets[:, 0], mtl_preds[:, 0])
    mtl_v_r2 = r2_score(mtl_targets[:, 1], mtl_preds[:, 1])
    
    print(f"\n   Optimized MTL: G R² = {mtl_g_r2:.4f}, V R² = {mtl_v_r2:.4f}")
    
    # ==================== SAVE TRAINING HISTORY ====================
    print("\n📊 Saving training history for Figure 3...")
    history_df = pd.DataFrame({
        'epoch': history_epochs,
        'train_loss': history_train_loss,
        'val_loss': history_val_loss,
        'val_g_r2': history_val_g_r2,
        'val_v_r2': history_val_v_r2,
        'lr': [best_params['learning_rate']] * len(history_epochs)
    })
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    print(f"   ✅ Training history saved to: {os.path.join(output_dir, 'training_history.csv')}")
    
    # Ensemble
    ensemble_g = 0.6 * mtl_preds[:, 0] + 0.4 * rf_pred_G
    ensemble_v = 0.6 * mtl_preds[:, 1] + 0.4 * rf_pred_V
    
    ensemble_g_r2 = r2_score(mtl_targets[:, 0], ensemble_g)
    ensemble_v_r2 = r2_score(mtl_targets[:, 1], ensemble_v)
    
    print(f"\n🎯 ENSEMBLE (MTL+RF) TEST RESULTS:")
    print(f"   G (J/m²): R² = {ensemble_g_r2:.4f}")
    print(f"   Velocity: R² = {ensemble_v_r2:.4f}")
    
    # Save final results
    final_results = {
        'cv_results': {
            'linear_regression': {'G_R2': cv_results['Linear Regression']['G_R2'], 
                                   'V_R2': cv_results['Linear Regression']['V_R2']},
            'ridge_regression': {'G_R2': cv_results['Ridge Regression']['G_R2'], 
                                  'V_R2': cv_results['Ridge Regression']['V_R2']},
            'random_forest': {'G_R2': cv_results['Random Forest']['G_R2'], 
                               'V_R2': cv_results['Random Forest']['V_R2']},
            'optimized_mtl': {'G_R2': cv_results['Optimized MTL']['G_R2'], 
                               'V_R2': cv_results['Optimized MTL']['V_R2']}
        },
        'final_test_results': {
            'random_forest': {'G_R2': float(rf_g_r2), 'V_R2': float(rf_v_r2)},
            'optimized_mtl': {'G_R2': float(mtl_g_r2), 'V_R2': float(mtl_v_r2)},
            'ensemble': {'G_R2': float(ensemble_g_r2), 'V_R2': float(ensemble_v_r2)}
        },
        'best_hyperparameters': best_params
    }
    
    with open(os.path.join(output_dir, 'complete_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save predictions
    preds_df = pd.DataFrame({
        'True_G': mtl_targets[:, 0],
        'Optimized_MTL_G': mtl_preds[:, 0],
        'Random_Forest_G': rf_pred_G,
        'Ensemble_G': ensemble_g,
        'True_Velocity': mtl_targets[:, 1],
        'Optimized_MTL_V': mtl_preds[:, 1],
        'Random_Forest_V': rf_pred_V,
        'Ensemble_V': ensemble_v
    })
    preds_df.to_csv(os.path.join(output_dir, 'optimized_predictions.csv'), index=False)
    
    print("\n" + "="*70)
    print("✅ OPTIMIZED TRAINING COMPLETED!")
    print("="*70)
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n📊 FILES CREATED FOR STATISTICAL ANALYSIS:")
    print("   - cv_results_all_folds.csv (Per-fold results for all models)")
    print("   - cv_results_summary.csv (Mean ± Std summary)")
    print("   - cv_matrix_G_R2.csv (Matrix for Friedman test - G)")
    print("   - cv_matrix_V_R2.csv (Matrix for Friedman test - V)")
    print("   - complete_results.json (All results in JSON)")
    print("   - optimized_predictions.csv (Test set predictions)")
    print("   - training_history.csv (Training history for Figure 3)")
    
    return final_results, output_dir


# ==================== MAIN ====================
if __name__ == "__main__":
    DATA_DIR = "E:/materials2/RAA/processed_data"
    OUTPUT_DIR = "E:/materials2/RAA/optimized_mtl_results"
    
    features_path = os.path.join(DATA_DIR, "processed_features.csv")
    targets_path = os.path.join(DATA_DIR, "processed_targets.csv")
    
    results, output_dir = train_optimized_mtl(features_path, targets_path, OUTPUT_DIR)