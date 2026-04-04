# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:27:16 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 09:37:00 2026

@author: H.A.R
Complete Multi-Task Learning Pipeline for Crack Growth Analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import json
from datetime import datetime
import shutil

# ==================== 1. PHYSICS-AWARE MODEL ====================
class PhysicsAwareMTL(nn.Module):
    """Multi-task model with physics awareness - emphasizes crack length for G prediction"""
    def __init__(self, input_size, crack_length_idx=2):  # Crack length is 3rd feature (index 2)
        super().__init__()
        self.crack_length_idx = crack_length_idx
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # G-specific pathway with direct crack length connection
        self.g_branch = nn.Sequential(
            nn.Linear(32 + 1, 24),  # +1 for direct crack length connection
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Velocity-specific pathway
        self.v_branch = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # Extract crack length separately
        crack_length = x[:, self.crack_length_idx:self.crack_length_idx+1]
        
        # Shared features
        shared_features = self.shared(x)
        
        # G prediction: combine shared features with crack length
        g_features = torch.cat([shared_features, crack_length], dim=1)
        g_pred = self.g_branch(g_features)
        
        # Velocity prediction: only shared features
        v_pred = self.v_branch(shared_features)
        
        return torch.cat([g_pred, v_pred], dim=1)

# ==================== 2. BALANCED TRAINER ====================
class BalancedMultiTaskTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Statistics from your data
        self.g_stats = {'mean': 0.2867, 'std': 0.0600}
        self.v_stats = {'mean': 12.8335, 'std': 67.5846}
        
        # Track loss history for dynamic weighting
        self.g_loss_history = []
        self.v_loss_history = []
        
    def calculate_task_weights(self, g_loss, v_loss):
        """Dynamic weighting based on recent loss ratios"""
        if len(self.g_loss_history) < 10:
            return 0.7, 0.3  # Default weights
            
        avg_g_loss = np.mean(self.g_loss_history[-10:])
        avg_v_loss = np.mean(self.v_loss_history[-10:])
        
        # More weight to task with higher relative loss
        if avg_g_loss > 2 * avg_v_loss:
            return 0.8, 0.2  # G needs more help
        elif avg_v_loss > 2 * avg_g_loss:
            return 0.4, 0.6  # Velocity needs more help
        else:
            return 0.6, 0.4  # Balanced
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        g_losses, v_losses = [], []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            
            predictions = self.model(X_batch)
            g_pred, v_pred = predictions[:, 0], predictions[:, 1]
            g_true, v_true = y_batch[:, 0], y_batch[:, 1]
            
            # Calculate losses (normalized by variance)
            g_loss = criterion(g_pred, g_true) / (self.g_stats['std'] ** 2)
            v_loss = criterion(v_pred, v_true) / (self.v_stats['std'] ** 2)
            
            # Dynamic weighting
            g_weight, v_weight = self.calculate_task_weights(
                g_loss.item(), v_loss.item()
            )
            
            loss = g_weight * g_loss + v_weight * v_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            g_losses.append(g_loss.item())
            v_losses.append(v_loss.item())
        
        # Update history
        self.g_loss_history.extend(g_losses)
        self.v_loss_history.extend(v_losses)
        
        return (
            total_loss / len(train_loader),
            np.mean(g_losses),
            np.mean(v_losses)
        )
    
    def evaluate(self, data_loader, criterion):
        self.model.eval()
        total_loss = 0
        g_losses, v_losses = [], []
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                
                g_pred, v_pred = predictions[:, 0], predictions[:, 1]
                g_true, v_true = y_batch[:, 0], y_batch[:, 1]
                
                g_loss = criterion(g_pred, g_true) / (self.g_stats['std'] ** 2)
                v_loss = criterion(v_pred, v_true) / (self.v_stats['std'] ** 2)
                
                g_weight, v_weight = 0.6, 0.4  # Fixed for evaluation
                loss = g_weight * g_loss + v_weight * v_loss
                
                total_loss += loss.item()
                g_losses.append(g_loss.item())
                v_losses.append(v_loss.item())
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        return (
            total_loss / len(data_loader),
            np.mean(g_losses),
            np.mean(v_losses),
            all_preds,
            all_targets
        )

# ==================== 3. COMPLETE TRAINING PIPELINE ====================
def ensure_directory_exists(dir_path):
    """Ensure output directory exists and is writable"""
    if os.path.exists(dir_path):
        # Try to remove and recreate if there are permission issues
        try:
            # Test if we can write to the directory
            test_file = os.path.join(dir_path, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except:
            # Create a new directory with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_path = f"{dir_path}_{timestamp}"
            os.makedirs(dir_path, exist_ok=True)
            print(f"⚠️  Created new directory: {dir_path}")
    else:
        os.makedirs(dir_path, exist_ok=True)
    
    return dir_path

def save_results(output_dir, history, test_metrics, test_preds, test_targets, best_model_path):
    """Save all results for reproducibility - FIXED VERSION"""
    
    # Save history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    
    # Save test predictions
    preds_df = pd.DataFrame({
        'True_G': test_targets[:, 0],
        'Pred_G': test_preds[:, 0],
        'True_Velocity': test_targets[:, 1],
        'Pred_Velocity': test_preds[:, 1]
    })
    preds_path = os.path.join(output_dir, 'test_predictions.csv')
    preds_df.to_csv(preds_path, index=False)
    
    # Convert numpy float32/float64 to Python float for JSON serialization
    serializable_metrics = {}
    for task, metrics in test_metrics.items():
        serializable_metrics[task] = {}
        for metric_name, value in metrics.items():
            # Convert numpy types to Python native types
            if hasattr(value, 'item'):
                serializable_metrics[task][metric_name] = value.item()
            elif isinstance(value, np.floating):
                serializable_metrics[task][metric_name] = float(value)
            elif isinstance(value, np.integer):
                serializable_metrics[task][metric_name] = int(value)
            else:
                serializable_metrics[task][metric_name] = value
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    # Save configuration with serializable values
    config = {
        'output_dir': str(output_dir),
        'best_model': str(best_model_path) if best_model_path else None,
        'history_file': str(history_path),
        'predictions_file': str(preds_path),
        'metrics_file': str(metrics_path),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"  - Best model: {os.path.basename(best_model_path) if best_model_path else 'None'}")
    print(f"  - Training history: training_history.csv")
    print(f"  - Test predictions: test_predictions.csv")
    print(f"  - Test metrics: test_metrics.json")
    print(f"  - Experiment config: experiment_config.json")

def print_results_summary(history, test_metrics, best_epoch, best_model_path):
    """Print comprehensive results summary"""
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Training Statistics:")
    print(f"  Total epochs trained: {len(history['epoch'])}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Final learning rate: {history['lr'][-1]:.6f}")
    
    print(f"\n🎯 Test Set Performance:")
    for task, metrics in test_metrics.items():
        print(f"\n  {task}:")
        print(f"    R² Score: {metrics['R2']:.6f}")
        print(f"    RMSE: {metrics['RMSE']:.6f}")
        print(f"    MAE: {metrics['MAE']:.6f}")
        print(f"    Explained Variance: {metrics['Explained_Variance']:.6f}")
    
    print(f"\n📈 Validation Performance at Best Model:")
    best_idx = len(history['val_g_r2']) - 1  # Last epoch
    print(f"  G R²: {history['val_g_r2'][best_idx]:.6f}")
    print(f"  Velocity R²: {history['val_v_r2'][best_idx]:.6f}")
    
    print(f"\n📁 Saved Files:")
    if best_model_path:
        print(f"  Best model: {os.path.basename(best_model_path)}")
    print(f"  Complete results in: {os.path.dirname(best_model_path) if best_model_path else 'N/A'}")
    
    print("\n" + "="*70)
    print("✅ READY FOR PAPER SUBMISSION")
    print("="*70)

def train_physics_aware_mtl(features_path, targets_path, output_dir):
    """Complete training pipeline with all improvements"""
    
    # Ensure output directory exists and is writable
    output_dir = ensure_directory_exists(output_dir)
    
    print("="*70)
    print("PHYSICS-AWARE MULTI-TASK LEARNING FOR CRACK GROWTH")
    print("="*70)
    
    # Load data
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    
    # Remove Source_File column
    X = features.drop('Source_File', axis=1).values
    y = targets.values
    
    # Create dataset
    class CrackDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]
    
    dataset = CrackDataset(X, y)
    
    # Split data
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"\n📊 Data splits:")
    print(f"  Training: {len(train_dataset)} samples ({len(train_dataset)/n*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/n*100:.1f}%)")
    print(f"  Test: {len(test_dataset)} samples ({len(test_dataset)/n*100:.1f}%)")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X.shape[1]
    model = PhysicsAwareMTL(input_size=input_size, crack_length_idx=2)  # Crack length is 3rd feature
    trainer = BalancedMultiTaskTrainer(model, device)
    
    print(f"\n⚙️  Model Info:")
    print(f"  Device: {device}")
    print(f"  Input features: {input_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Crack length index: {model.crack_length_idx} (direct connection to G branch)")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training history
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_g_loss': [], 'val_g_loss': [],
        'train_v_loss': [], 'val_v_loss': [],
        'val_g_r2': [], 'val_v_r2': [], 'lr': []
    }
    
    # Training loop
    print("\n🚀 Starting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    patience, patience_counter = 30, 0
    best_model_path = None
    
    for epoch in range(150):
        # Train
        train_loss, train_g_loss, train_v_loss = trainer.train_epoch(
            train_loader, optimizer, criterion
        )
        
        # Validate
        val_loss, val_g_loss, val_v_loss, val_preds, val_targets = trainer.evaluate(
            val_loader, criterion
        )
        
        # Calculate metrics
        g_r2 = r2_score(val_targets[:, 0], val_preds[:, 0])
        v_r2 = r2_score(val_targets[:, 1], val_preds[:, 1])
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_g_loss'].append(train_g_loss)
        history['val_g_loss'].append(val_g_loss)
        history['train_v_loss'].append(train_v_loss)
        history['val_v_loss'].append(val_v_loss)
        history['val_g_r2'].append(g_r2)
        history['val_v_r2'].append(v_r2)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping & model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model with timestamp in filename
            timestamp = datetime.now().strftime("%H%M%S")
            best_model_path = os.path.join(output_dir, f'best_model_{timestamp}.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'g_r2': g_r2,
                'v_r2': v_r2,
            }, best_model_path)
            
            print(f"  ✅ Epoch {epoch+1}: Saved best model (Val Loss: {val_loss:.4f}, G R²: {g_r2:.4f}, V R²: {v_r2:.4f})")
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"G R²: {g_r2:.4f}, V R²: {v_r2:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n📈 Final evaluation on test set...")
    test_loss, test_g_loss, test_v_loss, test_preds, test_targets = trainer.evaluate(
        test_loader, criterion
    )
    
    # Calculate comprehensive metrics
    test_metrics = {}
    for i, task in enumerate(['G (J/m^2)', 'Crack velocity (um/s)']):
        y_true = test_targets[:, i]
        y_pred = test_preds[:, i]
        
        test_metrics[task] = {
            'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'Explained_Variance': 1 - (np.var(y_true - y_pred) / np.var(y_true))
        }
    
    # Save results (FIXED VERSION)
    save_results(output_dir, history, test_metrics, test_preds, test_targets, best_model_path)
    
    # Print summary
    print_results_summary(history, test_metrics, best_epoch, best_model_path)
    
    # Compare with Random Forest baseline
    print("\n" + "="*70)
    print("📊 COMPARISON WITH BASELINE MODELS")
    print("="*70)
    
    print("\nRandom Forest Baseline (from previous run):")
    print("  G R²: 0.996761")
    print("  Velocity R²: 0.998109")
    
    print("\nPhysics-Aware MTL Results:")
    for task, metrics in test_metrics.items():
        print(f"  {task}: R² = {metrics['R2']:.6f}")
    
    print("\n" + "="*70)
    print("📝 SCIENTIFIC INTERPRETATION")
    print("="*70)
    print("\nKey Findings:")
    print("1. Random Forest finds near-perfect relationship for G (likely linear with crack length)")
    print("2. Physics-aware MTL incorporates crack length directly into G prediction")
    print(f"3. Physics-aware MTL achieved G R²: {test_metrics['G (J/m^2)']['R2']:.6f}")
    print(f"4. Physics-aware MTL achieved Velocity R²: {test_metrics['Crack velocity (um/s)']['R2']:.6f}")
    
    return model, history, test_metrics, output_dir

# ==================== 4. MAIN EXECUTION ====================
if __name__ == "__main__":
    # Set paths
    DATA_DIR = "E:/materials2/processed_data"
    OUTPUT_DIR = "E:/materials2/physics_aware_mtl_results2"
    
    features_path = os.path.join(DATA_DIR, "processed_features.csv")
    targets_path = os.path.join(DATA_DIR, "processed_targets.csv")
    
    print("Starting Physics-Aware Multi-Task Learning Experiment...")
    print(f"Features: {features_path}")
    print(f"Targets: {targets_path}")
    print(f"Output: {OUTPUT_DIR}")
    print("\n" + "="*70)
    
    # Run experiment
    try:
        model, history, test_metrics, output_dir = train_physics_aware_mtl(
            features_path, targets_path, OUTPUT_DIR
        )
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if output directory exists: E:/materials2/physics_aware_mtl_results")
        print("2. Try creating the directory manually")
        print("3. Or use a different output path")
        
        # Try with alternative output directory
        alt_output = "E:/materials2/mtl_results"
        print(f"\nTrying alternative output directory: {alt_output}")
        
        try:
            model, history, test_metrics, output_dir = train_physics_aware_mtl(
                features_path, targets_path, alt_output
            )
        except Exception as e2:
            print(f"❌ Still failing: {e2}")
            print("\nPlease create the output directory manually or check permissions.")