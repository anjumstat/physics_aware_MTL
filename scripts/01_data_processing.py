import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import warnings
import json
warnings.filterwarnings('ignore')

def parse_experimental_data_final(filepath):
    """
    Parse experimental data files with your specific format
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract metadata from first line
    metadata = {}
    if len(lines) > 0 and 'Spacer height at contact point:' in lines[0]:
        # First line format: 'Spacer height at contact point:,0.9317022199006851,Mica thickness = 2.097e-05 m\n'
        parts = lines[0].strip().split(',')
        
        # Parse spacer height (second element after colon)
        if len(parts) > 1:
            try:
                metadata['Spacer_Height'] = float(parts[1])
            except:
                metadata['Spacer_Height'] = np.nan
        
        # Parse mica thickness (third element, need to extract number)
        if len(parts) > 2 and '=' in parts[2]:
            thickness_part = parts[2].split('=')[1].strip().split()[0]
            try:
                metadata['Mica_Thickness'] = float(thickness_part)
            except:
                metadata['Mica_Thickness'] = np.nan
    
    # Read data starting from line 1 (header is at line 1)
    # Skip first line (metadata) and read from second line (header)
    try:
        data = pd.read_csv(filepath, skiprows=1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Clean column names (remove quotes if any)
    data.columns = [col.strip().replace('"', '') for col in data.columns]
    
    # Add metadata as columns
    for key, value in metadata.items():
        data[key] = value
    
    # Add source file identifier
    data['Source_File'] = os.path.basename(filepath)
    
    return data

def preprocess_crack_data_final(data_dir, target_tasks=None):
    """
    Main preprocessing function for your specific format
    """
    if target_tasks is None:
        target_tasks = ['G (J/m^2)', 'Crack velocity (um/s)']
    
    all_data = []
    file_count = 0
    
    # Process all CSV files in directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            print(f"Processing {filename}...")
            
            df = parse_experimental_data_final(filepath)
            if df is not None:
                print(f"  Success! Shape: {df.shape}")
                all_data.append(df)
                file_count += 1
    
    if file_count == 0:
        print("No CSV files successfully parsed!")
        return None
    
    # Combine all experiments
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined {file_count} files with {len(combined_df)} total rows")
    print(f"All columns: {combined_df.columns.tolist()}")
    
    # Check if target columns exist
    valid_targets = []
    for target in target_tasks:
        if target in combined_df.columns:
            valid_targets.append(target)
    
    if len(valid_targets) == 0:
        print("\nERROR: Target columns not found!")
        print(f"Looking for: {target_tasks}")
        print(f"Available columns: {combined_df.columns.tolist()}")
        return None
    
    print(f"\nTarget columns found: {valid_targets}")
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    
    # Define feature columns (excluding targets and metadata)
    exclude_cols = valid_targets + ['Source_File', 'Spacer_Height', 'Mica_Thickness', 
                                   'G error (J/m^2)', 'Crack Edge Error (um)']
    
    # Start with all columns except excluded ones
    feature_columns = [col for col in combined_df.columns if col not in exclude_cols]
    
    # Ensure metadata columns are included as features
    if 'Spacer_Height' in combined_df.columns and 'Spacer_Height' not in feature_columns:
        feature_columns.append('Spacer_Height')
    if 'Mica_Thickness' in combined_df.columns and 'Mica_Thickness' not in feature_columns:
        feature_columns.append('Mica_Thickness')
    
    print(f"Selected {len(feature_columns)} feature columns")
    print(f"Features: {feature_columns}")
    
    # Create feature matrix and target matrix
    X = combined_df[feature_columns].copy()
    y = pd.DataFrame()
    
    for target in valid_targets:
        y[target] = combined_df[target]
    
    # Handle missing values
    print(f"\nMissing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in targets: {y.isnull().sum().sum()}")
    
    if X.isnull().sum().sum() > 0:
        print("Filling missing values...")
        X = X.fillna(method='ffill').fillna(method='bfill')
        y = y.fillna(method='ffill').fillna(method='bfill')
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Add source file back for cross-validation
    X_scaled_df['Source_File'] = combined_df['Source_File'].values
    
    # Prepare multi-task output structure
    multi_task_data = {
        'features': X_scaled_df,
        'targets': y,
        'original_data': combined_df,
        'metadata': {
            'feature_names': X.columns.tolist(),
            'target_names': y.columns.tolist(),
            'experiment_files': combined_df['Source_File'].unique().tolist(),
            'total_experiments': file_count,
            'feature_columns': feature_columns,
            'target_columns': valid_targets,
            'scaler_mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
            'scaler_scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
        }
    }
    
    return multi_task_data

def save_processed_data(data, output_dir):
    """Save processed data to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features and targets
    data['features'].to_csv(os.path.join(output_dir, 'processed_features.csv'), index=False)
    data['targets'].to_csv(os.path.join(output_dir, 'processed_targets.csv'), index=False)
    
    # Save original combined data for reference
    data['original_data'].to_csv(os.path.join(output_dir, 'original_combined_data.csv'), index=False)
    
    # Save metadata as JSON (more reliable than DataFrame for mixed types)
    metadata_file = os.path.join(output_dir, 'processing_metadata.json')
    with open(metadata_file, 'w') as f:
        # Convert to JSON-serializable format
        metadata_serializable = {}
        for key, value in data['metadata'].items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                metadata_serializable[key] = value
            else:
                metadata_serializable[key] = str(value)
        
        json.dump(metadata_serializable, f, indent=2)
    
    # Also save a simplified metadata CSV
    simple_metadata = {
        'total_experiments': data['metadata']['total_experiments'],
        'total_samples': len(data['features']),
        'n_features': len(data['metadata']['feature_names']),
        'n_targets': len(data['metadata']['target_names']),
        'feature_columns': ', '.join(data['metadata']['feature_names']),
        'target_columns': ', '.join(data['metadata']['target_names']),
        'experiment_files': ', '.join(data['metadata']['experiment_files'])
    }
    
    pd.DataFrame([simple_metadata]).to_csv(
        os.path.join(output_dir, 'processing_summary.csv'), index=False
    )
    
    print(f"\nProcessed data saved to: {output_dir}")
    print(f"- Features shape: {data['features'].shape}")
    print(f"- Targets shape: {data['targets'].shape}")
    print(f"- Feature columns: {len(data['metadata']['feature_names'])}")
    print(f"- Target tasks: {data['metadata']['target_names']}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "E:/materials2/data"  # Your data directory
    OUTPUT_DIR = "E:/materials2/processed_data"  # Where to save processed data
    
    # Target tasks based on your file format
    TARGET_TASKS = [
        'G (J/m^2)',                # Fracture energy - PRIMARY TARGET
        'Crack velocity (um/s)',    # Crack velocity - SECONDARY TARGET
    ]
    
    print("="*60)
    print("CRACK GROWTH DATA PREPROCESSOR - FINAL VERSION (FIXED)")
    print("="*60)
    print("This script handles your specific file format:")
    print("1. Metadata line with spacer height and mica thickness")
    print("2. CSV header on line 1")
    print("3. Data rows starting from line 2")
    print("="*60)
    
    # Run preprocessing
    processed_data = preprocess_crack_data_final(DATA_DIR, TARGET_TASKS)
    
    if processed_data is not None:
        # Save results
        save_processed_data(processed_data, OUTPUT_DIR)
        
        # Display summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE - SUCCESS!")
        print("="*60)
        
        print(f"\n📊 DATA STATISTICS:")
        print(f"   Total experiments: {processed_data['metadata']['total_experiments']}")
        print(f"   Total data points: {len(processed_data['features'])}")
        print(f"   Feature dimensions: {len(processed_data['metadata']['feature_names'])}")
        
        print(f"\n🎯 TARGET VARIABLES:")
        for target in processed_data['metadata']['target_names']:
            target_data = processed_data['targets'][target]
            print(f"   {target}:")
            print(f"     Range: [{target_data.min():.4f}, {target_data.max():.4f}]")
            print(f"     Mean ± Std: {target_data.mean():.4f} ± {target_data.std():.4f}")
        
        print(f"\n🔧 FEATURES (physics-informed):")
        features = processed_data['metadata']['feature_names']
        for i, feat in enumerate(features):
            prefix = "• " if i < 9 else "  "
            print(f"   {prefix}{feat}")
        
        print(f"\n📁 OUTPUT FILES CREATED:")
        print(f"   1. processed_features.csv - Scaled features for ML")
        print(f"   2. processed_targets.csv - Target variables")
        print(f"   3. original_combined_data.csv - Raw combined data")
        print(f"   4. processing_metadata.json - Complete metadata")
        print(f"   5. processing_summary.csv - Quick summary")
        
        print("\n" + "="*60)
        print("🧠 READY FOR MULTI-TASK LEARNING MODEL")
        print("="*60)
        print("\nYour data is now prepared with:")
        print("✓ Physics features (Spacer_Height, Mica_Thickness)")
        print("✓ Time-series features for crack dynamics")
        print("✓ Standardized feature scaling")
        print("✓ Experiment traceability (Source_File column)")
        print("✓ Leave-One-File-Out validation capability")
        
        # Show a small sample
        print(f"\n📋 SAMPLE DATA (first 2 rows):")
        print("Features (scaled):")
        print(processed_data['features'].head(2).to_string())
        print("\nTargets:")
        print(processed_data['targets'].head(2).to_string())
        
    else:
        print("\n❌ Processing failed. Check the error messages above.")