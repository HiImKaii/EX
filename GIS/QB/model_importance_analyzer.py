import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.inspection import permutation_importance
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import argparse
import pickle
import rasterio
from rasterio.plot import show
import geopandas as gpd
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.patches as mpatches


def load_model(model_path):
    """
    Load a trained model from disk
    """
    print(f"Loading model from {model_path}...")
    try:
        # Try different loading methods
        if model_path.endswith('.pkl'):
            model = pickle.load(open(model_path, 'rb'))
        elif model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        else:
            # Try to determine model type from file
            model = joblib.load(model_path)
        
        # Identify model type
        if isinstance(model, RandomForestClassifier):
            model_type = "Random Forest"
        elif isinstance(model, xgb.XGBClassifier):
            model_type = "XGBoost"
        else:
            model_type = "Unknown"
            
        print(f"Successfully loaded {model_type} model")
        return model, model_type
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def load_data(data_path):
    """
    Load data from CSV file
    """
    print(f"Loading data from {data_path}...")
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded {data.shape[0]} samples with {data.shape[1]} features")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def analyze_feature_importance(model, model_type, feature_names, output_dir='.'):
    """
    Analyze and visualize feature importance from model
    """
    print(f"Analyzing feature importance for {model_type} model...")
    
    # Get feature importance
    if model_type == "Random Forest":
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        has_std = True
    elif model_type == "XGBoost":
        importances = model.feature_importances_
        has_std = False
    else:
        print("Unknown model type, can't extract feature importance")
        return
    
    # Sort importances
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Print feature importance
    print("\nFeature importance ranking:")
    for i, (feature, importance) in enumerate(zip(sorted_features, sorted_importances)):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    if has_std:
        sorted_std = std[indices]
        plt.bar(range(len(sorted_importances)), sorted_importances, 
                yerr=sorted_std, align='center', alpha=0.7)
    else:
        plt.bar(range(len(sorted_importances)), sorted_importances, align='center', alpha=0.7)
    
    plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance ({model_type})')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'feature_importance_{model_type}.png')
    plt.savefig(output_path, dpi=300)
    print(f"Feature importance plot saved to {output_path}")
    plt.close()
    
    # Return sorted features and importances for further analysis
    return sorted_features, sorted_importances


def compute_permutation_importance(model, X, y, feature_names, n_repeats=10, output_dir='.', model_type=""):
    """
    Compute permutation importance, which is model-agnostic
    """
    print(f"Computing permutation importance with {n_repeats} repeats...")
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    
    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    # Print permutation importance
    print("\nPermutation importance ranking:")
    for i, idx in enumerate(sorted_idx):
        print(f"{i+1}. {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} ± {perm_importance.importances_std[idx]:.4f}")
    
    # Plot permutation importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_idx)), 
            perm_importance.importances_mean[sorted_idx],
            yerr=perm_importance.importances_std[sorted_idx],
            align='center')
    plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Permutation Importance')
    plt.title(f'Permutation Feature Importance ({model_type})')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'permutation_importance_{model_type}.png')
    plt.savefig(output_path, dpi=300)
    print(f"Permutation importance plot saved to {output_path}")
    plt.close()
    
    return sorted_idx, perm_importance.importances_mean[sorted_idx], perm_importance.importances_std[sorted_idx]


def visualize_importance_on_map(raster_file, importance_value, feature_name, output_dir='.', model_type="", study_area_shapefile=None):
    """
    Visualize a raster layer with importance weighting
    """
    print(f"Visualizing importance for {feature_name} on map...")
    
    try:
        with rasterio.open(raster_file) as src:
            # Read raster data
            data = src.read(1)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Normalize data for visualization
            norm_data = data.copy()
            mask = ~np.isnan(norm_data)
            if mask.any():  # Check if there are any valid values
                vmin, vmax = np.nanmin(norm_data), np.nanmax(norm_data)
                norm_data = (norm_data - vmin) / (vmax - vmin) if vmax > vmin else norm_data
            
            # Create custom colormap with alpha channel based on importance
            cmap = plt.cm.viridis.copy()
            
            # Adjust alpha based on importance (higher importance = more opaque)
            alpha_factor = min(1.0, importance_value * 5)  # Scale importance to reasonable alpha
            cmap.set_bad(alpha=0)
            
            # Plot the raster
            im = show(norm_data, ax=ax, cmap=cmap, alpha=alpha_factor, title=f"{feature_name} (Importance: {importance_value:.4f})")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label(feature_name)
            
            # Add study area boundary if provided
            if study_area_shapefile and os.path.exists(study_area_shapefile):
                gdf = gpd.read_file(study_area_shapefile)
                gdf = gdf.to_crs(src.crs)
                gdf.boundary.plot(ax=ax, color='red', linewidth=2)
            
            # Add importance legend
            importance_patch = mpatches.Patch(color='black', alpha=alpha_factor, 
                                            label=f'Importance: {importance_value:.4f}')
            ax.legend(handles=[importance_patch], loc='lower right')
            
            # Save figure
            output_path = os.path.join(output_dir, f'map_importance_{model_type}_{feature_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Map saved to {output_path}")
            plt.close()
            
    except Exception as e:
        print(f"Error visualizing raster: {e}")


def create_combined_importance_map(raster_files, feature_names, importances, output_dir='.', model_type="", study_area_shapefile=None):
    """
    Create a combined importance map from multiple rasters weighted by their importance
    """
    print("Creating combined importance map...")
    
    # Check if we have raster files for all features
    if len(raster_files) != len(feature_names):
        print(f"Warning: Number of raster files ({len(raster_files)}) doesn't match number of features ({len(feature_names)})")
        # Match raster files to feature names based on name similarity
        matched_rasters = []
        for feature in feature_names:
            best_match = None
            for raster in raster_files:
                if feature.lower() in raster.lower():
                    best_match = raster
                    break
            matched_rasters.append(best_match)
        raster_files = matched_rasters
    
    # Check which raster files exist
    existing_rasters = []
    existing_features = []
    existing_importances = []
    
    for raster, feature, importance in zip(raster_files, feature_names, importances):
        if raster and os.path.exists(raster):
            existing_rasters.append(raster)
            existing_features.append(feature)
            existing_importances.append(importance)
        else:
            print(f"Raster file for {feature} not found: {raster}")
    
    if not existing_rasters:
        print("No valid raster files found, cannot create combined map")
        return
    
    try:
        # Create combined importance map
        combined_data = None
        geo_transform = None
        crs = None
        
        # Normalize importances to sum to 1
        norm_importances = np.array(existing_importances) / np.sum(existing_importances)
        
        # Read and combine rasters
        for i, (raster, importance) in enumerate(zip(existing_rasters, norm_importances)):
            with rasterio.open(raster) as src:
                data = src.read(1)
                
                # On first raster, initialize output
                if combined_data is None:
                    combined_data = np.zeros_like(data)
                    geo_transform = src.transform
                    crs = src.crs
                
                # Normalize data
                norm_data = data.copy()
                mask = ~np.isnan(norm_data)
                if mask.any():
                    vmin, vmax = np.nanmin(norm_data[mask]), np.nanmax(norm_data[mask])
                    if vmax > vmin:
                        norm_data[mask] = (norm_data[mask] - vmin) / (vmax - vmin)
                
                # Weight by importance and add to combined data
                combined_data += norm_data * importance
        
        # Create output visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create custom colormap
        cmap = plt.cm.plasma.copy()
        cmap.set_bad(alpha=0)
        
        # Plot combined map
        show(combined_data, ax=ax, cmap=cmap, title=f"Combined Feature Importance ({model_type})")
        
        # Add colorbar
        cbar = plt.colorbar(ax=ax, shrink=0.7)
        cbar.set_label("Weighted Importance")
        
        # Add study area boundary if provided
        if study_area_shapefile and os.path.exists(study_area_shapefile):
            gdf = gpd.read_file(study_area_shapefile)
            gdf = gdf.to_crs(crs)
            gdf.boundary.plot(ax=ax, color='red', linewidth=2)
        
        # Add importance legend for top 5 features
        handles = []
        for i in range(min(5, len(existing_features))):
            handles.append(mpatches.Patch(
                color=plt.cm.tab10(i), 
                label=f'{existing_features[i]}: {existing_importances[i]:.4f}'
            ))
        
        ax.legend(handles=handles, loc='lower right')
        
        # Save figure
        output_path = os.path.join(output_dir, f'combined_importance_map_{model_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined importance map saved to {output_path}")
        plt.close()
        
        # Save as GeoTIFF if we have geospatial information
        if geo_transform is not None and crs is not None:
            output_tiff = os.path.join(output_dir, f'combined_importance_map_{model_type}.tif')
            with rasterio.open(
                output_tiff, 'w',
                driver='GTiff',
                height=combined_data.shape[0],
                width=combined_data.shape[1],
                count=1,
                dtype=combined_data.dtype,
                crs=crs,
                transform=geo_transform,
                nodata=np.nan
            ) as dst:
                dst.write(combined_data, 1)
            print(f"Combined importance map GeoTIFF saved to {output_tiff}")
            
    except Exception as e:
        print(f"Error creating combined importance map: {e}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze feature importance for flood prediction models')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--data', required=True, help='Path to the CSV data file used for training')
    parser.add_argument('--output', default='output', help='Directory to save output files (default: output)')
    parser.add_argument('--raster_dir', help='Directory containing raster files for visualization')
    parser.add_argument('--study_area', help='Shapefile of study area for visualization')
    parser.add_argument('--permutation', action='store_true', help='Compute permutation importance')
    parser.add_argument('--n_repeats', type=int, default=10, help='Number of repeats for permutation importance')
    parser.add_argument('--visualize_maps', action='store_true', help='Create importance visualizations on maps')
    parser.add_argument('--scaler', help='Path to fitted scaler (if available)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model, model_type = load_model(args.model)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load data
    data = load_data(args.data)
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare feature names and data
    if 'flood' in data.columns:
        X = data.drop('flood', axis=1)
        y = data['flood']
    else:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    
    feature_names = X.columns.tolist()
    
    # Load scaler if provided
    scaler = None
    if args.scaler:
        try:
            scaler = joblib.load(args.scaler)
            X_scaled = scaler.transform(X)
        except Exception as e:
            print(f"Error loading scaler: {e}. Using unscaled data.")
            X_scaled = X.values
    else:
        X_scaled = X.values
    
    # Analyze built-in feature importance
    sorted_features, sorted_importances = analyze_feature_importance(
        model, model_type, feature_names, args.output
    )
    
    # Compute permutation importance if requested
    if args.permutation:
        sorted_idx, perm_importances, perm_std = compute_permutation_importance(
            model, X_scaled, y, feature_names, args.n_repeats, args.output, model_type
        )
    
    # Visualize importance on maps if requested
    if args.visualize_maps and args.raster_dir:
        # Get raster files
        raster_files = []
        for feature in feature_names:
            # Try to find a matching raster file
            potential_files = [
                os.path.join(args.raster_dir, f"{feature}.tif"),
                os.path.join(args.raster_dir, f"{feature.lower()}.tif"),
                os.path.join(args.raster_dir, f"{feature}.tiff"),
                # Add more potential patterns if needed
            ]
            
            found = False
            for file in potential_files:
                if os.path.exists(file):
                    raster_files.append(file)
                    found = True
                    break
            
            if not found:
                print(f"No raster file found for feature {feature}")
                raster_files.append(None)
        
        # Create individual importance maps
        importance_dict = dict(zip(sorted_features, sorted_importances))
        for feature, raster in zip(feature_names, raster_files):
            if raster and os.path.exists(raster):
                visualize_importance_on_map(
                    raster, importance_dict.get(feature, 0),
                    feature, args.output, model_type, args.study_area
                )
        
        # Create combined importance map
        create_combined_importance_map(
            raster_files, feature_names, [importance_dict.get(f, 0) for f in feature_names],
            args.output, model_type, args.study_area
        )
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 