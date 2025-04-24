import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point, shape
import os
import gc  # Garbage collector to manage memory
from tqdm import tqdm  # Progress bar

def create_sample_points(study_area_shapefile, num_points=500, output_file="sample_points.shp"):
    """
    Create random sample points within the study area.
    This is a method to create sample points if you don't have them already.
    """
    print("Creating random sample points...")
    
    # Read study area shapefile
    study_area = gpd.read_file(study_area_shapefile)
    
    # Create random points within the study area
    points = []
    bounds = study_area.total_bounds  # (minx, miny, maxx, maxy)
    
    for _ in range(num_points):
        x = np.random.uniform(bounds[0], bounds[2])
        y = np.random.uniform(bounds[1], bounds[3])
        point = Point(x, y)
        
        # Check if point is within study area
        if point.within(study_area.unary_union):
            points.append(point)
    
    # Create GeoDataFrame with points
    gdf = gpd.GeoDataFrame(geometry=points, crs=study_area.crs)
    
    # Add fire column with random values (0: no fire, 1: fire)
    # Note: This is just a placeholder, in practice you need real data about fire occurrence
    gdf['fire'] = np.random.choice([0, 1], size=len(gdf), p=[0.7, 0.3])  # Assume 30% points have fire
    
    # Save to shapefile
    gdf.to_file(output_file)
    print(f"Created and saved {len(gdf)} sample points to {output_file}")
    return gdf

def extract_values_from_known_points(points_shapefile, raster_directory, output_csv="forest_fire_data.csv", batch_size=10000):
    """
    Extract values from raster files at known points.
    Process in batches to avoid memory overflow with large datasets.
    
    Parameters:
    -----------
    points_shapefile : str
        Path to shapefile containing points
    raster_directory : str
        Path to directory containing raster files
    output_csv : str
        Path to CSV file to save results
    batch_size : int
        Batch size for processing (number of points per batch)
    """
    print("Extracting feature values...")
    
    # Read shapefile containing points
    print(f"Reading shapefile {points_shapefile}...")
    points = gpd.read_file(points_shapefile)
    total_points = len(points)
    print(f"Total points: {total_points}")
    
    # Check for fire column
    if 'fire' not in points.columns:
        print("Could not find 'fire' column in points file. Please add information about fire occurrence.")
        return
    
    # List of raster files to process
    raster_files = {
        "DEM": "DEM.tif",
        "AvgTemperature": "AvgTemperature.tif",
        "DroughtIndex": "DroughtIndex.tif",
        "RiverDensity": "RiverDensity.tif",
        "DistanceToRoad": "DistanceToRoad.tif",
        "LULC": "LULC.tif",
        "Slope": "Slope.tif",
        "Aspect": "Aspect.tif",
        "Elevation": "Elevation.tif",
        "Precipitation": "Precipitation.tif",
        "WindSpeed": "WindSpeed.tif"
    }
    
    # Check if output file already exists
    file_exists = os.path.exists(output_csv)
    
    # Calculate number of batches needed
    num_batches = (total_points + batch_size - 1) // batch_size
    
    # Process in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_points)
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (points {start_idx} to {end_idx-1})...")
        
        # Get batch points
        batch_points = points.iloc[start_idx:end_idx].copy()
        
        # Create DataFrame to store data for this batch
        batch_data = pd.DataFrame()
        batch_data['fire'] = batch_points['fire'].values
        
        # Extract values from raster files
        for column_name, raster_filename in raster_files.items():
            raster_path = os.path.join(raster_directory, raster_filename)
            
            if not os.path.exists(raster_path):
                print(f"File {raster_path} not found. Skipping.")
                continue
            
            try:
                with rasterio.open(raster_path) as src:
                    # Transform coordinates from points CRS to raster CRS if needed
                    if batch_points.crs != src.crs:
                        points_transformed = batch_points.to_crs(src.crs)
                    else:
                        points_transformed = batch_points
                    
                    # Prepare coordinate list for sampling
                    coords = [(point.x, point.y) for point in points_transformed.geometry]
                    
                    # Use sample method to extract values
                    values = [x[0] for x in src.sample(coords)]
                    
                    batch_data[column_name] = values
                    print(f"  Extracted {column_name} values from {raster_filename}")
            
            except Exception as e:
                print(f"Error processing file {raster_path}: {e}")
                batch_data[column_name] = np.nan
        
        # Remove rows with NaN values in this batch
        batch_data_cleaned = batch_data.dropna()
        print(f"  Removed {len(batch_data) - len(batch_data_cleaned)} rows with missing values in this batch.")
        
        # Save DataFrame to CSV (append mode for batches after first one)
        if batch_idx == 0 and file_exists:
            # If file exists and this is first batch, overwrite
            batch_data_cleaned.to_csv(output_csv, index=False, mode='w')
        elif batch_idx == 0:
            # First batch and file doesn't exist, create new with header
            batch_data_cleaned.to_csv(output_csv, index=False, mode='w')
        else:
            # Subsequent batches, append without header
            batch_data_cleaned.to_csv(output_csv, index=False, mode='a', header=False)
        
        print(f"  Saved {len(batch_data_cleaned)} data points from batch {batch_idx + 1} to {output_csv}")
        
        # Clean up memory
        del batch_points, batch_data, batch_data_cleaned
        gc.collect()
    
    print(f"Completed extraction for {total_points} points!")
    return

def process_fire_history_data(fire_history_shp, study_area_shapefile, raster_directory, output_csv="fire_history_data.csv"):
    """
    Process historical fire data and extract features at these locations.
    
    Parameters:
    -----------
    fire_history_shp : str
        Path to historical fire data shapefile
    study_area_shapefile : str
        Path to study area shapefile
    raster_directory : str
        Path to directory containing raster files
    output_csv : str
        Path to CSV file to save results
    """
    print("Processing historical fire data...")
    
    # Read fire history shapefile
    fire_history = gpd.read_file(fire_history_shp)
    study_area = gpd.read_file(study_area_shapefile)
    
    # Ensure the fire history points are in the study area
    if fire_history.crs != study_area.crs:
        fire_history = fire_history.to_crs(study_area.crs)
    
    # Clip fire history to study area
    fire_history = gpd.clip(fire_history, study_area)
    print(f"Found {len(fire_history)} historical fire points in study area")
    
    # Make sure we have a 'fire' column (1 for fire)
    if 'fire' not in fire_history.columns:
        fire_history['fire'] = 1
    
    # Save the processed fire history points
    temp_fire_points = "temp_fire_points.shp"
    fire_history.to_file(temp_fire_points)
    
    # Extract values from rasters at these points
    extract_values_from_known_points(temp_fire_points, raster_directory, output_csv)
    
    # Clean up temporary file
    if os.path.exists(temp_fire_points):
        import shutil
        shutil.rmtree(temp_fire_points.replace('.shp', ''))
    
    print(f"Completed processing historical fire data!")
    return

def generate_non_fire_points(fire_history_shp, study_area_shapefile, num_points=None, min_distance=1000, output_file="non_fire_points.shp"):
    """
    Generate points with no fire history, ensuring they're sufficiently far from known fire points.
    
    Parameters:
    -----------
    fire_history_shp : str
        Path to historical fire data shapefile
    study_area_shapefile : str
        Path to study area shapefile
    num_points : int, optional
        Number of non-fire points to generate. If None, will match number of fire points.
    min_distance : float
        Minimum distance (in projection units) from fire points
    output_file : str
        Path to output shapefile for non-fire points
    """
    print("Generating non-fire points...")
    
    # Read fire history and study area
    fire_history = gpd.read_file(fire_history_shp)
    study_area = gpd.read_file(study_area_shapefile)
    
    # Ensure same CRS
    if fire_history.crs != study_area.crs:
        fire_history = fire_history.to_crs(study_area.crs)
    
    # Determine number of points to generate
    if num_points is None:
        num_points = len(fire_history)
    
    # Create buffer around fire points
    fire_buffer = fire_history.buffer(min_distance).unary_union
    
    # Get area where we can place non-fire points (study area minus fire buffer)
    valid_area = study_area.unary_union.difference(fire_buffer)
    
    if valid_area.is_empty:
        print("WARNING: No valid area remains after applying minimum distance constraint.")
        print("Try reducing the minimum distance value.")
        return None
    
    # Generate random points in valid area
    non_fire_points = []
    bounds = study_area.total_bounds
    attempts = 0
    max_attempts = num_points * 100
    
    with tqdm(total=num_points) as pbar:
        while len(non_fire_points) < num_points and attempts < max_attempts:
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            point = Point(x, y)
            
            if point.within(valid_area):
                non_fire_points.append(point)
                pbar.update(1)
            
            attempts += 1
    
    if len(non_fire_points) < num_points:
        print(f"WARNING: Could only generate {len(non_fire_points)} non-fire points out of {num_points} requested.")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=non_fire_points, crs=study_area.crs)
    gdf['fire'] = 0  # 0 for no fire
    
    # Save to shapefile
    gdf.to_file(output_file)
    print(f"Created and saved {len(gdf)} non-fire points to {output_file}")
    return gdf

def create_balanced_dataset(fire_csv, non_fire_csv, output_csv="balanced_fire_data.csv"):
    """
    Create a balanced dataset from fire and non-fire point data.
    
    Parameters:
    -----------
    fire_csv : str
        Path to CSV with fire point data
    non_fire_csv : str
        Path to CSV with non-fire point data
    output_csv : str
        Path to output CSV for balanced dataset
    """
    print("Creating balanced dataset...")
    
    # Read data
    fire_data = pd.read_csv(fire_csv)
    non_fire_data = pd.read_csv(non_fire_csv)
    
    print(f"Fire data: {len(fire_data)} points")
    print(f"Non-fire data: {len(non_fire_data)} points")
    
    # Ensure we have a balanced dataset
    min_count = min(len(fire_data), len(non_fire_data))
    
    if len(fire_data) > min_count:
        fire_data = fire_data.sample(min_count, random_state=42)
    
    if len(non_fire_data) > min_count:
        non_fire_data = non_fire_data.sample(min_count, random_state=42)
    
    # Combine datasets
    balanced_data = pd.concat([fire_data, non_fire_data])
    
    # Shuffle data
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    balanced_data.to_csv(output_csv, index=False)
    print(f"Created balanced dataset with {len(balanced_data)} points and saved to {output_csv}")
    return balanced_data

def main():
    """
    Main function to demonstrate the workflow for forest fire feature extraction
    """
    # Parameters
    study_area_shapefile = "tuyen_quang_boundary.shp"
    raster_directory = "raster_data"
    fire_history_shp = "historical_fires.shp"
    
    # Process historical fire data
    process_fire_history_data(
        fire_history_shp, 
        study_area_shapefile, 
        raster_directory, 
        output_csv="fire_data.csv"
    )
    
    # Generate non-fire points
    generate_non_fire_points(
        fire_history_shp, 
        study_area_shapefile, 
        output_file="non_fire_points.shp"
    )
    
    # Process non-fire points
    extract_values_from_known_points(
        "non_fire_points.shp", 
        raster_directory, 
        output_csv="non_fire_data.csv"
    )
    
    # Create balanced dataset
    create_balanced_dataset(
        "fire_data.csv", 
        "non_fire_data.csv", 
        output_csv="balanced_fire_data.csv"
    )
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main() 