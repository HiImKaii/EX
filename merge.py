import pandas as pd
import os

def merge_csv():
    """Merge specific CSV files for flood point data"""
    files = [
        r"D:\Flood_event_cleaned.csv",
        r"D:\MultipleFlood_event_cleaned.csv",
        r"D:\Non_flood_cleaned.csv"
    ]

    output_dir = r"D:"
    output_file = "flood_point_merge.csv"
    output_path = os.path.join(output_dir, output_file)
    
    # Verify files exist
    for f in files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return None
    
    print(f"Found {len(files)} files")
    
    # Read and merge all files
    dfs = [pd.read_csv(f) for f in files]
    result = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates and save
    result = result.drop_duplicates()
    result.to_csv(output_path, index=False)
    
    print(f"Merged {len(result)} rows into {output_path}")
    return result

if __name__ == "__main__":
    merge_csv()