import pandas as pd
import numpy as np
import os

def process_csv(file_path):
    # Đọc file CSV
    df = pd.read_csv(file_path)
    
    # Xóa các dòng chứa dữ liệu bị khuyết
    df = df.dropna()
    
    # Chuẩn hóa tất cả các cột số về khoảng [0,1] theo công thức (x-xmin)/(xmax-xmin)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0
    
    # Lưu file đã xử lý
    base_name = os.path.splitext(file_path)[0]
    output_path = f"{base_name}_normalized.csv"
    df.to_csv(output_path, index=False)
    
    return output_path

# Chương trình chính
if __name__ == "__main__":
    file_path = r"d:\25-26_HKI_DATN_QuanVX\merged_flood_point_merge_cleaned_balanced_reordered_nonlatlon_normalized_delindexNB.csv"
    process_csv(file_path)