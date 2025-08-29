import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import gc
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import psutil

def process_large_csv(input_file, output_file, chunk_size=100000, n_workers=None):
    """
    Xử lý file CSV lớn theo chunks với multiprocessing
    
    Args:
        input_file (str): Đường dẫn file CSV đầu vào
        output_file (str): Đường dẫn file CSV đầu ra
        chunk_size (int): Kích thước mỗi chunk để xử lý (tăng lên cho máy mạnh)
        n_workers (int): Số worker processes (None = auto detect)
    """
    
    # Tự động phát hiện số worker tối ưu
    if n_workers is None:
        n_workers = min(24, cpu_count())  # Tối đa 24 cores như máy bạn
    
    print(f"🚀 Sử dụng {n_workers} workers với chunk size {chunk_size:,}")
    print(f"💾 RAM khả dụng: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # Kiểm tra file đầu vào có tồn tại không
    if not os.path.exists(input_file):
        print(f"Lỗi: File '{input_file}' không tồn tại!")
        return False
    
    # Danh sách các cột cần giữ lại (không bao gồm flood)
    required_columns = [
        'lulc', 'Density_River', 'Density_Road', 
        'Distan2river_met', 'Distan2road_met', 'aspect', 
        'curvature', 'dem', 'flowDir', 'slope', 'twi', 'NDVI'
    ]
    
    # Đọc header để phân tích các cột
    print("Đang đọc header và phân tích cấu trúc dữ liệu...")
    sample_df = pd.read_csv(input_file, nrows=0)  # Chỉ đọc header
    all_columns = sample_df.columns.tolist()
    
    print(f"Tổng số cột trong file: {len(all_columns)}")
    
    # Tìm các cột tọa độ (lat, lon)
    coord_columns = []
    for col in all_columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['lat', 'lon', 'longitude', 'latitude', 'x', 'y']):
            coord_columns.append(col)
    
    print(f"Các cột tọa độ tìm thấy: {coord_columns}")
    
    # Tìm cột NDVI 2024
    ndvi_2024_column = None
    for col in all_columns:
        col_lower = col.lower()
        if 'ndvi' in col_lower and '2024' in col:
            ndvi_2024_column = col
            break
    
    if ndvi_2024_column:
        print(f"Cột NDVI 2024 tìm thấy: {ndvi_2024_column}")
    else:
        print("Không tìm thấy cột NDVI 2024, sẽ tìm cột NDVI gần nhất")
        # Tìm cột NDVI gần nhất
        ndvi_columns = [col for col in all_columns if 'ndvi' in col.lower()]
        if ndvi_columns:
            ndvi_2024_column = ndvi_columns[-1]  # Lấy cột cuối cùng
            print(f"Sử dụng cột NDVI: {ndvi_2024_column}")
    
    # Xác định các cột cần đọc
    columns_to_read = []
    
    # Thêm cột tọa độ
    columns_to_read.extend(coord_columns)
    
    # Thêm các cột cần thiết
    for col in required_columns:
        if col == 'NDVI':
            if ndvi_2024_column and ndvi_2024_column in all_columns:
                columns_to_read.append(ndvi_2024_column)
        else:
            if col in all_columns:
                columns_to_read.append(col)
            else:
                print(f"Cảnh báo: Không tìm thấy cột '{col}'")
    
    # Loại bỏ cột trùng lặp
    columns_to_read = list(dict.fromkeys(columns_to_read))
    
    print(f"Tổng số cột sẽ được xử lý: {len(columns_to_read)}")
    
    # Xử lý file theo chunks
    first_chunk = True
    total_rows_processed = 0
    
    print("Bắt đầu xử lý file theo chunks...")
    
    try:
        # Đếm tổng số dòng để hiển thị progress
        print("Đang đếm tổng số dòng...")
        total_lines = sum(1 for line in open(input_file, 'r', encoding='utf-8')) - 1  # Trừ header
        total_chunks = (total_lines + chunk_size - 1) // chunk_size
        print(f"Tổng số dòng: {total_lines}, số chunks: {total_chunks}")
        
        chunk_reader = pd.read_csv(
            input_file, 
            chunksize=chunk_size,
            usecols=columns_to_read,
            low_memory=False
        )
        
        # Thu thập các chunk để xử lý batch
        print("🔄 Bắt đầu xử lý song song...")
        start_time = time.time()
        
        # Xử lý theo batch để tận dụng RAM
        batch_size = max(1, min(n_workers * 2, 32))  # Batch size tối ưu
        chunk_batch = []
        processed_rows = 0
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, chunk in enumerate(tqdm(chunk_reader, total=total_chunks, desc="Đọc chunks")):
                # Chuẩn bị dữ liệu cho multiprocessing
                chunk_args = (chunk, ndvi_2024_column, coord_columns)
                chunk_batch.append(chunk_args)
                
                # Xử lý batch khi đầy hoặc kết thúc
                if len(chunk_batch) >= batch_size or i == total_chunks - 1:
                    # Submit batch để xử lý song song
                    futures = [executor.submit(process_chunk_parallel, args) for args in chunk_batch]
                    
                    # Thu thập kết quả và ghi file
                    for j, future in enumerate(as_completed(futures)):
                        processed_chunk = future.result()
                        
                        if not processed_chunk.empty:
                            # Ghi chunk đã xử lý
                            if processed_rows == 0:
                                processed_chunk.to_csv(output_file, index=False, mode='w')
                            else:
                                processed_chunk.to_csv(output_file, index=False, mode='a', header=False)
                            
                            processed_rows += len(processed_chunk)
                    
                    # Dọn dẹp batch
                    chunk_batch = []
                    gc.collect()
                    
                    # Hiển thị tiến độ
                    elapsed = time.time() - start_time
                    speed = processed_rows / elapsed if elapsed > 0 else 0
                    print(f"⚡ Đã xử lý {processed_rows:,} dòng - Tốc độ: {speed:,.0f} dòng/giây")
        
        total_rows_processed = processed_rows
    
    except Exception as e:
        print(f"Lỗi khi xử lý file: {str(e)}")
        return False
    
    print(f"Hoàn thành! Đã xử lý {total_rows_processed:,} dòng")
    print(f"File kết quả được lưu tại: {output_file}")
    
    # Hiển thị thông tin file kết quả
    result_size = os.path.getsize(output_file) / (1024**3)  # GB
    print(f"Kích thước file kết quả: {result_size:.2f} GB")
    
    return True

def process_chunk_parallel(args):
    """
    Wrapper function để xử lý chunk trong multiprocessing
    """
    chunk_data, ndvi_2024_column, coord_columns = args
    return process_chunk(chunk_data, [], ndvi_2024_column, coord_columns)

def process_chunk(chunk, flood_columns, ndvi_2024_column, coord_columns):
    """
    Xử lý một chunk dữ liệu
    
    Args:
        chunk (DataFrame): Chunk dữ liệu cần xử lý
        flood_columns (list): Danh sách các cột lịch sử lũ (không sử dụng)
        ndvi_2024_column (str): Tên cột NDVI 2024
        coord_columns (list): Danh sách các cột tọa độ
    
    Returns:
        DataFrame: Chunk đã được xử lý
    """
    
    # Đổi tên cột NDVI 2024 thành NDVI
    if ndvi_2024_column and ndvi_2024_column in chunk.columns:
        chunk = chunk.rename(columns={ndvi_2024_column: 'NDVI'})
    
    # Sắp xếp lại thứ tự cột: lat, lon, lulc, rồi các feature khác
    final_columns = coord_columns + [
        'lulc', 'Density_River', 'Density_Road', 
        'Distan2river_met', 'Distan2road_met', 'aspect', 
        'curvature', 'dem', 'flowDir', 'slope', 'twi', 'NDVI'
    ]
    
    # Chỉ giữ các cột có trong chunk
    available_columns = [col for col in final_columns if col in chunk.columns]
    chunk = chunk[available_columns]
    
    return chunk

def analyze_csv_structure(input_file, num_rows=1000):
    """
    Phân tích cấu trúc file CSV để hiểu dữ liệu
    
    Args:
        input_file (str): Đường dẫn file CSV
        num_rows (int): Số dòng để phân tích mẫu
    """
    print("=== PHÂN TÍCH CẤU TRÚC FILE CSV ===")
    
    # Đọc mẫu dữ liệu
    sample_df = pd.read_csv(input_file, nrows=num_rows)
    
    print(f"Kích thước mẫu: {sample_df.shape}")
    print(f"Các cột trong file ({len(sample_df.columns)}):")
    
    # Phân loại các cột
    coord_cols = []
    flood_cols = []
    ndvi_cols = []
    other_cols = []
    
    for col in sample_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['lat', 'lon', 'longitude', 'latitude', 'x', 'y']):
            coord_cols.append(col)
        elif 'flood' in col_lower:
            flood_cols.append(col)
        elif 'ndvi' in col_lower:
            ndvi_cols.append(col)
        else:
            other_cols.append(col)
    
    print(f"\nCác cột tọa độ ({len(coord_cols)}): {coord_cols}")
    print(f"Các cột lũ ({len(flood_cols)}): {flood_cols[:10]}{'...' if len(flood_cols) > 10 else ''}")
    print(f"Các cột NDVI ({len(ndvi_cols)}): {ndvi_cols[:10]}{'...' if len(ndvi_cols) > 10 else ''}")
    print(f"Các cột khác ({len(other_cols)}): {other_cols[:10]}{'...' if len(other_cols) > 10 else ''}")
    
    # Hiển thị thống kê dữ liệu mẫu
    print(f"\nThống kê mẫu dữ liệu:")
    print(sample_df.describe())
    
    return sample_df

# Cách sử dụng
if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn
    INPUT_FILE = "your_large_file.csv"  # Thay bằng đường dẫn file 73GB của bạn
    OUTPUT_FILE = "processed_flood_data.csv"  # File kết quả
    
    # Tối ưu cho máy mạnh: 24 cores, 29GB RAM trống
    CHUNK_SIZE = 500000  # Tăng lên 500k dòng mỗi chunk (từ 10k)
    N_WORKERS = 20  # Sử dụng 20/24 cores, để lại 4 cores cho hệ thống
    
    print("🚀 === XỬ LÝ FILE CSV 73GB VỚI MULTIPROCESSING ===")
    print(f"💻 Cấu hình: {N_WORKERS} workers, chunk size {CHUNK_SIZE:,}")
    print(f"💾 RAM khả dụng: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # Phân tích cấu trúc file trước (tùy chọn)
    try:
        analyze_csv_structure(INPUT_FILE, num_rows=1000)
    except Exception as e:
        print(f"Không thể phân tích cấu trúc: {e}")
    
    print("\n" + "="*60)
    
    # Xử lý file chính với multiprocessing
    start_total = time.time()
    success = process_large_csv(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, N_WORKERS)
    end_total = time.time()
    
    if success:
        print(f"✅ Xử lý hoàn thành thành công trong {end_total - start_total:.1f} giây!")
        
        # Kiểm tra kết quả
    
    if success:
        print("✅ Xử lý hoàn thành thành công!")
        
        # Kiểm tra kết quả
        result_sample = pd.read_csv(OUTPUT_FILE, nrows=5)
        print(f"\nMẫu dữ liệu kết quả:")
        print(result_sample)
        print(f"Các cột trong file kết quả: {list(result_sample.columns)}")
    else:
        print("❌ Có lỗi xảy ra trong quá trình xử lý!")