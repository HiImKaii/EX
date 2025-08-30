import pandas as pd
import numpy as np
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import psutil
import time
import gc
import os
import warnings
warnings.filterwarnings('ignore')

def extract_rainfall_parallel(input_csv, rainfall_tiff, output_csv, chunk_size=1000000, n_workers=20):
    """
    Trích xuất giá trị lượng mưa từ TIFF và thêm vào CSV - Tối ưu cao
    
    Args:
        input_csv: File CSV đầu vào (sau khi chạy xoa_cot.py)
        rainfall_tiff: File TIFF chứa dữ liệu lượng mưa
        output_csv: File CSV đầu ra với cột rainfall
        chunk_size: Kích thước chunk (1M cho máy mạnh)
        n_workers: Số workers (20 cho máy 24 cores)
    """
    print(f"🌧️ === TRÍCH XUẤT LƯỢNG MƯA VỚI MULTIPROCESSING ===")
    print(f"💻 Cấu hình: {n_workers} workers, chunk size {chunk_size:,}")
    print(f"💾 RAM khả dụng: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(input_csv):
        print(f"❌ File CSV không tồn tại: {input_csv}")
        return False
    
    if not os.path.exists(rainfall_tiff):
        print(f"❌ File TIFF không tồn tại: {rainfall_tiff}")
        return False
    
    # Đọc thông tin TIFF
    print("📊 Đọc thông tin TIFF...")
    with rasterio.open(rainfall_tiff) as src:
        tiff_info = {
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'data': src.read(1)  # Đọc band đầu tiên
        }
    
    print(f"📐 TIFF: {tiff_info['width']}×{tiff_info['height']}, CRS: {tiff_info['crs']}")
    print(f"🌍 Bounds: {tiff_info['bounds']}")
    
    # Kiểm tra cấu trúc CSV
    print("📋 Kiểm tra cấu trúc CSV...")
    sample = pd.read_csv(input_csv, nrows=1000)
    columns = list(sample.columns)
    
    # Tìm cột tọa độ
    lat_col = None
    lon_col = None
    
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['lat', 'latitude', 'y']):
            lat_col = col
        elif any(keyword in col_lower for keyword in ['lon', 'long', 'longitude', 'x']):
            lon_col = col
    
    if not lat_col or not lon_col:
        print("❌ Không tìm thấy cột tọa độ (lat/lon)!")
        print(f"🔍 Các cột hiện có: {columns}")
        return False
    
    print(f"📍 Cột tọa độ: Lat={lat_col}, Lon={lon_col}")
    print(f"📊 Tổng số cột: {len(columns)}")
    
    # Xử lý với multiprocessing
    print("🔄 Bắt đầu trích xuất song song...")
    start_time = time.time()
    
    first_chunk = True
    total_processed = 0
    batch_size = n_workers * 2
    chunk_batch = []
    
    chunk_reader = pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False, engine='c')
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i, chunk in enumerate(chunk_reader):
            # Chuẩn bị args cho worker
            chunk_args = (chunk, lat_col, lon_col, tiff_info)
            chunk_batch.append(chunk_args)
            
            # Xử lý batch khi đầy
            if len(chunk_batch) >= batch_size:
                futures = [executor.submit(extract_rainfall_chunk, args) for args in chunk_batch]
                
                # Thu thập kết quả
                batch_results = []
                for future in as_completed(futures):
                    processed_chunk = future.result()
                    batch_results.append(processed_chunk)
                
                # Ghi batch
                if batch_results:
                    combined_batch = pd.concat(batch_results, ignore_index=True)
                    
                    if first_chunk:
                        combined_batch.to_csv(output_csv, index=False, mode='w')
                        first_chunk = False
                    else:
                        combined_batch.to_csv(output_csv, index=False, mode='a', header=False)
                    
                    total_processed += len(combined_batch)
                
                # Dọn dẹp
                chunk_batch = []
                del batch_results
                gc.collect()
                
                # Progress
                elapsed = time.time() - start_time
                speed = total_processed / elapsed if elapsed > 0 else 0
                print(f"⚡ {total_processed:,} dòng | {speed:,.0f} dòng/s")
    
    # Xử lý batch cuối
    if chunk_batch:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(extract_rainfall_chunk, args) for args in chunk_batch]
            
            for future in as_completed(futures):
                processed_chunk = future.result()
                
                if first_chunk:
                    processed_chunk.to_csv(output_csv, index=False, mode='w')
                    first_chunk = False
                else:
                    processed_chunk.to_csv(output_csv, index=False, mode='a', header=False)
                
                total_processed += len(processed_chunk)
    
    elapsed_total = time.time() - start_time
    print(f"\n🎉 Hoàn thành! Đã xử lý {total_processed:,} dòng trong {elapsed_total:.1f} giây")
    print(f"⚡ Tốc độ trung bình: {total_processed/elapsed_total:,.0f} dòng/giây")
    print(f"📁 File kết quả: {output_csv}")
    
    return True

def extract_rainfall_chunk(args):
    """
    Worker function để trích xuất lượng mưa cho một chunk
    """
    chunk, lat_col, lon_col, tiff_info = args
    
    # Lấy tọa độ
    lats = chunk[lat_col].values
    lons = chunk[lon_col].values
    
    # Chuyển đổi tọa độ thành pixel
    transform_matrix = tiff_info['transform']
    rainfall_data = tiff_info['data']
    
    # Trích xuất giá trị lượng mưa
    rainfall_values = []
    
    for lat, lon in zip(lats, lons):
        try:
            # Chuyển đổi tọa độ địa lý thành pixel
            col, row = ~transform_matrix * (lon, lat)
            col, row = int(col), int(row)
            
            # Kiểm tra trong bounds
            if 0 <= row < rainfall_data.shape[0] and 0 <= col < rainfall_data.shape[1]:
                rainfall_value = rainfall_data[row, col]
                
                # Kiểm tra NoData value (thường là NaN, -9999, hoặc rất lớn)
                if np.isnan(rainfall_value) or rainfall_value < -9000 or rainfall_value > 10000:
                    rainfall_values.append(0.0)  # Gán 0 cho NoData
                else:
                    rainfall_values.append(float(rainfall_value))
            else:
                rainfall_values.append(0.0)  # Ngoài bounds
                
        except Exception:
            rainfall_values.append(0.0)  # Lỗi thì gán 0
    
    # Thêm cột rainfall vào chunk
    chunk['rainfall'] = rainfall_values
    
    return chunk.copy()

def show_sample_output(output_csv):
    """
    Hiển thị 10 dòng đầu của file kết quả
    """
    print(f"\n📋 === 10 DÒNG ĐẦU CỦA FILE {output_csv} ===")
    
    try:
        sample = pd.read_csv(output_csv, nrows=10)
        print(f"📊 Kích thước file: {sample.shape[0]} dòng (mẫu) × {sample.shape[1]} cột")
        print(f"📋 Các cột: {list(sample.columns)}")
        
        # Hiển thị dữ liệu với format đẹp
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.6f}'.format)
        
        print("\n📊 DỮ LIỆU:")
        print(sample.to_string(index=True))
        
        # Thống kê cột rainfall
        if 'rainfall' in sample.columns:
            rainfall_stats = sample['rainfall'].describe()
            print(f"\n🌧️ THỐNG KÊ CỘT RAINFALL:")
            print(f"  Min: {rainfall_stats['min']:.2f}")
            print(f"  Max: {rainfall_stats['max']:.2f}")
            print(f"  Mean: {rainfall_stats['mean']:.2f}")
            print(f"  Std: {rainfall_stats['std']:.2f}")
            
            # Đếm số lượng giá trị 0 (NoData)
            zero_count = (sample['rainfall'] == 0).sum()
            print(f"  Số điểm có rainfall = 0: {zero_count}/{len(sample)}")
        
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")

# Chương trình chính
if __name__ == "__main__":
    # Đường dẫn file
    INPUT_CSV = "processed_flood_data.csv"  # File sau khi chạy xoa_cot.py
    RAINFALL_TIFF = "rainfall_data.tif"     # File TIFF lượng mưa
    OUTPUT_CSV = "flood_points.csv"         # File kết quả
    
    # Cấu hình cho máy mạnh
    CHUNK_SIZE = 1000000  # 1M dòng cho tối ưu tốc độ
    N_WORKERS = 20        # 20/24 cores
    
    print("🌧️ === CHƯƠNG TRÌNH TRÍCH XUẤT LƯỢNG MƯA ===")
    print(f"📂 Input CSV: {INPUT_CSV}")
    print(f"🗺️ Rainfall TIFF: {RAINFALL_TIFF}")
    print(f"📁 Output CSV: {OUTPUT_CSV}")
    print(f"💻 Cấu hình: {N_WORKERS} workers, chunk {CHUNK_SIZE:,}")
    
    # Kiểm tra file đầu vào
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Không tìm thấy file CSV: {INPUT_CSV}")
        print("💡 Hãy đảm bảo bạn đã chạy xoa_cot.py trước!")
        exit(1)
    
    if not os.path.exists(RAINFALL_TIFF):
        print(f"❌ Không tìm thấy file TIFF: {RAINFALL_TIFF}")
        print("💡 Hãy đặt file TIFF lượng mưa vào cùng thư mục!")
        exit(1)
    
    # Trích xuất lượng mưa
    success = extract_rainfall_parallel(
        input_csv=INPUT_CSV,
        rainfall_tiff=RAINFALL_TIFF,
        output_csv=OUTPUT_CSV,
        chunk_size=CHUNK_SIZE,
        n_workers=N_WORKERS
    )
    
    if success:
        print(f"\n✅ TRÍCH XUẤT HOÀN THÀNH!")
        
        # Hiển thị 10 dòng đầu
        show_sample_output(OUTPUT_CSV)
        
        print(f"\n🎯 File kết quả đã được lưu: {OUTPUT_CSV}")
        print("👋 Chương trình hoàn thành!")
    else:
        print("❌ Có lỗi xảy ra trong quá trình trích xuất!")
