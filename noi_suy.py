import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import psutil
import time
import gc
import os
import warnings
warnings.filterwarnings('ignore')

def quick_impute_large_csv(input_file, output_file, chunk_size=1000000, n_workers=20):
    """
    Nội suy nhanh cho file CSV lớn với multiprocessing - Tối ưu cao
    
    Args:
        input_file: Đường dẫn file đầu vào
        output_file: Đường dẫn file đầu ra
        chunk_size: Kích thước chunk (1M cho máy mạnh)
        n_workers: Số workers (20 cho máy 24 cores)
    """
    print(f"🚀 === NỘI SUY FILE LỚN VỚI MULTIPROCESSING ===")
    print(f"💻 Cấu hình: {n_workers} workers, chunk size {chunk_size:,}")
    print(f"💾 RAM khả dụng: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    print(f"📁 File: {input_file}")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(input_file):
        print(f"❌ File không tồn tại: {input_file}")
        return False
    
    # Đọc mẫu để kiểm tra missing data
    print("📊 Kiểm tra dữ liệu thiếu...")
    sample = pd.read_csv(input_file, nrows=10000)
    missing_info = sample.isnull().sum()
    missing_cols = missing_info[missing_info > 0]
    
    if len(missing_cols) == 0:
        print("✅ Không có dữ liệu thiếu!")
        return True
    
    print(f"🔧 Tìm thấy {len(missing_cols)} cột có dữ liệu thiếu:")
    for col, count in missing_cols.items():
        pct = (count / len(sample)) * 100
        print(f"   {col}: {pct:.1f}%")
    
    # Xử lý từng chunk với multiprocessing
    print("🔄 Bắt đầu xử lý song song...")
    start_time = time.time()
    first_chunk = True
    total_processed = 0
    
    # Batch processing
    batch_size = n_workers * 2
    chunk_batch = []
    
    try:
        chunk_reader = pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, engine='c', buffer_lines=50000)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, chunk in enumerate(chunk_reader):
                chunk_batch.append(chunk)
                
                # Xử lý batch khi đầy
                if len(chunk_batch) >= batch_size:
                    # Submit batch để xử lý song song
                    futures = [executor.submit(fast_impute_chunk, chunk_data) for chunk_data in chunk_batch]
                    
                    # Thu thập kết quả và ghi file
                    batch_results = []
                    for future in as_completed(futures):
                        processed_chunk = future.result()
                        batch_results.append(processed_chunk)
                    
                    # Ghi tất cả kết quả của batch cùng lúc
                    if batch_results:
                        combined_batch = pd.concat(batch_results, ignore_index=True)
                        
                        if first_chunk:
                            combined_batch.to_csv(output_file, index=False, mode='w')
                            first_chunk = False
                        else:
                            combined_batch.to_csv(output_file, index=False, mode='a', header=False)
                        
                        total_processed += len(combined_batch)
                    
                    # Dọn dẹp batch
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
                futures = [executor.submit(fast_impute_chunk, chunk_data) for chunk_data in chunk_batch]
                
                for future in as_completed(futures):
                    processed_chunk = future.result()
                    
                    if first_chunk:
                        processed_chunk.to_csv(output_file, index=False, mode='w')
                        first_chunk = False
                    else:
                        processed_chunk.to_csv(output_file, index=False, mode='a', header=False)
                    
                    total_processed += len(processed_chunk)
    
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False
    
    elapsed_total = time.time() - start_time
    print(f"\n🎉 Hoàn thành! Đã xử lý {total_processed:,} dòng trong {elapsed_total:.1f} giây")
    print(f"⚡ Tốc độ trung bình: {total_processed/elapsed_total:,.0f} dòng/giây")
    print(f"📁 File kết quả: {output_file}")
    return True

def fast_impute_chunk(chunk):
    """
    Nội suy nhanh cho một chunk - Tối ưu tốc độ cao
    """
    # Lấy cột số nhanh
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return chunk.copy()
    
    # Kiểm tra missing nhanh
    missing_cols = [col for col in numeric_cols if chunk[col].isnull().any()]
    
    if len(missing_cols) == 0:
        return chunk.copy()
    
    # Chiến lược nội suy tối ưu:
    for col in missing_cols:
        missing_pct = chunk[col].isnull().sum() / len(chunk) * 100
        
        if missing_pct > 70:
            # Quá nhiều thiếu -> median nhanh
            chunk[col].fillna(chunk[col].median(), inplace=True)
        
        elif missing_pct > 30:
            # Thiếu nhiều -> KNN nhanh (k=3)
            if len(numeric_cols) > 1:
                try:
                    imputer = KNNImputer(n_neighbors=3)
                    chunk.loc[:, numeric_cols] = imputer.fit_transform(chunk[numeric_cols])
                    break  # Đã impute tất cả cột số
                except:
                    chunk[col].fillna(chunk[col].median(), inplace=True)
            else:
                chunk[col].fillna(chunk[col].median(), inplace=True)
        
        else:
            # Thiếu ít -> KNN chính xác (k=5)
            if len(numeric_cols) > 3:
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    chunk.loc[:, numeric_cols] = imputer.fit_transform(chunk[numeric_cols])
                    break  # Đã impute tất cả cột số
                except:
                    chunk[col].fillna(chunk[col].median(), inplace=True)
            else:
                chunk[col].fillna(chunk[col].median(), inplace=True)
    
    return chunk.copy()

def show_sample_output(output_file, num_rows=10):
    """
    Hiển thị dữ liệu mẫu từ file kết quả
    """
    print(f"\n📋 === {num_rows} DÒNG ĐẦU CỦA FILE {output_file} ===")
    
    try:
        # Đọc file kết quả
        sample = pd.read_csv(output_file, nrows=num_rows)
        
        print(f"📊 Kích thước file: {sample.shape[0]} dòng (mẫu) × {sample.shape[1]} cột")
        print(f"📋 Các cột: {list(sample.columns)}")
        
        # Set display options for better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.6f}'.format)
        
        print(f"\n📊 DỮ LIỆU {num_rows} DÒNG ĐẦU:")
        print(sample.to_string(index=True))
        
        # Thống kê missing values sau nội suy
        print(f"\n📈 THỐNG KÊ SAU NỘI SUY:")
        missing_info = sample.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            print("⚠️ Các cột vẫn còn missing:")
            for col, count in missing_cols.items():
                pct = (count / len(sample)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print("✅ Không còn missing values!")
        
        # Thống kê cột số
        numeric_cols = sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📊 THỐNG KÊ CÁC CỘT SỐ:")
            for col in numeric_cols[:3]:  # Hiển thị 3 cột đầu
                col_stats = sample[col].describe()
                print(f"  {col}: Min={col_stats['min']:.3f}, Max={col_stats['max']:.3f}, Mean={col_stats['mean']:.3f}")
        
        # Kiểm tra file size
        file_size = os.path.getsize(output_file) / (1024**3)  # GB
        print(f"\n💾 Kích thước file: {file_size:.2f} GB")
        
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")

def quick_twi_fix(input_file, output_file, chunk_size=1000000, n_workers=20):
    """
    Sửa nhanh cột TWI bị thiếu với multiprocessing - Đặc biệt tối ưu
    """
    print("🌊 === SỬA NHANH CỘT TWI VỚI MULTIPROCESSING ===")
    print(f"💻 Cấu hình: {n_workers} workers, chunk size {chunk_size:,}")
    
    start_time = time.time()
    first_chunk = True
    total_processed = 0
    batch_size = n_workers * 2
    chunk_batch = []
    
    chunk_reader = pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, engine='c', buffer_lines=50000)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i, chunk in enumerate(chunk_reader):
            chunk_batch.append(chunk)
            
            # Xử lý batch khi đầy
            if len(chunk_batch) >= batch_size:
                futures = [executor.submit(process_twi_chunk, chunk_data) for chunk_data in chunk_batch]
                
                batch_results = []
                for future in as_completed(futures):
                    processed_chunk = future.result()
                    batch_results.append(processed_chunk)
                
                # Ghi batch
                if batch_results:
                    combined_batch = pd.concat(batch_results, ignore_index=True)
                    
                    if first_chunk:
                        combined_batch.to_csv(output_file, index=False, mode='w')
                        first_chunk = False
                    else:
                        combined_batch.to_csv(output_file, index=False, mode='a', header=False)
                    
                    total_processed += len(combined_batch)
                
                # Dọn dẹp
                chunk_batch = []
                del batch_results
                gc.collect()
                
                # Progress
                elapsed = time.time() - start_time
                speed = total_processed / elapsed if elapsed > 0 else 0
                print(f"⚡ TWI: {total_processed:,} dòng | {speed:,.0f} dòng/s")
    
    # Xử lý batch cuối
    if chunk_batch:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_twi_chunk, chunk_data) for chunk_data in chunk_batch]
            
            for future in as_completed(futures):
                processed_chunk = future.result()
                
                if first_chunk:
                    processed_chunk.to_csv(output_file, index=False, mode='w')
                    first_chunk = False
                else:
                    processed_chunk.to_csv(output_file, index=False, mode='a', header=False)
                
                total_processed += len(processed_chunk)
    
    elapsed_total = time.time() - start_time
    print(f"\n🎉 Hoàn thành TWI! {total_processed:,} dòng trong {elapsed_total:.1f} giây")
    print(f"⚡ Tốc độ: {total_processed/elapsed_total:,.0f} dòng/giây")

def process_twi_chunk(chunk):
    """
    Worker function xử lý TWI cho một chunk
    """
    # Xử lý TWI
    if 'twi' in chunk.columns:
        missing_pct = chunk['twi'].isnull().sum() / len(chunk) * 100
        
        if missing_pct > 0:
            # Tìm cột địa hình liên quan
            topo_cols = [col for col in ['dem', 'slope', 'curvature', 'aspect'] 
                       if col in chunk.columns and chunk[col].isnull().sum() < len(chunk) * 0.1]
            
            if len(topo_cols) >= 2 and missing_pct < 50:
                # Dùng RF nhanh
                mask_missing = chunk['twi'].isnull()
                chunk_complete = chunk[~mask_missing]
                chunk_missing = chunk[mask_missing]
                
                if len(chunk_complete) > 20:
                    try:
                        rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=2)
                        rf.fit(chunk_complete[topo_cols], chunk_complete['twi'])
                        predicted = rf.predict(chunk_missing[topo_cols])
                        chunk.loc[mask_missing, 'twi'] = predicted
                    except:
                        chunk['twi'].fillna(chunk['twi'].median(), inplace=True)
                else:
                    chunk['twi'].fillna(chunk['twi'].median(), inplace=True)
            else:
                # Dùng median
                chunk['twi'].fillna(chunk['twi'].median(), inplace=True)
    
    return chunk.copy()

# Sử dụng
if __name__ == "__main__":
    # Cấu hình cho máy mạnh
    INPUT_FILE = "your_large_file.csv"
    OUTPUT_FILE = "data_imputed.csv"
    CHUNK_SIZE = 1000000  # 1M dòng cho máy mạnh
    N_WORKERS = 20  # 20/24 cores
    
    print("🚀 === CHƯƠNG TRÌNH NỘI SUY TỐI ƯU CAO ===")
    print(f"💻 Cấu hình: {N_WORKERS} workers, chunk {CHUNK_SIZE:,}")
    print(f"💾 RAM khả dụng: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # Chọn một trong hai:
    
    # 1. Nội suy tổng quát (multiprocessing)
    success = quick_impute_large_csv(INPUT_FILE, OUTPUT_FILE, 
                                   chunk_size=CHUNK_SIZE, n_workers=N_WORKERS)
    
    # 2. Chỉ sửa TWI (multiprocessing siêu nhanh)
    # success = quick_twi_fix(INPUT_FILE, OUTPUT_FILE, 
    #                        chunk_size=CHUNK_SIZE, n_workers=N_WORKERS)
    
    if success:
        print("✅ Nội suy hoàn thành thành công!")
        
        # Hiển thị 10 dòng đầu của file kết quả
        show_sample_output(OUTPUT_FILE, num_rows=10)
        
        print(f"\n🎯 File kết quả đã được lưu: {OUTPUT_FILE}")
        print("👋 Chương trình hoàn thành!")
    else:
        print("❌ Có lỗi xảy ra!")