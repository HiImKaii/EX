import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import psutil
import os
import warnings
warnings.filterwarnings('ignore')

class CSVDataNormalizer:
    def __init__(self, file_path, chunk_size=1000000, n_workers=None):
        """
        Khởi tạo với đường dẫn file CSV - Tối ưu tốc độ cao
        
        Args:
            file_path: Đường dẫn file CSV
            chunk_size: Kích thước chunk (1M cho máy mạnh)
            n_workers: Số workers (None = auto detect 20/24 cores)
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.n_workers = min(20, cpu_count() - 4) if n_workers is None else n_workers  # Để lại 4 cores
        self.df = None
        self.original_df = None
        self.flood_column = None
        self.processing_columns = []
        
        print(f"🚀 Normalizer tối ưu: {self.n_workers} workers, chunk {self.chunk_size:,}")
        print(f"💾 RAM khả dụng: {psutil.virtual_memory().available / (1024**3):.1f}GB")
        
    def load_data(self):
        """Tải dữ liệu từ file CSV"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.file_path, encoding=encoding)
                    self.original_df = self.df.copy()  # Lưu bản gốc
                    print(f"✅ Đọc file thành công: {self.df.shape[0]} hàng, {self.df.shape[1]} cột")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise Exception("Không thể đọc file với bất kỳ encoding nào")
            
            self._identify_columns()
            return True
            
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
            return False
    
    def _identify_columns(self):
        """Xác định các cột cần xử lý"""
        columns = list(self.df.columns)
        print(f"\n📋 Danh sách tất cả các cột: {columns}")
        
        if len(columns) >= 1:
            self.flood_column = columns[0]
            self.processing_columns = columns[1:]  # Tất cả các cột trừ cột flood
        else:
            print("⚠️ File phải có ít nhất 1 cột!")
            return
        
        print(f"🌊 Cột nhãn lũ (không xử lý): {self.flood_column}")
        print(f"🔧 Các cột cần xử lý ({len(self.processing_columns)}): {self.processing_columns}")
        
        # Hiển thị thông tin chi tiết về từng cột
        print(f"\n📊 Thông tin chi tiết từng cột:")
        for i, col in enumerate(self.processing_columns):
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            unique_count = self.df[col].nunique()
            print(f"  {i+1:2d}. {col:20s} | Kiểu: {str(dtype):10s} | Null: {null_count:4d} | Unique: {unique_count:4d}")
    
    def show_column_statistics(self, col_name):
        """Hiển thị thống kê của một cột"""
        if col_name not in self.df.columns:
            print(f"❌ Cột '{col_name}' không tồn tại")
            return
        
        print(f"\n📈 Thống kê cột '{col_name}':")
        print(f"  Kiểu dữ liệu: {self.df[col_name].dtype}")
        print(f"  Số giá trị null: {self.df[col_name].isnull().sum()}")
        print(f"  Số giá trị unique: {self.df[col_name].nunique()}")
        
        if pd.api.types.is_numeric_dtype(self.df[col_name]):
            stats = self.df[col_name].describe()
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
        else:
            print(f"  Top values: {dict(self.df[col_name].value_counts().head(3))}")
    
    def handle_missing_values_for_column(self, col_name):
        """Xử lý giá trị thiếu cho một cột cụ thể"""
        if col_name not in self.processing_columns:
            print(f"❌ Cột '{col_name}' không trong danh sách xử lý")
            return False
        
        null_count = self.df[col_name].isnull().sum()
        if null_count == 0:
            print(f"✅ Cột '{col_name}': Không có giá trị thiếu")
            return True
        
        print(f"🔧 Xử lý {null_count} giá trị thiếu trong cột '{col_name}'")
        
        if pd.api.types.is_numeric_dtype(self.df[col_name]):
            # Cột số: điền bằng mean
            mean_val = self.df[col_name].mean()
            self.df[col_name].fillna(mean_val, inplace=True)
            print(f"  ➤ Điền bằng mean: {mean_val:.4f}")
        else:
            # Cột phân loại: điền bằng mode
            mode_val = self.df[col_name].mode()[0] if not self.df[col_name].mode().empty else 'Unknown'
            self.df[col_name].fillna(mode_val, inplace=True)
            print(f"  ➤ Điền bằng mode: '{mode_val}'")
        
        return True
    
    def normalize_numeric_column(self, col_name, method='minmax'):
        """Chuẩn hóa một cột số về khoảng [0,1]"""
        if col_name not in self.processing_columns:
            print(f"❌ Cột '{col_name}' không trong danh sách xử lý")
            return False
        
        if not pd.api.types.is_numeric_dtype(self.df[col_name]):
            print(f"⚠️ Cột '{col_name}' không phải là số - bỏ qua chuẩn hóa")
            return False
        
        original_stats = {
            'min': self.df[col_name].min(),
            'max': self.df[col_name].max(),
            'mean': self.df[col_name].mean(),
            'std': self.df[col_name].std()
        }
        
        print(f"🎯 Chuẩn hóa cột '{col_name}' bằng phương pháp '{method}'")
        print(f"  Trước: min={original_stats['min']:.4f}, max={original_stats['max']:.4f}")
        
        if method == 'minmax':
            # Min-Max: (x - min) / (max - min) -> [0, 1]
            min_val = self.df[col_name].min()
            max_val = self.df[col_name].max()
            if max_val != min_val:
                self.df[col_name] = (self.df[col_name] - min_val) / (max_val - min_val)
            else:
                print(f"  ⚠️ Tất cả giá trị đều bằng nhau: {min_val}")
                self.df[col_name] = 0  # Gán về 0 nếu tất cả giá trị giống nhau
                
        elif method == 'standard':
            # Z-score: (x - mean) / std, sau đó áp dụng sigmoid để đưa về [0,1]
            mean_val = self.df[col_name].mean()
            std_val = self.df[col_name].std()
            if std_val != 0:
                z_scores = (self.df[col_name] - mean_val) / std_val
                # Áp dụng sigmoid: 1 / (1 + exp(-z))
                self.df[col_name] = 1 / (1 + np.exp(-z_scores))
            else:
                self.df[col_name] = 0.5  # Gán về 0.5 nếu std = 0
        
        new_stats = {
            'min': self.df[col_name].min(),
            'max': self.df[col_name].max(),
            'mean': self.df[col_name].mean()
        }
        
        print(f"  Sau:  min={new_stats['min']:.4f}, max={new_stats['max']:.4f}, mean={new_stats['mean']:.4f}")
        return True
    
    def encode_categorical_column(self, col_name):
        """Mã hóa một cột phân loại thành số trong khoảng [0,1]"""
        if col_name not in self.processing_columns:
            print(f"❌ Cột '{col_name}' không trong danh sách xử lý")
            return False
        
        if pd.api.types.is_numeric_dtype(self.df[col_name]):
            print(f"⚠️ Cột '{col_name}' đã là số - bỏ qua mã hóa")
            return False
        
        print(f"🏷️ Mã hóa cột phân loại '{col_name}'")
        unique_vals = self.df[col_name].nunique()
        print(f"  Số lượng nhóm: {unique_vals}")
        
        # Xử lý NaN trước khi mã hóa
        if self.df[col_name].isnull().any():
            self.df[col_name] = self.df[col_name].fillna('__MISSING__')
        
        # Mã hóa bằng LabelEncoder
        le = LabelEncoder()
        encoded_values = le.fit_transform(self.df[col_name])
        
        # Chuẩn hóa về [0,1] nếu có nhiều hơn 1 nhóm
        if len(le.classes_) > 1:
            encoded_values = encoded_values / (len(le.classes_) - 1)
        else:
            encoded_values = np.zeros_like(encoded_values)
        
        self.df[col_name] = encoded_values
        
        print(f"  Ánh xạ: {dict(zip(le.classes_, np.unique(encoded_values)))}")
        return True
    
    def process_column(self, col_name, normalization_method='minmax'):
        """Xử lý hoàn chỉnh một cột: missing values + chuẩn hóa/mã hóa"""
        print(f"\n{'='*60}")
        print(f"🔄 XỬ LÝ CỘT: '{col_name}'")
        print(f"{'='*60}")
        
        # Hiển thị thống kê ban đầu
        self.show_column_statistics(col_name)
        
        # Xử lý giá trị thiếu
        if not self.handle_missing_values_for_column(col_name):
            return False
        
        # Chuẩn hóa hoặc mã hóa
        if pd.api.types.is_numeric_dtype(self.df[col_name]):
            success = self.normalize_numeric_column(col_name, normalization_method)
        else:
            success = self.encode_categorical_column(col_name)
        
        if success:
            print(f"✅ Hoàn thành xử lý cột '{col_name}'")
        
        return success
    
    def is_column_normalized(self, col_name):
        """Kiểm tra xem một cột đã được chuẩn hóa hay chưa"""
        if not pd.api.types.is_numeric_dtype(self.df[col_name]):
            return False
        
        min_val = self.df[col_name].min()
        max_val = self.df[col_name].max()
        
        # Kiểm tra xem giá trị có nằm trong khoảng [0,1] không
        return (min_val >= -1e-10) and (max_val <= 1 + 1e-10)
    
    def process_all_columns(self, normalization_method='minmax'):
        """Xử lý tất cả các cột chưa được chuẩn hóa (trừ cột flood)"""
        print("🚀 BẮT ĐẦU KIỂM TRA VÀ XỬ LÝ CÁC CỘT")
        print("=" * 80)
        
        if not self.processing_columns:
            print("⚠️ Không có cột nào để xử lý")
            return False
        
        columns_to_process = []
        for col in self.processing_columns:
            if not self.is_column_normalized(col):
                columns_to_process.append(col)
                
        if not columns_to_process:
            print("✅ Tất cả các cột đã được chuẩn hóa")
            return True
            
        print(f"\n🎯 Phát hiện {len(columns_to_process)} cột cần chuẩn hóa:")
        for col in columns_to_process:
            print(f"  - {col}")
        
        success_count = 0
        for i, col in enumerate(columns_to_process):
            print(f"\n[{i+1}/{len(columns_to_process)}] Đang xử lý cột: {col}")
            if self.process_column(col, normalization_method):
                success_count += 1
        
        print(f"\n📊 KẾT QUẢ TỔNG QUAN:")
        print(f"  Tổng số cột cần xử lý: {len(columns_to_process)}")
        print(f"  Xử lý thành công: {success_count}")
        print(f"  Xử lý thất bại: {len(columns_to_process) - success_count}")
        
        return success_count == len(columns_to_process)
    
    def save_data(self, suffix="_normalized"):
        """Lưu dữ liệu đã xử lý"""
        base_name = os.path.splitext(self.file_path)[0]
        output_path = f"{base_name}{suffix}.csv"
        
        try:
            self.df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"💾 Đã lưu file: {output_path}")
            print(f"📏 Kích thước: {self.df.shape[0]} hàng x {self.df.shape[1]} cột")
            return output_path
        except Exception as e:
            print(f"❌ Lỗi lưu file: {str(e)}")
            return None
    
    def show_comparison(self, col_name, sample_size=10):
        """So sánh dữ liệu trước và sau xử lý"""
        if col_name not in self.df.columns:
            print(f"❌ Cột '{col_name}' không tồn tại")
            return
        
        print(f"\n🔍 SO SÁNH DỮ LIỆU CỘT '{col_name}':")
        comparison_df = pd.DataFrame({
            'Trước': self.original_df[col_name].head(sample_size),
            'Sau': self.df[col_name].head(sample_size)
        })
        print(comparison_df)

def process_chunk_normalization(args):
    """
    Worker function để chuẩn hóa chunk song song
    """
    chunk_data, processing_columns, flood_column, lulc_column, normalization_method = args
    
    # Xử lý missing values
    for col in processing_columns:
        if col in chunk_data.columns and chunk_data[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(chunk_data[col]):
                chunk_data[col].fillna(chunk_data[col].mean(), inplace=True)
            else:
                mode_val = chunk_data[col].mode()[0] if not chunk_data[col].mode().empty else 'Unknown'
                chunk_data[col].fillna(mode_val, inplace=True)
    
    # Chuẩn hóa các cột số
    for col in processing_columns:
        if col in chunk_data.columns and pd.api.types.is_numeric_dtype(chunk_data[col]):
            if normalization_method == 'minmax':
                min_val = chunk_data[col].min()
                max_val = chunk_data[col].max()
                if max_val != min_val:
                    chunk_data[col] = (chunk_data[col] - min_val) / (max_val - min_val)
                else:
                    chunk_data[col] = 0
            elif normalization_method == 'standard':
                mean_val = chunk_data[col].mean()
                std_val = chunk_data[col].std()
                if std_val != 0:
                    z_scores = (chunk_data[col] - mean_val) / std_val
                    chunk_data[col] = 1 / (1 + np.exp(-z_scores))
                else:
                    chunk_data[col] = 0.5
    
    return chunk_data

def normalize_large_csv_parallel(input_file, output_file, chunk_size=1000000, n_workers=20, normalization_method='minmax'):
    """
    Chuẩn hóa file CSV lớn với multiprocessing - Tối ưu tốc độ cao
    
    Args:
        input_file: File đầu vào
        output_file: File đầu ra
        chunk_size: Kích thước chunk (1M cho máy mạnh)
        n_workers: Số workers (20 cho máy 24 cores)
        normalization_method: 'minmax' hoặc 'standard'
    """
    print(f"🚀 === CHUẨN HÓA FILE LỚN VỚI MULTIPROCESSING TỐI ƯU ===")
    print(f"💻 Cấu hình: {n_workers} workers, chunk size {chunk_size:,}")
    print(f"💾 RAM khả dụng: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    # Đọc sample để xác định cấu trúc
    print("📊 Phân tích cấu trúc file...")
    sample = pd.read_csv(input_file, nrows=10000)
    columns = list(sample.columns)
    
    if len(columns) < 2:
        print("❌ File phải có ít nhất 2 cột!")
        return False
    
    flood_column = columns[0]
    lulc_column = columns[1] if len(columns) > 1 else None
    processing_columns = columns[2:] if lulc_column else columns[1:]
    
    print(f"🌊 Cột nhãn lũ (giữ nguyên): {flood_column}")
    print(f"🏠 Cột LULC (giữ nguyên): {lulc_column}")
    print(f"🔧 Số cột cần chuẩn hóa: {len(processing_columns)}")
    
    # Tính thống kê toàn cục trước (cần cho minmax)
    print("📈 Tính thống kê toàn cục...")
    global_stats = {}
    
    if normalization_method == 'minmax':
        chunk_reader = pd.read_csv(input_file, chunksize=chunk_size*2, low_memory=False, engine='c', buffer_lines=50000)
        
        for col in processing_columns:
            global_stats[col] = {'min': float('inf'), 'max': float('-inf')}
        
        for chunk in chunk_reader:
            for col in processing_columns:
                if col in chunk.columns and pd.api.types.is_numeric_dtype(chunk[col]):
                    col_min = chunk[col].min()
                    col_max = chunk[col].max()
                    if not pd.isna(col_min):
                        global_stats[col]['min'] = min(global_stats[col]['min'], col_min)
                    if not pd.isna(col_max):
                        global_stats[col]['max'] = max(global_stats[col]['max'], col_max)
        
        print(f"✓ Đã tính thống kê cho {len(global_stats)} cột")
    
    # Xử lý file với multiprocessing tối ưu
    print("🔄 Bắt đầu xử lý song song tối ưu...")
    start_time = time.time()
    
    chunk_reader = pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, engine='c', buffer_lines=50000)
    total_processed = 0
    first_chunk = True
    
    # Batch processing lớn để tối ưu RAM và CPU
    batch_size = n_workers * 3  # Tăng batch size
    chunk_batch = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i, chunk in enumerate(chunk_reader):
            # Áp dụng thống kê toàn cục cho minmax
            if normalization_method == 'minmax':
                for col in processing_columns:
                    if col in chunk.columns and col in global_stats:
                        min_val = global_stats[col]['min']
                        max_val = global_stats[col]['max']
                        if max_val != min_val and not (pd.isna(min_val) or pd.isna(max_val)):
                            chunk[col] = (chunk[col] - min_val) / (max_val - min_val)
                        else:
                            chunk[col] = 0
            
            # Chuẩn bị args cho worker
            chunk_args = (chunk, processing_columns, flood_column, lulc_column, normalization_method)
            chunk_batch.append(chunk_args)
            
            # Xử lý batch khi đầy
            if len(chunk_batch) >= batch_size:
                futures = [executor.submit(process_chunk_normalization, args) for args in chunk_batch]
                
                # Thu thập kết quả và ghi file theo batch
                batch_results = []
                for future in as_completed(futures):
                    processed_chunk = future.result()
                    batch_results.append(processed_chunk)
                
                # Ghi tất cả kết quả của batch cùng lúc để tối ưu I/O
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
                import gc
                gc.collect()
                
                # Progress mỗi 10 batch để giảm overhead
                if i % (batch_size * 10) == 0:
                    elapsed = time.time() - start_time
                    speed = total_processed / elapsed if elapsed > 0 else 0
                    print(f"⚡ {total_processed:,} dòng | {speed:,.0f} dòng/s")
        
        # Xử lý batch cuối
        if chunk_batch:
            futures = [executor.submit(process_chunk_normalization, args) for args in chunk_batch]
            
            for future in as_completed(futures):
                processed_chunk = future.result()
                
                if first_chunk:
                    processed_chunk.to_csv(output_file, index=False, mode='w')
                    first_chunk = False
                else:
                    processed_chunk.to_csv(output_file, index=False, mode='a', header=False)
                
                total_processed += len(processed_chunk)
    
    elapsed_total = time.time() - start_time
    print(f"\n🎉 Hoàn thành! Đã xử lý {total_processed:,} dòng trong {elapsed_total:.1f} giây")
    print(f"⚡ Tốc độ trung bình: {total_processed/elapsed_total:,.0f} dòng/giây")
    print(f"📁 File kết quả: {output_file}")
    
    return True

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
        
        # Thống kê chuẩn hóa
        print(f"\n📈 THỐNG KÊ SAU CHUẨN HÓA:")
        numeric_cols = sample.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            print("📊 Khoảng giá trị các cột số:")
            for col in numeric_cols:
                col_min = sample[col].min()
                col_max = sample[col].max()
                col_mean = sample[col].mean()
                
                # Kiểm tra xem có được chuẩn hóa chưa
                if col_min >= -0.01 and col_max <= 1.01:
                    status = "✅ Đã chuẩn hóa"
                else:
                    status = "⚠️ Chưa chuẩn hóa"
                
                print(f"  {col}: [{col_min:.6f}, {col_max:.6f}] Mean={col_mean:.6f} {status}")
        
        # Kiểm tra missing values
        missing_info = sample.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            print(f"\n⚠️ Cột còn missing values:")
            for col, count in missing_cols.items():
                pct = (count / len(sample)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print(f"\n✅ Không còn missing values!")
        
        # Kiểm tra file size
        file_size = os.path.getsize(output_file) / (1024**3)  # GB
        print(f"\n💾 Kích thước file: {file_size:.2f} GB")
        
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")

# Chương trình chính
if __name__ == "__main__":
    # Đường dẫn file CSV - Hỗ trợ cả file lớn và nhỏ
    file_path = r"C:\Users\Admin\Downloads\prj\Flood_point\merged_flood_point_merge_cleaned_balanced_reordered_nonlatlon.csv"
    
    # Kiểm tra kích thước file
    import os
    file_size_gb = os.path.getsize(file_path) / (1024**3) if os.path.exists(file_path) else 0
    
    print(f"📁 File: {os.path.basename(file_path)}")
    print(f"💾 Kích thước: {file_size_gb:.2f}GB")
    
    # Quyết định phương pháp xử lý dựa trên kích thước file
    if file_size_gb > 5:  # File lớn hơn 5GB - dùng multiprocessing
        print(f"\n🚀 File lớn ({file_size_gb:.1f}GB) - Sử dụng multiprocessing")
        
        # Tạo tên file output
        base_name = os.path.splitext(file_path)[0]
        output_file = f"{base_name}_normalized_multiprocessing.csv"
        
        # Cấu hình tối ưu cho máy mạnh
        n_workers = 20  # 20/24 cores
        chunk_size = 1000000  # 1M dòng cho tối ưu tốc độ
        normalization_method = 'minmax'  # Hoặc 'standard'
        
        print(f"⚙️ Cấu hình: {n_workers} workers, chunk {chunk_size:,} dòng")
        
        # Xử lý với multiprocessing
        success = normalize_large_csv_parallel(
            input_file=file_path,
            output_file=output_file, 
            chunk_size=chunk_size,
            n_workers=n_workers,
            normalization_method=normalization_method
        )
        
        if success:
            print(f"\n✅ MULTIPROCESSING HOÀN THÀNH!")
            
            # Hiển thị 10 dòng đầu của file kết quả
            show_sample_output(output_file, num_rows=10)
            
            print(f"\n🎯 File kết quả: {output_file}")
            print("👋 Chương trình hoàn thành!")
        else:
            print("❌ Có lỗi trong quá trình multiprocessing!")
            
    else:  # File nhỏ - dùng phương pháp thông thường
        print(f"\n🔧 File nhỏ ({file_size_gb:.1f}GB) - Sử dụng phương pháp thông thường")
        
        # Khởi tạo normalizer
        normalizer = CSVDataNormalizer(file_path, chunk_size=1000000, n_workers=20)
        
        # Tải dữ liệu
        if normalizer.load_data():
            print("\n" + "="*80)
            print("🎛️ TỰ ĐỘNG XỬ LÝ TẤT CẢ CÁC CỘT")
            print("=" * 80)
            
            # Tự động xử lý tất cả các cột với phương pháp minmax
            success = normalizer.process_all_columns(normalization_method='minmax')
            
            if success:
                # Tự động lưu kết quả
                output_path = normalizer.save_data()
                if output_path:
                    print(f"\n✅ HOÀN THÀNH THÀNH CÔNG!")
                    
                    # Hiển thị 10 dòng đầu của file kết quả
                    show_sample_output(output_path, num_rows=10)
                    
                    print(f"\n🎯 File đã được lưu tại: {output_path}")
                    print("👋 Chương trình hoàn thành!")
                else:
                    print("❌ Có lỗi khi lưu file!")
            else:
                print("❌ Có lỗi trong quá trình xử lý!")
        else:
            print("❌ Không thể tải dữ liệu!")
    
    print("\n👋 Chương trình kết thúc!")