import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

class CSVDataNormalizer:
    def __init__(self, file_path):
        """Khởi tạo với đường dẫn file CSV"""
        self.file_path = file_path
        self.df = None
        self.original_df = None  # Lưu bản gốc để so sánh
        self.flood_column = None
        self.processing_columns = []
        
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

# Chương trình chính
if __name__ == "__main__":
    # Đường dẫn file CSV
    file_path = r"C:\Users\Admin\Downloads\prj\Flood_point\merged_flood_point_merge_cleaned_balanced_reordered_nonlatlon.csv"
    
    # Khởi tạo normalizer
    normalizer = CSVDataNormalizer(file_path)
    
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
                print(f"📁 File đã được lưu tại: {output_path}")
                print(f"📊 Dữ liệu cuối cùng: {normalizer.df.shape[0]} hàng x {normalizer.df.shape[1]} cột")
                
                # Hiển thị thống kê tổng quan
                print(f"\n📈 THỐNG KÊ TỔNG QUAN:")
                print(f"  - Cột nhãn lũ (giữ nguyên): {normalizer.flood_column}")
                print(f"  - Cột LULC (giữ nguyên): {normalizer.lulc_column}")
                print(f"  - Số cột đã chuẩn hóa: {len(normalizer.processing_columns)}")
                
                # Hiển thị khoảng giá trị của các cột đã xử lý
                print(f"\n🎯 KHOẢNG GIÁ TRỊ SAU CHUẨN HÓA:")
                for col in normalizer.processing_columns[:5]:  # Hiển thị 5 cột đầu
                    min_val = normalizer.df[col].min()
                    max_val = normalizer.df[col].max()
                    print(f"  {col}: [{min_val:.6f}, {max_val:.6f}]")
                if len(normalizer.processing_columns) > 5:
                    print(f"  ... và {len(normalizer.processing_columns) - 5} cột khác")
            else:
                print("❌ Có lỗi khi lưu file!")
        else:
            print("❌ Có lỗi trong quá trình xử lý!")
    else:
        print("❌ Không thể tải dữ liệu!")
    
    print("\n👋 Chương trình kết thúc!")