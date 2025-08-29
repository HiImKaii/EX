import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import os
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, output_folder="results"):
        """Khởi tạo bộ làm sạch dữ liệu"""
        self.output_folder = output_folder
        self.cleaning_report = {}
        os.makedirs(output_folder, exist_ok=True)
        
    def detect_target_column(self, df):
        """Tự động phát hiện cột mục tiêu nhị phân (0/1)"""
        binary_cols = []
        
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                binary_cols.append(col)
        
        if binary_cols:
            # Ưu tiên cột có từ khóa liên quan
            keywords = ['flood', 'lũ', 'event', 'target', 'label', 'class']
            for col in binary_cols:
                if any(keyword in col.lower() for keyword in keywords):
                    return col
            return binary_cols[0]
        
        return None
    
    def remove_outliers_iqr(self, df, multiplier=2.0, target_col=None):
        """Loại bỏ outliers sử dụng IQR method"""
        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        outlier_counts = {}
        for column in numeric_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
            before_count = len(df_clean)
            df_clean = df_clean[(df_clean[column] >= lower) & (df_clean[column] <= upper)]
            outlier_counts[column] = before_count - len(df_clean)
            
        self.cleaning_report['outlier_removal'] = outlier_counts
        return df_clean
    
    def remove_high_correlation(self, df, threshold=0.9, target_col=None):
        """Loại bỏ features có tương quan cao"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            feature_cols = [col for col in numeric_cols if col != target_col]
        else:
            feature_cols = numeric_cols
            
        if len(feature_cols) < 2:
            return df
            
        corr_matrix = df[feature_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        self.cleaning_report['high_correlation'] = to_drop
        
        keep_cols = [col for col in df.columns if col not in to_drop]
        return df[keep_cols]
    
    def select_best_features(self, df, target_col, k=8):
        """Chọn k features tốt nhất"""
        if not target_col:
            return df
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if len(feature_cols) <= k:
            return df
            
        X = df[feature_cols]
        y = df[target_col]
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        self.cleaning_report['feature_selection'] = selected_features
        return df[selected_features + [target_col]]
    
    def clean_data(self, df, target_col=None):
        """Quy trình làm sạch dữ liệu chính"""
        print("🧹 Bắt đầu làm sạch dữ liệu...")
        print(f"Dữ liệu ban đầu: {df.shape}")
        
        # Tự động phát hiện target column nếu không được cung cấp
        if not target_col:
            target_col = self.detect_target_column(df)
            if target_col:
                print(f"🎯 Tự động phát hiện cột mục tiêu: '{target_col}'")
        
        df_clean = df.copy()
        
        # Bước 1: Xử lý missing values
        df_clean = df_clean.dropna()
        print(f"Sau khi xử lý missing values: {df_clean.shape}")
        
        # Bước 2: Loại bỏ duplicates
        df_clean = df_clean.drop_duplicates()
        print(f"Sau khi loại bỏ duplicates: {df_clean.shape}")
        
        # Bước 3: Loại bỏ outliers
        df_clean = self.remove_outliers_iqr(df_clean, target_col=target_col)
        print(f"Sau khi loại bỏ outliers: {df_clean.shape}")
        
        # Bước 4: Loại bỏ tương quan cao
        df_clean = self.remove_high_correlation(df_clean, target_col=target_col)
        print(f"Sau khi loại bỏ tương quan cao: {df_clean.shape}")
        
        # Bước 5: Feature selection (chỉ khi có target column)
        if target_col:
            df_clean = self.select_best_features(df_clean, target_col)
            print(f"Sau feature selection: {df_clean.shape}")
        
        return df_clean, target_col
    
    def plot_cleaning_comparison(self, df_original, df_clean, target_col=None):
        """Vẽ biểu đồ so sánh trước và sau làm sạch"""
        print("📊 Đang vẽ biểu đồ so sánh...")
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            feature_cols = [col for col in numeric_cols if col != target_col]
        else:
            feature_cols = numeric_cols
            
        plot_cols = feature_cols[:4]  # Chỉ vẽ 4 cột đầu
        
        if len(plot_cols) == 0:
            print("⚠️ Không có cột số để vẽ biểu đồ")
            return
        
        fig, axes = plt.subplots(2, len(plot_cols), figsize=(4*len(plot_cols), 8))
        if len(plot_cols) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(plot_cols):
            # Histogram so sánh
            axes[0, i].hist(df_original[col], bins=30, alpha=0.6, label='Trước', color='#e74c3c')
            axes[0, i].hist(df_clean[col], bins=30, alpha=0.6, label='Sau', color='#2ecc71')
            axes[0, i].set_title(f'{col} - Phân bố')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Boxplot so sánh
            data_to_plot = [df_original[col].dropna(), df_clean[col].dropna()]
            axes[1, i].boxplot(data_to_plot, labels=['Trước', 'Sau'])
            axes[1, i].set_title(f'{col} - Boxplot')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle('So sánh Dữ liệu: Trước và Sau Làm sạch', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/cleaning_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

class DataAnalyzer:
    def __init__(self, output_folder="results"):
        """Khởi tạo bộ phân tích dữ liệu"""
        self.output_folder = output_folder
        self.df = None
        self.target_col = None
        self.numeric_cols = []
        os.makedirs(output_folder, exist_ok=True)
        
    def load_data(self, data_source):
        """Tải dữ liệu từ file hoặc DataFrame"""
        if isinstance(data_source, str):
            print("📊 Đang tải dữ liệu từ file...")
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            print("📊 Đang tải dữ liệu từ DataFrame...")
            self.df = data_source.copy()
        else:
            raise ValueError("Data source phải là đường dẫn file hoặc DataFrame")
            
        print(f"Tải thành công: {len(self.df)} dòng, {len(self.df.columns)} cột")
        
    def detect_target_column(self):
        """Tự động phát hiện cột mục tiêu nhị phân"""
        binary_cols = []
        
        for col in self.df.columns:
            unique_vals = self.df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                binary_cols.append(col)
        
        if binary_cols:
            # Ưu tiên cột có từ khóa liên quan
            keywords = ['flood', 'lũ', 'event', 'target', 'label', 'class']
            for col in binary_cols:
                if any(keyword in col.lower() for keyword in keywords):
                    self.target_col = col
                    break
            
            if not self.target_col:
                self.target_col = binary_cols[0]
                
            print(f"🎯 Phát hiện cột mục tiêu: '{self.target_col}'")
        else:
            print("⚠️ Không tìm thấy cột mục tiêu nhị phân")
            
    def identify_columns(self):
        """Phân loại các cột số"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.target_col and self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)
            
        print(f"📋 Tìm thấy {len(self.numeric_cols)} cột số để phân tích")
        
    def plot_correlation_matrix(self):
        """Vẽ ma trận tương quan (nửa dưới)"""
        if len(self.numeric_cols) < 2:
            print("⚠️ Không đủ cột số để vẽ ma trận tương quan")
            return
            
        print("🔗 Đang vẽ ma trận tương quan...")
        
        corr_matrix = self.df[self.numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, fmt='.2f', square=True, linewidths=0.5)
        plt.title('Ma trận Tương quan', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_pairplot(self):
        """Vẽ ma trận mật độ xác suất"""
        if len(self.numeric_cols) < 2:
            print("⚠️ Không đủ cột số để vẽ pairplot")
            return
            
        print("📊 Đang vẽ ma trận mật độ xác suất...")
        
        # Chọn tối đa 6 cột để tránh biểu đồ quá phức tạp
        cols_to_plot = self.numeric_cols[:6]
        plot_data = cols_to_plot.copy()
        
        if self.target_col:
            plot_data.append(self.target_col)
            
        g = sns.pairplot(
            data=self.df[plot_data],
            hue=self.target_col if self.target_col else None,
            diag_kind='kde',
            plot_kws={'alpha': 0.7, 's': 8, 'edgecolors': 'none'},
            diag_kws={'fill': True, 'alpha': 0.8, 'linewidth': 1.5},
            height=2,
            aspect=1,
            corner=True,
            palette=['#e74c3c', '#2ecc71'] if self.target_col else None
        )
        
        g.fig.suptitle('Ma trận Mật độ Xác suất', fontsize=16, y=1.02, fontweight='bold')
        
        if self.target_col and g._legend:
            g._legend.set_title('Phân loại', prop={'size': 12, 'weight': 'bold'})
            legend_labels = ['Nhóm 0', 'Nhóm 1']
            for t, l in zip(g._legend.texts, legend_labels):
                t.set_text(l)
                
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/pairplot_density.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_target_correlations(self):
        """Vẽ tương quan với cột mục tiêu"""
        if not self.target_col or len(self.numeric_cols) == 0:
            print("⚠️ Không thể vẽ tương quan với cột mục tiêu")
            return
            
        print("📈 Đang vẽ tương quan với cột mục tiêu...")
        
        target_corr = self.df[self.numeric_cols].corrwith(self.df[self.target_col])
        target_corr = target_corr.sort_values(key=abs, ascending=True)
        
        plt.figure(figsize=(10, max(6, len(target_corr) * 0.4)))
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in target_corr.values]
        target_corr.plot(kind='barh', color=colors)
        plt.title(f'Tương quan với "{self.target_col}"', fontsize=14, fontweight='bold')
        plt.xlabel('Hệ số tương quan')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/target_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

class IntegratedDataPipeline:
    def __init__(self, output_folder="results"):
        """Pipeline tích hợp làm sạch và phân tích dữ liệu"""
        self.output_folder = output_folder
        self.cleaner = DataCleaner(output_folder)
        self.analyzer = DataAnalyzer(output_folder)
        
    def run_cleaning(self, file_path, save_cleaned=True):
        """Chạy quy trình làm sạch dữ liệu"""
        print("=" * 60)
        print("🧹 BƯỚC 1: LÀM SẠCH DỮ LIỆU")
        print("=" * 60)
        
        # Tải dữ liệu gốc
        df_original = pd.read_csv(file_path)
        print(f"📊 Dữ liệu gốc: {df_original.shape}")
        
        # Tự động phát hiện target column
        target_col = self.cleaner.detect_target_column(df_original)
        
        # Làm sạch dữ liệu
        df_cleaned, target_col = self.cleaner.clean_data(df_original, target_col)
        
        # Vẽ biểu đồ so sánh
        self.cleaner.plot_cleaning_comparison(df_original, df_cleaned, target_col)
        
        # Lưu dữ liệu đã làm sạch
        if save_cleaned:
            cleaned_path = file_path.replace('.csv', '_cleaned.csv')
            df_cleaned.to_csv(cleaned_path, index=False)
            print(f"💾 Dữ liệu đã làm sạch được lưu: {cleaned_path}")
            
        return df_cleaned, target_col, cleaned_path if save_cleaned else None
        
    def run_analysis(self, data_source):
        """Chạy quy trình phân tích dữ liệu"""
        print("\n" + "=" * 60)
        print("📊 BƯỚC 2: PHÂN TÍCH DỮ LIỆU")
        print("=" * 60)
        
        # Tải dữ liệu
        self.analyzer.load_data(data_source)
        self.analyzer.detect_target_column()
        self.analyzer.identify_columns()
        
        # Vẽ các biểu đồ phân tích
        self.analyzer.plot_correlation_matrix()
        self.analyzer.plot_pairplot()
        self.analyzer.plot_target_correlations()
        
        # Tạo báo cáo tóm tắt
        self.generate_summary()
        
    def generate_summary(self):
        """Tạo báo cáo tóm tắt tổng hợp"""
        print("\n" + "=" * 60)
        print("📋 BÁO CÁO TÓM TẮT")
        print("=" * 60)
        
        # Thông tin cơ bản
        print(f"📊 Tổng số dòng: {len(self.analyzer.df)}")
        print(f"📊 Tổng số cột: {len(self.analyzer.df.columns)}")
        print(f"📊 Số cột số: {len(self.analyzer.numeric_cols)}")
        print(f"🎯 Cột mục tiêu: {self.analyzer.target_col or 'Không phát hiện'}")
        
        if self.analyzer.target_col:
            target_rate = self.analyzer.df[self.analyzer.target_col].mean()
            print(f"📈 Tỷ lệ nhóm 1: {target_rate:.1%}")
            
        # Báo cáo làm sạch
        if hasattr(self.cleaner, 'cleaning_report') and self.cleaner.cleaning_report:
            print("\n🧹 Chi tiết làm sạch:")
            for step, details in self.cleaner.cleaning_report.items():
                if step == 'outlier_removal':
                    total_outliers = sum(details.values()) if isinstance(details, dict) else 0
                    print(f"  - Outliers loại bỏ: {total_outliers}")
                elif step == 'high_correlation':
                    removed_count = len(details) if isinstance(details, list) else 0
                    print(f"  - Features tương quan cao: {removed_count}")
                elif step == 'feature_selection':
                    selected_count = len(details) if isinstance(details, list) else 0
                    print(f"  - Features được chọn: {selected_count}")
        
        print("=" * 60)
        print(f"📁 Tất cả kết quả được lưu tại: {self.output_folder}")
        
    def run_full_pipeline(self, file_path):
        """Chạy toàn bộ pipeline: làm sạch -> phân tích"""
        try:
            # Bước 1: Làm sạch dữ liệu
            df_cleaned, target_col, cleaned_path = self.run_cleaning(file_path)
            
            # Bước 2: Phân tích dữ liệu đã làm sạch
            self.run_analysis(df_cleaned)
            
            print("\n🎉 PIPELINE HOÀN THÀNH!")
            return df_cleaned, target_col
            
        except Exception as e:
            print(f"❌ Lỗi trong pipeline: {e}")
            return None, None

# Hàm tiện ích
def clean_data(file_path, output_folder="results"):
    """Chỉ làm sạch dữ liệu"""
    cleaner = DataCleaner(output_folder)
    df = pd.read_csv(file_path)
    df_cleaned, target_col = cleaner.clean_data(df)
    
    # Lưu kết quả
    cleaned_path = file_path.replace('.csv', '_cleaned.csv')
    df_cleaned.to_csv(cleaned_path, index=False)
    
    return df_cleaned, cleaned_path

def analyze_data(data_source, output_folder="results"):
    """Chỉ phân tích dữ liệu"""
    analyzer = DataAnalyzer(output_folder)
    analyzer.load_data(data_source)
    analyzer.detect_target_column()
    analyzer.identify_columns()
    analyzer.plot_correlation_matrix()
    analyzer.plot_pairplot()
    analyzer.plot_target_correlations()
    return analyzer

def full_pipeline(file_path, output_folder="results"):
    """Chạy toàn bộ pipeline"""
    pipeline = IntegratedDataPipeline(output_folder)
    return pipeline.run_full_pipeline(file_path)

# Chạy chương trình
if __name__ == "__main__":
    # Thay đổi đường dẫn file theo nhu cầu
    file_path = "data.csv"
    
    # Chọn chế độ chạy:
    # 1. Chỉ làm sạch
    # df_cleaned, cleaned_path = clean_data(file_path)
    
    # 2. Chỉ phân tích (với dữ liệu đã sạch)
    # analyzer = analyze_data("data_cleaned.csv")
    
    # 3. Chạy toàn bộ pipeline
    df_result, target_col = full_pipeline(file_path)