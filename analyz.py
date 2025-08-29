import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class DataAnalyzer:
    def __init__(self, file_path, output_folder="results"):
        """Khởi tạo bộ phân tích dữ liệu"""
        self.file_path = file_path
        self.output_folder = output_folder
        self.df = None
        self.target_col = None
        self.numeric_cols = []
        
        # Tạo thư mục kết quả
        os.makedirs(output_folder, exist_ok=True)
        
    def load_data(self):
        """Tải dữ liệu từ file CSV"""
        print("📊 Đang tải dữ liệu...")
        self.df = pd.read_csv(self.file_path)
        print(f"Tải thành công: {len(self.df)} dòng, {len(self.df.columns)} cột")
        
    def detect_target_column(self):
        """Tự động phát hiện vector mục tiêu (0/1) - có thể là cột lũ hoặc bất kỳ cột nhị phân nào"""
        binary_cols = []
        
        for col in self.df.columns:
            unique_vals = self.df[col].dropna().unique()
            # Kiểm tra cột có giá trị 0/1 hoặc True/False
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                binary_cols.append(col)
        
        if binary_cols:
            # Ưu tiên cột có từ khóa liên quan đến lũ
            flood_keywords = ['flood', 'lũ', 'event', 'disaster']
            for col in binary_cols:
                if any(keyword in col.lower() for keyword in flood_keywords):
                    self.target_col = col
                    break
            
            # Nếu không tìm thấy, chọn cột nhị phân đầu tiên
            if not self.target_col:
                self.target_col = binary_cols[0]
                
            print(f"🎯 Phát hiện cột mục tiêu: '{self.target_col}'")
        else:
            print("⚠️ Không tìm thấy cột mục tiêu nhị phân (0/1)")
            
    def identify_columns(self):
        """Phân loại các cột số"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Loại bỏ cột mục tiêu khỏi danh sách cột số
        if self.target_col and self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)
            
        print(f"📋 Tìm thấy {len(self.numeric_cols)} cột số để phân tích")
        
    def plot_correlation_matrix(self):
        """Vẽ ma trận tương quan (nửa dưới)"""
        if len(self.numeric_cols) < 2:
            print("⚠️ Không đủ cột số để vẽ ma trận tương quan")
            return
            
        print("🔗 Đang vẽ ma trận tương quan...")
        
        # Tính ma trận tương quan
        corr_data = self.df[self.numeric_cols]
        corr_matrix = corr_data.corr()
        
        # Tạo mask để chỉ hiển thị nửa dưới
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Vẽ heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, fmt='.2f', square=True, linewidths=0.5)
        plt.title('Ma trận Tương quan', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/correlation_matrix_invers.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_pairplot(self):
        """Vẽ ma trận mật độ xác suất (pairplot)"""
        if len(self.numeric_cols) < 2:
            print("⚠️ Không đủ cột số để vẽ pairplot")
            return
            
        print("📊 Đang vẽ ma trận mật độ xác suất...")
        
        # Sử dụng tất cả các cột số
        cols_to_plot = self.numeric_cols.copy()
        plot_data = cols_to_plot.copy()
        
        # Thêm cột mục tiêu nếu có
        if self.target_col:
            plot_data.append(self.target_col)
            
        # Vẽ pairplot
        g = sns.pairplot(
            data=self.df[plot_data],
            hue=self.target_col if self.target_col else None,
            diag_kind='kde',
            plot_kws={'alpha': 0.7, 's': 8, 'edgecolors': 'none'},
            diag_kws={'fill': True, 'alpha': 0.8, 'linewidth': 1.5},
            height=2,
            aspect=1,
            corner=True,  # Chỉ vẽ nửa dưới ma trận để tránh trùng lặp
            palette=['#e74c3c', '#2ecc71'] if self.target_col else None
        )
        
        g.fig.suptitle('Ma trận Mật độ Xác suất', fontsize=16, y=1.02, fontweight='bold')
        
        # Cập nhật legend nếu có cột mục tiêu
        if self.target_col and g._legend:
            g._legend.set_title('Phân loại', prop={'size': 12, 'weight': 'bold'})
            legend_labels = ['Nhóm 0', 'Nhóm 1']
            for t, l in zip(g._legend.texts, legend_labels):
                t.set_text(l)
                
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/pairplot_density_invers.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_target_correlations(self):
        """Vẽ biểu đồ tương quan với cột mục tiêu"""
        if not self.target_col or len(self.numeric_cols) == 0:
            print("⚠️ Không thể vẽ tương quan với cột mục tiêu")
            return
            
        print("📈 Đang vẽ tương quan với cột mục tiêu...")
        
        # Tính tương quan với cột mục tiêu
        target_corr = self.df[self.numeric_cols].corrwith(self.df[self.target_col])
        target_corr = target_corr.sort_values(key=abs, ascending=True)
        
        # Vẽ biểu đồ thanh ngang
        plt.figure(figsize=(10, max(6, len(target_corr) * 0.4)))
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in target_corr.values]
        target_corr.plot(kind='barh', color=colors)
        plt.title(f'Tương quan với cột "{self.target_col}"', fontsize=14, fontweight='bold')
        plt.xlabel('Hệ số tương quan')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/target_correlations_invers.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary(self):
        """Tạo báo cáo tóm tắt"""
        print("📋 Tạo báo cáo tóm tắt...")
        
        summary = {
            'Tổng số dòng': len(self.df),
            'Tổng số cột': len(self.df.columns),
            'Số cột số': len(self.numeric_cols),
            'Cột mục tiêu': self.target_col or 'Không phát hiện'
        }
        
        if self.target_col:
            target_rate = self.df[self.target_col].mean()
            summary[f'Tỷ lệ nhóm 1 ({self.target_col})'] = f"{target_rate:.1%}"
            
        print("\n" + "="*50)
        print("📊 BÁO CÁO TÓM TẮT")
        print("="*50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("="*50)
        
    def run_analysis(self):
        """Chạy toàn bộ quá trình phân tích"""
        try:
            # Thiết lập style
            plt.style.use('default')
            sns.set_palette("Set2")
            
            # Các bước phân tích
            self.load_data()
            self.detect_target_column()
            self.identify_columns()
            
            # Vẽ các biểu đồ
            self.plot_correlation_matrix()
            self.plot_pairplot()
            self.plot_target_correlations()
            
            # Tạo báo cáo
            self.generate_summary()
            
            print(f"\n✅ PHÂN TÍCH HOÀN THÀNH!")
            print(f"📁 Kết quả được lưu tại: {self.output_folder}")
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình phân tích: {e}")

# Hàm chính để sử dụng
def analyze_data(file_path, output_folder="results"):
    """Hàm tiện ích để phân tích dữ liệu"""
    analyzer = DataAnalyzer(file_path, output_folder)
    analyzer.run_analysis()
    return analyzer

# Chạy phân tích
if __name__ == "__main__":
    # Thay đổi đường dẫn file theo nhu cầu
    file_path = r"D:\25-26_HKI_DATN_QuanVX\merged_flood_point_merge_cleaned_balanced_reordered_nonlatlon_normalized_delindexNB.csv"
    
    # Chạy phân tích
    analyzer = analyze_data(file_path)