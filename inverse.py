import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

def reverse_data_functions():
    """Các hàm toán học để đảo chiều dữ liệu"""
    return {
        'nghịch_đảo': lambda x: 1/x if x != 0 else np.nan,
        'âm': lambda x: -x,
        'logarit_âm': lambda x: -np.log(x) if x > 0 else np.nan,
        'căn_bậc_hai_nghịch_đảo': lambda x: 1/np.sqrt(x) if x > 0 else np.nan,
        'hàm_mũ_nghịch_đảo': lambda x: np.exp(-x)
    }

def load_csv():
    """Đọc file CSV cố định và hiển thị thông tin các cột"""
    file_path = r"C:\Users\Admin\Downloads\prj\Flood_point\merged_flood_point_merge_cleaned_balanced_reordered_nonlatlon_normalized_delindexNB.csv"
    
    try:
        df = pd.read_csv(file_path)
        print(f"Đã tải file: {file_path}")
        print(f"Số hàng: {len(df)}, Số cột: {len(df.columns)}")
        
        print("\nDANH SÁCH CÁC CỘT:")
        for i, column in enumerate(df.columns, 1):
            data_type = "Số" if pd.api.types.is_numeric_dtype(df[column]) else "Không phải số"
            print(f"{i:2d}. {column:<30} | {data_type}")
        
        return df
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

def get_column_choice(df):
    """Hỏi người dùng chọn cột"""
    while True:
        try:
            choice = int(input(f"\nChọn cột (1-{len(df.columns)}): "))
            if 1 <= choice <= len(df.columns):
                column = df.columns[choice - 1]
                if pd.api.types.is_numeric_dtype(df[column]):
                    return column
                else:
                    print("Cột này không phải kiểu số!")
            else:
                print(f"Nhập số từ 1 đến {len(df.columns)}")
        except ValueError:
            print("Nhập số nguyên hợp lệ!")

def process_all_functions(df, column):
    """Áp dụng tất cả hàm và tạo các cột mới"""
    functions = reverse_data_functions()
    new_columns = []
    
    print(f"\nXử lý cột '{column}' với {len(functions)} hàm:")
    
    for name, func in functions.items():
        new_col = f"{column}_{name}"
        df[new_col] = df[column].apply(func)
        df[new_col] = pd.to_numeric(df[new_col], errors='coerce')
        
        valid_count = df[new_col].count()
        print(f"→ {new_col}: {valid_count}/{len(df)} giá trị hợp lệ")
        new_columns.append(new_col)
    
    return df, new_columns

def plot_matrices(df, columns):
    """Vẽ ma trận mật độ xác suất"""
    n_cols = len(columns)
    fig, axes = plt.subplots(4, n_cols, figsize=(4*n_cols, 12))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Ma Trận Mật Độ Xác Suất', fontsize=14, fontweight='bold')
    
    for i, col in enumerate(columns):
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        # Histogram + KDE
        axes[0,i].hist(data, bins=20, density=True, alpha=0.7, color='skyblue')
        if len(data) > 1:
            kde = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 100)
            axes[0,i].plot(x, kde(x), 'r-', linewidth=2)
        axes[0,i].set_title(f'Histogram\n{col}', fontsize=8)
        axes[0,i].grid(True, alpha=0.3)
        
        # Box plot
        axes[1,i].boxplot(data, patch_artist=True, 
                         boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[1,i].set_title(f'Box Plot\n{col}', fontsize=8)
        axes[1,i].grid(True, alpha=0.3)
        
        # Q-Q plot
        if len(data) > 1:
            stats.probplot(data, dist="norm", plot=axes[2,i])
        axes[2,i].set_title(f'Q-Q Plot\n{col}', fontsize=8)
        axes[2,i].grid(True, alpha=0.3)
        
        # Violin plot
        if len(data) > 1:
            parts = axes[3,i].violinplot([data], positions=[1], showmeans=True)
            for pc in parts['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.7)
        axes[3,i].set_title(f'Violin Plot\n{col}', fontsize=8)
        axes[3,i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_statistics(df, columns):
    """In thống kê cho các cột"""
    print("\nTHỐNG KÊ:")
    print("="*60)
    
    for col in columns:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"\n{col}:")
            print(f"Mẫu: {len(data)}, TB: {data.mean():.4f}, ĐLC: {data.std():.4f}")
            print(f"Min: {data.min():.4f}, Max: {data.max():.4f}")
            if len(data) > 1:
                print(f"Skew: {stats.skew(data):.4f}, Kurt: {stats.kurtosis(data):.4f}")

def save_result(df):
    """Lưu kết quả"""
    output_file = r"C:\Users\Admin\Downloads\prj\Flood_point\processed_data.csv"
    try:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nĐã lưu: {output_file}")
    except Exception as e:
        print(f"Lỗi lưu file: {e}")

def main():
    """Hàm chính"""
    print("CHƯƠNG TRÌNH XỬ LÝ ĐẢO CHIỀU DỮ LIỆU CSV")
    print("="*50)
    
    # Tải dữ liệu
    df = load_csv()
    if df is None:
        return
    
    # Chọn cột và xử lý
    column = get_column_choice(df)
    df, new_columns = process_all_functions(df, column)
    
    # Vẽ biểu đồ và in thống kê
    plot_matrices(df, new_columns)
    print_statistics(df, new_columns)
    
    # Lưu file
    if input("\nLưu kết quả? (y/n): ").lower().startswith('y'):
        save_result(df)
    
    print("Hoàn tất!")

if __name__ == "__main__":
    main()