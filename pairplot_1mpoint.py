import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Cấu hình giao diện
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(file_path):
    """Tải và xử lý dữ liệu ML cơ bản"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Đã tải dữ liệu: {df.shape[0]} hàng, {df.shape[1]} cột")

        # Các nhóm đặc trưng
        terrain_features = ['dem', 'slope', 'aspect', 'curvature', 'twi', 'flowDir']
        proximity_features = ['Density_River', 'Density_Road', 'Distan2river_met', 'Distan2road_met']
        categorical_features = ['lulc']

        # Tính trung bình và độ lệch chuẩn cho các chỉ số phổ
        spectral_features = []
        for index in ['NDVI', 'NDBI', 'NDWI']:
            year_cols = [col for col in df.columns if index in col and any(year in col for year in ['2020', '2021', '2022', '2023', '2024'])]
            if year_cols:
                df[f'avg_{index}'] = df[year_cols].mean(axis=1, skipna=True)
                df[f'std_{index}'] = df[year_cols].std(axis=1, skipna=True)
                spectral_features.extend([f'avg_{index}', f'std_{index}'])

        # Tạo biến mục tiêu
        flood_cols = [col for col in df.columns if 'floodevent' in col]
        df['flood_binary'] = (df[flood_cols].sum(axis=1) > 0).astype(int)
        df['flood_frequency'] = df[flood_cols].sum(axis=1)

        # Tổng hợp đặc trưng
        all_features = terrain_features + spectral_features + proximity_features + categorical_features
        target_features = ['flood_binary', 'flood_frequency']

        # Chỉ giữ lại các cột có sẵn
        available_features = [col for col in all_features if col in df.columns]
        ml_data = df[available_features + target_features].copy()

        print(f"✅ Đã chọn {len(available_features)} đặc trưng ML")
        return ml_data, available_features, target_features

    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        return None, None, None

def plot_correlation_matrix(ml_data, features, save_path="correlation_matrix.png"):
    """Vẽ ma trận tương quan chỉ hiện nửa dưới"""
    numerical_features = [col for col in features if col != 'lulc']
    corr_matrix = ml_data[numerical_features].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Ẩn nửa trên
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
    plt.title("Ma trận tương quan (nửa dưới)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Đã lưu ma trận tương quan: {save_path}")

def plot_pairplot_lower_triangle(ml_data, features, save_path="pairplot_lower.png"):
    """Vẽ pairplot chỉ hiện nửa dưới"""
    continuous_features = [col for col in features if col != 'lulc']
    plot_data = ml_data[continuous_features + ['flood_binary']].dropna()

    if len(plot_data) > 5000:
        plot_data = plot_data.sample(5000, random_state=42)

    g = sns.pairplot(plot_data, hue='flood_binary', diag_kind='kde', plot_kws={'alpha': 0.6})
    n_vars = len(continuous_features)
    for i in range(n_vars):
        for j in range(n_vars):
            if i < j:
                g.axes[i, j].set_visible(False)
    g.fig.suptitle('Pairplot (nửa dưới)', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Đã lưu pairplot: {save_path}")

def analyze_targets(ml_data, save_path="target_analysis.png"):
    """Phân tích phân bố biến mục tiêu"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    flood_counts = ml_data['flood_binary'].value_counts()
    axes[0].pie(flood_counts.values, labels=['Không lũ', 'Có lũ'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'coral'])
    axes[0].set_title("Tỷ lệ xảy ra lũ")

    # Histogram
    axes[1].hist(ml_data['flood_frequency'], bins=30, color='skyblue', edgecolor='black')
    axes[1].set_title("Tần suất lũ")
    axes[1].set_xlabel("Số lần lũ")
    axes[1].set_ylabel("Số lượng")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Đã lưu phân tích mục tiêu: {save_path}")

def feature_importance_proxy(ml_data, features, save_path="feature_importance.png"):
    """Ước lượng độ quan trọng của đặc trưng qua hệ số tương quan với mục tiêu"""
    numerical_features = [col for col in features if col != 'lulc']
    correlations = ml_data[numerical_features].corrwith(ml_data['flood_binary']).abs().sort_values(ascending=False)

    top_features = correlations.head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
    plt.title("Top đặc trưng theo độ tương quan với flood_binary")
    plt.xlabel("Hệ số tương quan")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Đã lưu đánh giá đặc trưng: {save_path}")

def run_analysis(file_path):
    """Chạy toàn bộ phân tích ML cơ bản"""
    ml_data, features, targets = load_and_prepare_data(file_path)
    if ml_data is None:
        return

    print("\n📊 Bắt đầu phân tích...")
    plot_correlation_matrix(ml_data, features)
    plot_pairplot_lower_triangle(ml_data, features)
    analyze_targets(ml_data)
    feature_importance_proxy(ml_data, features)

    print("\n✅ Hoàn tất phân tích ML cơ bản.")

# Chạy chương trình
if __name__ == "__main__":
    file_path = r"C:\Users\Admin\Downloads\prj\BD_PointGrid_10m_aoi_sample.csv"
    run_analysis(file_path)