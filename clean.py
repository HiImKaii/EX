import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def filter_and_prepare_data(input_file, output_file, sample_size=10000, random_state=42):
    """
    Lọc dữ liệu từ file gốc để tạo file nhỏ với các đặc trưng đã chọn
    và cân bằng mẫu giữa lũ và không lũ
    """
    
    print("🔄 Đang tải dữ liệu...")
    # Đọc dữ liệu
    df = pd.read_csv(input_file)
    print(f"✅ Đã tải: {len(df)} điểm dữ liệu")
    
    # Các đặc trưng cần giữ (đã chọn từ phân tích tương quan)
    features_to_keep = [
        'dem', 'slope', 'twi', 
        'avg_NDVI', 'avg_NDBI', 'std_NDWI',
        'Density_River', 'Density_Road',
        'Distan2river_met', 'Distan2road_met'
    ]
    
    # Các cột lũ cần gộp
    flood_columns = [col for col in df.columns if 'floodevent' in col]
    print(f"🌊 Tìm thấy {len(flood_columns)} cột lũ: {flood_columns}")
    
    # Tính toán các đặc trưng trung bình và độ lệch chuẩn
    print("📊 Đang tính toán các đặc trưng phổ...")
    
    # Tính trung bình và độ lệch chuẩn cho các chỉ số phổ
    for index in ['NDVI', 'NDBI', 'NDWI']:
        year_cols = [col for col in df.columns if index in col and any(year in col for year in ['2020', '2021', '2022', '2023', '2024'])]
        if year_cols:
            df[f'avg_{index}'] = df[year_cols].mean(axis=1, skipna=True)
            df[f'std_{index}'] = df[year_cols].std(axis=1, skipna=True)
    
    # Tạo cột lũ nhị phân (1 nếu có ít nhất 1 lần lũ, 0 nếu không)
    print("⛈️ Đang gộp các cột lũ...")
    df['flood_event'] = (df[flood_columns].sum(axis=1) > 0).astype(int)
    
    # Kiểm tra phân bố lũ
    flood_count = df['flood_event'].sum()
    no_flood_count = len(df) - flood_count
    print(f"📊 Phân bố lũ: {flood_count} điểm có lũ ({flood_count/len(df)*100:.1f}%), {no_flood_count} điểm không lũ ({no_flood_count/len(df)*100:.1f}%)")
    
    # Kiểm tra các đặc trưng cần giữ có tồn tại không
    available_features = [col for col in features_to_keep if col in df.columns]
    missing_features = [col for col in features_to_keep if col not in df.columns]
    
    if missing_features:
        print(f"⚠️ Cảnh báo: Thiếu các đặc trưng: {missing_features}")
        print(f"✅ Sẽ sử dụng các đặc trưng có sẵn: {available_features}")
    
    # Loại bỏ các hàng có giá trị NaN trong các đặc trưng cần thiết
    clean_data = df[available_features + ['flood_event']].dropna()
    print(f"🧹 Sau khi loại bỏ NaN: {len(clean_data)} điểm dữ liệu")
    
    # Phân tách dữ liệu lũ và không lũ
    flood_data = clean_data[clean_data['flood_event'] == 1]
    no_flood_data = clean_data[clean_data['flood_event'] == 0]
    
    print(f"📊 Sau khi làm sạch:")
    print(f"   - Có lũ: {len(flood_data)} điểm")
    print(f"   - Không lũ: {len(no_flood_data)} điểm")
    
    # Tính số lượng mẫu cho mỗi lớp (50% lũ, 50% không lũ)
    samples_per_class = sample_size // 2
    
    # Lấy mẫu từ mỗi lớp
    if len(flood_data) >= samples_per_class and len(no_flood_data) >= samples_per_class:
        # Lấy mẫu ngẫu nhiên từ mỗi lớp
        sampled_flood = flood_data.sample(n=samples_per_class, random_state=random_state)
        sampled_no_flood = no_flood_data.sample(n=samples_per_class, random_state=random_state)
        
        # Kết hợp hai mẫu
        final_sample = pd.concat([sampled_flood, sampled_no_flood])
        
        # Xáo trộn dữ liệu
        final_sample = final_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        print(f"🎯 Đã lấy mẫu cân bằng: {len(final_sample)} điểm (50% lũ, 50% không lũ)")
        
    else:
        print("⚠️ Dữ liệu không đủ để lấy mẫu cân bằng theo yêu cầu")
        print("💡 Sẽ lấy mẫu tối đa có thể với tỷ lệ cân bằng")
        
        min_class_size = min(len(flood_data), len(no_flood_data))
        if min_class_size > 0:
            sampled_flood = flood_data.sample(n=min(min_class_size, sample_size//2), random_state=random_state)
            sampled_no_flood = no_flood_data.sample(n=min(min_class_size, sample_size//2), random_state=random_state)
            final_sample = pd.concat([sampled_flood, sampled_no_flood])
            final_sample = final_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
            print(f"🎯 Đã lấy mẫu: {len(final_sample)} điểm")
        else:
            print("❌ Không có đủ dữ liệu để tạo mẫu")
            return None
    
    # Lưu file kết quả
    final_sample.to_csv(output_file, index=False)
    print(f"✅ Đã lưu file kết quả: {output_file}")
    print(f"📁 Kích thước file: {len(final_sample)} điểm × {len(final_sample.columns)} cột")
    
    # Thống kê cuối cùng
    print("\n📈 THỐNG KÊ KẾT QUẢ:")
    print(f"   - Tổng số điểm: {len(final_sample)}")
    print(f"   - Số đặc trưng: {len(available_features)}")
    print(f"   - Điểm có lũ: {final_sample['flood_event'].sum()} ({final_sample['flood_event'].mean()*100:.1f}%)")
    print(f"   - Điểm không lũ: {len(final_sample) - final_sample['flood_event'].sum()} ({(1-final_sample['flood_event'].mean())*100:.1f}%)")
    print(f"   - Các cột trong file: {list(final_sample.columns)}")
    
    return final_sample

# Sử dụng chương trình
if __name__ == "__main__":
    input_file = r"C:\Users\Admin\Downloads\prj\BD_PointGrid_10m_aoi_sample.csv"
    output_file = r"C:\Users\Admin\Downloads\filtered_flood_data.csv"
    
    # Lọc dữ liệu với 10000 điểm (5000 lũ, 5000 không lũ)
    result = filter_and_prepare_data(input_file, output_file, sample_size=10000)