import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class SmartDataBalancer:
    def __init__(self):
        # ==== BỘ THAM SỐ ĐIỀU CHỈNH ====
        self.params = {
            # Phát hiện outlier (như cũ)
            'zscore_threshold': 3.5,
            'iqr_multiplier': 2.0,
            'isolation_contamination': 0.05,
            
            # Xác thực outlier (như cũ)
            'max_noise_rate': 0.12,
            'min_improvement': 0.005,
            'min_samples': 30,
            
            # Normality test (như cũ)
            'normality_alpha': 0.01,
            'max_skew': 2.5,
            'max_kurt': 8.0,
            
            # Consensus (như cũ)
            'min_votes': 2,
            'max_iterations': 2,
            
            # ==== THAM SỐ MỚI CHO BALANCING ====
            'target_ratio': 3.0,           # Tỷ lệ mục tiêu non-flood:flood (3:1)
            'clustering_ratio': 0.3,       # 30% dùng clustering
            'boundary_ratio': 0.3,         # 30% dùng boundary analysis
            'random_ratio': 0.4,           # 40% random (sau khi lọc noise)
            'min_cluster_size': 5,         # Kích thước cluster tối thiểu
            'boundary_neighbors': 10,      # Số láng giềng cho boundary analysis
            'preserve_outliers': True      # Giữ lại outliers có thể quan trọng
        }
    
    def is_normal_distributed(self, data):
        """Kiểm tra phân phối chuẩn với multiple tests"""
        if len(data) < 8:
            return False, 0
        
        try:
            _, jb_p = stats.jarque_bera(data)
            _, sw_p = stats.shapiro(data) if len(data) < 5000 else (0, 0.5)
            _, da_p = stats.normaltest(data)
            
            p_values = [p for p in [jb_p, sw_p, da_p] if p > 0]
            avg_p = np.mean(p_values) if p_values else 0
            
            skew = abs(stats.skew(data))
            kurt = abs(stats.kurtosis(data))
            
            is_normal = (avg_p > self.params['normality_alpha'] and 
                        skew < self.params['max_skew'] and 
                        kurt < self.params['max_kurt'])
            
            return is_normal, avg_p
        except:
            return False, 0
    
    def detect_outliers_consensus(self, data):
        """Phát hiện outliers bằng consensus voting"""
        if len(data) < self.params['min_samples']:
            return pd.Series([False] * len(data), index=data.index)
        
        outlier_votes = []
        
        # Method 1: Modified Z-score
        try:
            median_val = data.median()
            mad = np.median(np.abs(data - median_val))
            if mad > 0:
                mod_z = 0.6745 * (data - median_val) / mad
                outlier_votes.append(np.abs(mod_z) > self.params['zscore_threshold'])
        except:
            pass
        
        # Method 2: IQR method
        try:
            Q1, Q3 = data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - self.params['iqr_multiplier'] * IQR
                upper = Q3 + self.params['iqr_multiplier'] * IQR
                outlier_votes.append((data < lower) | (data > upper))
        except:
            pass
        
        # Method 3: Isolation Forest
        if len(data.dropna()) >= 50:
            try:
                iso_forest = IsolationForest(
                    contamination=self.params['isolation_contamination'],
                    random_state=42,
                    n_estimators=50
                )
                clean_data = data.dropna()
                predictions = iso_forest.fit_predict(clean_data.values.reshape(-1, 1))
                iso_outliers = pd.Series([False] * len(data), index=data.index)
                iso_outliers.loc[clean_data.index] = (predictions == -1)
                outlier_votes.append(iso_outliers)
            except:
                pass
        
        if len(outlier_votes) == 0:
            return pd.Series([False] * len(data), index=data.index)
        
        vote_matrix = pd.DataFrame(outlier_votes).T.fillna(False)
        consensus = vote_matrix.sum(axis=1) >= self.params['min_votes']
        
        return pd.Series(consensus, index=data.index)
    
    def validate_outliers(self, data, outliers):
        """Xác thực outliers để tránh over-filtering"""
        if outliers.sum() == 0:
            return outliers
        
        noise_rate = outliers.sum() / len(data)
        
        if noise_rate > self.params['max_noise_rate']:
            median_val = data.median()
            mad = np.median(np.abs(data - median_val))
            if mad > 0:
                extreme_z = np.abs(0.6745 * (data - median_val) / mad)
                threshold = np.percentile(extreme_z, 98)
                return extreme_z > threshold
            return pd.Series([False] * len(data), index=data.index)
        
        # Bảo vệ consecutive patterns
        validated = outliers.copy()
        outlier_indices = data[outliers].index.tolist()
        
        if len(outlier_indices) > 3:
            for i, idx in enumerate(outlier_indices[:-1]):
                cluster_size = 1
                for j in range(i+1, len(outlier_indices)):
                    if outlier_indices[j] - outlier_indices[j-1] <= 2:
                        cluster_size += 1
                    else:
                        break
                
                if cluster_size > 4:
                    cluster_indices = outlier_indices[i:i+cluster_size]
                    for cluster_idx in cluster_indices:
                        validated.loc[cluster_idx] = False
        
        return validated
    
    def identify_cluster_representatives(self, df_non_flood, n_samples):
        """Sử dụng clustering để tìm điểm đại diện"""
        print(f"  🔍 Clustering analysis...")
        
        numeric_cols = df_non_flood.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df_non_flood.sample(n_samples, random_state=42)
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_non_flood[numeric_cols].fillna(0))
        
        # Xác định số cluster tối ưu
        n_clusters = max(self.params['min_cluster_size'], 
                        min(n_samples // self.params['min_cluster_size'], len(df_non_flood) // 20))
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Chọn điểm đại diện từ mỗi cluster
        representatives = []
        samples_per_cluster = n_samples // n_clusters
        extra_samples = n_samples % n_clusters
        
        for i in range(n_clusters):
            cluster_mask = (clusters == i)
            cluster_data = df_non_flood[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Số mẫu cho cluster này
            cluster_samples = samples_per_cluster + (1 if i < extra_samples else 0)
            cluster_samples = min(cluster_samples, len(cluster_data))
            
            if cluster_samples == 1:
                # Chọn điểm gần centroid nhất
                cluster_center = kmeans.cluster_centers_[i]
                cluster_X = X_scaled[cluster_mask]
                distances = np.sum((cluster_X - cluster_center) ** 2, axis=1)
                closest_idx = cluster_data.iloc[np.argmin(distances)].name
                representatives.append(closest_idx)
            else:
                # Chọn nhiều điểm đại diện
                selected = cluster_data.sample(cluster_samples, random_state=42)
                representatives.extend(selected.index.tolist())
        
        print(f"    ✅ Đã chọn {len(representatives)} điểm từ {n_clusters} clusters")
        return df_non_flood.loc[representatives]
    
    def identify_boundary_samples(self, df_non_flood, df_flood, n_samples):
        """Tìm các mẫu non-flood gần boundary với flood"""
        print(f"  🎯 Boundary analysis...")
        
        numeric_cols = df_non_flood.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols or len(df_flood) == 0:
            return df_non_flood.sample(n_samples, random_state=42)
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        
        # Fit trên toàn bộ dữ liệu
        all_data = pd.concat([df_non_flood[numeric_cols], df_flood[numeric_cols]])
        scaler.fit(all_data.fillna(0))
        
        X_non_flood = scaler.transform(df_non_flood[numeric_cols].fillna(0))
        X_flood = scaler.transform(df_flood[numeric_cols].fillna(0))
        
        # Tìm k nearest neighbors từ flood data
        k = min(self.params['boundary_neighbors'], len(df_flood))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_flood)
        
        # Tính khoảng cách đến flood samples
        distances, _ = nn.kneighbors(X_non_flood)
        avg_distances = np.mean(distances, axis=1)
        
        # Chọn những điểm gần boundary nhất
        boundary_indices = np.argsort(avg_distances)[:n_samples]
        selected_samples = df_non_flood.iloc[boundary_indices]
        
        print(f"    ✅ Đã chọn {len(selected_samples)} điểm gần boundary")
        return selected_samples
    
    def smart_balance(self, df, flood_column='flood', target_total=2000):
        """Cân bằng dữ liệu thông minh"""
        print(f"📊 SMART DATA BALANCING")
        print(f"   Target: {target_total} total samples")
        
        # Phân tách dữ liệu
        df_flood = df[df[flood_column] == 1].copy()
        df_non_flood = df[df[flood_column] == 0].copy()
        
        n_flood = len(df_flood)
        n_non_flood = len(df_non_flood)
        
        print(f"   Hiện tại: {n_flood} flood, {n_non_flood} non-flood")
        
        # Tính toán target
        target_flood = n_flood  # Giữ nguyên số lượng flood
        target_non_flood = target_total - target_flood
        
        if target_non_flood >= n_non_flood:
            print(f"   ⚠️  Target non-flood ({target_non_flood}) >= hiện tại ({n_non_flood})")
            print(f"   Sẽ giữ nguyên và chỉ lọc noise...")
            return self.filter_noise_only(df)
        
        print(f"   Target: {target_flood} flood, {target_non_flood} non-flood")
        print(f"   Cần giảm: {n_non_flood - target_non_flood} non-flood samples")
        
        # BƯỚC 1: Lọc noise từ non-flood trước
        print(f"\n🔧 BƯỚC 1: Lọc noise từ non-flood data...")
        df_non_flood_clean = self.filter_noise_targeted(df_non_flood)
        n_after_noise = len(df_non_flood_clean)
        print(f"   Sau lọc noise: {n_after_noise} non-flood samples")
        
        if n_after_noise <= target_non_flood:
            print(f"   ✅ Đã đạt target sau lọc noise!")
            return pd.concat([df_flood, df_non_flood_clean]).reset_index(drop=True)
        
        # BƯỚC 2: Áp dụng chiến lược balancing
        print(f"\n🎯 BƯỚC 2: Smart sampling...")
        remaining_to_remove = n_after_noise - target_non_flood
        
        # Phân bổ theo tỷ lệ
        n_cluster = int(target_non_flood * self.params['clustering_ratio'])
        n_boundary = int(target_non_flood * self.params['boundary_ratio'])  
        n_random = target_non_flood - n_cluster - n_boundary
        
        print(f"   Phân bổ: {n_cluster} (cluster) + {n_boundary} (boundary) + {n_random} (random)")
        
        selected_samples = []
        
        # Clustering-based selection
        if n_cluster > 0:
            cluster_samples = self.identify_cluster_representatives(df_non_flood_clean, n_cluster)
            selected_samples.append(cluster_samples)
        
        # Boundary-based selection  
        if n_boundary > 0:
            boundary_samples = self.identify_boundary_samples(df_non_flood_clean, df_flood, n_boundary)
            selected_samples.append(boundary_samples)
        
        # Random selection từ phần còn lại
        if n_random > 0:
            used_indices = set()
            for sample in selected_samples:
                used_indices.update(sample.index)
            
            remaining_data = df_non_flood_clean.drop(index=list(used_indices))
            if len(remaining_data) > 0:
                n_random_actual = min(n_random, len(remaining_data))
                random_samples = remaining_data.sample(n_random_actual, random_state=42)
                selected_samples.append(random_samples)
        
        # Kết hợp tất cả
        final_non_flood = pd.concat(selected_samples).drop_duplicates()
        
        # Đảm bảo đúng số lượng
        if len(final_non_flood) > target_non_flood:
            final_non_flood = final_non_flood.sample(target_non_flood, random_state=42)
        
        # Kết quả cuối cùng
        result = pd.concat([df_flood, final_non_flood]).reset_index(drop=True)
        
        print(f"\n📈 KẾT QUẢ BALANCING:")
        print(f"   • Flood: {len(df_flood)} (giữ nguyên)")
        print(f"   • Non-flood: {n_non_flood} → {len(final_non_flood)} (-{n_non_flood - len(final_non_flood)})")
        print(f"   • Tổng: {len(df)} → {len(result)} ({len(result)/len(df):.1%})")
        print(f"   • Tỷ lệ: 1:{len(final_non_flood)/len(df_flood):.1f}")
        
        return result
    
    def filter_noise_targeted(self, df):
        """Lọc noise có mục tiêu cụ thể"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df
        
        current_df = df.copy()
        total_removed = 0
        
        for iteration in range(self.params['max_iterations']):
            iteration_outliers = pd.Series([False] * len(current_df), index=current_df.index)
            any_improvement = False
            
            for col in numeric_cols:
                col_data = current_df[col].dropna()
                if len(col_data) < self.params['min_samples']:
                    continue
                
                is_normal_before, p_before = self.is_normal_distributed(col_data)
                if is_normal_before:
                    continue
                
                outliers = self.detect_outliers_consensus(current_df[col])
                validated_outliers = self.validate_outliers(current_df[col], outliers)
                
                if validated_outliers.sum() == 0:
                    continue
                
                temp_data = current_df.loc[~validated_outliers, col].dropna()
                if len(temp_data) >= self.params['min_samples']:
                    is_normal_after, p_after = self.is_normal_distributed(temp_data)
                    improvement = p_after - p_before
                    
                    if improvement > self.params['min_improvement']:
                        iteration_outliers = iteration_outliers | validated_outliers
                        any_improvement = True
            
            if not any_improvement:
                break
            
            removed_this_iter = iteration_outliers.sum()
            current_df = current_df[~iteration_outliers].copy()
            total_removed += removed_this_iter
        
        print(f"    Đã lọc {total_removed} noise samples")
        return current_df
    
    def filter_noise_only(self, df):
        """Chỉ lọc noise, không balance"""
        print(f"📊 CHỈ LỌC NOISE - KHÔNG BALANCE")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df
        
        current_df = df.copy()
        total_removed = 0
        
        for iteration in range(self.params['max_iterations']):
            print(f"\n🔄 Lần lặp {iteration + 1}:")
            
            iteration_outliers = pd.Series([False] * len(current_df), index=current_df.index)
            any_improvement = False
            
            for col in numeric_cols:
                col_data = current_df[col].dropna()
                if len(col_data) < self.params['min_samples']:
                    continue
                
                is_normal_before, p_before = self.is_normal_distributed(col_data)
                if is_normal_before:
                    print(f"  ✅ {col}: Đã chuẩn (p={p_before:.3f})")
                    continue
                
                outliers = self.detect_outliers_consensus(current_df[col])
                validated_outliers = self.validate_outliers(current_df[col], outliers)
                
                if validated_outliers.sum() == 0:
                    continue
                
                temp_data = current_df.loc[~validated_outliers, col].dropna()
                if len(temp_data) >= self.params['min_samples']:
                    is_normal_after, p_after = self.is_normal_distributed(temp_data)
                    improvement = p_after - p_before
                    
                    if improvement > self.params['min_improvement']:
                        iteration_outliers = iteration_outliers | validated_outliers
                        any_improvement = True
                        noise_rate = validated_outliers.sum() / len(current_df)
                        print(f"  🎯 {col}: {validated_outliers.sum()} outliers ({noise_rate:.2%}) | "
                              f"p: {p_before:.3f} → {p_after:.3f}")
            
            if not any_improvement:
                print(f"  🎉 Hội tụ! Không còn cải thiện được nữa.")
                break
            
            removed_this_iter = iteration_outliers.sum()
            current_df = current_df[~iteration_outliers].copy()
            total_removed += removed_this_iter
            print(f"  📉 Loại bỏ: {removed_this_iter} hàng")
        
        preservation_rate = len(current_df) / len(df)
        print(f"\n📈 KẾT QUẢ CUỐI CÙNG:")
        print(f"   • Loại bỏ: {total_removed} hàng ({total_removed/len(df):.1%})")
        print(f"   • Bảo toàn: {preservation_rate:.1%} dữ liệu")
        print(f"   • Kích thước: {df.shape} → {current_df.shape}")
        
        return current_df


def balance_flood_data(file_path, flood_column='flood', target_total=2000, custom_params=None):
    """
    Hàm chính - Cân bằng dữ liệu lũ thông minh
    
    Args:
        file_path: Đường dẫn file CSV
        flood_column: Tên cột chứa label lũ (1=lũ, 0=không lũ)
        target_total: Tổng số mẫu mục tiêu
        custom_params: Dict tham số tùy chỉnh
    
    Returns:
        DataFrame đã được cân bằng
    """
    print("📂 Đang đọc dữ liệu...")
    df = pd.read_csv(file_path)
    
    # Kiểm tra cột flood
    if flood_column not in df.columns:
        print(f"❌ Không tìm thấy cột '{flood_column}' trong dữ liệu!")
        print(f"Available columns: {list(df.columns)}")
        return df
    
    # Kiểm tra giá trị trong cột flood
    flood_values = df[flood_column].unique()
    print(f"📊 Giá trị trong cột '{flood_column}': {flood_values}")
    
    if not set(flood_values).issubset({0, 1}):
        print(f"⚠️  Cột '{flood_column}' chứa giá trị không phải 0/1: {flood_values}")
        print("Sẽ chuyển đổi: 0 = không lũ, khác 0 = lũ")
        df[flood_column] = (df[flood_column] != 0).astype(int)
    
    # Khởi tạo balancer
    balancer = SmartDataBalancer()
    
    # Áp dụng tham số tùy chỉnh
    if custom_params:
        balancer.params.update(custom_params)
        print(f"⚙️  Đã cập nhật tham số: {custom_params}")
    
    # Thực hiện cân bằng
    df_balanced = balancer.smart_balance(df, flood_column, target_total)
    
    # Lưu kết quả
    output_path = file_path.replace('.csv', '_balanced.csv')
    df_balanced.to_csv(output_path, index=False)
    print(f"\n💾 Đã lưu kết quả: {output_path}")
    
    return df_balanced


# CÁCH SỬ DỤNG
if __name__ == "__main__":
    # Sử dụng cơ bản - cân bằng xuống 2000 mẫu
    file_path = r"D:\Vscode\flood_point_merge.csv"
    df_balanced = balance_flood_data(file_path, flood_column='flood', target_total=2000)
    
    # Tùy chỉnh các tham số
    # custom_params = {
    #     'target_ratio': 2.5,           # Tỷ lệ non-flood:flood = 2.5:1
    #     'clustering_ratio': 0.4,       # 40% dùng clustering
    #     'boundary_ratio': 0.4,         # 40% dùng boundary analysis
    #     'random_ratio': 0.2,           # 20% random
    #     'zscore_threshold': 3.0,       # Nghiêm ngặt hơn trong lọc noise
    # }
    # df_balanced = balance_flood_data(file_path, 'flood', 2000, custom_params)