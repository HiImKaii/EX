import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import rasterio
from rasterio.transform import from_origin
import os
import geopandas as gpd
from rasterio.mask import mask
import warnings
warnings.filterwarnings('ignore')

def load_data(path_to_csv):
    """
    Tải dữ liệu từ file CSV
    """
    print("Đang tải dữ liệu...")
    data = pd.read_csv(path_to_csv)
    print(f"Đã tải {data.shape[0]} mẫu với {data.shape[1]} thuộc tính")
    return data

def preprocess_data(data):
    """
    Tiền xử lý dữ liệu
    """
    print("Đang tiền xử lý dữ liệu...")
    
    # Kiểm tra giá trị thiếu
    print("Số lượng giá trị thiếu theo cột:")
    print(data.isnull().sum())
    
    # Xử lý giá trị thiếu nếu có
    if data.isnull().sum().sum() > 0:
        # Điền giá trị thiếu bằng giá trị trung bình của cột
        data = data.fillna(data.mean())
    
    # Tách đặc trưng và nhãn
    features = data.drop('flood', axis=1) if 'flood' in data.columns else data.iloc[:, :-1]
    labels = data['flood'] if 'flood' in data.columns else data.iloc[:, -1]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, labels, scaler, features.columns

def train_random_forest(X_train, y_train):
    """
    Huấn luyện mô hình Random Forest
    """
    print("Huấn luyện mô hình Random Forest...")
    
    # Định nghĩa tham số cần tìm kiếm
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Tìm kiếm lưới cho các tham số tối ưu
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Lấy mô hình tốt nhất
    best_rf = grid_search.best_estimator_
    print(f"Tham số tốt nhất cho RF: {grid_search.best_params_}")
    
    return best_rf

def train_xgboost(X_train, y_train):
    """
    Huấn luyện mô hình XGBoost
    """
    print("Huấn luyện mô hình XGBoost...")
    
    # Định nghĩa tham số cần tìm kiếm
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Tìm kiếm lưới cho các tham số tối ưu
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Lấy mô hình tốt nhất
    best_xgb = grid_search.best_estimator_
    print(f"Tham số tốt nhất cho XGB: {grid_search.best_params_}")
    
    return best_xgb

def evaluate_model(model, X_test, y_test, model_name):
    """
    Đánh giá hiệu suất của mô hình
    """
    print(f"Đánh giá mô hình {model_name}...")
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Tính toán độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác của {model_name}: {accuracy:.4f}")
    
    # Tạo ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    print(f"Ma trận nhầm lẫn của {model_name}:")
    print(cm)
    
    # Tạo báo cáo phân loại
    report = classification_report(y_test, y_pred)
    print(f"Báo cáo phân loại của {model_name}:")
    print(report)
    
    # Tính toán đường cong ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Vẽ đường cong ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.close()
    
    return accuracy, roc_auc

def plot_feature_importance(model, feature_names, model_name):
    """
    Vẽ biểu đồ tầm quan trọng của các đặc trưng
    """
    # Lấy tầm quan trọng của các đặc trưng
    if model_name == 'Random Forest':
        importances = model.feature_importances_
    else:  # XGBoost
        importances = model.feature_importances_
    
    # Sắp xếp tầm quan trọng
    indices = np.argsort(importances)[::-1]
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 8))
    plt.title(f'Tầm quan trọng của các đặc trưng - {model_name}')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name}.png')
    plt.close()

def create_flood_risk_map(model, raster_files, output_file, scaler, study_area_shapefile=None):
    """
    Tạo bản đồ cảnh báo ngập lụt từ mô hình
    """
    print(f"Đang tạo bản đồ cảnh báo ngập lụt...")
    
    # Đọc raster đầu tiên để lấy thông tin meta
    with rasterio.open(raster_files[0]) as src:
        meta = src.meta.copy()
        if study_area_shapefile:
            # Cắt theo ranh giới khu vực nghiên cứu
            gdf = gpd.read_file(study_area_shapefile)
            try:
                # Đảm bảo cùng hệ tọa độ
                gdf = gdf.to_crs(src.crs)
                # Tạo mask
                shapes = [geom for geom in gdf.geometry]
                out_image, out_transform = mask(src, shapes, crop=True)
                meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
            except Exception as e:
                print(f"Lỗi khi cắt theo ranh giới: {e}")
        
        # Đọc tất cả các raster
        raster_data = []
        for raster_file in raster_files:
            with rasterio.open(raster_file) as src_i:
                if study_area_shapefile:
                    # Cắt theo ranh giới khu vực nghiên cứu
                    band, _, _ = mask(src_i, shapes, crop=True)
                    band = band[0]
                else:
                    band = src_i.read(1)
                raster_data.append(band)
        
        # Chuẩn bị dữ liệu cho dự đoán
        # Chuyển đổi dữ liệu thành dạng điểm
        height, width = raster_data[0].shape
        features = np.zeros((height * width, len(raster_data)))
        
        for i, band in enumerate(raster_data):
            features[:, i] = band.flatten()
        
        # Loại bỏ các giá trị NoData
        valid_pixels = ~np.isnan(features).any(axis=1)
        valid_features = features[valid_pixels]
        
        # Chuẩn hóa dữ liệu
        valid_features_scaled = scaler.transform(valid_features)
        
        # Dự đoán nguy cơ ngập lụt
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(valid_features_scaled)[:, 1]
        else:
            predictions = model.predict(valid_features_scaled)
        
        # Tạo bản đồ nguy cơ
        result = np.full((height * width), np.nan)
        result[valid_pixels] = predictions
        result = result.reshape((height, width))
        
        # Cập nhật meta data cho bản đồ nguy cơ
        meta.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': np.nan
        })
        
        # Lưu bản đồ
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(result.astype('float32'), 1)
    
    print(f"Đã lưu bản đồ cảnh báo ngập lụt vào {output_file}")

def main():
    """
    Hàm chính để chạy toàn bộ quá trình
    """
    # Đường dẫn đến dữ liệu
    data_path = "flood_data.csv"  # Thay đổi theo đường dẫn thực tế của bạn
    
    # Tải và tiền xử lý dữ liệu
    data = load_data(data_path)
    X, y, scaler, feature_names = preprocess_data(data)
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Huấn luyện và đánh giá mô hình Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_accuracy, rf_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    plot_feature_importance(rf_model, feature_names, "Random Forest")
    
    # Huấn luyện và đánh giá mô hình XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_accuracy, xgb_auc = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    plot_feature_importance(xgb_model, feature_names, "XGBoost")
    
    # So sánh hai mô hình
    print("\nSo sánh hiệu suất mô hình:")
    print(f"Random Forest - Độ chính xác: {rf_accuracy:.4f}, AUC: {rf_auc:.4f}")
    print(f"XGBoost - Độ chính xác: {xgb_accuracy:.4f}, AUC: {xgb_auc:.4f}")
    
    # Danh sách các file raster cho các đặc trưng
    raster_files = [
        "Rainfall.tif",
        "Elevation.tif",
        "Slope.tif",
        "Aspect.tif",
        "FlowDirection.tif",
        "FlowAccumulation.tif",
        "TWI.tif",
        "DistanceToRiver.tif",
        "DrainageCapacity.tif",
        "LandCover.tif",
        "ImperviousSurface.tif",
        "SurfaceTemperature.tif",
        "HydrologicSoilGroup.tif"
    ]
    
    # Tạo bản đồ cảnh báo ngập lụt
    # Sử dụng mô hình có hiệu suất tốt hơn
    best_model = rf_model if rf_accuracy >= xgb_accuracy else xgb_model
    model_name = "RandomForest" if rf_accuracy >= xgb_accuracy else "XGBoost"
    
    create_flood_risk_map(
        best_model,
        raster_files,
        f"flood_risk_map_{model_name}.tif",
        scaler,
        "study_area.shp"  # Thay đổi theo đường dẫn thực tế của bạn
    )
    
    # Tạo bản đồ từ cả hai mô hình để so sánh
    create_flood_risk_map(rf_model, raster_files, "flood_risk_map_RF.tif", scaler, "study_area.shp")
    create_flood_risk_map(xgb_model, raster_files, "flood_risk_map_XGB.tif", scaler, "study_area.shp")
    
    print("Đã hoàn thành!")

if __name__ == "__main__":
    main()