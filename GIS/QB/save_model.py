import os
import sys
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Nhập các hàm cần thiết từ rf_xgb.py
# Giả sử rf_xgb.py nằm cùng thư mục
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rf_xgb import load_data, preprocess_data, train_random_forest, train_xgboost

def save_models(data_path, output_dir='models'):
    """
    Tải dữ liệu, huấn luyện và lưu mô hình
    """
    print(f"=== Bắt đầu huấn luyện và lưu mô hình từ {data_path} ===")
    
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Tải và xử lý dữ liệu
    data = load_data(data_path)
    X, y, scaler, feature_names = preprocess_data(data)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Huấn luyện mô hình Random Forest
    print("\nHuấn luyện mô hình Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    # Huấn luyện mô hình XGBoost 
    print("\nHuấn luyện mô hình XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    
    # Lưu các mô hình
    print("\nLưu mô hình...")
    joblib.dump(rf_model, os.path.join(output_dir, 'rf_model.joblib'))
    joblib.dump(xgb_model, os.path.join(output_dir, 'xgb_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    # Lưu thông tin về các đặc trưng
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names)
    }
    with open(os.path.join(output_dir, 'feature_info.pkl'), 'wb') as f:
        pickle.dump(feature_info, f)
    
    # Lưu tập dữ liệu kiểm tra để đánh giá sau này
    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(test_data, os.path.join(output_dir, 'test_data.joblib'))
    
    print(f"=== Đã lưu tất cả mô hình và dữ liệu liên quan vào thư mục {output_dir} ===")
    print(f"Các file đã tạo:")
    print(f"  - {output_dir}/rf_model.joblib")
    print(f"  - {output_dir}/xgb_model.joblib")
    print(f"  - {output_dir}/scaler.joblib")
    print(f"  - {output_dir}/feature_info.pkl")
    print(f"  - {output_dir}/test_data.joblib")
    
    return rf_model, xgb_model, scaler, feature_names, X_test, y_test

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Huấn luyện và lưu mô hình từ dữ liệu CSV')
    parser.add_argument('--data', required=True, help='Đường dẫn đến file dữ liệu CSV')
    parser.add_argument('--output', default='models', help='Thư mục lưu mô hình (mặc định: models)')
    
    args = parser.parse_args()
    
    save_models(args.data, args.output) 