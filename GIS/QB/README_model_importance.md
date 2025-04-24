# Model Importance Analyzer for Flood Prediction

Công cụ phân tích tầm quan trọng của các đặc trưng trong mô hình dự báo ngập lụt. Công cụ này hoạt động độc lập với chương trình huấn luyện chính (`rf_xgb.py`).

## Chức năng chính

- Phân tích tầm quan trọng của các đặc trưng trong mô hình đã huấn luyện
- Tính toán Permutation Importance (tầm quan trọng hoán vị) - một phương pháp model-agnostic
- Tạo ra các biểu đồ trực quan về tầm quan trọng của các đặc trưng
- Tạo bản đồ trực quan hóa tầm quan trọng của các đặc trưng trên dữ liệu không gian
- Kết hợp các lớp raster với trọng số là tầm quan trọng của đặc trưng

## Cách sử dụng

### Các tham số dòng lệnh

```
python model_importance_analyzer.py --model MODEL_PATH --data DATA_PATH [options]
```

#### Tham số bắt buộc
- `--model`: Đường dẫn đến file mô hình đã huấn luyện (định dạng .pkl hoặc .joblib)
- `--data`: Đường dẫn đến file dữ liệu CSV đã sử dụng để huấn luyện mô hình

#### Tham số tùy chọn
- `--output`: Thư mục lưu kết quả đầu ra (mặc định: 'output')
- `--raster_dir`: Thư mục chứa các file raster cho trực quan hóa
- `--study_area`: File shapefile của khu vực nghiên cứu để hiển thị trên bản đồ
- `--permutation`: Nếu được đặt, sẽ tính toán Permutation Importance 
- `--n_repeats`: Số lần lặp lại cho phương pháp Permutation Importance (mặc định: 10)
- `--visualize_maps`: Nếu được đặt, sẽ tạo các bản đồ trực quan hóa tầm quan trọng
- `--scaler`: Đường dẫn đến file scaler đã được fit (nếu có)

### Ví dụ

1. Phân tích cơ bản tầm quan trọng của đặc trưng:
   ```
   python model_importance_analyzer.py --model models/rf_model.joblib --data flood_data.csv
   ```

2. Phân tích toàn diện với tất cả các tùy chọn:
   ```
   python model_importance_analyzer.py --model models/xgb_model.joblib --data flood_data.csv --output results --raster_dir rasters --study_area study_area.shp --permutation --visualize_maps --scaler models/scaler.joblib
   ```

## Kết quả đầu ra

Công cụ sẽ tạo ra các kết quả sau:

1. **Báo cáo dạng văn bản** hiển thị thứ hạng tầm quan trọng của các đặc trưng
2. **Biểu đồ tầm quan trọng đặc trưng** (`feature_importance_<model_type>.png`)
3. **Biểu đồ tầm quan trọng hoán vị** (`permutation_importance_<model_type>.png`) - nếu yêu cầu
4. **Bản đồ tầm quan trọng cho từng đặc trưng** (`map_importance_<model_type>_<feature_name>.png`) - nếu yêu cầu
5. **Bản đồ tầm quan trọng tổng hợp** (`combined_importance_map_<model_type>.png` và `.tif`) - nếu yêu cầu

## Lưu ý

- Mô hình cần phải là Random Forest hoặc XGBoost
- Thư mục chứa raster cần có các file với tên tương ứng với tên đặc trưng, hoặc có pattern tương tự
- Dữ liệu CSV cần có cùng cấu trúc với dữ liệu đã dùng để huấn luyện mô hình

## Cách để lưu mô hình từ rf_xgb.py

Để sử dụng công cụ này, bạn cần lưu mô hình từ chương trình rf_xgb.py. Thêm đoạn code sau vào hàm `main()` trong file rf_xgb.py:

```python
# Lưu mô hình
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/rf_model.joblib')
joblib.dump(xgb_model, 'models/xgb_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
print("Đã lưu mô hình vào thư mục 'models'")
``` 