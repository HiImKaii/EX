import numpy as np
import pandas as pd

# Xác định phạm vi tọa độ của tỉnh Tuyên Quang
min_lat, max_lat = 21.30, 22.25  # Vĩ độ
min_lon, max_lon = 104.50, 105.70  # Kinh độ

# Độ phân giải 100m (0.001 độ ~ 100m)
resolution = 0.2

# Tạo lưới tọa độ
latitudes = np.arange(min_lat, max_lat, resolution)
longitudes = np.arange(min_lon, max_lon, resolution)

# Tạo danh sách tọa độ
coordinates = [(lat, lon, "", "Asia/Ho_Chi_Minh", "2020-10-01", "2020-10-31") 
               for lat in latitudes for lon in longitudes]

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(coordinates, columns=["Latitude", "Longitude", "", "Timezone", "Start Date", "End Date"])

# Lưu thành tệp CSV
file_path = "tuyen_quang_coordinates_100m.csv"
df.to_csv(file_path, index=False)

print(f"Đã tạo file: {file_path}")
