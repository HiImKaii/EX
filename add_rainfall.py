import pandas as pd
import rasterio

# Đọc CSV và TIFF, thêm cột rainfall
df = pd.read_csv("input.csv")
with rasterio.open("rainfall.tif") as src:
    rainfall_data = src.read(1)
    transform = src.transform

# Trích xuất rainfall cho từng điểm
rainfall_values = []
for _, row in df.iterrows():
    lat, lon = row['lat'], row['lon']
    col, row_idx = ~transform * (lon, lat)
    col, row_idx = int(col), int(row_idx)
    
    if 0 <= row_idx < rainfall_data.shape[0] and 0 <= col < rainfall_data.shape[1]:
        rainfall_values.append(rainfall_data[row_idx, col])
    else:
        rainfall_values.append(0)

df['rainfall'] = rainfall_values
df.to_csv("output.csv", index=False)
