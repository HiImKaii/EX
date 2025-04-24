import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point, shape
import os
import gc  # Garbage collector để quản lý bộ nhớ
from tqdm import tqdm  # Thanh tiến trình

def create_sample_points(study_area_shapefile, num_points=500, output_file="sample_points.shp"):
    """
    Tạo các điểm mẫu ngẫu nhiên trong phạm vi khu vực nghiên cứu.
    Đây là một cách để tạo điểm mẫu nếu bạn chưa có.
    """
    print("Đang tạo điểm mẫu ngẫu nhiên...")
    
    # Đọc shapefile khu vực nghiên cứu
    study_area = gpd.read_file(study_area_shapefile)
    
    # Tạo điểm ngẫu nhiên trong phạm vi khu vực
    points = []
    bounds = study_area.total_bounds  # (minx, miny, maxx, maxy)
    
    for _ in range(num_points):
        x = np.random.uniform(bounds[0], bounds[2])
        y = np.random.uniform(bounds[1], bounds[3])
        point = Point(x, y)
        
        # Kiểm tra xem điểm có nằm trong khu vực nghiên cứu không
        if point.within(study_area.unary_union):
            points.append(point)
    
    # Tạo GeoDataFrame với các điểm
    gdf = gpd.GeoDataFrame(geometry=points, crs=study_area.crs)
    
    # Thêm cột flood ngẫu nhiên (0: không ngập, 1: ngập)
    # Lưu ý: Đây chỉ là giả định, trong thực tế bạn cần dữ liệu thực về tình trạng ngập lụt
    gdf['flood'] = np.random.choice([0, 1], size=len(gdf), p=[0.7, 0.3])  # Giả sử 30% điểm bị ngập
    
    # Lưu file shapefile
    gdf.to_file(output_file)
    print(f"Đã tạo và lưu {len(gdf)} điểm mẫu vào {output_file}")
    return gdf

def extract_values_from_known_points(points_shapefile, raster_directory, output_csv="flood_data.csv", batch_size=10000):
    """
    Trích xuất giá trị từ các file raster tại các điểm đã biết.
    Xử lý theo lô để tránh tràn bộ nhớ với bộ dữ liệu lớn.
    
    Parameters:
    -----------
    points_shapefile : str
        Đường dẫn đến file shapefile chứa các điểm
    raster_directory : str
        Đường dẫn đến thư mục chứa các file raster
    output_csv : str
        Đường dẫn đến file CSV để lưu kết quả
    batch_size : int
        Kích thước lô để xử lý (số điểm mỗi lô)
    """
    print("Đang trích xuất giá trị đặc trưng...")
    
    # Đọc file shapefile chứa các điểm
    print(f"Đang đọc file shapefile {points_shapefile}...")
    points = gpd.read_file(points_shapefile)
    total_points = len(points)
    print(f"Tổng số điểm: {total_points}")
    
    # Kiểm tra cột flood
    if 'flood' not in points.columns:
        print("Không tìm thấy cột 'flood' trong file điểm. Vui lòng thêm thông tin về tình trạng ngập lụt.")
        return
    
    # Danh sách các file raster cần xử lý
    raster_files = {
        "Rainfall": "Rainfall.tif",
        "Elevation": "Elevation.tif",
        "Slope": "Slope.tif",
        "Aspect": "Aspect.tif",
        "FlowDirection": "FlowDirection.tif",
        "FlowAccumulation": "FlowAccumulation.tif",
        "TWI": "TWI.tif",
        "DistanceToRiver": "DistanceToRiver.tif",
        "DrainageCapacity": "DrainageCapacity.tif",
        "LandCover": "LandCover.tif",
        "ImperviousSurface": "ImperviousSurface.tif",
        "SurfaceTemperature": "SurfaceTemperature.tif",
        "HydrologicSoilGroup": "HydrologicSoilGroup.tif"
    }
    
    # Kiểm tra xem file output đã tồn tại chưa
    file_exists = os.path.exists(output_csv)
    
    # Tính số lô cần xử lý
    num_batches = (total_points + batch_size - 1) // batch_size
    
    # Xử lý theo lô
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_points)
        
        print(f"Đang xử lý lô {batch_idx + 1}/{num_batches} (điểm {start_idx} đến {end_idx-1})...")
        
        # Lấy batch points
        batch_points = points.iloc[start_idx:end_idx].copy()
        
        # Tạo DataFrame để lưu dữ liệu cho lô này
        batch_data = pd.DataFrame()
        batch_data['flood'] = batch_points['flood'].values
        
        # Trích xuất giá trị từ các file raster
        for column_name, raster_filename in raster_files.items():
            raster_path = os.path.join(raster_directory, raster_filename)
            
            if not os.path.exists(raster_path):
                print(f"Không tìm thấy file {raster_path}. Bỏ qua.")
                continue
            
            try:
                with rasterio.open(raster_path) as src:
                    # Chuyển tọa độ từ hệ tọa độ của điểm sang hệ tọa độ của raster nếu cần
                    if batch_points.crs != src.crs:
                        points_transformed = batch_points.to_crs(src.crs)
                    else:
                        points_transformed = batch_points
                    
                    # Chuẩn bị danh sách tọa độ để lấy mẫu
                    coords = [(point.x, point.y) for point in points_transformed.geometry]
                    
                    # Sử dụng phương thức sample để trích xuất giá trị
                    values = [x[0] for x in src.sample(coords)]
                    
                    batch_data[column_name] = values
                    print(f"  Đã trích xuất giá trị {column_name} từ {raster_filename}")
            
            except Exception as e:
                print(f"Lỗi khi xử lý file {raster_path}: {e}")
                batch_data[column_name] = np.nan
        
        # Loại bỏ các hàng có giá trị NaN trong lô
        batch_data_cleaned = batch_data.dropna()
        print(f"  Đã loại bỏ {len(batch_data) - len(batch_data_cleaned)} hàng có giá trị thiếu trong lô này.")
        
        # Lưu DataFrame thành file CSV (append mode cho các lô sau lô đầu tiên)
        if batch_idx == 0 and file_exists:
            # Nếu file đã tồn tại và đây là lô đầu tiên, ghi đè
            batch_data_cleaned.to_csv(output_csv, index=False, mode='w')
        elif batch_idx == 0:
            # Lô đầu tiên và file chưa tồn tại, tạo mới với header
            batch_data_cleaned.to_csv(output_csv, index=False, mode='w')
        else:
            # Các lô tiếp theo, ghi thêm vào không cần header
            batch_data_cleaned.to_csv(output_csv, index=False, mode='a', header=False)
        
        print(f"  Đã lưu {len(batch_data_cleaned)} điểm dữ liệu từ lô {batch_idx + 1} vào {output_csv}")
        
        # Dọn dẹp bộ nhớ
        del batch_points, batch_data, batch_data_cleaned
        gc.collect()
    
    print(f"Đã hoàn thành trích xuất giá trị cho {total_points} điểm!")
    return

def extract_values_from_rasters_alternative(points_shapefile, raster_directory, output_csv="flood_data.csv", batch_size=10000):
    """
    Phương pháp trích xuất giá trị thay thế sử dụng rasterio.sample().
    Xử lý theo lô để tránh tràn bộ nhớ với bộ dữ liệu lớn.
    """
    print("Đang trích xuất giá trị đặc trưng (phương pháp thay thế)...")
    
    # Đọc file shapefile chứa các điểm
    print(f"Đang đọc file shapefile {points_shapefile}...")
    points = gpd.read_file(points_shapefile)
    total_points = len(points)
    print(f"Tổng số điểm: {total_points}")
    
    # Kiểm tra cột flood
    if 'flood' not in points.columns:
        print("Không tìm thấy cột 'flood' trong file điểm. Vui lòng thêm thông tin về tình trạng ngập lụt.")
        return
    
    # Danh sách các file raster cần xử lý
    raster_files = {
        "Rainfall": "Rainfall.tif",
        "Elevation": "Elevation.tif",
        "Slope": "Slope.tif",
        "Aspect": "Aspect.tif",
        "FlowDirection": "FlowDirection.tif",
        "FlowAccumulation": "FlowAccumulation.tif",
        "TWI": "TWI.tif",
        "DistanceToRiver": "DistanceToRiver.tif",
        "DrainageCapacity": "DrainageCapacity.tif",
        "LandCover": "LandCover.tif",
        "ImperviousSurface": "ImperviousSurface.tif",
        "SurfaceTemperature": "SurfaceTemperature.tif",
        "HydrologicSoilGroup": "HydrologicSoilGroup.tif"
    }
    
    # Kiểm tra xem file output đã tồn tại chưa
    file_exists = os.path.exists(output_csv)
    
    # Tính số lô cần xử lý
    num_batches = (total_points + batch_size - 1) // batch_size
    
    # Xử lý theo lô
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_points)
        
        print(f"Đang xử lý lô {batch_idx + 1}/{num_batches} (điểm {start_idx} đến {end_idx-1})...")
        
        # Lấy batch points
        batch_points = points.iloc[start_idx:end_idx].copy()
        
        # Tạo DataFrame để lưu dữ liệu cho lô này
        batch_data = pd.DataFrame()
        batch_data['flood'] = batch_points['flood'].values
        
        # Trích xuất giá trị từ các file raster
        for column_name, raster_filename in raster_files.items():
            raster_path = os.path.join(raster_directory, raster_filename)
            
            if not os.path.exists(raster_path):
                print(f"Không tìm thấy file {raster_path}. Bỏ qua.")
                continue
            
            try:
                with rasterio.open(raster_path) as src:
                    # Chuyển tọa độ từ hệ tọa độ của điểm sang hệ tọa độ của raster nếu cần
                    if batch_points.crs != src.crs:
                        points_transformed = batch_points.to_crs(src.crs)
                    else:
                        points_transformed = batch_points
                    
                    # Chuẩn bị danh sách tọa độ để lấy mẫu
                    coords = [(point.x, point.y) for point in points_transformed.geometry]
                    
                    # Sử dụng phương thức sample để trích xuất giá trị
                    values = [x[0] for x in src.sample(coords)]
                    
                    batch_data[column_name] = values
                    print(f"  Đã trích xuất giá trị {column_name} từ {raster_filename}")
            
            except Exception as e:
                print(f"Lỗi khi xử lý file {raster_path}: {e}")
                batch_data[column_name] = np.nan
        
        # Loại bỏ các hàng có giá trị NaN hoặc nodata
        nodata_value = getattr(src, 'nodata', None)
        if nodata_value is not None:
            batch_data_cleaned = batch_data.replace(nodata_value, np.nan).dropna()
        else:
            batch_data_cleaned = batch_data.dropna()
            
        print(f"  Đã loại bỏ {len(batch_data) - len(batch_data_cleaned)} hàng có giá trị thiếu trong lô này.")
        
        # Lưu DataFrame thành file CSV (append mode cho các lô sau lô đầu tiên)
        if batch_idx == 0 and file_exists:
            # Nếu file đã tồn tại và đây là lô đầu tiên, ghi đè
            batch_data_cleaned.to_csv(output_csv, index=False, mode='w')
        elif batch_idx == 0:
            # Lô đầu tiên và file chưa tồn tại, tạo mới với header
            batch_data_cleaned.to_csv(output_csv, index=False, mode='w')
        else:
            # Các lô tiếp theo, ghi thêm vào không cần header
            batch_data_cleaned.to_csv(output_csv, index=False, mode='a', header=False)
        
        print(f"  Đã lưu {len(batch_data_cleaned)} điểm dữ liệu từ lô {batch_idx + 1} vào {output_csv}")
        
        # Dọn dẹp bộ nhớ
        del batch_points, batch_data, batch_data_cleaned
        gc.collect()
    
    print(f"Đã hoàn thành trích xuất giá trị cho {total_points} điểm!")
    return

def manually_create_flood_points(raster_directory, study_area_shapefile, 
                                flood_points_coords, non_flood_points_coords, 
                                output_shapefile="manual_flood_points.shp"):
    """
    Tạo file điểm thủ công từ danh sách tọa độ của điểm ngập và không ngập.
    """
    print("Đang tạo file điểm ngập lụt thủ công...")
    
    # Đọc shapefile khu vực nghiên cứu để lấy CRS
    study_area = gpd.read_file(study_area_shapefile)
    crs = study_area.crs
    
    # Tạo danh sách điểm và nhãn
    geometries = []
    flood_labels = []
    
    # Thêm các điểm ngập lụt
    for x, y in flood_points_coords:
        geometries.append(Point(x, y))
        flood_labels.append(1)
    
    # Thêm các điểm không ngập lụt
    for x, y in non_flood_points_coords:
        geometries.append(Point(x, y))
        flood_labels.append(0)
    
    # Tạo GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'flood': flood_labels,
        'geometry': geometries
    }, crs=crs)
    
    # Lưu file shapefile
    gdf.to_file(output_shapefile)
    print(f"Đã tạo và lưu {len(gdf)} điểm (ngập: {sum(flood_labels)}, không ngập: {len(flood_labels) - sum(flood_labels)}) vào {output_shapefile}")
    
    return gdf

def process_grid_points_with_flood_history(
    grid_points_csv, 
    flood_history_shp, 
    raster_directory, 
    output_csv="final_flood_data.csv", 
    batch_size=50000
):
    """
    Xử lý các điểm lưới từ file CSV, kết hợp với dữ liệu ngập lụt lịch sử từ shapefile,
    và trích xuất giá trị từ các file raster, tạo ra file CSV kết quả cuối cùng.
    
    Parameters:
    -----------
    grid_points_csv : str
        Đường dẫn đến file CSV chứa các điểm lưới
    flood_history_shp : str
        Đường dẫn đến file Shapefile chứa dữ liệu ngập lụt lịch sử
    raster_directory : str
        Đường dẫn đến thư mục chứa các file raster
    output_csv : str
        Đường dẫn đến file CSV kết quả cuối cùng
    batch_size : int
        Kích thước lô để xử lý (số điểm mỗi lô)
    """
    print("Bắt đầu xử lý điểm lưới và dữ liệu ngập lụt...")
    
    # 1. Đọc dữ liệu điểm lưới từ file CSV
    print(f"Đang đọc file điểm lưới {grid_points_csv}...")
    grid_points_df = pd.read_csv(grid_points_csv)
    total_points = len(grid_points_df)
    print(f"Tổng số điểm lưới: {total_points}")
    
    # Kiểm tra xem cột tọa độ có tồn tại không
    # Điều chỉnh tên cột nếu cần thiết
    x_col = None
    y_col = None
    
    possible_x_cols = ['x', 'X', 'longitude', 'Longitude', 'long', 'Long', 'lon', 'Lon']
    possible_y_cols = ['y', 'Y', 'latitude', 'Latitude', 'lat', 'Lat']
    
    for col in possible_x_cols:
        if col in grid_points_df.columns:
            x_col = col
            break
    
    for col in possible_y_cols:
        if col in grid_points_df.columns:
            y_col = col
            break
    
    if x_col is None or y_col is None:
        print("Không tìm thấy cột tọa độ x/y trong file CSV. Vui lòng kiểm tra tên cột.")
        print(f"Tên cột trong file: {grid_points_df.columns.tolist()}")
        return
    
    print(f"Sử dụng cột '{x_col}' cho kinh độ và '{y_col}' cho vĩ độ.")
    
    # 2. Đọc dữ liệu ngập lụt lịch sử từ file Shapefile
    print(f"Đang đọc file ngập lụt lịch sử {flood_history_shp}...")
    flood_history = gpd.read_file(flood_history_shp)
    
    # 3. Chuẩn bị danh sách các file raster cần xử lý
    raster_files = {
        "Rainfall": "Rainfall.tif",
        "Elevation": "Elevation.tif",
        "Slope": "Slope.tif",
        "Aspect": "Aspect.tif",
        "FlowDirection": "FlowDirection.tif",
        "FlowAccumulation": "FlowAccumulation.tif",
        "TWI": "TWI.tif",
        "DistanceToRiver": "DistanceToRiver.tif",
        "DrainageCapacity": "DrainageCapacity.tif",
        "LandCover": "LandCover.tif",
        "ImperviousSurface": "ImperviousSurface.tif",
        "SurfaceTemperature": "SurfaceTemperature.tif",
        "HydrologicSoilGroup": "HydrologicSoilGroup.tif"
    }
    
    # Kiểm tra các file raster có tồn tại không
    for feature_name, filename in raster_files.items():
        file_path = os.path.join(raster_directory, filename)
        if not os.path.exists(file_path):
            print(f"Cảnh báo: Không tìm thấy file {file_path}")
    
    # 4. Kiểm tra xem file output đã tồn tại chưa
    file_exists = os.path.exists(output_csv)
    
    # 5. Tính số lô cần xử lý
    num_batches = (total_points + batch_size - 1) // batch_size
    
    # 6. Xử lý theo lô
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_points)
        
        print(f"Đang xử lý lô {batch_idx + 1}/{num_batches} (điểm {start_idx} đến {end_idx-1})...")
        
        # Lấy batch points
        batch_points = grid_points_df.iloc[start_idx:end_idx].copy()
        
        # Tạo GeoDataFrame từ batch_points để kiểm tra điểm nằm trong vùng ngập lụt
        geometry = [Point(xy) for xy in zip(batch_points[x_col], batch_points[y_col])]
        batch_gdf = gpd.GeoDataFrame(batch_points, geometry=geometry, crs=flood_history.crs)
        
        # Xác định điểm nào nằm trong vùng ngập lụt
        print("Đang xác định các điểm trong vùng ngập lụt...")
        # Phương pháp 1: Sử dụng spatial join
        # Chỉ hoạt động tốt nếu flood_history chứa các vùng ngập (polygon)
        try:
            # Tạo buffer nhỏ cho các điểm để tính toán giao điểm chính xác hơn
            batch_gdf['geometry'] = batch_gdf.geometry.buffer(0.00001)
            
            # Sử dụng spatial join để xác định điểm nào nằm trong vùng ngập
            joined = gpd.sjoin(batch_gdf, flood_history, how="left", predicate='intersects')
            batch_gdf['flood'] = ~joined.index.isin(joined[joined.index.duplicated(keep=False)].index)
            batch_gdf['flood'] = batch_gdf['flood'].astype(int)
            
            # Nếu kết quả có quá ít điểm ngập, có thể có vấn đề với dữ liệu
            flood_count = batch_gdf['flood'].sum()
            if flood_count == 0:
                print("Cảnh báo: Không có điểm nào nằm trong vùng ngập lụt.")
            else:
                print(f"Tìm thấy {flood_count} điểm nằm trong vùng ngập lụt.")
        except Exception as e:
            print(f"Lỗi khi xác định điểm ngập: {e}")
            print("Sử dụng phương pháp thay thế...")
            
            # Phương pháp 2: Kiểm tra từng điểm có nằm trong vùng ngập lụt không
            # Phương pháp này chậm hơn nhưng đáng tin cậy hơn
            flood_geom = flood_history.unary_union
            batch_gdf['flood'] = batch_gdf.geometry.apply(lambda x: 1 if x.intersects(flood_geom) else 0)
            
            flood_count = batch_gdf['flood'].sum()
            print(f"Tìm thấy {flood_count} điểm nằm trong vùng ngập lụt.")
        
        # Tạo DataFrame kết quả với cột STT, kinh độ, vĩ độ và ngập lụt
        result_df = pd.DataFrame()
        result_df['STT'] = range(start_idx + 1, end_idx + 1)  # 1-based index
        result_df['KinhDo'] = batch_points[x_col].values
        result_df['ViDo'] = batch_points[y_col].values
        result_df['NgapLut'] = batch_gdf['flood'].values
        
        # Trích xuất giá trị từ các file raster
        for feature_name, filename in raster_files.items():
            raster_path = os.path.join(raster_directory, filename)
            
            if not os.path.exists(raster_path):
                print(f"Bỏ qua {feature_name}: File không tồn tại")
                result_df[feature_name] = np.nan
                continue
            
            try:
                with rasterio.open(raster_path) as src:
                    # Chuyển tọa độ từ hệ tọa độ của điểm sang hệ tọa độ của raster nếu cần
                    if batch_gdf.crs != src.crs and batch_gdf.crs is not None and src.crs is not None:
                        print(f"Đang chuyển đổi hệ tọa độ cho {feature_name}...")
                        points_transformed = batch_gdf.to_crs(src.crs)
                        coords = [(point.x, point.y) for point in points_transformed.geometry]
                    else:
                        coords = [(x, y) for x, y in zip(batch_points[x_col], batch_points[y_col])]
                    
                    # Sử dụng phương thức sample để trích xuất giá trị
                    print(f"Đang trích xuất giá trị {feature_name}...")
                    sample_gen = src.sample(coords)
                    values = [x[0] if x[0] != src.nodata else np.nan for x in sample_gen]
                    
                    result_df[feature_name] = values
                    
            except Exception as e:
                print(f"Lỗi khi xử lý {feature_name}: {e}")
                result_df[feature_name] = np.nan
        
        # Lưu kết quả vào file CSV
        print(f"Đang lưu kết quả lô {batch_idx + 1}...")
        
        if batch_idx == 0 and file_exists:
            # Nếu file đã tồn tại và đây là lô đầu tiên, ghi đè
            result_df.to_csv(output_csv, index=False, mode='w')
        elif batch_idx == 0:
            # Lô đầu tiên và file chưa tồn tại, tạo mới với header
            result_df.to_csv(output_csv, index=False, mode='w')
        else:
            # Các lô tiếp theo, ghi thêm vào không cần header
            result_df.to_csv(output_csv, index=False, mode='a', header=False)
        
        # Dọn dẹp bộ nhớ
        del batch_points, batch_gdf, result_df
        gc.collect()
    
    print(f"Đã hoàn thành xử lý {total_points} điểm và lưu kết quả vào {output_csv}")
    
    return

def main():
    """
    Hàm chính để thực hiện quy trình trích xuất giá trị đặc trưng
    """
    # Đường dẫn đến thư mục chứa dữ liệu và tệp tin
    raster_directory = "raster_data"  # Thay đổi theo đường dẫn thực tế
    grid_points_csv = "grid_points.csv"  # Thay đổi thành tên file CSV chứa 1.55tr điểm lưới
    flood_history_shp = "flood_history.shp"  # Thay đổi thành tên file shapefile chứa dữ liệu ngập lụt lịch sử
    output_csv = "final_flood_data.csv"  # File CSV kết quả cuối cùng
    
    # Xử lý điểm lưới với dữ liệu ngập lụt lịch sử và trích xuất giá trị từ raster
    process_grid_points_with_flood_history(
        grid_points_csv=grid_points_csv,
        flood_history_shp=flood_history_shp,
        raster_directory=raster_directory,
        output_csv=output_csv,
        batch_size=50000  # Điều chỉnh kích thước lô phù hợp với RAM máy tính
    )
    
    print("Đã hoàn thành toàn bộ quy trình!")

if __name__ == "__main__":
    main()