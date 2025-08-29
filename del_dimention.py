import pandas as pd
import os

def remove_columns(input_file_path, columns_to_remove, output_file_path=None):
    """
    Xóa các cột được chỉ định của file CSV
    
    Args:
        input_file_path (str): Đường dẫn đến file CSV đầu vào
        columns_to_remove (list): Danh sách các chỉ số cột cần xóa (bắt đầu từ 0)
        output_file_path (str, optional): Đường dẫn file CSV đầu ra. 
                                        Nếu None, sẽ ghi đè lên file gốc
    """
    try:
        # Đọc file CSV
        print(f"Đang đọc file: {input_file_path}")
        df = pd.read_csv(input_file_path)
        
        print(f"File có {len(df)} dòng và {len(df.columns)} cột")
        print(f"Tên các cột ban đầu: {list(df.columns)}")
        
        # Kiểm tra xem có đủ cột để xóa không
        if len(df.columns) <= max(columns_to_remove):
            print(f"Lỗi: File không có đủ cột để xóa (cần có ít nhất {max(columns_to_remove) + 1} cột)")
            return False
            
        # Tạo danh sách các cột cần giữ lại
        keep_columns = [i for i in range(len(df.columns)) if i not in columns_to_remove]
        df_new = df.iloc[:, keep_columns]
        
        print(f"Các cột sau khi xóa: {list(df_new.columns)}")
        
        # Xác định tên file đầu ra
        if output_file_path is None:
            # Tạo tên file đầu ra với đuôi _nonlatlon
            input_dir = os.path.dirname(input_file_path)
            input_filename = os.path.basename(input_file_path)
            name, ext = os.path.splitext(input_filename)
            output_file_path = os.path.join(input_dir, f"{name}_delindexNB{ext}")
        
        # Lưu file CSV mới
        print(f"Đang lưu file: {output_file_path}")
        df_new.to_csv(output_file_path, index=False)
        
        print("Hoàn thành! File đã được lưu thành công.")
        print(f"Số cột ban đầu: {len(df.columns)}")
        print(f"Số cột sau khi xóa: {len(df_new.columns)}")
        
        return True
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {input_file_path}")
        return False
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return False

def main():
    # Đường dẫn đến file CSV
    input_file = r"C:\Users\Admin\Downloads\prj\Flood_point\merged_flood_point_merge_cleaned_balanced_reordered_nonlatlon_normalized.csv"
    
    # Chỉ định các cột cần xóa (cột 14 và 15, index bắt đầu từ 0)
    columns_to_remove = [13, 14]  # index 13 và 14 tương ứng với cột 14 và 15
    
    # Gọi hàm xử lý
    success = remove_columns(input_file, columns_to_remove)
    
    if success:
        print("\n✅ Chương trình đã chạy thành công!")
    else:
        print("\n❌ Có lỗi xảy ra trong quá trình xử lý!")

if __name__ == "__main__":
    main()