import pandas as pd
import numpy as np
import os

def analyze_and_clean_csv(file_path):
    """
    Phân tích cấu trúc CSV và làm sạch dữ liệu
    
    Args:
        file_path (str): Đường dẫn đến file CSV
    
    Returns:
        pd.DataFrame: DataFrame đã được làm sạch
    """
    
    try:
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")
        
        print(f"Đang đọc file: {file_path}")
        
        # Đọc file CSV với các tùy chọn linh hoạt
        # Thử các encoding phổ biến
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Đọc thành công với encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Không thể đọc file với các encoding đã thử")
        
        print("\n=== THÔNG TIN CẤU TRÚC DỮ LIỆU ===")
        print(f"Kích thước dữ liệu: {df.shape[0]} hàng, {df.shape[1]} cột")
        print(f"Tên các cột: {list(df.columns)}")
        
        # Hiển thị kiểu dữ liệu
        print("\n=== KIỂU DỮ LIỆU CÁC CỘT ===")
        print(df.dtypes)
        
        # Hiển thị một vài hàng đầu
        print("\n=== 5 HÀNG ĐẦU TIÊN ===")
        print(df.head())
        
        # Phân tích dữ liệu thiếu
        print("\n=== PHÂN TÍCH DỮ LIỆU THIẾU ===")
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        missing_summary = pd.DataFrame({
            'Cột': missing_data.index,
            'Số hàng thiếu': missing_data.values,
            'Phần trăm thiếu': missing_percentage.values
        })
        missing_summary = missing_summary[missing_summary['Số hàng thiếu'] > 0]
        
        if len(missing_summary) > 0:
            print("Các cột có dữ liệu thiếu:")
            print(missing_summary.to_string(index=False))
        else:
            print("Không có dữ liệu thiếu!")
        
        # Làm sạch dữ liệu
        print("\n=== BẮT ĐẦU LÀM SẠCH DỮ LIỆU ===")
        
        # Lưu số hàng ban đầu
        original_rows = len(df)
        
        # Loại bỏ các hàng hoàn toàn trống
        df_cleaned = df.dropna(how='all')
        rows_after_empty = len(df_cleaned)
        
        if original_rows != rows_after_empty:
            print(f"Đã loại bỏ {original_rows - rows_after_empty} hàng hoàn toàn trống")
        
        # Loại bỏ các hàng có ít nhất 1 giá trị thiếu
        df_cleaned = df_cleaned.dropna()
        final_rows = len(df_cleaned)
        
        if rows_after_empty != final_rows:
            print(f"Đã loại bỏ {rows_after_empty - final_rows} hàng có dữ liệu thiếu")
        
        # Thống kê cuối cùng
        print("\n=== KẾT QUẢ LÀM SẠCH ===")
        print(f"Số hàng ban đầu: {original_rows}")
        print(f"Số hàng sau khi làm sạch: {final_rows}")
        print(f"Số hàng đã loại bỏ: {original_rows - final_rows}")
        print(f"Tỷ lệ dữ liệu giữ lại: {(final_rows/original_rows)*100:.2f}%")
        
        # Hiển thị thống kê mô tả cho dữ liệu số
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            print("\n=== THỐNG KÊ MÔ TẢ CÁC CỘT SỐ ===")
            print(df_cleaned[numeric_columns].describe())
        
        return df_cleaned
        
    except Exception as e:
        print(f"Lỗi khi xử lý file: {str(e)}")
        return None

def save_cleaned_data(df, original_path):
    """
    Lưu dữ liệu đã làm sạch
    
    Args:
        df (pd.DataFrame): DataFrame đã làm sạch
        original_path (str): Đường dẫn file gốc
    """
    if df is None or len(df) == 0:
        print("Không có dữ liệu để lưu!")
        return
    
    # Tạo tên file mới
    base_name = os.path.splitext(original_path)[0]
    cleaned_path = f"{base_name}_cleaned.csv"
    
    try:
        df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
        print(f"\nĐã lưu dữ liệu đã làm sạch vào: {cleaned_path}")
        return cleaned_path
    except Exception as e:
        print(f"Lỗi khi lưu file: {str(e)}")
        return None

def main():
    """
    Hàm chính
    """
    # Đường dẫn file đầu vào
    file_path = r"D:\Non_flood.csv"
    
    print("=== CHƯƠNG TRÌNH LÀM SẠCH DỮ LIỆU CSV ===\n")
    
    # Phân tích và làm sạch dữ liệu
    cleaned_df = analyze_and_clean_csv(file_path)
    
    if cleaned_df is not None:
        # Lưu dữ liệu đã làm sạch
        saved_path = save_cleaned_data(cleaned_df, file_path)
        
        if saved_path:
            print("\n=== HOÀN THÀNH ===")
            print("Chương trình đã xử lý thành công!")
            
            # Hỏi người dùng có muốn xem dữ liệu mẫu không
            print("\nBạn có muốn xem 10 hàng đầu tiên của dữ liệu đã làm sạch? (y/n)")
            choice = input().lower().strip()
            
            if choice in ['y', 'yes', 'có']:
                print("\n=== 10 HÀNG ĐẦU TIÊN CỦA DỮ LIỆU ĐÃ LÀM SẠCH ===")
                print(cleaned_df.head(10).to_string(index=False))
    else:
        print("\nKhông thể xử lý file. Vui lòng kiểm tra lại đường dẫn và định dạng file.")

if __name__ == "__main__":
    main()