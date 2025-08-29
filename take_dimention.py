#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chương trình ghép dữ liệu CSV
Lấy cột 2 và 3 từ file CSV lớn để bổ sung vào file CSV nhỏ
"""

import pandas as pd
import os
import sys
from pathlib import Path

def read_csv_file(file_path, encoding='utf-8'):
    """Đọc file CSV với nhiều encoding khác nhau"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"✓ Đã đọc file: {file_path}")
            print(f"  - Encoding: {enc}")
            print(f"  - Số dòng: {len(df)}")
            print(f"  - Số cột: {len(df.columns)}")
            print(f"  - Tên cột: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            return df, enc
        except Exception as e:
            print(f"  Thử encoding {enc}... ✗ ({e})")
            continue
    
    print(f"✗ Không thể đọc file {file_path} với các encoding thử nghiệm")
    return None, None

def merge_csv_files(large_file_path, small_file_path, output_file_path):
    """
    Ghép dữ liệu từ 2 file CSV
    
    Args:
        large_file_path: Đường dẫn file lớn (có đầy đủ cột)
        small_file_path: Đường dẫn file nhỏ (thiếu 2 cột đầu)
        output_file_path: Đường dẫn file kết quả
    """
    
    print("=" * 60)
    print("🔗 BẮT ĐẦU XỬ LÝ GHÉP DỮ LIỆU CSV")
    print("=" * 60)
    
    # Đọc file lớn
    print("\n1. Đọc file CSV lớn...")
    large_df, large_encoding = read_csv_file(large_file_path)
    if large_df is None:
        return False
    
    # Đọc file nhỏ
    print("\n2. Đọc file CSV nhỏ...")
    small_df, small_encoding = read_csv_file(small_file_path)
    if small_df is None:
        return False
    
    # Hiển thị thông tin chi tiết
    print(f"\n📊 THÔNG TIN CHI TIẾT:")
    print(f"   File lớn: {len(large_df)} dòng, {len(large_df.columns)} cột")
    print(f"   File nhỏ: {len(small_df)} dòng, {len(small_df.columns)} cột")
    
    # Kiểm tra số lượng dòng
    if len(small_df) > len(large_df):
        print("⚠️  Cảnh báo: File nhỏ có nhiều dòng hơn file lớn!")
        print("   Sẽ chỉ lấy số dòng tương ứng với file nhỏ")
    
    # Lấy cột thứ 2 và thứ 3 từ file lớn
    print("\n3. Lấy cột thứ 2 và thứ 3 từ file lớn...")
    
    if len(large_df.columns) < 3:
        print("✗ Lỗi: File lớn không có đủ 3 cột!")
        print(f"   Chỉ có {len(large_df.columns)} cột: {list(large_df.columns)}")
        return False
    
    # Lấy cột thứ 2 và thứ 3 (index 1 và 2)
    col2_name = large_df.columns[1]
    col3_name = large_df.columns[2]
    
    print(f"  - Cột thứ 2: '{col2_name}'")
    print(f"  - Cột thứ 3: '{col3_name}'")
    
    # Lấy dữ liệu cột 2 và 3, chỉ lấy số dòng bằng với file nhỏ
    num_rows = len(small_df)
    col2_data = large_df[col2_name].iloc[:num_rows]
    col3_data = large_df[col3_name].iloc[:num_rows]
    
    print(f"  - Lấy {num_rows} dòng dữ liệu đầu tiên")
    
    # Hiển thị preview dữ liệu sẽ lấy
    print(f"\n📋 PREVIEW DỮ LIỆU SẼ LẤY (5 dòng đầu):")
    preview_data = pd.DataFrame({
        col2_name: col2_data.head(),
        col3_name: col3_data.head()
    })
    print(preview_data.to_string())
    
    # Tạo DataFrame kết quả
    print("\n4. Tạo file kết quả...")
    
    # Tạo DataFrame mới với cột 2 và 3 từ file lớn + tất cả cột từ file nhỏ
    result_df = pd.DataFrame()
    
    # Thêm cột thứ 2 và thứ 3 từ file lớn
    result_df[col2_name] = col2_data.reset_index(drop=True)
    result_df[col3_name] = col3_data.reset_index(drop=True)
    
    # Thêm tất cả cột từ file nhỏ
    for col in small_df.columns:
        result_df[col] = small_df[col].reset_index(drop=True)
    
    print(f"  - File kết quả có {len(result_df)} dòng và {len(result_df.columns)} cột")
    print(f"  - Thứ tự cột: {list(result_df.columns)}")
    
    # Lưu file kết quả
    print(f"\n5. Lưu file kết quả: {output_file_path}")
    try:
        result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print("✓ Đã lưu file thành công!")
        
        # Hiển thị preview kết quả
        print("\n6. PREVIEW KẾT QUẢ (5 dòng đầu):")
        print("=" * 100)
        print(result_df.head().to_string())
        print("=" * 100)
        
        # Thống kê
        print(f"\n📈 THỐNG KÊ KẾT QUẢ:")
        print(f"   ✅ Đã ghép {len(result_df)} dòng dữ liệu")
        print(f"   ✅ Tổng số cột: {len(result_df.columns)}")
        print(f"   ✅ Dung lượng file: {os.path.getsize(output_file_path) / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"✗ Lỗi khi lưu file: {e}")
        return False

def main():
    """Hàm chính với đường dẫn được chỉ định sẵn"""
    print("🔗 CHƯƠNG TRÌNH GHÉP DỮ LIỆU CSV")
    print("Tác giả: Claude AI")
    print("Mô tả: Lấy cột 2 và 3 từ file lớn để bổ sung vào file nhỏ\n")
    
    # Đường dẫn file được chỉ định
    large_file = r"D:\Vscode\flood_point_merge.csv"
    small_file = r"C:\Users\Admin\Downloads\prj\Flood_point\flood_point_merge_cleaned_balanced.csv"
    
    print(f"📁 File lớn: {large_file}")
    print(f"📁 File nhỏ: {small_file}")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(large_file):
        print(f"✗ File lớn không tồn tại: {large_file}")
        print("💡 Vui lòng kiểm tra lại đường dẫn!")
        return
        
    if not os.path.exists(small_file):
        print(f"✗ File nhỏ không tồn tại: {small_file}")
        print("💡 Vui lòng kiểm tra lại đường dẫn!")
        return
    
    # Tạo tên file output
    output_dir = os.path.dirname(small_file)
    output_name = "merged_" + os.path.basename(small_file)
    output_file = os.path.join(output_dir, output_name)
    
    print(f"\n📤 File kết quả sẽ được lưu tại:")
    print(f"   {output_file}")
    
    # Xác nhận
    print(f"\n🔍 THÔNG TIN FILE:")
    try:
        large_size = os.path.getsize(large_file) / (1024*1024)  # MB
        small_size = os.path.getsize(small_file) / (1024*1024)  # MB
        print(f"   File lớn: {large_size:.1f} MB")
        print(f"   File nhỏ: {small_size:.1f} MB")
    except:
        pass
    
    confirm = input("\n⚡ Bắt đầu xử lý? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes', 'có']:
        print("🚫 Đã hủy!")
        return
    
    # Thực hiện ghép dữ liệu
    success = merge_csv_files(large_file, small_file, output_file)
    
    if success:
        print(f"\n🎉 HOÀN THÀNH!")
        print(f"📂 File kết quả: {output_file}")
        print("✨ Dữ liệu đã được ghép thành công!")
        
        # Mở thư mục chứa file kết quả (Windows)
        if sys.platform.startswith('win'):
            try:
                os.startfile(os.path.dirname(output_file))
                print("📁 Đã mở thư mục chứa file kết quả!")
            except:
                pass
    else:
        print("\n💥 XỬ LÝ THẤT BẠI!")
        print("❌ Vui lòng kiểm tra lại dữ liệu và thử lại.")

def quick_analysis():
    """Phân tích nhanh 2 file trước khi ghép"""
    large_file = r"D:\Vscode\flood_point_merge.csv"
    small_file = r"C:\Users\Admin\Downloads\prj\Flood_point\flood_point_merge_cleaned_balanced.csv"
    
    print("🔍 PHÂN TÍCH NHANH 2 FILE CSV")
    print("=" * 50)
    
    for i, file_path in enumerate([large_file, small_file], 1):
        print(f"\n{i}. File: {os.path.basename(file_path)}")
        if os.path.exists(file_path):
            try:
                df, encoding = read_csv_file(file_path)
                if df is not None:
                    print(f"   📊 Shape: {df.shape}")
                    print(f"   🏷️  Columns: {list(df.columns)}")
                    print(f"   📝 Sample data:")
                    print(f"      {df.iloc[0].to_dict()}")
            except Exception as e:
                print(f"   ✗ Lỗi: {e}")
        else:
            print(f"   ✗ File không tồn tại!")

if __name__ == "__main__":
    # Cài đặt encoding cho Windows
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    
    # Kiểm tra thư viện cần thiết
    try:
        import pandas as pd
    except ImportError:
        print("❌ Thiếu thư viện pandas!")
        print("📦 Vui lòng cài đặt: pip install pandas")
        sys.exit(1)
    
    # Chạy chương trình
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        quick_analysis()
    else:
        main()