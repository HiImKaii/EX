from PIL import Image
import os

def convert_image(input_path, output_format="jpeg", output_directory=None, resize=None):
    """
    Chuyển đổi định dạng và kích thước hình ảnh
    
    Args:
        input_path: Đường dẫn đến hình ảnh gốc
        output_format: Định dạng đầu ra (jpeg, png, bmp, gif)
        output_directory: Thư mục lưu hình ảnh đã chuyển đổi, mặc định là thư mục hiện tại
        resize: Tuple chứa (width, height) nếu muốn thay đổi kích thước
    
    Returns:
        Path to the converted image
    """
    try:
        # Mở hình ảnh gốc
        img = Image.open(input_path)
        
        # Lấy tên file
        file_name = os.path.basename(input_path)
        name, _ = os.path.splitext(file_name)
        
        # Đặt thư mục đầu ra
        if output_directory is None:
            output_directory = os.path.dirname(input_path)
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(output_directory, exist_ok=True)
        
        # Đường dẫn đầu ra
        output_path = os.path.join(output_directory, f"{name}.{output_format}")
        
        # Thay đổi kích thước nếu được yêu cầu
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        
        # Lưu hình ảnh với định dạng mới
        img.save(output_path, format=output_format.upper())
        
        return output_path
    
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

def batch_convert(input_directory, output_format="jpeg", output_directory=None, resize=None, recursive=False):
    """
    Chuyển đổi tất cả hình ảnh trong thư mục
    
    Args:
        input_directory: Thư mục chứa hình ảnh cần chuyển đổi
        output_format: Định dạng đầu ra (jpeg, png, bmp, gif)
        output_directory: Thư mục lưu trữ hình ảnh đã chuyển đổi
        resize: Tuple chứa (width, height) nếu muốn thay đổi kích thước
        recursive: Có nên tìm kiếm trong các thư mục con hay không
    """
    
    # Định dạng hình ảnh được hỗ trợ
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    # Tạo thư mục đầu ra nếu cần
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    # Đếm số lượng hình ảnh đã chuyển đổi
    converted_count = 0
    
    # Lấy danh sách tất cả các tệp
    if recursive:
        for root, _, files in os.walk(input_directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                
                if ext.lower() in image_extensions:
                    rel_path = os.path.relpath(root, input_directory)
                    out_dir = os.path.join(output_directory, rel_path) if output_directory else None
                    
                    result = convert_image(file_path, output_format, out_dir, resize)
                    if result:
                        converted_count += 1
    else:
        for file in os.listdir(input_directory):
            file_path = os.path.join(input_directory, file)
            
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file)
                
                if ext.lower() in image_extensions:
                    result = convert_image(file_path, output_format, output_directory, resize)
                    if result:
                        converted_count += 1
    
    return converted_count

def main():
    print("===== CHƯƠNG TRÌNH CHUYỂN ĐỔI HÌNH ẢNH =====")
    print("1. Chuyển đổi một hình ảnh")
    print("2. Chuyển đổi hàng loạt")
    
    choice = input("\nNhập lựa chọn của bạn: ")
    
    if choice == "1":
        input_path = input("Đường dẫn đến hình ảnh: ")
        
        if not os.path.exists(input_path):
            print("Không tìm thấy hình ảnh!")
            return
        
        output_format = input("Định dạng đầu ra (jpeg, png, bmp, gif) [mặc định: jpeg]: ").lower() or "jpeg"
        output_dir = input("Thư mục đầu ra (để trống nếu muốn sử dụng thư mục hiện tại): ")
        
        resize_choice = input("Bạn có muốn thay đổi kích thước không? (y/n): ").lower()
        resize = None
        
        if resize_choice == 'y':
            try:
                width = int(input("Chiều rộng (pixel): "))
                height = int(input("Chiều cao (pixel): "))
                resize = (width, height)
            except ValueError:
                print("Kích thước không hợp lệ, sẽ giữ nguyên kích thước gốc!")
        
        output_path = convert_image(input_path, output_format, output_dir if output_dir else None, resize)
        
        if output_path:
            print(f"Đã chuyển đổi thành công! Hình ảnh mới: {output_path}")
    
    elif choice == "2":
        input_dir = input("Thư mục chứa hình ảnh: ")
        
        if not os.path.isdir(input_dir):
            print("Thư mục không tồn tại!")
            return
        
        output_format = input("Định dạng đầu ra (jpeg, png, bmp, gif) [mặc định: jpeg]: ").lower() or "jpeg"
        output_dir = input("Thư mục đầu ra (để trống nếu muốn sử dụng thư mục gốc): ")
        
        resize_choice = input("Bạn có muốn thay đổi kích thước không? (y/n): ").lower()
        resize = None
        
        if resize_choice == 'y':
            try:
                width = int(input("Chiều rộng (pixel): "))
                height = int(input("Chiều cao (pixel): "))
                resize = (width, height)
            except ValueError:
                print("Kích thước không hợp lệ, sẽ giữ nguyên kích thước gốc!")
        
        recursive = input("Tìm kiếm trong các thư mục con? (y/n): ").lower() == 'y'
        
        count = batch_convert(input_dir, output_format, output_dir if output_dir else None, resize, recursive)
        print(f"Đã chuyển đổi {count} hình ảnh!")
    
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 