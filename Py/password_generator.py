import random
import string
import argparse
import pyperclip
import os
import json
from datetime import datetime

class PasswordGenerator:
    def __init__(self):
        self.lowercase_chars = string.ascii_lowercase
        self.uppercase_chars = string.ascii_uppercase
        self.digit_chars = string.digits
        self.special_chars = "!@#$%^&*()-_=+[]{}|;:,.<>?/~"
        self.similar_chars = "il1Lo0O"
        self.ambiguous_chars = "{}[]()/\\'\"`~,;:.<>"
        
        # Thiết lập mặc định
        self.length = 16
        self.use_lowercase = True
        self.use_uppercase = True
        self.use_digits = True
        self.use_special = True
        self.avoid_similar = False
        self.avoid_ambiguous = False
        
        # File lưu trữ các mật khẩu đã tạo
        self.history_file = os.path.join(os.path.expanduser("~"), ".password_history.json")
    
    def set_options(self, length=16, use_lowercase=True, use_uppercase=True, 
                   use_digits=True, use_special=True, avoid_similar=False, 
                   avoid_ambiguous=False):
        """Đặt các tùy chọn cho việc tạo mật khẩu"""
        self.length = max(4, length)  # Tối thiểu 4 ký tự
        self.use_lowercase = use_lowercase
        self.use_uppercase = use_uppercase
        self.use_digits = use_digits
        self.use_special = use_special
        self.avoid_similar = avoid_similar
        self.avoid_ambiguous = avoid_ambiguous
    
    def generate(self):
        """Tạo mật khẩu ngẫu nhiên dựa trên các tùy chọn đã cài đặt"""
        # Tạo bộ ký tự cho mật khẩu
        charset = ""
        
        if self.use_lowercase:
            charset += self.lowercase_chars
        
        if self.use_uppercase:
            charset += self.uppercase_chars
        
        if self.use_digits:
            charset += self.digit_chars
        
        if self.use_special:
            charset += self.special_chars
        
        # Loại bỏ các ký tự tương tự nếu được yêu cầu
        if self.avoid_similar:
            for char in self.similar_chars:
                charset = charset.replace(char, "")
        
        # Loại bỏ các ký tự có thể gây nhầm lẫn nếu được yêu cầu
        if self.avoid_ambiguous:
            for char in self.ambiguous_chars:
                charset = charset.replace(char, "")
        
        # Kiểm tra xem còn ký tự nào không
        if not charset:
            return "Lỗi: Không còn ký tự nào thỏa mãn các điều kiện"
        
        # Tạo mật khẩu
        password = "".join(random.choice(charset) for _ in range(self.length))
        
        # Đảm bảo mật khẩu có ít nhất một ký tự từ mỗi loại nếu được yêu cầu
        if self.length >= 4:
            has_lower = any(c in self.lowercase_chars for c in password) if self.use_lowercase else True
            has_upper = any(c in self.uppercase_chars for c in password) if self.use_uppercase else True
            has_digit = any(c in self.digit_chars for c in password) if self.use_digits else True
            has_special = any(c in self.special_chars for c in password) if self.use_special else True
            
            # Nếu không thỏa mãn, tạo lại
            if not (has_lower and has_upper and has_digit and has_special):
                return self.generate()
        
        return password
    
    def save_to_history(self, password, label=None):
        """Lưu mật khẩu đã tạo vào lịch sử"""
        # Đọc lịch sử hiện tại
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []
        
        # Thêm mật khẩu mới
        history.append({
            "password": password,
            "label": label,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "length": self.length,
            "options": {
                "lowercase": self.use_lowercase,
                "uppercase": self.use_uppercase,
                "digits": self.use_digits,
                "special": self.use_special,
                "avoid_similar": self.avoid_similar,
                "avoid_ambiguous": self.avoid_ambiguous
            }
        })
        
        # Lưu lại lịch sử
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_history(self, limit=10):
        """Lấy lịch sử các mật khẩu đã tạo"""
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            # Trả về lịch sử gần nhất
            return history[-limit:]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def clear_history(self):
        """Xóa lịch sử các mật khẩu đã tạo"""
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
            return True
        return False
    
    def analyze_password_strength(self, password):
        """Phân tích độ mạnh của mật khẩu"""
        length = len(password)
        has_lower = any(c in self.lowercase_chars for c in password)
        has_upper = any(c in self.uppercase_chars for c in password)
        has_digit = any(c in self.digit_chars for c in password)
        has_special = any(c in self.special_chars for c in password)
        
        # Tính điểm
        score = 0
        
        # Điểm theo độ dài
        if length >= 12:
            score += 3
        elif length >= 8:
            score += 2
        elif length >= 6:
            score += 1
        
        # Điểm theo loại ký tự
        if has_lower:
            score += 1
        if has_upper:
            score += 1
        if has_digit:
            score += 1
        if has_special:
            score += 1
        
        # Đánh giá
        if score >= 6:
            strength = "Rất mạnh"
        elif score >= 4:
            strength = "Mạnh"
        elif score >= 3:
            strength = "Trung bình"
        else:
            strength = "Yếu"
        
        return {
            "score": score,
            "strength": strength,
            "details": {
                "length": length,
                "has_lowercase": has_lower,
                "has_uppercase": has_upper,
                "has_digits": has_digit,
                "has_special": has_special
            }
        }

def interactive_mode():
    """Chạy chương trình trong chế độ tương tác"""
    generator = PasswordGenerator()
    
    while True:
        print("\n===== CHƯƠNG TRÌNH TẠO MẬT KHẨU =====")
        print("1. Tạo mật khẩu mới")
        print("2. Tùy chỉnh các thiết lập")
        print("3. Phân tích độ mạnh của mật khẩu")
        print("4. Xem lịch sử các mật khẩu đã tạo")
        print("5. Xóa lịch sử")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn của bạn: ")
        
        if choice == "1":
            # Tạo mật khẩu mới
            password = generator.generate()
            print(f"\nMật khẩu của bạn: {password}")
            
            # Phân tích độ mạnh
            analysis = generator.analyze_password_strength(password)
            print(f"Độ mạnh: {analysis['strength']} ({analysis['score']}/7 điểm)")
            
            # Hỏi có muốn sao chép vào clipboard không
            copy = input("Sao chép vào clipboard? (y/n): ").lower() == 'y'
            if copy:
                try:
                    pyperclip.copy(password)
                    print("Đã sao chép mật khẩu vào clipboard!")
                except:
                    print("Không thể sao chép vào clipboard!")
            
            # Hỏi có muốn lưu vào lịch sử không
            save = input("Lưu mật khẩu vào lịch sử? (y/n): ").lower() == 'y'
            if save:
                label = input("Nhập nhãn cho mật khẩu này (để trống nếu không cần): ")
                generator.save_to_history(password, label if label else None)
                print("Đã lưu vào lịch sử!")
        
        elif choice == "2":
            # Tùy chỉnh thiết lập
            print("\n=== TÙY CHỈNH THIẾT LẬP ===")
            
            try:
                length = int(input(f"Độ dài ({generator.length}): ") or generator.length)
                use_lowercase = input(f"Sử dụng chữ thường (y/n) [{('y' if generator.use_lowercase else 'n')}]: ").lower()
                use_uppercase = input(f"Sử dụng chữ hoa (y/n) [{('y' if generator.use_uppercase else 'n')}]: ").lower()
                use_digits = input(f"Sử dụng chữ số (y/n) [{('y' if generator.use_digits else 'n')}]: ").lower()
                use_special = input(f"Sử dụng ký tự đặc biệt (y/n) [{('y' if generator.use_special else 'n')}]: ").lower()
                avoid_similar = input(f"Tránh các ký tự tương tự (y/n) [{('y' if generator.avoid_similar else 'n')}]: ").lower()
                avoid_ambiguous = input(f"Tránh các ký tự có thể gây nhầm lẫn (y/n) [{('y' if generator.avoid_ambiguous else 'n')}]: ").lower()
                
                generator.set_options(
                    length=length,
                    use_lowercase=use_lowercase != 'n',
                    use_uppercase=use_uppercase != 'n',
                    use_digits=use_digits != 'n',
                    use_special=use_special != 'n',
                    avoid_similar=avoid_similar == 'y',
                    avoid_ambiguous=avoid_ambiguous == 'y'
                )
                
                print("Đã cập nhật thiết lập!")
                
            except ValueError:
                print("Giá trị không hợp lệ, sử dụng giá trị mặc định!")
        
        elif choice == "3":
            # Phân tích độ mạnh của mật khẩu
            password = input("Nhập mật khẩu cần phân tích: ")
            
            if not password:
                print("Mật khẩu không được để trống!")
                continue
            
            analysis = generator.analyze_password_strength(password)
            
            print("\n=== PHÂN TÍCH ĐỘ MẠNH ===")
            print(f"Mật khẩu: {password}")
            print(f"Độ mạnh: {analysis['strength']} ({analysis['score']}/7 điểm)")
            print(f"Độ dài: {analysis['details']['length']} ký tự")
            print(f"Có chữ thường: {'Có' if analysis['details']['has_lowercase'] else 'Không'}")
            print(f"Có chữ hoa: {'Có' if analysis['details']['has_uppercase'] else 'Không'}")
            print(f"Có chữ số: {'Có' if analysis['details']['has_digits'] else 'Không'}")
            print(f"Có ký tự đặc biệt: {'Có' if analysis['details']['has_special'] else 'Không'}")
        
        elif choice == "4":
            # Xem lịch sử
            history = generator.get_history()
            
            if not history:
                print("Chưa có mật khẩu nào trong lịch sử!")
                continue
            
            print("\n=== LỊCH SỬ MẬT KHẨU ===")
            for i, item in enumerate(history):
                label = f" ({item['label']})" if item.get('label') else ""
                print(f"{i+1}. {item['password']}{label} - {item['created_at']}")
        
        elif choice == "5":
            # Xóa lịch sử
            confirm = input("Bạn có chắc chắn muốn xóa lịch sử? (y/n): ").lower() == 'y'
            
            if confirm:
                if generator.clear_history():
                    print("Đã xóa lịch sử!")
                else:
                    print("Không có lịch sử để xóa!")
        
        elif choice == "0":
            # Thoát
            print("Tạm biệt!")
            break
        
        else:
            print("Lựa chọn không hợp lệ!")

def command_line_mode():
    """Chạy chương trình trong chế độ dòng lệnh"""
    parser = argparse.ArgumentParser(description="Chương trình tạo mật khẩu mạnh")
    
    parser.add_argument('-l', '--length', type=int, default=16, help='Độ dài của mật khẩu')
    parser.add_argument('-nl', '--no-lowercase', action='store_true', help='Không sử dụng chữ thường')
    parser.add_argument('-nu', '--no-uppercase', action='store_true', help='Không sử dụng chữ hoa')
    parser.add_argument('-nd', '--no-digits', action='store_true', help='Không sử dụng chữ số')
    parser.add_argument('-ns', '--no-special', action='store_true', help='Không sử dụng ký tự đặc biệt')
    parser.add_argument('-as', '--avoid-similar', action='store_true', help='Tránh các ký tự tương tự')
    parser.add_argument('-aa', '--avoid-ambiguous', action='store_true', help='Tránh các ký tự có thể gây nhầm lẫn')
    parser.add_argument('-c', '--copy', action='store_true', help='Sao chép mật khẩu vào clipboard')
    parser.add_argument('-s', '--save', action='store_true', help='Lưu mật khẩu vào lịch sử')
    parser.add_argument('--label', type=str, help='Nhãn cho mật khẩu khi lưu vào lịch sử')
    parser.add_argument('-a', '--analyze', action='store_true', help='Phân tích độ mạnh của mật khẩu')
    parser.add_argument('-p', '--password', type=str, help='Mật khẩu cần phân tích')
    parser.add_argument('--history', action='store_true', help='Xem lịch sử các mật khẩu đã tạo')
    parser.add_argument('--clear-history', action='store_true', help='Xóa lịch sử các mật khẩu đã tạo')
    
    args = parser.parse_args()
    
    generator = PasswordGenerator()
    
    if args.history:
        # Xem lịch sử
        history = generator.get_history()
        
        if not history:
            print("Chưa có mật khẩu nào trong lịch sử!")
            return
        
        print("=== LỊCH SỬ MẬT KHẨU ===")
        for i, item in enumerate(history):
            label = f" ({item['label']})" if item.get('label') else ""
            print(f"{i+1}. {item['password']}{label} - {item['created_at']}")
        
        return
    
    if args.clear_history:
        # Xóa lịch sử
        if generator.clear_history():
            print("Đã xóa lịch sử!")
        else:
            print("Không có lịch sử để xóa!")
        
        return
    
    if args.analyze:
        # Phân tích độ mạnh của mật khẩu
        password = args.password
        
        if not password:
            print("Vui lòng cung cấp mật khẩu cần phân tích bằng tham số --password")
            return
        
        analysis = generator.analyze_password_strength(password)
        
        print("=== PHÂN TÍCH ĐỘ MẠNH ===")
        print(f"Mật khẩu: {password}")
        print(f"Độ mạnh: {analysis['strength']} ({analysis['score']}/7 điểm)")
        print(f"Độ dài: {analysis['details']['length']} ký tự")
        print(f"Có chữ thường: {'Có' if analysis['details']['has_lowercase'] else 'Không'}")
        print(f"Có chữ hoa: {'Có' if analysis['details']['has_uppercase'] else 'Không'}")
        print(f"Có chữ số: {'Có' if analysis['details']['has_digits'] else 'Không'}")
        print(f"Có ký tự đặc biệt: {'Có' if analysis['details']['has_special'] else 'Không'}")
        
        return
    
    # Thiết lập các tùy chọn
    generator.set_options(
        length=args.length,
        use_lowercase=not args.no_lowercase,
        use_uppercase=not args.no_uppercase,
        use_digits=not args.no_digits,
        use_special=not args.no_special,
        avoid_similar=args.avoid_similar,
        avoid_ambiguous=args.avoid_ambiguous
    )
    
    # Tạo mật khẩu
    password = generator.generate()
    
    # Phân tích độ mạnh
    analysis = generator.analyze_password_strength(password)
    
    # Hiển thị kết quả
    print(f"Mật khẩu: {password}")
    print(f"Độ mạnh: {analysis['strength']} ({analysis['score']}/7 điểm)")
    
    # Sao chép vào clipboard nếu được yêu cầu
    if args.copy:
        try:
            pyperclip.copy(password)
            print("Đã sao chép mật khẩu vào clipboard!")
        except:
            print("Không thể sao chép vào clipboard!")
    
    # Lưu vào lịch sử nếu được yêu cầu
    if args.save:
        generator.save_to_history(password, args.label)
        print("Đã lưu vào lịch sử!")

def main():
    # Kiểm tra xem có đối số dòng lệnh hay không
    if len(sys.argv) > 1:
        command_line_mode()
    else:
        interactive_mode()

if __name__ == "__main__":
    import sys
    main() 