import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

class CSVAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Đọc dữ liệu từ file CSV"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Đã đọc file CSV với {len(self.df)} dòng và {len(self.df.columns)} cột")
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            self.df = None
    
    def get_summary(self):
        """Trả về thông tin tóm tắt về dữ liệu"""
        if self.df is None:
            return None
        
        summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_columns": list(self.df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object']).columns),
            "datetime_columns": list(self.df.select_dtypes(include=['datetime']).columns)
        }
        
        # Thống kê mô tả cho các cột số
        if len(summary["numeric_columns"]) > 0:
            summary["numeric_stats"] = self.df[summary["numeric_columns"]].describe().to_dict()
        
        # Thống kê cho các cột phân loại
        if len(summary["categorical_columns"]) > 0:
            cat_stats = {}
            for col in summary["categorical_columns"]:
                value_counts = self.df[col].value_counts()
                if len(value_counts) <= 10:  # Chỉ hiển thị nếu có ít hơn 10 giá trị khác nhau
                    cat_stats[col] = value_counts.to_dict()
                else:
                    cat_stats[col] = {"unique_values": len(value_counts)}
            summary["categorical_stats"] = cat_stats
        
        return summary
    
    def plot_histogram(self, column, bins=10, output_file=None):
        """Vẽ biểu đồ histogram cho một cột số"""
        if self.df is None or column not in self.df.columns:
            print(f"Không tìm thấy cột {column}")
            return False
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print(f"Cột {column} không phải là cột số")
            return False
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.df[column].dropna(), bins=bins)
        plt.title(f'Histogram của {column}')
        plt.xlabel(column)
        plt.ylabel('Tần suất')
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file)
            print(f"Đã lưu biểu đồ vào file {output_file}")
        else:
            plt.show()
        
        plt.close()
        return True
    
    def plot_bar(self, column, top_n=10, output_file=None):
        """Vẽ biểu đồ cột cho các giá trị phân loại"""
        if self.df is None or column not in self.df.columns:
            print(f"Không tìm thấy cột {column}")
            return False
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        value_counts.plot(kind='bar')
        plt.title(f'Top {top_n} giá trị phổ biến nhất của {column}')
        plt.xlabel(column)
        plt.ylabel('Số lượng')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Đã lưu biểu đồ vào file {output_file}")
        else:
            plt.show()
        
        plt.close()
        return True
    
    def plot_scatter(self, x_column, y_column, output_file=None):
        """Vẽ biểu đồ scatter giữa 2 cột số"""
        if self.df is None:
            print("Không có dữ liệu")
            return False
        
        if x_column not in self.df.columns or y_column not in self.df.columns:
            print(f"Không tìm thấy cột {x_column} hoặc {y_column}")
            return False
        
        if not pd.api.types.is_numeric_dtype(self.df[x_column]) or not pd.api.types.is_numeric_dtype(self.df[y_column]):
            print(f"Cả hai cột {x_column} và {y_column} phải là cột số")
            return False
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[x_column], self.df[y_column], alpha=0.5)
        plt.title(f'Biểu đồ scatter giữa {x_column} và {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file)
            print(f"Đã lưu biểu đồ vào file {output_file}")
        else:
            plt.show()
        
        plt.close()
        return True
    
    def filter_data(self, conditions):
        """Lọc dữ liệu theo điều kiện và trả về DataFrame mới"""
        if self.df is None:
            print("Không có dữ liệu")
            return None
        
        filtered_df = self.df.copy()
        
        for column, condition in conditions.items():
            if column not in self.df.columns:
                print(f"Không tìm thấy cột {column}")
                continue
            
            operator, value = condition
            
            if operator == "==":
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == "!=":
                filtered_df = filtered_df[filtered_df[column] != value]
            elif operator == ">":
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == ">=":
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif operator == "<":
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == "<=":
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif operator == "in":
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            elif operator == "not in":
                filtered_df = filtered_df[~filtered_df[column].isin(value)]
            elif operator == "contains":
                filtered_df = filtered_df[filtered_df[column].str.contains(value, na=False)]
            else:
                print(f"Toán tử không hỗ trợ: {operator}")
        
        print(f"Đã lọc được {len(filtered_df)} dòng")
        return filtered_df
    
    def export_to_csv(self, df, output_file):
        """Xuất DataFrame ra file CSV"""
        try:
            df.to_csv(output_file, index=False)
            print(f"Đã xuất dữ liệu ra file {output_file}")
            return True
        except Exception as e:
            print(f"Lỗi khi xuất file: {e}")
            return False

def main():
    # Kiểm tra đối số dòng lệnh
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Nhập đường dẫn đến file CSV: ")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} không tồn tại")
        return
    
    analyzer = CSVAnalyzer(file_path)
    
    if analyzer.df is None:
        return
    
    while True:
        print("\n===== PHÂN TÍCH DỮ LIỆU CSV =====")
        print("1. Xem thông tin tóm tắt")
        print("2. Xem mẫu dữ liệu")
        print("3. Vẽ biểu đồ histogram")
        print("4. Vẽ biểu đồ cột")
        print("5. Vẽ biểu đồ scatter")
        print("6. Lọc dữ liệu")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn của bạn: ")
        
        if choice == "1":
            summary = analyzer.get_summary()
            if summary:
                print("\n=== THÔNG TIN TÓM TẮT ===")
                print(f"Số dòng: {summary['shape'][0]}")
                print(f"Số cột: {summary['shape'][1]}")
                
                print("\nDanh sách các cột:")
                for col in summary['columns']:
                    print(f"- {col} ({summary['dtypes'][col]})")
                
                print("\nGiá trị thiếu:")
                for col, count in summary['missing_values'].items():
                    if count > 0:
                        print(f"- {col}: {count} ({count/summary['shape'][0]*100:.2f}%)")
                
                if 'numeric_stats' in summary:
                    print("\nThống kê cho các cột số:")
                    for col in summary['numeric_columns']:
                        stats = {k: summary['numeric_stats'][col][k] for k in ['min', 'max', 'mean', 'std']}
                        print(f"- {col}: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Trung bình={stats['mean']:.2f}, Độ lệch chuẩn={stats['std']:.2f}")
                
                if 'categorical_stats' in summary:
                    print("\nThống kê cho các cột phân loại:")
                    for col, stats in summary['categorical_stats'].items():
                        if isinstance(stats, dict) and 'unique_values' in stats:
                            print(f"- {col}: {stats['unique_values']} giá trị duy nhất")
                        else:
                            print(f"- {col}: {len(stats)} giá trị duy nhất")
                            for val, count in list(stats.items())[:5]:
                                print(f"  + {val}: {count} ({count/summary['shape'][0]*100:.2f}%)")
                            if len(stats) > 5:
                                print("  + ...")
        
        elif choice == "2":
            rows = input("Số dòng muốn xem (mặc định: 5): ")
            try:
                rows = int(rows) if rows else 5
                print(analyzer.df.head(rows))
            except ValueError:
                print("Số dòng không hợp lệ")
        
        elif choice == "3":
            # Hiển thị các cột số
            numeric_cols = analyzer.df.select_dtypes(include=['number']).columns
            print("Các cột số:")
            for i, col in enumerate(numeric_cols):
                print(f"{i+1}. {col}")
            
            col_idx = input("Chọn cột (nhập số thứ tự): ")
            try:
                col_idx = int(col_idx) - 1
                if 0 <= col_idx < len(numeric_cols):
                    column = numeric_cols[col_idx]
                    bins = input("Số bins (mặc định: 10): ")
                    bins = int(bins) if bins else 10
                    save = input("Lưu biểu đồ? (y/n): ").lower() == 'y'
                    output_file = None
                    if save:
                        output_file = input("Tên file (mặc định: histogram.png): ") or "histogram.png"
                    analyzer.plot_histogram(column, bins, output_file)
                else:
                    print("Lựa chọn không hợp lệ")
            except ValueError:
                print("Số thứ tự không hợp lệ")
        
        elif choice == "4":
            # Hiển thị tất cả các cột
            columns = analyzer.df.columns
            print("Các cột:")
            for i, col in enumerate(columns):
                print(f"{i+1}. {col}")
            
            col_idx = input("Chọn cột (nhập số thứ tự): ")
            try:
                col_idx = int(col_idx) - 1
                if 0 <= col_idx < len(columns):
                    column = columns[col_idx]
                    top_n = input("Số giá trị hiển thị (mặc định: 10): ")
                    top_n = int(top_n) if top_n else 10
                    save = input("Lưu biểu đồ? (y/n): ").lower() == 'y'
                    output_file = None
                    if save:
                        output_file = input("Tên file (mặc định: bar_chart.png): ") or "bar_chart.png"
                    analyzer.plot_bar(column, top_n, output_file)
                else:
                    print("Lựa chọn không hợp lệ")
            except ValueError:
                print("Số thứ tự không hợp lệ")
        
        elif choice == "5":
            # Hiển thị các cột số
            numeric_cols = analyzer.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                print("Cần ít nhất 2 cột số để vẽ biểu đồ scatter")
                continue
            
            print("Các cột số:")
            for i, col in enumerate(numeric_cols):
                print(f"{i+1}. {col}")
            
            x_col_idx = input("Chọn cột X (nhập số thứ tự): ")
            y_col_idx = input("Chọn cột Y (nhập số thứ tự): ")
            
            try:
                x_col_idx = int(x_col_idx) - 1
                y_col_idx = int(y_col_idx) - 1
                
                if 0 <= x_col_idx < len(numeric_cols) and 0 <= y_col_idx < len(numeric_cols):
                    x_column = numeric_cols[x_col_idx]
                    y_column = numeric_cols[y_col_idx]
                    
                    save = input("Lưu biểu đồ? (y/n): ").lower() == 'y'
                    output_file = None
                    if save:
                        output_file = input("Tên file (mặc định: scatter_plot.png): ") or "scatter_plot.png"
                    
                    analyzer.plot_scatter(x_column, y_column, output_file)
                else:
                    print("Lựa chọn không hợp lệ")
            except ValueError:
                print("Số thứ tự không hợp lệ")
        
        elif choice == "6":
            # Lọc dữ liệu
            conditions = {}
            
            while True:
                # Hiển thị các cột
                columns = analyzer.df.columns
                print("\nCác cột:")
                for i, col in enumerate(columns):
                    print(f"{i+1}. {col}")
                
                col_idx = input("\nChọn cột để lọc (nhập số thứ tự, nhấn Enter để kết thúc): ")
                if not col_idx:
                    break
                
                try:
                    col_idx = int(col_idx) - 1
                    if 0 <= col_idx < len(columns):
                        column = columns[col_idx]
                        
                        # Hiển thị các toán tử có thể sử dụng
                        print("\nCác toán tử:")
                        operators = ["==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains"]
                        for i, op in enumerate(operators):
                            print(f"{i+1}. {op}")
                        
                        op_idx = input("Chọn toán tử (nhập số thứ tự): ")
                        try:
                            op_idx = int(op_idx) - 1
                            if 0 <= op_idx < len(operators):
                                operator = operators[op_idx]
                                
                                # Nhập giá trị
                                if operator in ["in", "not in"]:
                                    value_input = input("Nhập các giá trị (cách nhau bởi dấu phẩy): ")
                                    value = [v.strip() for v in value_input.split(",")]
                                    # Chuyển đổi sang số nếu cần
                                    if pd.api.types.is_numeric_dtype(analyzer.df[column]):
                                        value = [float(v) if v.replace('.', '', 1).isdigit() else v for v in value]
                                else:
                                    value = input("Nhập giá trị: ")
                                    # Chuyển đổi sang số nếu cần
                                    if pd.api.types.is_numeric_dtype(analyzer.df[column]) and value.replace('.', '', 1).isdigit():
                                        value = float(value)
                                
                                conditions[column] = (operator, value)
                            else:
                                print("Lựa chọn không hợp lệ")
                        except ValueError:
                            print("Số thứ tự không hợp lệ")
                    else:
                        print("Lựa chọn không hợp lệ")
                except ValueError:
                    print("Số thứ tự không hợp lệ")
            
            if conditions:
                filtered_df = analyzer.filter_data(conditions)
                if filtered_df is not None and len(filtered_df) > 0:
                    print("\n=== KẾT QUẢ LỌC ===")
                    rows = input("Số dòng muốn xem (mặc định: 5): ")
                    try:
                        rows = int(rows) if rows else 5
                        print(filtered_df.head(rows))
                        
                        save = input("Xuất kết quả ra file CSV? (y/n): ").lower() == 'y'
                        if save:
                            output_file = input("Tên file (mặc định: filtered_data.csv): ") or "filtered_data.csv"
                            analyzer.export_to_csv(filtered_df, output_file)
                    except ValueError:
                        print("Số dòng không hợp lệ")
        
        elif choice == "0":
            print("Tạm biệt!")
            break
        
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 