import os
import json
from datetime import datetime

class TaskManager:
    def __init__(self, file_path="tasks.json"):
        self.file_path = file_path
        self.tasks = self._load_tasks()
    
    def _load_tasks(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []
    
    def _save_tasks(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, ensure_ascii=False, indent=2)
    
    def add_task(self, title, description="", due_date=None):
        task = {
            "id": len(self.tasks) + 1,
            "title": title,
            "description": description,
            "due_date": due_date,
            "completed": False,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.tasks.append(task)
        self._save_tasks()
        return task
    
    def list_tasks(self, show_completed=True):
        if not show_completed:
            return [task for task in self.tasks if not task["completed"]]
        return self.tasks
    
    def complete_task(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                self._save_tasks()
                return True
        return False
    
    def delete_task(self, task_id):
        for i, task in enumerate(self.tasks):
            if task["id"] == task_id:
                del self.tasks[i]
                self._save_tasks()
                return True
        return False

def main():
    manager = TaskManager()
    
    while True:
        print("\n===== QUẢN LÝ CÔNG VIỆC =====")
        print("1. Thêm công việc mới")
        print("2. Xem danh sách công việc")
        print("3. Đánh dấu hoàn thành")
        print("4. Xóa công việc")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn của bạn: ")
        
        if choice == "1":
            title = input("Tiêu đề: ")
            description = input("Mô tả (có thể bỏ trống): ")
            due_date = input("Hạn hoàn thành (YYYY-MM-DD, có thể bỏ trống): ")
            due_date = due_date if due_date else None
            manager.add_task(title, description, due_date)
            print("Đã thêm công việc mới!")
            
        elif choice == "2":
            show_all = input("Hiển thị cả công việc đã hoàn thành? (y/n): ").lower() == 'y'
            tasks = manager.list_tasks(show_all)
            
            if not tasks:
                print("Không có công việc nào!")
            else:
                print("\n===== DANH SÁCH CÔNG VIỆC =====")
                for task in tasks:
                    status = "✓" if task["completed"] else "□"
                    print(f"{task['id']}. [{status}] {task['title']} - {task['due_date'] or 'Không có hạn'}")
                    if task["description"]:
                        print(f"   {task['description']}")
                        
        elif choice == "3":
            task_id = int(input("Nhập ID công việc cần đánh dấu hoàn thành: "))
            if manager.complete_task(task_id):
                print("Đã đánh dấu hoàn thành!")
            else:
                print("Không tìm thấy công việc!")
                
        elif choice == "4":
            task_id = int(input("Nhập ID công việc cần xóa: "))
            if manager.delete_task(task_id):
                print("Đã xóa công việc!")
            else:
                print("Không tìm thấy công việc!")
                
        elif choice == "0":
            print("Tạm biệt!")
            break
            
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 