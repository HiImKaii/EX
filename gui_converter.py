import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from PIL import Image
import threading

class JPGtoPNGConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("JPG to PNG Converter")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Set up the UI
        self.setup_ui()
        
        # Variables to track conversion
        self.files_to_convert = []
        self.output_directory = ""
        self.conversion_in_progress = False
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="JPG to PNG Converter", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # File Selection Section
        file_frame = ttk.LabelFrame(main_frame, text="Select Files or Directory")
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Buttons for selecting files
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.select_files_btn = ttk.Button(btn_frame, text="Select Files", command=self.select_files)
        self.select_files_btn.pack(side=tk.LEFT, padx=5)
        
        self.select_dir_btn = ttk.Button(btn_frame, text="Select Directory", command=self.select_directory)
        self.select_dir_btn.pack(side=tk.LEFT, padx=5)
        
        self.selected_label = ttk.Label(file_frame, text="No files selected")
        self.selected_label.pack(padx=10, pady=(0, 10))
        
        # Output Directory Section
        output_frame = ttk.LabelFrame(main_frame, text="Output Directory")
        output_frame.pack(fill=tk.X, pady=(0, 20))
        
        output_btn_frame = ttk.Frame(output_frame)
        output_btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.output_dir_btn = ttk.Button(output_btn_frame, text="Select Output Directory", command=self.select_output_directory)
        self.output_dir_btn.pack(side=tk.LEFT)
        
        self.output_label = ttk.Label(output_frame, text="Same as input")
        self.output_label.pack(padx=10, pady=(0, 10))
        
        # Progress Section
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=550, mode="determinate")
        self.progress_bar.pack(padx=10, pady=10)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(padx=10)
        
        # Convert Button
        self.convert_btn = ttk.Button(main_frame, text="Convert", command=self.start_conversion)
        self.convert_btn.pack(pady=10)
    
    def select_files(self):
        if self.conversion_in_progress:
            return
            
        filetypes = [("JPEG files", "*.jpg *.jpeg")]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if files:
            self.files_to_convert = list(files)
            self.selected_label.config(text=f"{len(files)} files selected")
    
    def select_directory(self):
        if self.conversion_in_progress:
            return
            
        directory = filedialog.askdirectory()
        
        if directory:
            # Find all jpg/jpeg files in the directory
            jpg_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg')):
                        jpg_files.append(os.path.join(root, file))
            
            if jpg_files:
                self.files_to_convert = jpg_files
                self.selected_label.config(text=f"{len(jpg_files)} files found in directory")
            else:
                messagebox.showinfo("No Files Found", "No JPG/JPEG files found in the selected directory.")
                self.files_to_convert = []
                self.selected_label.config(text="No files selected")
    
    def select_output_directory(self):
        if self.conversion_in_progress:
            return
            
        directory = filedialog.askdirectory()
        
        if directory:
            self.output_directory = directory
            self.output_label.config(text=directory)
        else:
            self.output_directory = ""
            self.output_label.config(text="Same as input")
    
    def start_conversion(self):
        if self.conversion_in_progress:
            return
            
        if not self.files_to_convert:
            messagebox.showinfo("No Files", "Please select files to convert.")
            return
        
        # Disable buttons during conversion
        self.toggle_ui_elements(False)
        self.conversion_in_progress = True
        
        # Start conversion in a separate thread to keep UI responsive
        conversion_thread = threading.Thread(target=self.convert_files)
        conversion_thread.daemon = True
        conversion_thread.start()
    
    def convert_files(self):
        total_files = len(self.files_to_convert)
        converted = 0
        failed = 0
        
        # Reset progress bar
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = total_files
        
        for i, file_path in enumerate(self.files_to_convert):
            try:
                # Update status
                file_name = os.path.basename(file_path)
                self.update_status(f"Converting {i+1}/{total_files}: {file_name}")
                
                # Determine output path
                if self.output_directory:
                    out_filename = os.path.splitext(os.path.basename(file_path))[0] + ".png"
                    output_path = os.path.join(self.output_directory, out_filename)
                else:
                    output_path = os.path.splitext(file_path)[0] + ".png"
                
                # Convert the image
                img = Image.open(file_path)
                img.save(output_path, "PNG")
                converted += 1
                
            except Exception as e:
                failed += 1
                print(f"Error converting {file_path}: {e}")
            
            # Update progress bar
            self.progress_bar["value"] = i + 1
        
        # Conversion completed
        self.update_status(f"Completed: {converted} converted, {failed} failed")
        messagebox.showinfo("Conversion Complete", f"Successfully converted {converted} images.\nFailed: {failed}")
        
        # Reset and enable UI
        self.conversion_in_progress = False
        self.toggle_ui_elements(True)
    
    def update_status(self, message):
        # This needs to be run in the main thread
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def toggle_ui_elements(self, enabled):
        state = "normal" if enabled else "disabled"
        self.select_files_btn.config(state=state)
        self.select_dir_btn.config(state=state)
        self.output_dir_btn.config(state=state)
        self.convert_btn.config(state=state)

if __name__ == "__main__":
    root = tk.Tk()
    app = JPGtoPNGConverter(root)
    root.mainloop() 