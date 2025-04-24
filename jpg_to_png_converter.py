from PIL import Image
import os
import sys

def convert_jpg_to_png(input_path, output_path=None):
    """
    Convert a JPG image to PNG format
    
    Args:
        input_path: Path to the input JPG file
        output_path: Path to save the output PNG file. If None, will use the same name but with .png extension
    """
    try:
        # If output path is not specified, create one from the input path
        if output_path is None:
            file_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(file_name)[0]
            output_dir = os.path.dirname(input_path)
            output_path = os.path.join(output_dir, name_without_ext + ".png")
        
        # Open the image and convert it
        img = Image.open(input_path)
        img.save(output_path, "PNG")
        print(f"Successfully converted {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def convert_directory(input_dir, output_dir=None):
    """
    Convert all JPG images in a directory to PNG format
    
    Args:
        input_dir: Directory containing JPG files
        output_dir: Directory to save PNG files. If None, will save in the same directory
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return
    
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    converted = 0
    failed = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            
            if output_dir is not None:
                name_without_ext = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, name_without_ext + ".png")
            else:
                output_path = None
            
            success = convert_jpg_to_png(input_path, output_path)
            if success:
                converted += 1
            else:
                failed += 1
    
    print(f"Conversion complete. {converted} images converted, {failed} failed.")

if __name__ == "__main__":
    # Simple command line interface
    if len(sys.argv) == 1:
        print("Usage:")
        print("  Single file: python jpg_to_png_converter.py image.jpg [output.png]")
        print("  Directory:   python jpg_to_png_converter.py input_dir [output_dir]")
    elif len(sys.argv) == 2:
        # Single argument - could be file or directory
        path = sys.argv[1]
        if os.path.isdir(path):
            convert_directory(path)
        else:
            convert_jpg_to_png(path)
    elif len(sys.argv) == 3:
        # Two arguments - both files or both directories
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        if os.path.isdir(input_path):
            convert_directory(input_path, output_path)
        else:
            convert_jpg_to_png(input_path, output_path) 