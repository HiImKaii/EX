# JPG to PNG Converter

A simple Python utility to convert JPG/JPEG images to PNG format.

## Requirements

- Python 3.6 or higher
- Pillow library

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

### Command Line Interface

#### Convert a single image

```
python jpg_to_png_converter.py image.jpg
```

This will create a PNG file with the same name in the same directory.

You can also specify the output file:

```
python jpg_to_png_converter.py image.jpg output.png
```

#### Convert all images in a directory

```
python jpg_to_png_converter.py input_directory
```

This will convert all JPG/JPEG files in the input directory and save the PNG files in the same directory.

You can also specify the output directory:

```
python jpg_to_png_converter.py input_directory output_directory
```

### Graphical User Interface

For a more user-friendly experience, you can use the GUI version of the converter:

```
python gui_converter.py
```

The GUI allows you to:
- Select individual files or an entire directory to convert
- Choose an output directory
- Monitor conversion progress
- See conversion statistics

## Features

- Preserves image quality
- Supports both single file and batch conversion
- Simple command-line interface
- User-friendly graphical interface
- Error handling for invalid files 