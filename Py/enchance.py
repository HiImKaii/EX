import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import argparse
import cv2

class ImageEnhancer:
    def __init__(self, model_path=None, target_resolution=(3840, 2160), preserve_objects=True):
        self.model = None
        self.target_resolution = target_resolution  # Default to 4K resolution
        self.preserve_objects = preserve_objects
        self.object_detector = None
        
        if preserve_objects:
            self._initialize_object_detector()
            
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def _initialize_object_detector(self):
        """Initialize object detector for object preservation"""
        try:
            # Thử tải một mô hình đơn giản từ OpenCV để phát hiện vật thể
            print("Initializing object detector for object preservation...")
            # Sử dụng mô hình YOLOv4 từ OpenCV
            config_path = "yolov4.cfg" 
            weights_path = "yolov4.weights"
            
            # Kiểm tra xem file có tồn tại không
            if not (os.path.exists(config_path) and os.path.exists(weights_path)):
                print("YOLOv4 model files not found, downloading...")
                # Nếu chưa có file, sử dụng DNN từ OpenCV
                self.object_detector = cv2.dnn.readNetFromDarknet(
                    "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
                    "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
                )
            else:
                self.object_detector = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            
            # Sử dụng COCO dataset classes
            with open("coco.names", "r") as f:
                self.classes = f.read().strip().split("\n")
                
            # Fallback nếu không tìm thấy file coco.names
            if not os.path.exists("coco.names"):
                self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                               "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                               "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                               "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                               "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                               "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                               "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
                
            print("Object detector initialized successfully!")
        except Exception as e:
            print(f"Error initializing object detector: {e}")
            print("Continuing without object preservation...")
            self.object_detector = None
            self.preserve_objects = False
    
    def _detect_objects(self, image_array):
        """Detect objects in the image for preservation"""
        if not self.preserve_objects or self.object_detector is None:
            return None
            
        try:
            height, width = image_array.shape[:2]
            
            # Tạo blob từ ảnh để đưa vào mô hình
            blob = cv2.dnn.blobFromImage(image_array, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.object_detector.setInput(blob)
            
            # Lấy các layer đầu ra
            layer_names = self.object_detector.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.object_detector.getUnconnectedOutLayers()]
            
            # Chạy forward pass
            outputs = self.object_detector.forward(output_layers)
            
            # Khởi tạo danh sách kết quả
            boxes = []
            confidences = []
            class_ids = []
            
            # Xử lý mỗi output
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:  # Ngưỡng tin cậy
                        # Tọa độ object
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Tọa độ góc trái trên
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Non-maximum suppression để loại bỏ các bbox trùng lặp
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            detected_objects = []
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    x, y, w, h = box
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    detected_objects.append({
                        'box': box,
                        'label': label,
                        'confidence': confidence
                    })
            
            return detected_objects
        
        except Exception as e:
            print(f"Error in object detection: {e}")
            return None
    
    def _apply_object_preservation(self, original_image, enhanced_image, objects):
        """Apply object preservation to ensure objects don't get distorted"""
        if objects is None or len(objects) == 0:
            return enhanced_image
            
        try:
            # Convert to numpy arrays if they are PIL images
            if isinstance(original_image, Image.Image):
                original_array = np.array(original_image)
            else:
                original_array = original_image
                
            if isinstance(enhanced_image, Image.Image):
                enhanced_array = np.array(enhanced_image)
            else:
                enhanced_array = enhanced_image
            
            # Đảm bảo kích thước phù hợp
            if original_array.shape[:2] != enhanced_array.shape[:2]:
                enhanced_array = cv2.resize(enhanced_array, (original_array.shape[1], original_array.shape[0]))
            
            # Tạo một mask để theo dõi các vùng đã thay đổi
            mask = np.zeros(original_array.shape[:2], dtype=np.uint8)
            
            # Xử lý từng đối tượng
            for obj in objects:
                x, y, w, h = obj['box']
                
                # Đảm bảo tọa độ nằm trong phạm vi ảnh
                x = max(0, x)
                y = max(0, y)
                w = min(w, original_array.shape[1] - x)
                h = min(h, original_array.shape[0] - y)
                
                # Extract original and enhanced object regions
                original_obj = original_array[y:y+h, x:x+w]
                enhanced_obj = enhanced_array[y:y+h, x:x+w]
                
                # Apply structural similarity-based blending
                # Tính toán cấu trúc giống nhau giữa 2 vùng
                original_gray = cv2.cvtColor(original_obj, cv2.COLOR_BGR2GRAY)
                enhanced_gray = cv2.cvtColor(enhanced_obj, cv2.COLOR_BGR2GRAY)
                
                # Áp dụng bộ lọc Gaussian để tính toán cấu trúc
                original_structure = cv2.GaussianBlur(original_gray, (5, 5), 0)
                enhanced_structure = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
                
                # Tính sự khác biệt của cấu trúc
                structure_diff = cv2.absdiff(original_structure, enhanced_structure)
                _, structure_mask = cv2.threshold(structure_diff, 30, 255, cv2.THRESH_BINARY)
                
                # Tạo mặt nạ ưu tiên bảo toàn cấu trúc
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                structure_mask = cv2.dilate(structure_mask, kernel, iterations=2)
                
                # Blend based on structure mask - bảo toàn cấu trúc từ ảnh gốc
                for c in range(3):  # process each channel
                    enhanced_obj[:,:,c] = np.where(
                        structure_mask == 255,
                        original_obj[:,:,c],  # Keep original structure
                        enhanced_obj[:,:,c]   # Use enhanced details
                    )
                
                # Cập nhật vùng đối tượng trong ảnh đã tăng cường
                enhanced_array[y:y+h, x:x+w] = enhanced_obj
                mask[y:y+h, x:x+w] = 255
            
            # Hiển thị thông tin về số lượng đối tượng được bảo toàn
            print(f"Preserved {len(objects)} objects in the image")
            
            # Convert back to PIL image
            return Image.fromarray(enhanced_array)
            
        except Exception as e:
            print(f"Error in object preservation: {e}")
            return enhanced_image
    
    def build_model(self):
        # Input layer - allow any size input
        inputs = keras.layers.Input(shape=(None, None, 3))
        
        # Improved model architecture with residual connections for better quality
        # Initial feature extraction
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        
        # First residual block
        skip1 = x
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.Add()([x, skip1])
        x = keras.layers.Activation('relu')(x)
        
        # Second residual block
        skip2 = x
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.Add()([x, skip2])
        x = keras.layers.Activation('relu')(x)
        
        # Third residual block
        skip3 = x
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.Add()([x, skip3])
        x = keras.layers.Activation('relu')(x)
        
        # Feature reconstruction
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        
        # Output layer
        outputs = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Create model
        self.model = keras.models.Model(inputs, outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                          loss='mse',
                          metrics=['mae'])
        print("Enhanced model built successfully")
    
    def prepare_data(self, high_quality_dir, low_quality_dir, img_size=(512, 512), augment=True):
        """Prepare training data from directories of high and low quality images with larger patch size"""
        high_quality_images = glob(os.path.join(high_quality_dir, "*.jpg")) + glob(os.path.join(high_quality_dir, "*.png"))
        low_quality_images = glob(os.path.join(low_quality_dir, "*.jpg")) + glob(os.path.join(low_quality_dir, "*.png"))
        
        if len(high_quality_images) != len(low_quality_images):
            raise ValueError("The number of high and low quality images must be the same")
        
        x_data = []
        y_data = []
        
        for i in range(len(high_quality_images)):
            # Load images while preserving original size
            low_img = Image.open(low_quality_images[i])
            high_img = Image.open(high_quality_images[i])
            
            # For training, we extract patches to handle memory constraints
            low_img_array = np.array(low_img)
            high_img_array = np.array(high_img)
            
            # Ensure same size for paired images
            min_height = min(low_img_array.shape[0], high_img_array.shape[0])
            min_width = min(low_img_array.shape[1], high_img_array.shape[1])
            
            low_img_array = low_img_array[:min_height, :min_width]
            high_img_array = high_img_array[:min_height, :min_width]
            
            # Extract random patches for training
            h, w = img_size
            for _ in range(4):  # Extract multiple patches from each image
                if min_height > h and min_width > w:
                    y = np.random.randint(0, min_height - h)
                    x = np.random.randint(0, min_width - w)
                    
                    low_patch = low_img_array[y:y+h, x:x+w]
                    high_patch = high_img_array[y:y+h, x:x+w]
                    
                    # Data augmentation if enabled
                    if augment and np.random.rand() > 0.5:
                        if np.random.rand() > 0.5:
                            # Horizontal flip
                            low_patch = np.fliplr(low_patch)
                            high_patch = np.fliplr(high_patch)
                        else:
                            # Vertical flip
                            low_patch = np.flipud(low_patch)
                            high_patch = np.flipud(high_patch)
                    
                    x_data.append(low_patch / 255.0)
                    y_data.append(high_patch / 255.0)
        
        return np.array(x_data), np.array(y_data)
    
    def train(self, high_quality_dir, low_quality_dir, epochs=100, batch_size=16, validation_split=0.2, patch_size=(512, 512)):
        """Train the model using the provided image directories with larger patches"""
        x_data, y_data = self.prepare_data(high_quality_dir, low_quality_dir, patch_size)
        
        # Model checkpoint callback
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            'best_model.h5', save_best_only=True, monitor='val_loss'
        )
        
        # Early stopping callback
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            x_data, y_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        
        return history
    
    def enhance_image(self, input_image_path, output_image_path=None, target_resolution=None):
        """Enhance a single image with optional upscaling to target resolution"""
        if target_resolution is None:
            target_resolution = self.target_resolution
            
        # Load image
        img = Image.open(input_image_path)
        original_width, original_height = img.size
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Store original image for object preservation
        original_img_array = np.array(img)
        
        # Detect objects in original image if object preservation is enabled
        detected_objects = None
        if self.preserve_objects and self.object_detector is not None:
            detected_objects = self._detect_objects(original_img_array)
            if detected_objects:
                print(f"Detected {len(detected_objects)} objects for preservation")
            
        img_array = np.array(img) / 255.0
        
        # Add batch dimension if not present
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Process image with model for enhancement
        enhanced_array = self.model.predict(img_array)
        enhanced_array = np.clip(enhanced_array[0] * 255, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        enhanced_img = Image.fromarray(enhanced_array)
        
        # Apply object preservation if enabled and objects were detected
        if self.preserve_objects and detected_objects:
            enhanced_img = self._apply_object_preservation(original_img_array, enhanced_img, detected_objects)
        
        # Upscale to target resolution if needed
        current_width, current_height = enhanced_img.size
        if (current_width < target_resolution[0] or current_height < target_resolution[1]):
            # Calculate aspect ratio
            aspect_ratio = current_width / current_height
            
            # Determine new dimensions maintaining aspect ratio
            if aspect_ratio > target_resolution[0] / target_resolution[1]:
                # Width constrained by target width
                new_width = target_resolution[0]
                new_height = int(new_width / aspect_ratio)
            else:
                # Height constrained by target height
                new_height = target_resolution[1]
                new_width = int(new_height * aspect_ratio)
            
            # Store upscaled image for object preservation
            enhanced_img_array = np.array(enhanced_img)
            
            # Use high-quality upscaling
            enhanced_img_cv = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
            enhanced_img_cv = cv2.resize(enhanced_img_cv, (new_width, new_height), 
                                         interpolation=cv2.INTER_LANCZOS4)
            enhanced_img = Image.fromarray(cv2.cvtColor(enhanced_img_cv, cv2.COLOR_BGR2RGB))
            
            # Apply object preservation on upscaled image if objects were detected
            if self.preserve_objects and detected_objects:
                # Scale the object boxes according to the resize ratio
                width_ratio = new_width / current_width
                height_ratio = new_height / current_height
                
                scaled_objects = []
                for obj in detected_objects:
                    box = obj['box']
                    x, y, w, h = box
                    scaled_box = [
                        int(x * width_ratio),
                        int(y * height_ratio),
                        int(w * width_ratio),
                        int(h * height_ratio)
                    ]
                    scaled_obj = obj.copy()
                    scaled_obj['box'] = scaled_box
                    scaled_objects.append(scaled_obj)
                
                # Apply object preservation on the upscaled image
                upscaled_original = cv2.resize(original_img_array, (new_width, new_height), 
                                              interpolation=cv2.INTER_LANCZOS4)
                enhanced_img = self._apply_object_preservation(upscaled_original, enhanced_img, scaled_objects)
            
            print(f"Upscaled from {current_width}x{current_height} to {new_width}x{new_height}")
        
        # Save if output path is provided
        if output_image_path:
            enhanced_img.save(output_image_path, quality=95)  # Higher JPEG quality
            print(f"Enhanced image saved to {output_image_path}")
        
        return enhanced_img
    
    def enhance_directory(self, input_dir, output_dir, target_resolution=None):
        """Enhance all images in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))
        
        for i, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing image {i+1}/{len(image_files)}: {filename}")
            self.enhance_image(img_path, output_path, target_resolution)
    
    def save_model(self, model_path):
        """Save the model"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")


def download_yolo_files():
    """Download YOLOv4 model files if not present"""
    # Download config file
    config_path = "yolov4.cfg"
    if not os.path.exists(config_path):
        print("Downloading YOLOv4 config file...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            config_path
        )
    
    # Download weights file
    weights_path = "yolov4.weights"
    if not os.path.exists(weights_path):
        print("Downloading YOLOv4 weights file (this may take a while)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
            weights_path
        )
    
    # Download class names
    coco_names_path = "coco.names"
    if not os.path.exists(coco_names_path):
        print("Downloading COCO class names...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
            coco_names_path
        )
    
    print("YOLOv4 files downloaded successfully")


def main():
    parser = argparse.ArgumentParser(description='Image Enhancement with AI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--high-quality', required=True, help='Directory containing high quality images')
    train_parser.add_argument('--low-quality', required=True, help='Directory containing low quality images')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    train_parser.add_argument('--patch-size', type=int, default=512, help='Size of image patches for training')
    train_parser.add_argument('--model-output', required=True, help='Path to save the trained model')
    
    # Enhance parser
    enhance_parser = subparsers.add_parser('enhance', help='Enhance images')
    enhance_parser.add_argument('--model', required=True, help='Path to the trained model')
    enhance_parser.add_argument('--input', required=True, help='Input image or directory to enhance')
    enhance_parser.add_argument('--output', required=True, help='Output image or directory')
    enhance_parser.add_argument('--resolution', default='4k', 
                               help='Target resolution: 4k (3840x2160), 2k (2560x1440), or WxH format')
    enhance_parser.add_argument('--preserve-objects', action='store_true', 
                               help='Enable object preservation to prevent distortion of objects')
    enhance_parser.add_argument('--download-models', action='store_true',
                               help='Download required object detection models before enhancement')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        enhancer = ImageEnhancer()
        patch_size = (args.patch_size, args.patch_size)
        enhancer.train(args.high_quality, args.low_quality, args.epochs, args.batch_size, 
                       patch_size=patch_size)
        enhancer.save_model(args.model_output)
    
    elif args.command == 'enhance':
        # Download models if requested
        if getattr(args, 'download_models', False):
            download_yolo_files()
        
        # Parse resolution
        if args.resolution.lower() == '4k':
            target_resolution = (3840, 2160)
        elif args.resolution.lower() == '2k':
            target_resolution = (2560, 1440)
        elif 'x' in args.resolution.lower():
            parts = args.resolution.lower().split('x')
            target_resolution = (int(parts[0]), int(parts[1]))
        else:
            target_resolution = (3840, 2160)  # Default to 4K
            
        enhancer = ImageEnhancer(args.model, target_resolution, 
                                preserve_objects=getattr(args, 'preserve_objects', False))
        
        if os.path.isdir(args.input):
            enhancer.enhance_directory(args.input, args.output, target_resolution)
        else:
            enhancer.enhance_image(args.input, args.output, target_resolution)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
