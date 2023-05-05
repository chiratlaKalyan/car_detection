from ultralytics import YOLO
import cv2
import numpy as np

custom_model="runs/detect/yolov8n_cars_custom/weights/best.pt"

model = YOLO(custom_model)

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0")




