import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ====================================================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ YOLO И УСТРОЙСТВА
# ====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство для YOLO: {device}")
yolo_model = YOLO('yolov8n-pose.pt')
yolo_model.to(device)

# ====================================================
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ КАДРОВ
# ====================================================
def resize_with_aspect_ratio(image, target_height=480):
    """Безопасное изменение размера кадра с сохранением пропорций"""
    h, w = image.shape[:2]
    if h == target_height: return image
    scale = target_height / h
    return cv2.resize(image, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)

def extract_single_person(video_path, person_idx):
    """Извлекает кадры и координаты (позы) для одного человека из видео"""
    cap = cv2.VideoCapture(video_path)
    poses, frames = [], []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
        
        frame = resize_with_aspect_ratio(frame, target_height=480) 
        results = yolo_model(frame, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) >= person_idx:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            kps = results[0].keypoints.xy.cpu().numpy()
            # Сортируем людей слева направо
            sorted_idx = np.argsort(boxes[:, 0]) 
            poses.append(kps[sorted_idx[person_idx - 1]].flatten())
            # Сохраняем кадр в памяти
            _, enc_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frames.append(enc_frame)
    cap.release()
    return np.array(poses), frames

def extract_two_people(video_path, c_idx, u_idx):
    """Извлекает кадры и координаты сразу двух людей (коуча и ученика) из одного видео"""
    cap = cv2.VideoCapture(video_path)
    c_poses, u_poses, frames = [], [], []
    max_idx = max(c_idx, u_idx)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
        
        frame = resize_with_aspect_ratio(frame, target_height=480)
        results = yolo_model(frame, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) >= max_idx:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            kps = results[0].keypoints.xy.cpu().numpy()
            # Сортируем людей слева направо
            sorted_idx = np.argsort(boxes[:, 0])
            c_poses.append(kps[sorted_idx[c_idx - 1]].flatten())
            u_poses.append(kps[sorted_idx[u_idx - 1]].flatten())
            # Сохраняем кадр в памяти
            _, enc_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frames.append(enc_frame)
    cap.release()
    return np.array(c_poses), np.array(u_poses), frames