#import gc
#import os
#import shutil
#import time
import cv2
import numpy as np



# ====================================================
# МАТЕМАТИЧЕСКИЙ АППАРАТ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ====================================================
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
# Индексы YOLOv8: 
# 0-4 (Лицо), 5-6 (Плечи), 7-8 (Локти), 9-10 (Запястья)
# 11-12 (Бедра), 13-14 (Колени), 15-16 (Лодыжки)
JOINT_WEIGHTS = np.array([
    0.1, 0.1, 0.1, 0.1, 0.1,  # Лицо почти не влияет на счет
    1.0, 1.0,                 # Корпус (плечи) — стандартный вес
    2.0, 2.0,                 # Локти — двойной штраф за ошибку
    3.0, 3.0,                 # Запястья — тройной штраф (самые подвижные части)
    1.0, 1.0,                 # Корпус (бедра)
    2.0, 2.0,                 # Колени
    3.0, 3.0                  # Лодыжки
])

def get_normalized_pose(pose_flat):
    pose = pose_flat.reshape(17, 2)
    valid_points = pose[pose[:, 0] > 0]
    if len(valid_points) == 0: return pose_flat
    
    center = np.mean(valid_points, axis=0)
    centered = pose - center
    centered[pose[:, 0] == 0] = 0 
    
    norm = np.linalg.norm(centered)
    if norm > 1e-6: return (centered / norm).flatten()
    return centered.flatten()

def get_smooth_color(score_percent, threshold=85):
    score = max(0, min(100, score_percent))
    if score >= threshold: return (0, 255, 0)
    elif score <= 40: return (0, 0, 255)
    else:
        normalized = (score - 40) / (threshold - 40.0)
        if normalized >= 0.5: return (0, 255, int((1.0 - normalized) * 2 * 255))
        else: return (0, int(normalized * 2 * 255), 255)

def get_lag_color(lag_sec):
    lag = abs(lag_sec)
    if lag <= 0.1: return (0, 255, 0)
    elif lag <= 0.3: return (0, 255, int((lag - 0.1) * (255 / 0.2)))
    elif lag <= 0.5: return (0, int(255 - (lag - 0.3) * (255 / 0.2)), 255)
    else: return (0, 0, 255)

def format_lag_info(lag_sec, lag_frames, match_score):
    """Определяет текст и цвет задержки в зависимости от качества выполнения"""
    if match_score < 40:
        return "Lag: N/A", (0, 0, 255)
    
    if abs(lag_sec) <= 0.2:
        return "Lag: Good", (0, 255, 0)
        
    color_lag = get_lag_color(lag_sec)
    if lag_frames > 0:
        return f"Lag: {lag_sec:.2f}s (S)", color_lag
    else:
        return f"Lag: {abs(lag_sec):.2f}s (F)", color_lag

def mpjpe_distances(p1_flat, p2_flat):
    p1 = get_normalized_pose(p1_flat).reshape(17, 2)
    p2 = get_normalized_pose(p2_flat).reshape(17, 2)
    mask = (p1[:, 0] != 0) & (p2[:, 0] != 0)
    
    distances = np.zeros(17)
    if not np.any(mask): return distances, 1.0
    
    # 1. Считаем чистые физические дистанции для отрисовки цвета
    distances[mask] = np.linalg.norm(p1[mask] - p2[mask], axis=1)
    
    # 2. Считаем глобальную оценку с учетом ВЕСОВ суставов
    weighted_distances = distances[mask] * JOINT_WEIGHTS[mask]
    
    # Взвешенное среднее: делим сумму ошибок на сумму активных весов
    weighted_mean_error = np.sum(weighted_distances) / np.sum(JOINT_WEIGHTS[mask])
    
    return distances, weighted_mean_error

def custom_pose_distance(p1_flat, p2_flat):
    _, mean_error = mpjpe_distances(p1_flat, p2_flat)
    return mean_error

# ====================================================
# ФУНКЦИИ ОТРИСОВКИ И ИНТЕРФЕЙСА
# ====================================================
def draw_text_with_shadow(frame, text, pos, font_scale, color, thick, shadow):
    """Отрисовка текста с черной обводкой/тенью"""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), shadow)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)

def draw_skeleton_with_gradient(frame, kp_user, kp_coach, threshold=85):
    kp_u = kp_user.reshape(17, 2)
    distances, _ = mpjpe_distances(kp_user, kp_coach)
    pt_scores = np.clip((1.0 - (distances / 0.4)) * 100, 0, 100)

    for p1, p2 in SKELETON_EDGES:
        x1, y1 = kp_u[p1]
        x2, y2 = kp_u[p2]
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            edge_score = (pt_scores[p1] + pt_scores[p2]) / 2.0
            color = get_smooth_color(edge_score, threshold)
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
    for i, (x, y) in enumerate(kp_u):
        if x > 0 and y > 0:
            color = get_smooth_color(pt_scores[i], threshold)
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

def draw_skeleton_static(frame, keypoints, color=(0, 255, 0)):
    keypoints = keypoints.reshape(17, 2)
    for p1, p2 in SKELETON_EDGES:
        x1, y1 = keypoints[p1]
        x2, y2 = keypoints[p2]
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
