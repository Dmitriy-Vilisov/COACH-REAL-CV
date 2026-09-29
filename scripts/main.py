import os
import sys
import cv2
import numpy as np
from fastdtw import fastdtw

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.MyProject.math_and_vis import (
    custom_pose_distance, mpjpe_distances, draw_skeleton_static,
    draw_skeleton_with_gradient, get_smooth_color, format_lag_info, draw_text_with_shadow
)
from src.myproject_models.pose_estimation import extract_single_person, extract_two_people
from src.myproject_data.video_utils import apply_audio_and_save

# ====================================================
# НАСТРОЙКА ПУТЕЙ (относительно папки scripts)
# ====================================================
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ====================================================
# ОСНОВНАЯ ФУНКЦИЯ РЕНДЕРА И СИНХРОНИЗАЦИИ
# ====================================================
def main_func2(vidio_coach, vidio_user, COACH_PERSON_IDX, USER_PERSON_IDX, THRESHOLD_SCORE, save_slot:str):
    TEMP_VIDEO_PATH = 'temp_output.mp4'
    COACH_VIDEO = os.path.join(DATA_DIR, 'coach', vidio_coach)
    USER_VIDEO  = os.path.join(DATA_DIR, 'user', vidio_user)
    
    is_single_video = (vidio_coach == vidio_user)
    FINAL_VIDEO_PATH = os.path.join(OUTPUT_DIR, f'{save_slot}_yolo8_hybrid_sync_mpjpe_one_v.mp4') if is_single_video else os.path.join(OUTPUT_DIR, f'{save_slot}_yolo8_hybrid_sync_mpjpe_dual_v.mp4')
    
    if not is_single_video:
        print(f"Режим: РАЗНЫЕ ВИДЕО ({vidio_coach}, {vidio_user}). Честный рендер + Расчет отставания")
        coach_poses, coach_frames = extract_single_person(COACH_VIDEO, COACH_PERSON_IDX)
        user_poses, user_frames = extract_single_person(USER_VIDEO, USER_PERSON_IDX)
        
        if len(coach_poses) > 0 and len(user_poses) > 0:
            cap_c = cv2.VideoCapture(COACH_VIDEO)
            coach_fps = cap_c.get(cv2.CAP_PROP_FPS) or 30.0
            cap_c.release()
            
            cap_u = cv2.VideoCapture(USER_VIDEO)
            user_fps = cap_u.get(cv2.CAP_PROP_FPS) or 30.0
            cap_u.release()

            print("Анализ стартового сдвига (Фоновый DTW)...")
            _, path = fastdtw(coach_poses, user_poses, dist=custom_pose_distance)
            sync_map_dtw = {c: u for c, u in path}
            
            sync_duration = min(int(coach_fps * 10), len(coach_poses))
            initial_offsets = [sync_map_dtw[c] - int(c * (user_fps / coach_fps)) for c in range(sync_duration) if c in sync_map_dtw]
            start_offset_u = int(np.median(initial_offsets)) if initial_offsets else 0

            print("Сборка Side-by-Side...")
            first_c = cv2.imdecode(coach_frames[0], cv2.IMREAD_COLOR)
            h_c, w_c = first_c.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = None 
            
            font_scale = max(0.5, h_c / 800.0)
            thick = max(2, int(font_scale * 2.5))
            shadow = thick + 2
            
            y_pos_1 = int(h_c * 0.1)
            y_pos_2 = y_pos_1 + int(40 * font_scale) + 10
            y_pos_3 = y_pos_2 + int(40 * font_scale) + 10
            
            accumulated_scores = []
            
            for coach_idx in range(len(coach_poses)):
                linear_u_idx = int(coach_idx * (user_fps / coach_fps)) + start_offset_u
                user_idx = max(0, min(linear_u_idx, len(user_poses) - 1)) 
                
                ideal_u_idx = sync_map_dtw.get(coach_idx, user_idx)
                lag_frames = ideal_u_idx - user_idx
                lag_sec = lag_frames / user_fps
                
                _, match_error = mpjpe_distances(user_poses[ideal_u_idx], coach_poses[coach_idx])
                match_score = max(0, min(100, int((1.0 - (match_error / 0.4)) * 100)))
                
                frame_c = cv2.imdecode(coach_frames[coach_idx], cv2.IMREAD_COLOR)
                frame_u = cv2.imdecode(user_frames[user_idx], cv2.IMREAD_COLOR)
                
                h_u, w_u = frame_u.shape[:2]
                if h_c != h_u:
                    scale = h_c / h_u
                    frame_u = cv2.resize(frame_u, (int(w_u * scale), h_c))

                if out_video is None:
                    out_video = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, coach_fps, (w_c + frame_u.shape[1], h_c))

                _, mean_error = mpjpe_distances(user_poses[user_idx], coach_poses[coach_idx])
                score = max(0, min(100, int((1.0 - (mean_error / 0.4)) * 100)))
                
                accumulated_scores.append(score)
                running_avg = int(np.mean(accumulated_scores))
                
                draw_skeleton_static(frame_c, coach_poses[coach_idx], (255, 255, 255))
                draw_skeleton_with_gradient(frame_u, user_poses[user_idx], coach_poses[coach_idx], THRESHOLD_SCORE)
                
                color_score = get_smooth_color(score, THRESHOLD_SCORE)
                color_avg = get_smooth_color(running_avg, THRESHOLD_SCORE)
                lag_text, color_lag = format_lag_info(lag_sec, lag_frames, match_score)
                
                # Текст Коуча
                draw_text_with_shadow(frame_c, "COACH", (int(w_c*0.05), y_pos_1), font_scale, (255, 255, 255), thick, shadow)
                
                # Оценки Ученика
                draw_text_with_shadow(frame_u, f"Score: {score}%", (int(w_u*0.05), y_pos_1), font_scale, color_score, thick, shadow)
                draw_text_with_shadow(frame_u, lag_text, (int(w_u*0.05), y_pos_2), font_scale, color_lag, thick, shadow)
                draw_text_with_shadow(frame_u, f"Avg Score: {running_avg}%", (int(w_u*0.05), y_pos_3), font_scale, color_avg, thick, shadow)
                
                out_video.write(cv2.hconcat([frame_c, frame_u]))
                
            out_video.release()
            apply_audio_and_save(TEMP_VIDEO_PATH, FINAL_VIDEO_PATH, COACH_VIDEO)
        else: print("Ошибка: Люди не найдены.")

    else:
        print(f"Режим: ОДНО ВИДЕО ({vidio_coach})")
        coach_poses, user_poses, frames = extract_two_people(COACH_VIDEO, COACH_PERSON_IDX, USER_PERSON_IDX)
        if len(coach_poses) > 0:
            
            cap_orig = cv2.VideoCapture(COACH_VIDEO)
            orig_fps = cap_orig.get(cv2.CAP_PROP_FPS) or 30.0
            cap_orig.release()
            
            print("Анализ отставания (Фоновый DTW)...")
            _, path = fastdtw(coach_poses, user_poses, dist=custom_pose_distance)
            sync_map_dtw = {c: u for c, u in path}

            print("Сборка видео и расчет ошибки (MPJPE)...")
            first_frame = cv2.imdecode(frames[0], cv2.IMREAD_COLOR)
            h, w = first_frame.shape[:2]
            
            font_scale = max(0.5, h / 800.0)
            thick = max(2, int(font_scale * 2.5))
            shadow = thick + 2
            
            y_pos_1 = int(h * 0.1)
            y_pos_2 = y_pos_1 + int(40 * font_scale) + 10
            y_pos_3 = y_pos_2 + int(40 * font_scale) + 10
            
            accumulated_scores = []
            
            out_dtw = cv2.VideoWriter(TEMP_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (w, h))
            for idx in range(len(frames)):
                frame = cv2.imdecode(frames[idx], cv2.IMREAD_COLOR)
                
                _, mean_error = mpjpe_distances(user_poses[idx], coach_poses[idx])
                score = max(0, min(100, int((1.0 - (mean_error / 0.4)) * 100)))
                
                accumulated_scores.append(score)
                running_avg = int(np.mean(accumulated_scores))
                
                ideal_u_idx = sync_map_dtw.get(idx, idx)
                lag_frames = ideal_u_idx - idx
                lag_sec = lag_frames / orig_fps
                
                _, match_error = mpjpe_distances(user_poses[ideal_u_idx], coach_poses[idx])
                match_score = max(0, min(100, int((1.0 - (match_error / 0.4)) * 100)))
                
                draw_skeleton_static(frame, coach_poses[idx], (255, 255, 255))
                draw_skeleton_with_gradient(frame, user_poses[idx], coach_poses[idx], THRESHOLD_SCORE)
                
                color_score = get_smooth_color(score, THRESHOLD_SCORE)
                color_avg = get_smooth_color(running_avg, THRESHOLD_SCORE)
                lag_text, color_lag = format_lag_info(lag_sec, lag_frames, match_score)
                
                draw_text_with_shadow(frame, f"Score: {score}%", (int(w*0.05), y_pos_1), font_scale, color_score, thick, shadow)
                draw_text_with_shadow(frame, lag_text, (int(w*0.05), y_pos_2), font_scale, color_lag, thick, shadow)
                draw_text_with_shadow(frame, f"Avg Score: {running_avg}%", (int(w*0.05), y_pos_3), font_scale, color_avg, thick, shadow)
                
                out_dtw.write(frame)
            out_dtw.release()
            apply_audio_and_save(TEMP_VIDEO_PATH, FINAL_VIDEO_PATH, COACH_VIDEO)
        else: print("Ошибка: Люди не найдены.")


if __name__ == "__main__":
    print("=== ИНИЦИАЛИЗАЦИЯ ===")
    
    # Настройки для запуска
    COACH_VIDEO = 'GGS_30fps 1918-1078.mp4'  
    USER_VIDEO =  'GGS_30fps 1918-1078.mp4'    
    COACH_PERSON_IDX = 2             # Порядковый номер тренера слева направо 
    USER_PERSON_IDX = 1              # Порядковый номер ученика
    THRESHOLD_SCORE = 85             # Порог строгой оценки
    SAVE_SLOT = 'run01'              # Префикс для сохранения
    
    # Запуск основного пайплайна
    #main_func2(COACH_VIDEO, USER_VIDEO, COACH_PERSON_IDX, USER_PERSON_IDX, THRESHOLD_SCORE, SAVE_SLOT)
    print()
    
    # Настройки для второго запуска (разные видео)
    COACH_VIDEO = 'coach_2.mp4'  
    USER_VIDEO =  'student_2.mp4'    
    COACH_PERSON_IDX = 2             # Порядковый номер тренера слева направо 
    USER_PERSON_IDX = 3              # Порядковый номер ученика
    THRESHOLD_SCORE = 85             # Порог строгой оценки
    SAVE_SLOT = 'run02'              # Префикс для сохранения
    main_func2(COACH_VIDEO, USER_VIDEO, COACH_PERSON_IDX, USER_PERSON_IDX, THRESHOLD_SCORE, SAVE_SLOT)