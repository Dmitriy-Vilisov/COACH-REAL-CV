import os
import shutil
import time
from moviepy.editor import VideoFileClip, AudioFileClip

# ====================================================
# ОБРАБОТКА МЕДИА И ПЕРЕНОС АУДИО
# ====================================================
def apply_audio_and_save(temp_video_path, final_video_path, coach_video_path):
    """
    Перенос оригинальной аудиодорожки из видео коуча на итоговое видео.
    Включает защиту от падения (Broken Pipe), если звук в исходнике отсутствует.
    """
    print("Перенос аудио...")
    try:
        video_clip = VideoFileClip(temp_video_path)
        audio_clip = AudioFileClip(coach_video_path)
        final_clip = video_clip.set_audio(audio_clip)
        
        # Сохраняем финальный файл со звуком
        final_clip.write_videofile(
            final_video_path, 
            codec="libx264", 
            audio_codec="aac", 
            threads=2, 
            preset="ultrafast", 
            logger=None
        )
        
        # Освобождаем ресурсы
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        time.sleep(1) # Даем ОС время на освобождение файла
        
        # Удаляем временный файл без звука
        if os.path.exists(temp_video_path): 
            os.remove(temp_video_path)
            
        print(f"Готово! Видео (С ЗВУКОМ) сохранено в: {final_video_path}")
        
    except Exception as e: 
        print(f"Аудио не перенесено ({e}). Сохраняем без звука...")
        
        # Пытаемся безопасно закрыть клипы при возникновении ошибки
        try: video_clip.close() 
        except: pass
        try: audio_clip.close() 
        except: pass
        time.sleep(1)
        
        # Если звук наложить не удалось, просто переименовываем временный файл в финальный
        if os.path.exists(temp_video_path): 
            shutil.move(temp_video_path, final_video_path)
            
        print(f"Готово! Видео (БЕЗ ЗВУКА) сохранено в: {final_video_path}")