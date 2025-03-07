import cv2
import os
import uuid

os.makedirs('frames', exist_ok=True)

for video_file in os.listdir('videos'):
    video_path = os.path.join('videos', video_file)
    
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        continue
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps//2
    
    frame_count = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join('frames', f'{os.path.splitext(video_file)[0]}--{uuid.uuid4()}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        frame_count += frame_interval
    
    cap.release()

print("Extração de frames concluída!")