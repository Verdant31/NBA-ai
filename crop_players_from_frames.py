import cv2
import os
import uuid
from ultralytics import YOLO
import torch
import base64
from openai import OpenAI
import json
import dotenv
import numpy as np

dotenv.load_dotenv()
client = OpenAI()

def encode_image(image_array: np.ndarray) -> str:
    """
    Encode a NumPy image array into a base64 string.
    
    :param image_array: NumPy array representing the image.
    :return: Base64-encoded string of the image.
    """
    _, buffer = cv2.imencode(".png", image_array)  
    return base64.b64encode(buffer).decode("utf-8")

model_path = os.path.join('runs/detect/train16/weights', 'best.pt')
model = YOLO(model_path, verbose=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

for video_file in os.listdir('videos'):
    video_path = os.path.join('videos', video_file)
    
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        continue
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_interval = int(fps * 0.1)
    frame_count = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            original_frame = frame.copy()
            results = model.predict(frame, device=device, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    label = f"{model.names[cls]} {conf:.2f}"
                    
                    croped = original_frame[y1:y2, x1:x2]
                    base64_image = encode_image(croped)

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_format={"type": "json_object"},  
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Analyze the basketball player's jersey in the image. Extract ONLY the numerical jersey number. Return JSON: { 'jerseyNumber': number | 'undefined' }. IT IS CRUCIAL TO RETURN 'undefined' IF NO NUMBER IS VISIBLE.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                    },
                                ],
                            }
                        ],
                    )
                    response_as_json = json.loads(response.choices[0].message.content)

                    if(model.names[cls] == "player" and conf > 0.6):
                        if(response_as_json['jerseyNumber'] != 'undefined'):
                            number = int(response_as_json['jerseyNumber'])
                            cv2.imwrite(f'./croppedWithAi/good/{video_file}--{number}--{uuid.uuid4()}.jpg', croped)

        frame_count += frame_interval
    
    cap.release()

print("Extração de frames concluída!")