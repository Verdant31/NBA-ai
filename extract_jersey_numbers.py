import torch
import cv2
import os
from my_modules import log
from tqdm import tqdm  
from ultralytics import YOLO
import shutil

def process_frame(frame, file_name):
    try:
        results = model.predict(frame, verbose=False)[0]
        if(len(results) == 0):
            return []
        
        # if(file_name == "009dfd56-4633-4907-ab2f-dbf5591a0cfa.png"):
        #     results.show()
        kp = results.keypoints.xy[0].cpu().numpy()
            
        left_shoulder = int(kp[5][0]), int(kp[5][1])
        right_shoulder = int(kp[6][0]), int(kp[6][1])
        
        left_elbow = int(kp[7][0]), int(kp[7][1])
        right_elbow = int(kp[8][0]), int(kp[8][1])
        
        right_hip = int(kp[11][0]), int(kp[11][1])
        
        # Add some additional width to make sure that contains the whole jersey number
        # and decrease the height to centralize more the number
        left_shoulder = (left_shoulder[0] - 5, left_shoulder[1] + 5)
        right_shoulder = (right_shoulder[0] + 5, right_shoulder[1] + 5)
        
        if(left_elbow[0] - right_elbow[0] < 10 and left_shoulder[0] - right_shoulder[0] < 10):
            x_min = 20
            x_max = frame.shape[0] - 20
            y_min = min(left_shoulder[1], right_shoulder[1])
        elif((left_elbow[0] - right_elbow[0] < left_shoulder[0] - right_shoulder[0]) or (left_elbow[0] == 0 or right_elbow[0] == 0)):
            x_min = min(left_shoulder[0], right_shoulder[0])
            x_max = max(left_shoulder[0], right_shoulder[0])
            y_min = min(left_shoulder[1], right_shoulder[1])
        else:
            x_min = min(left_elbow[0], right_elbow[0])
            x_max = max(left_elbow[0], right_elbow[0])
            y_min = min(left_shoulder[1], right_shoulder[1]) 
 
        y_max = right_hip[1]  

        cropped_frame = frame[y_min:y_max, x_min:x_max]
        
        return cropped_frame
    except:
        cv2.imwrite("./cropped_jersey_numbers/error.jpg", frame)
        result = model(frame)[0]  
        result.save("./cropped_jersey_numbers/error_with_keypoints.jpg")
        raise Exception(f"Error on file: {file_name}")    
        
        
if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    frames = []
    torso_connections = [(5, 6), (6, 12), (12, 11), (11, 5)]
    model = YOLO("yolo11n-pose.pt")  
    
    if os.path.exists("./cropped_jersey_numbers"):
        shutil.rmtree("./cropped_jersey_numbers")
    os.makedirs("./cropped_jersey_numbers")

    for img_file in os.listdir("./cropped"):
        frame = cv2.imread(os.path.join("./cropped", img_file))
        frames.append({
            "frame": frame,
            "file_name": img_file
        })

    try:
        with tqdm(total=len(frames), desc="Cropping frames", unit="frame") as pbar:
            for index, frame in enumerate(frames):
                frame_width = frame['frame'].shape[1]
                frame_height = frame['frame'].shape[0]
                if(frame_width < 75):
                    continue
                frame_cpy = frame['frame'].copy()
                cropped = process_frame(frame_cpy, frame['file_name'])

                if(len(cropped) == 0): 
                    continue
                cv2.imwrite(f"./cropped_jersey_numbers/{index}.jpg", cropped)
                # cv2.imwrite(f"./cropped_jersey_numbers/{index}_original.jpg", frame_cpy)
                pbar.update(1)  
        
    except Exception as e:
        log(f"Error occurred: {e}")
    finally:
        log("Finished processing.")