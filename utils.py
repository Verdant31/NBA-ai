import os
import re
import logging
from ultralytics import YOLO
import base64
from io import BytesIO
from IPython.display import HTML, display
from PIL import Image
from matplotlib import cm
import numpy as np

logging.basicConfig(format='%(levelname)s - %(asctime)s.%(msecs)03d: %(message)s',datefmt='%H:%M:%S', level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("ssl").setLevel(logging.WARNING)

def convert_to_base64(cv2Image):
    pil_image = Image.fromarray(cv2Image.astype('uint8'), 'RGB')
    
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def log(msg):
    if "error" in msg.lower():
        logging.error(msg)
        return
    logging.info(msg)

def get_last_model(model_name = "player_recognition", verbose = False):
    print(os.listdir())
    print(os.listdir(os.path.join(os.getcwd(), f'./models/{model_name}/runs/train/weights')))
    files = os.listdir(os.path.join(os.getcwd(), f'./models/{model_name}/runs/train/weights'))

    numbers = [int(re.findall(r'\d+', item)[0]) if re.findall(r'\d+', item) else None for item in files]
    numbers = [num for num in numbers if num is not None]
    
    if(len(numbers) == 0):
        return YOLO(f'./models/{model_name}/runs/train/weights/best.pt', verbose=verbose)

    last_model = f'runs/detect/train{max(numbers)}/weights/best.pt'
    return YOLO(last_model, verbose=verbose)