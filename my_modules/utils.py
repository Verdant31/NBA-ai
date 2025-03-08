from typing import Literal
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

logging.basicConfig(format='%(levelname)s - %(asctime)s.%(msecs)03d: %(message)s',
                    datefmt='%H:%M:%S', level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("ssl").setLevel(logging.WARNING)
logging.getLogger("deep_sort_realtime").setLevel(logging.WARNING)


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


def get_root_path():
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_script_path, '..'))

    return root_path


def get_last_model(
    model_name: Literal["player_recognition", "jersey_number_recognition"],
    verbose=False
):
    model = f'{get_root_path()}/models/{model_name}/runs'
    files = os.listdir(model)

    numbers = [int(re.findall(r'\d+', item)[0])
               if re.findall(r'\d+', item) else None for item in files]
    numbers = [num for num in numbers if num is not None]
    run_number = '' if len(numbers) == 0 else max(numbers)

    # log(f'Getting model from: {f'models/{model_name}/runs/train{run_number}/weights/best.pt'}')

    last_model = f'{model}/train{run_number}/weights/best.pt'
    return YOLO(last_model, verbose=verbose)
