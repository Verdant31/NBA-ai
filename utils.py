import os
import re
import logging
from ultralytics import YOLO
logging.basicConfig(format='%(levelname)s - %(asctime)s.%(msecs)03d: %(message)s',datefmt='%H:%M:%S', level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def log(msg):
    if "error" in msg.lower():
        logging.error(msg)
        return
    logging.info(msg)

def get_last_model(verbose = False):
    files = os.listdir('runs/detect')

    numbers = [int(re.findall(r'\d+', item)[0]) if re.findall(r'\d+', item) else None for item in files]
    numbers = [num for num in numbers if num is not None]

    last_model = f'runs/detect/train{max(numbers)}/weights/best.pt'

    return YOLO(last_model, verbose=verbose)