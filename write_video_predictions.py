import time
import torch
import cv2
import os
import signal
from multiprocessing import Pool
from utils import get_last_model, log
from tqdm import tqdm  

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pool = None  

def init_worker():
    global model
    model = get_last_model()
    signal.signal(signal.SIGINT, signal.SIG_IGN) 

def process_frame(frame):
    if frame is None:
        return None, 0
    
    start_time = time.time()
    results = model.predict(frame, device=device, verbose=False)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, inference_time

def cleanup(cap, out):
    global pool
    if cap:
        cap.release()
    if out:
        out.release()

    if pool:
        pool.terminate()
        pool.join()
        log("Pool terminated.")

    log("Cleanup complete. Resources released.")

def signal_handler():
    log("Interrupt received, terminating...")
    cleanup(cap, out, inference_times)
    exit(0)

def frame_generator():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()  
        
if __name__ == '__main__':
    input_video_path = './test/small_input.mp4'
    output_video_path = './test/output.mp4'

    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps,
        (frame_width, frame_height)
    )
    inference_times = []

    num_processes = 4 if device == "cuda" else os.cpu_count()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        pool = Pool(processes=num_processes, initializer=init_worker)
        processed_frames = pool.imap(process_frame, frame_generator(), chunksize=2)

        with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
            for index, result in enumerate(processed_frames):
                processed_frame, inference_time = result

                if processed_frame is not None:
                    out.write(processed_frame)
                    inference_times.append(inference_time)

                pbar.update(1)  # Update progress bar
        
        model = get_last_model()
        model.val()
        
    except KeyboardInterrupt:
        log("KeyboardInterrupt detected. Cleaning up...")
    except Exception as e:
        log(f"Error occurred: {e}")
    finally:
        if inference_times:
            mean_inference_time = sum(inference_times) / len(inference_times)
            log(f"Mean Inference time: {mean_inference_time:.2f} ms per frame")
            log(f"Elapsed time to predict {len(inference_times)}: {sum(inference_times):.2f} ms")
