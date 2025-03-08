import time
import torch
import cv2
import os
import signal
from multiprocessing import Pool, freeze_support
from my_modules import get_last_model, log
from tqdm import tqdm
import sys
from tabulate import tabulate

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pool = None


def init_worker():
    global model
    try:
        model = get_last_model(model_name="player_recognition")
    except Exception as e:
        log("Error loading player recognition model: {}".format(e))


jersey_number_model = get_last_model(
    model_name="jersey_number_recognition", verbose=False)
jersey_numbers = {}


def get_jersey_number_from_frame(frame):
    try:
        results = jersey_number_model.predict(frame, conf=0.5, verbose=False)
        jersey_number = ''
        for result in results:
            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            labels = result.boxes.cls
            for _, _, label in zip(boxes, scores, labels):
                number = jersey_number_model.names[int(label)]
                jersey_number += str(number)
        return jersey_number if jersey_number else None
    except Exception as e:
        log("Error extracting jersey number: {}".format(e))
        return None


def process_frame(frame):
    jersey_numbers_frame = {}
    try:
        if frame is None:
            return None, 0, jersey_numbers_frame
        start_time = time.time()
        results = model.predict(frame, device=device, verbose=False)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                if model.names[cls] == "player" and conf > 0.6:
                    cropped = frame[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{str(model.names[cls])}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    jersey_number = get_jersey_number_from_frame(cropped)
                    if jersey_number:
                        jersey_numbers_frame[jersey_number] = jersey_numbers_frame.get(
                            jersey_number, 0) + 1

        return frame, inference_time, jersey_numbers_frame
    except Exception as e:
        log("Error processing frame: {}".format(e))
        return frame, 0, jersey_numbers_frame


def cleanup(cap):
    global pool
    log("Starting cleanup...")
    if cap:
        cap.release()
        log("Video capture released.")
    if pool:
        pool.terminate()
        pool.join()
        log("Multiprocessing pool terminated and joined.")
    cv2.destroyAllWindows()
    log("Destroyed all OpenCV windows.")
    log("Cleanup complete. Resources released.")


def signal_handler():
    log("Interrupt received, terminating...")
    cleanup(cap)
    sys.exit(0)


def frame_generator():
    while True:
        ret, frame = cap.read()
        if not ret:
            log("No more frames to read from video.")
            break
        yield frame
    cap.release()


if __name__ == '__main__':
    freeze_support()

    input_video_path = './test/input.mp4'
    output_video_path = './test/output.mp4'

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        log("Error opening video file: {}".format(input_video_path))
        sys.exit(1)
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
    identified_jersey_numbers = {}
    log("Video opened. Dimensions: {}x{}, FPS: {}, Total Frames: {}".format(
        frame_width, frame_height, fps, total_frames))
    PREBUFFER_COUNT = fps

    num_processes = 4 if device == "cuda" else os.cpu_count()
    signal.signal(signal.SIGINT, signal_handler)

    inference_times = []
    identified_jersey_numbers = {}

    try:
        pool = Pool(processes=num_processes, initializer=init_worker)
        processed_frames = pool.imap(
            process_frame, frame_generator(), chunksize=2)

        with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
            for index, result in enumerate(processed_frames):
                processed_frame, inference_time, jersey_numbers_frame = result

                if processed_frame is not None:
                    out.write(processed_frame)
                    inference_times.append(inference_time)
                    for jersey, count in jersey_numbers_frame.items():
                        if jersey not in identified_jersey_numbers:
                            identified_jersey_numbers[jersey] = count
                        else:
                            identified_jersey_numbers[jersey] += count

                pbar.update(1)

    except KeyboardInterrupt:
        log("KeyboardInterrupt detected. Cleaning up...")
    except Exception as e:
        import sys
        import os
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log(f"{e} File: {fname, exc_tb.tb_lineno}")
    finally:
        if inference_times:
            mean_inference_time = sum(inference_times) / len(inference_times)
            log(f"Mean Inference time: {mean_inference_time:.2f} ms per frame")
            log(f"Elapsed time to predict {len(inference_times)} frames: {sum(inference_times):.2f} ms")
        log("Identified jersey numbers:")
        table_data = sorted(
            [[key, value]
             for key, value in identified_jersey_numbers.items()],
            key=lambda x: x[1],
            reverse=True
        )
        print(tabulate(table_data, headers=[
            'Jersey Number', 'Count'], tablefmt='pretty'))

        cleanup(cap)
