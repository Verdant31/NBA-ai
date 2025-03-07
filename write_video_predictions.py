from uuid import uuid4
import time
import torch
import cv2
import os
import signal
from multiprocessing import Pool
from my_modules import get_last_model, log
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
from sys import exit
import sys
from tabulate import tabulate
import signal
from multiprocessing import Pool, freeze_support

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pool = None
tracker = DeepSort(
    max_age=20,
    n_init=2,
    nms_max_overlap=0.3,
    max_cosine_distance=0.8,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None,
)


def init_worker():
    global model
    try:
        model = get_last_model(model_name="player_recognition")
    except:
        log("Error loading model.")


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
                jersey_number = f'{jersey_number}{number}'

        if len(jersey_number) > 0:
            return jersey_number
        else:
            return None
    except Exception:
        print("Erro na extração do número da camisa")
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

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                detections.append(([x1, y1, w, h], conf, cls))

                if model.names[cls] == "player" and conf > 0.6:
                    cropped = frame[y1:y2, x1:x2]
                    jersey_number = get_jersey_number_from_frame(cropped)
                    if jersey_number:
                        if jersey_number not in jersey_numbers_frame:
                            jersey_numbers_frame[jersey_number] = 1
                        else:
                            jersey_numbers_frame[jersey_number] += 1

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(
                ltrb[2]), int(ltrb[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{str(track.track_id)}", (int(ltrb[0]), int(ltrb[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, inference_time, jersey_numbers_frame
    except Exception:
        print('Erro no processamento do frame')
        return frame, 0, jersey_numbers_frame


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
    freeze_support()

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
    identified_jersey_numbers = {}

    num_processes = 4 if device == "cuda" else os.cpu_count()

    signal.signal(signal.SIGINT, signal_handler)

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

        cleanup(cap, out)
