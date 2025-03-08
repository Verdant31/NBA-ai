from uuid import uuid4
import time
import torch
import cv2
import os
import signal
import threading
import queue
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{str(model.names[cls])}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    jersey_number = get_jersey_number_from_frame(cropped)
                    if jersey_number:
                        jersey_numbers_frame[jersey_number] = jersey_numbers_frame.get(
                            jersey_number, 0) + 1

        log("Frame processed in {:.2f} ms".format(inference_time))
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


def frame_generator(cap):
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            log("No more frames to read from video.")
            break
        frame_count += 1
        log("Frame {} read from video.".format(frame_count))
        yield frame
    cap.release()
    log("Video capture released in frame_generator.")

# Producer thread that fills a queue with processed frames


def frame_producer(frame_iter, frame_queue):
    log("Frame producer thread started.")
    try:
        for processed in frame_iter:
            frame_queue.put(processed)
            log("Processed frame enqueued.")
    except Exception as e:
        log(f"Exception in frame producer: {e}")
    finally:
        # Signal end of stream
        frame_queue.put(None)
        log("Frame producer thread ended; sentinel added to queue.")


if __name__ == '__main__':
    freeze_support()

    input_video_path = './test/small_input.mp4'
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        log("Error opening video file: {}".format(input_video_path))
        sys.exit(1)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log("Video opened. Dimensions: {}x{}, FPS: {}, Total Frames: {}".format(
        frame_width, frame_height, fps, total_frames))
    PREBUFFER_COUNT = fps

    num_processes = 4 if device == "cuda" else os.cpu_count()
    signal.signal(signal.SIGINT, signal_handler)

    inference_times = []
    identified_jersey_numbers = {}

    # Create a queue to buffer processed frames
    frame_queue = queue.Queue(maxsize=PREBUFFER_COUNT * 2)
    log("Frame queue created with maxsize: {}".format(PREBUFFER_COUNT * 2))

    try:
        pool = Pool(processes=num_processes, initializer=init_worker)
        frame_iter = pool.imap(
            process_frame, frame_generator(cap), chunksize=2)

        # Start a producer thread to continuously fill the queue with processed frames
        producer_thread = threading.Thread(
            target=frame_producer, args=(frame_iter, frame_queue))
        producer_thread.daemon = True
        producer_thread.start()
        log("Producer thread started.")

        # Prebuffer frames before starting playback
        prebuffer = []
        log("Starting prebuffering of {} frames.".format(PREBUFFER_COUNT))
        prebuffer_start_time = time.time()

        while len(prebuffer) < PREBUFFER_COUNT:
            item = frame_queue.get()
            if item is None:
                log("End of stream reached during prebuffering.")
                break
            prebuffer.append(item)
            log("Prebuffered frame {}.".format(len(prebuffer)))
        log("Prebuffering complete with {} frames.".format(len(prebuffer)))
        prebuffer_end_time = time.time()
        prebuffer_duration = prebuffer_end_time - prebuffer_start_time
        log("Prebuffering complete with {} frames in {:.2f} seconds.".format(
            len(prebuffer), prebuffer_duration))

        log("Starting video playback with prebuffered frames.")
        # Display the prebuffered frames first
        frame_display_count = 0
        for processed_frame, inference_time, jersey_numbers_frame in prebuffer:
            if processed_frame is not None:
                cv2.imshow("Predictions", processed_frame)
                log("Displaying prebuffered frame {}.".format(
                    frame_display_count + 1))
            else:
                log("Skipping a prebuffered frame (None).")
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                log("User pressed 'q' during prebuffer display; exiting.")
                raise KeyboardInterrupt
            inference_times.append(inference_time)
            for jersey, count in jersey_numbers_frame.items():
                identified_jersey_numbers[jersey] = identified_jersey_numbers.get(
                    jersey, 0) + count
            frame_display_count += 1

        log("Prebuffered frames displayed. Continuing with live playback.")

        # Now continuously display frames from the queue at a fixed rate
        while True:
            item = frame_queue.get()
            if item is None:  # End of stream signaled
                log("End of stream signal received from frame queue.")
                break
            processed_frame, inference_time, jersey_numbers_frame = item
            if processed_frame is not None:
                cv2.imshow("Predictions", processed_frame)
                log("Displaying live frame.")
            else:
                log("Received a None frame in live playback; skipping display.")
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                log("User pressed 'q' during live playback; exiting.")
                break
            inference_times.append(inference_time)
            for jersey, count in jersey_numbers_frame.items():
                identified_jersey_numbers[jersey] = identified_jersey_numbers.get(
                    jersey, 0) + count

    except KeyboardInterrupt:
        log("KeyboardInterrupt detected. Cleaning up...")
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log(f"Exception: {e} File: {fname}, Line: {exc_tb.tb_lineno}")
    finally:
        if inference_times:
            mean_inference_time = sum(inference_times) / len(inference_times)
            log("Mean Inference time: {:.2f} ms per frame".format(
                mean_inference_time))
            log("Elapsed time to predict {} frames: {:.2f} ms".format(
                len(inference_times), sum(inference_times)))
        log("Identified jersey numbers:")
        table_data = sorted(
            [[key, value] for key, value in identified_jersey_numbers.items()],
            key=lambda x: x[1],
            reverse=True
        )
        print(tabulate(table_data, headers=[
              'Jersey Number', 'Count'], tablefmt='pretty'))
        cleanup(cap)
