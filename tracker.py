from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
  def __init__(self):
    self.object_tracker = DeepSort(
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
        today=None
    )
