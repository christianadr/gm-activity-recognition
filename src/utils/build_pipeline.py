import decord
import mmcv
import numpy as np
from mmaction.structures import ActionDataSample
from mmcv.transforms import TRANSFORMS, BaseTransform, to_tensor


@TRANSFORMS.register_module()
class VideoInit(BaseTransform):
    def transform(self, results):
        container = decord.VideoReader(results["filename"])
        results["total_frames"] = len(container)
        results["video_reader"] = container
        return results


@TRANSFORMS.register_module()
class VideoSample(BaseTransform):
    def __init__(self, clip_len, num_clips, test_mode=False):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def transform(self, results):
        total_frames = results["total_frames"]
        interval = total_frames // self.clip_len

        if self.test_mode:
            np.random.seed(42)  # Deterministic sampling during testing

        inds_of_all_clips = list()

        for _ in range(self.num_clips):
            bids = np.arange(self.clip_len) * interval
            offset = np.random.randint(interval, size=bids.shape)
            inds = bids + offset
            inds_of_all_clips.append(inds)

        results["frame_inds"] = np.concatenate(inds_of_all_clips)
        results["clip_len"] = self.clip_len
        results["num_clips"] = self.num_clips
        return results
