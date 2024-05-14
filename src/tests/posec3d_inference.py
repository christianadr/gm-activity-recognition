import os
import tempfile

import cv2
import mmcv
import mmengine
import torch
from mmaction.apis import (
    detection_inference,
    inference_skeleton,
    init_recognizer,
    pose_inference,
)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract
from mmengine.utils import track_iter_progress

try:
    import moviepy.video.io.ImageSequenceClip as mpy
except ImportError:
    raise ImportError("Please install moviepy to enable output file")


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)
THICKNESS = 1
LINETYPE = 1
BACKGROUND_COLOR = (0, 0, 0)
BACKGROUND_ALPHA = 0.5


class RunInference:
    def __init__(
        self,
        video_path,
        out_filename,
        config: str = "configs/pretrained_slowonly_resnet50_gym-keypoint_v2.py",
        checkpoint: str = "models/best_acc_top1_epoch_17.pth",
        det_config: str = "mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py",
        det_checkpoint: str = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth",
        pose_config: str = "mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py",
        pose_checkpoint: str = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
        det_score_thr: float = 0.9,
        label_map: str = "src/tests/label_map_grossmotor.txt",
        short_side: int = 480,
        cfg_options: dict = {},
        device: str = None,
    ):
        self.video_path = video_path
        self.out_filename = out_filename
        self.config = config
        self.checkpoint = checkpoint
        self.det_config = det_config
        self.det_checkpoint = det_checkpoint
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.det_score_thr = det_score_thr
        self.label_map = label_map
        self.short_side = short_side
        self.cfg_options = cfg_options

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"

            else:
                self.device = "cpu"
                print("CUDA not supported, using CPU instead.")

    def draw_percentage_bar(self, frame, scores, label_map):
        h, w, _ = frame.shape
        bar_height = 15
        bar_length = 100
        padding = 5
        x_offset = w - bar_length - padding
        y_offset = h - (len(scores) * (bar_height + padding)) - padding
        max_score = max(scores)

        overlay = frame.copy()
        for i, score in enumerate(scores):
            label = label_map[i]
            percentage = "{:.4f}".format(score * 100)
            bar_width = int(score * bar_length / max_score)
            bar_y1 = y_offset + i * (bar_height + padding)
            bar_y2 = bar_y1 + bar_height

            text = f"{label}: {percentage}%"
            text_size = cv2.getTextSize(text, FONTFACE, FONTSCALE, THICKNESS)[0]
            text_x = x_offset - text_size[0] - padding * 2
            text_y = bar_y1 + bar_height - 3

            text_x = max(text_x, 0)
            text_y = min(text_y, h - 3)

            cv2.rectangle(
                overlay,
                (text_x - padding, bar_y1),
                (text_x + text_size[0] + padding, bar_y2),
                BACKGROUND_COLOR,
                cv2.FILLED,
            )

            cv2.rectangle(
                overlay,
                (x_offset, bar_y1),
                (x_offset + bar_width, bar_y2),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                overlay,
                text,
                (text_x, text_y),
                FONTFACE,
                FONTSCALE,
                FONTCOLOR,
                THICKNESS,
                LINETYPE,
            )

        cv2.addWeighted(
            overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame
        )

    def visualize(self, frames, data_samples, action_label, scores, label_map):
        pose_config = mmengine.Config.fromfile(self.pose_config)
        visualizer = VISUALIZERS.build(pose_config.visualizer)
        visualizer.set_dataset_meta(data_samples[0].dataset_meta)

        vis_frames = []
        print("Drawing skeleton and percentage bar for each frame")
        for d, f in track_iter_progress(list(zip(data_samples, frames))):
            f = mmcv.imconvert(f, "bgr", "rgb")
            visualizer.add_datasample(
                "result",
                f,
                data_sample=d,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show=False,
                wait_time=0,
                out_file=None,
                kpt_thr=0.3,
            )
            vis_frame = visualizer.get_image()
            self.draw_percentage_bar(vis_frame, scores, label_map)
            vis_frames.append(vis_frame)

        # Save the frames as a video
        print("Saving the video")
        height, width, layers = vis_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
        video = cv2.VideoWriter(self.out_filename, fourcc, 30, (width, height))

        for frame in vis_frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert back to BGR

        video.release()
        print(f"Video saved to {self.out_filename}")

    def run(self, show_visualization: bool = False):
        tmp_dir = tempfile.TemporaryDirectory()
        frame_paths, frames = frame_extract(
            self.video_path, self.short_side, tmp_dir.name
        )
        h, w, _ = frames[0].shape

        # Get human detection results
        det_results, _ = detection_inference(
            self.det_config,
            self.det_checkpoint,
            frame_paths,
            self.det_score_thr,
            0,
            self.device,
        )
        torch.cuda.empty_cache()

        # Get pose estimation results
        pose_results, pose_data_samples = pose_inference(
            self.pose_config,
            self.pose_checkpoint,
            frame_paths,
            det_results,
            self.device,
        )
        torch.cuda.empty_cache()

        config = mmengine.Config.fromfile(self.config)
        config.merge_from_dict(self.cfg_options)

        model = init_recognizer(config, self.checkpoint, self.device)
        result = inference_skeleton(model, pose_results, (h, w))

        max_pred_index = result.pred_score.argmax().item()
        scores = result.pred_score.cpu().numpy()
        label_map = [x.strip() for x in open(self.label_map).readlines()]
        action_label = label_map[max_pred_index]
        action_score = scores[max_pred_index]

        if show_visualization:
            self.visualize(frames, pose_data_samples, action_label, scores, label_map)

        tmp_dir.cleanup()
        return action_label, action_score


def main():
    from constants import globals as g

    filename = "center-4_run_2_3.mp4"
    src_video = os.path.join(g.RAW_DIR.__str__(), filename)
    out_file = os.path.join(g.TEST_DATA_DIR.__str__(), f"{filename}-output.mp4")
    inference = RunInference(video_path=src_video, out_filename=out_file)

    pred_action, pred_score = inference.run(show_visualization=True)
    print(
        f"Predicted action: {pred_action.upper()} with confidence score of {round((pred_score*100), 4)}%."
    )


if __name__ == "__main__":
    main()
