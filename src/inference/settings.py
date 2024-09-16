from pathlib import Path

import cv2

from constants import globals as g

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)
THICKNESS = 1
LINETYPE = 1
BACKGROUND_COLOR = (0, 0, 0)
BACKGROUND_ALPHA = 0.5

MODEL_CONFIG = Path(
    g.PROJECT_DIR, "configs/pretrained_slowonly_resnet50_gym-keypoint_v2.py"
).__str__()
MODEL_CHECKPOINT = Path(g.PROJECT_DIR, "models/PoseC3D-best.pth").__str__()

LABEL_MAP = ["gallop", "hop", "jump", "leap", "run", "skip", "slide"]

DET_CONFIG = {
    "det_config": Path(
        g.PROJECT_DIR,
        "mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py",
    ).__str__(),
    "det_checkpoint": Path(
        g.PROJECT_DIR,
        "models/skeleton/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth",
    ).__str__(),
    "det_score_thr": 0.9,
    "det_cat_id": 0,
}

POSE_CONFIG = {
    "pose_config": Path(
        g.PROJECT_DIR,
        "mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py",
    ).__str__(),
    "pose_checkpoint": Path(
        g.PROJECT_DIR, "models/skeleton/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
    ).__str__(),
}

DET_DEPLOY_CFG = "mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py"
DET_BACKEND_MODEL = "/models/det/ort/end2end.onnx"
