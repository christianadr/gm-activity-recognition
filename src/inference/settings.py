import cv2
from constants import globals as g

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)
THICKNESS = 1
LINETYPE = 1
BACKGROUND_COLOR = (0, 0, 0)
BACKGROUND_ALPHA = 0.5

MODEL_CONFIG = "configs/pretrained_slowonly_resnet50_gym-keypoint_v2.py"
MODEL_CHECKPOINT = "models/best_acc_top1_epoch_17.pth"

LABEL_MAP = ["gallop", "hop", "jump", "leap", "run", "skip", "slide"]

DET_CONFIG = {
    "det_config": "mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py",
    "det_checkpoint": "models/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth",
    "det_score_thr": 0.9,
    "det_cat_id": 0,
}

POSE_CONFIG = {
    "pose_config": "mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py",
    "pose_checkpoint": "models/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
}

DET_DEPLOY_CFG = "mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py"
DET_BACKEND_MODEL = "/models/det/ort/end2end.onnx"
