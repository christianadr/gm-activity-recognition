from pathlib import Path

from constants import globals as g

det_config = Path(
    g.PROJECT_DIR,
    "mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py",
).__str__()

det_checkpoint = Path(
    g.PROJECT_DIR,
    "models/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth",
).__str__()

det_score_thr = 0.5

pose_config = Path(
    g.PROJECT_DIR,
    "mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py",
).__str__()

pose_checkpoint = Path(
    g.PROJECT_DIR, "models/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
).__str__()
