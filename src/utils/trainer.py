# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import time

from mmaction.registry import RUNNERS
from mmengine.config import Config
from mmengine.runner import Runner


def merge_args(cfg, work_dir):
    """Merge CLI arguments to config."""
    cfg.work_dir = work_dir  # Set the work_dir to save the results


def main(config_path, work_dir):
    # Load config from file
    cfg = Config.fromfile(config_path)

    # Merge config if needed
    merge_args(cfg, work_dir)

    # Build the runner from config
    if "runner_type" not in cfg:
        # Build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # Start training
    runner.train()


if __name__ == "__main__":
    config_path = "../../configs/pretrained_slowonly_resnet50_gym-keypoint_v2.py"
    work_dir = "../../results/posec3d"
    start_train_time = time.time()
    main(config_path, work_dir)
    end_train_time = time.time()
