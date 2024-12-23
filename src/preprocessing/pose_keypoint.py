import os.path as osp
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mmengine
import numpy as np
from mmaction.apis import detection_inference, pose_inference
from mmaction.utils import frame_extract
from tqdm import tqdm

from constants import globals as g

# SETUP VARIABLES
DET_CONFIG = Path(
    g.PROJECT_DIR,
    "mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py",
).__str__()

DET_CHECKPOINT = Path(
    g.PROJECT_DIR,
    "models/skeleton/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth",
).__str__()

DET_SCORE_THR = 0.5
POSE_CONFIG = Path(
    g.PROJECT_DIR,
    "mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py",
).__str__()

POSE_CHECKPOINT = Path(
    g.PROJECT_DIR,
    "models/skeleton/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
).__str__()


def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):
    def inside(box0, box1, threshold=0.8):
        return intersection(box0, box1) / area(box0) > threshold

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i], bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, threshold=threshold):
        shape = [sum(bbox[:, -1] > threshold) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) < 10
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets


def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.0
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.0
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox


def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox[:, None, :]


def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, : item.shape[0]] = item
        else:
            inds = sorted(list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret


def ntu_det_postproc(vid, det_results):
    det_results = [removedup(x) for x in det_results]
    # print(g.CLASS_NAMES[vid.split("_")[1]])
    label = g.CLASS_NAMES[vid.split("_")[1]]
    mpaction = list(range(50, 61)) + list(range(106, 121))
    n_person = 2 if label in mpaction else 1
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        print("\nEasy Example")
        return bboxes

    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    print(f"\nHard {n_person}-person Example, found {len(tracklets)} tracklet")
    if n_person == 1:
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results
    # n_person is 2
    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        return bboxes2bbox(det_results, len(det_results))


def pose_inference_with_align(frame_paths, det_results):
    # filter frame without det bbox
    det_results = [frm_dets for frm_dets in det_results if frm_dets.shape[0] > 0]

    pose_results, _ = pose_inference(
        POSE_CONFIG,
        POSE_CHECKPOINT,
        frame_paths,
        det_results,
        "cuda:0",
    )
    # align the num_person among frames
    num_persons = max([pose["keypoints"].shape[0] for pose in pose_results])
    num_points = pose_results[0]["keypoints"].shape[1]
    num_frames = len(pose_results)
    keypoints = np.zeros((num_persons, num_frames, num_points, 2), dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose["keypoints"].shape[0]
        for p_idx in range(frm_num_persons):
            keypoints[p_idx, f_idx] = frm_pose["keypoints"][p_idx]
            scores[p_idx, f_idx] = frm_pose["keypoint_scores"][p_idx]

    return keypoints, scores


def ntu_pose_extraction(vid, skip_postproc=False):
    tmp_dir = TemporaryDirectory()
    frame_paths, _ = frame_extract(vid, out_dir=tmp_dir.name)
    det_results, _ = detection_inference(
        det_config=DET_CONFIG,
        det_checkpoint=DET_CHECKPOINT,
        det_score_thr=DET_SCORE_THR,
        frame_paths=frame_paths,
        device="cuda:0",
        with_score=True,
    )

    if not skip_postproc:
        det_results = ntu_det_postproc(vid, det_results)

    anno = dict()
    shape = (1080, 1920)
    keypoints, scores = pose_inference_with_align(frame_paths, det_results)
    anno["keypoint"] = keypoints
    anno["keypoint_score"] = scores
    anno["frame_dir"] = osp.splitext(osp.basename(vid))[0]
    anno["img_shape"] = shape
    anno["original_shape"] = shape
    anno["total_frames"] = keypoints.shape[1]
    anno["label"] = g.CLASS_NAMES[osp.basename(vid).split("-")[0]]
    tmp_dir.cleanup()

    return anno


def main():
    import os

    src_dir = Path(g.RAW_DIR, "gm-activities")
    sources = [src for src in src_dir.iterdir()]

    for source in tqdm(sources, desc="Extracting keypoints", total=len(sources)):
        destination = os.path.join(
            Path(g.GROSSMOTOR_DIR_PK),
            os.path.splitext(os.path.basename(source))[0] + ".pkl",
        )

        if not os.path.exists(destination):
            anno = ntu_pose_extraction(source.__str__(), skip_postproc=True)
            mmengine.dump(anno, destination)

        else:
            print(f"{destination} already exists")

    # video_filepaths = list(g.PROCESSED_DIR.iterdir())

    # for src_vid in tqdm(
    #     video_filepaths,
    #     desc="Extracting keypoints",
    #     total=len(video_filepaths),
    # ):
    #     dst = osp.join(
    #         Path(g.GROSSMOTOR_DIR_PK, "center-cropped"),
    #         osp.splitext(osp.basename(src_vid))[0] + ".pkl",
    #     )
    #     if not osp.exists(dst):
    #         anno = ntu_pose_extraction(src_vid.__str__(), skip_postproc=True)
    #         mmengine.dump(anno, dst)
    #     else:
    #         print(f"{dst} already exists")


if __name__ == "__main__":
    main()
