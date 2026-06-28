import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import cv2
from ultralytics import YOLO
import torch
import os
from pathlib import Path

from rpg_baselines.common.setup_logger import setup_logging

def convert_to_yolo_kp_labels(env):
    """Convert known gate positions if they are in camera view to YOLO-style
    pose estimation positions for a dataset."""
    
    fx = 180 
    fy = 180
    img_w = 640
    img_h = 360
    
    # check each gate is in camera frame then input for label
    yolo_labels = []
    for i in range(len(env.venv.gates)):
        # get gates world pos relative to drone
        center = env.venv.gates[i]
        right = env.venv.rot_mats[i, :, 0] * env.venv.half_w
        up = env.venv.rot_mats[i, :, 2] * env.venv.half_h
        drone_pos = env.venv.drone_pos[0]

        p_world = np.stack([
            (center - right + up) - drone_pos,  # TL
            (center + right + up) - drone_pos,  # TR
            (center + right - up) - drone_pos,  # BR
            (center - right - up) - drone_pos  # BL
        ], axis=0)

        # world pos relative to drone -> drone local position
        R_world = env.venv._full_obs[0, 3:12].reshape(3, 3)
        p_body = (R_world.T @ p_world.T).T
        
        # drone local position -> camera local position
        p_body_offset = p_body - [0, 0, 0.3]  # still in drone pos, just offset by cam z
        p_cam = (env.venv.R_body_cam.T @ p_body_offset.T).T
        
        # forward axis
        if (np.all(p_cam[:, 2] > 0)):
            # camera local pos projected to image pixels
            u = 320 + fx * p_cam[:, 0] / p_cam[:, 2]
            v = 180 + fy * p_cam[:, 1] / p_cam[:, 2]
            # per corner bool mask, normalize & check visibility
            in_bounds = (u >= 0) & (u <= 640) & (v >= 0) & (v <= 360)
            
            if np.sum(in_bounds) >= 2:
                # grab all in bound coordinates, normalize
                u_vis = u[in_bounds]
                v_vis = v[in_bounds]
                x_min, x_max = u_vis.min(), u_vis.max()
                y_min, y_max = v_vis.min(), v_vis.max()
                
                w = (x_max - x_min) / img_w
                h = (y_max - y_min) / img_h
                cx = (x_min + x_max) / 2 / img_w
                cy = (y_min + y_max) / 2 / img_h

                # if not below 20 pixels in width / height
                if (w > 0.03 and h > 0.03):
                    u /= 640
                    v /= 360
                    # yolo format for pose:
                    # <class> <xc> <yc> <w> <h> 
                    # <kp1_x> <kp1_y> <kp1_vis> <kp2_x> <kp2_y> <kp2_vis> 
                    # <kp3_x> <kp3_y> <kp3_vis> <kp4_x> <kp4_y> <kp4_vis>
                    temp_str = f"0 {cx:.6f} {cy:.6f} {w} {h}"
                    for j in range(len(u)):
                        vis = 2 if in_bounds[j] else 0
                        kp_x = float(u[j]) if in_bounds[j] else 0.0
                        kp_y = float(v[j]) if in_bounds[j] else 0.0
                        temp_str += f" {kp_x:.6f} {kp_y:.6f} {vis}"
                    yolo_labels.append(temp_str)
    
    return yolo_labels

def convert_to_yolo_seg_labels(env):
    """Convert known gate positions if they are in camera view to YOLO-style
    segmentation masks for a dataset.
    """
    fx = 180 
    fy = 180
    img_w = 640
    img_h = 360
    
    # check each gate is in camera frame then input for label
    yolo_labels = []
    for i in range(len(env.venv.gates)):
        # get gates world pos relative to drone
        center = env.venv.gates[i]
        right = env.venv.rot_mats[i, :, 0] * 1.35
        up = env.venv.rot_mats[i, :, 2] * 1.35
        drone_pos = env.venv.drone_pos[0]

        p_world = np.stack([
            (center - right + up) - drone_pos,  # TL
            (center + right + up) - drone_pos,  # TR
            (center + right - up) - drone_pos,  # BR
            (center - right - up) - drone_pos  # BL
        ], axis=0)

        # world pos relative to drone -> drone local position
        R_world = env.venv._full_obs[0, 3:12].reshape(3, 3)
        p_body = (R_world.T @ p_world.T).T
        
        # drone local position -> camera local position
        p_body_offset = p_body - [0, 0, 0.3]  # still in drone pos, just offset by cam z
        p_cam = (env.venv.R_body_cam.T @ p_body_offset.T).T
        
        # forward axis
        if (np.any(p_cam[:, 2] > 0)):
            # camera local pos projected to image pixels
            u = 320 + fx * p_cam[:, 0] / p_cam[:, 2]
            v = 180 + fy * p_cam[:, 1] / p_cam[:, 2]
            # per corner bool mask, normalize & check visibility
            in_bounds = (u >= 0) & (u <= 640) & (v >= 0) & (v <= 360)
            
            if np.sum(in_bounds) >= 3:
                # grab all in bound coordinates, normalize
                u_vis = u[in_bounds]
                v_vis = v[in_bounds]
                x_min, x_max = u_vis.min(), u_vis.max()
                y_min, y_max = v_vis.min(), v_vis.max()
                
                w = (x_max - x_min) / img_w
                h = (y_max - y_min) / img_h
                cx = (x_min + x_max) / 2 / img_w
                cy = (y_min + y_max) / 2 / img_h

                # if not below 20 pixels in width / height
                if (w > 0.03 and h > 0.03):
                    u /= 640
                    v /= 360
                    # yolo format for seg:
                    # <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
                    temp_str = f"0"
                    for j in range(len(u)):
                        if in_bounds[j]:
                            temp_str += f" {u[j]:.6f} {v[j]:.6f}"
                    yolo_labels.append(temp_str)
    
    return yolo_labels
    
    
def test_model(
        env, 
        model, 
        render: bool = False, 
        num_rollouts: int = 5, 
        weight_path: str = "", 
        vid: bool = False, 
        vision_weights: str = "",
        build_dataset: bool = False
    ):
    """Inferences model for a number of rollouts.
    
    Can use/record camera with the option of a vision model inferencing.
    Additionally, can build a YOLO-style pose dataset using camera 
    and known absolute gate positions.
    Args:
        render (bool): whether to render unity standalone
        num_rollouts (int): # of rollouts to inference on
        weight_path (string): weight path to use for inference
        vid (bool): whether to record video via camera
        vision_weights (string): whether to use vision model inference (cam must be true) 
        build_dataset (bool): whether to build dataset (cam must be true)
    """
    max_ep_length = env.max_episode_steps
    logger = setup_logging(name=weight_path.replace(".zip", "_logger"), filename=weight_path.replace(".zip", ".log"))
    
    if render:
        env.connectUnity()

    if vision_weights != "":
        vis_model = YOLO(vision_weights)

    if build_dataset:
        base_dir = Path(weight_path.replace(".zip", f"_dataset"))
        dataset_iter = 0
        dataset_dir = base_dir
        while dataset_dir.exists():
            dataset_dir = base_dir.parent / f"{base_dir.name}_{dataset_iter}"
            dataset_iter += 1
        # note that all files go into `train` for ease of access.
        dataset_dir.mkdir(parents=True, exist_ok=True)
        images_dir = dataset_dir / "images" / "train"
        labels_dir = dataset_dir / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0

        # create dataset if vid + dataset flags, else just video if flagged.
        if vid:
            if not build_dataset:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(
                    weight_path.replace(".zip", f"_rollout_{n_roll}.avi"),
                    fourcc, 30.0, (640, 360)
                )
        
        # inference of policy
        total_rew = 0
        while not (done or (ep_len >= max_ep_length)):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = env.step(act) # act
            
            labels = ['gate_x','gate_y','gate_z',
                'R00','R01','R02','R10','R11','R12','R20','R21','R22',
                'vel_x','vel_y','vel_z','wx','wy','wz',
                'pa0','pa1','pa2','pa3',
                'c0x','c0y','c0z','c1x','c1y','c1z','c2x','c2y','c2z','c3x','c3y','c3z',
                'next_x','next_y','next_z']
            for l, v in zip(labels, env.venv._full_obs[0]):
                logger.info(f"  {l}: {v:.3f}")
            
            total_rew += rew
            ep_len += 1
            # capture video output and perform an action based on flags
            if vid:
                frame = np.array(env.venv.rgb_image[0], dtype=np.uint8)
                if vision_weights:
                    results = vis_model(
                        frame, 
                        device=("cuda" if torch.cuda.is_available() else "cpu"),
                        verbose=False
                    )
                    out.write(results[0].plot())
                elif build_dataset and ep_len % 25 == 0:
                    labels = convert_to_yolo_seg_labels(env)
                    if labels:
                        cv2.imwrite(str(images_dir / f"{dataset_iter}_{n_roll}_{ep_len:010d}.jpg"), frame)
                        with open(str(labels_dir / f"{dataset_iter}_{n_roll}_{ep_len:010d}.txt"), "w", encoding="utf-8") as f:
                            for line in labels:
                                f.write(f"{line}\n")
                elif not build_dataset:
                    out.write(frame)

        if vid and not build_dataset:
            out.release()
        print(f"\n\nEpisode ended: step={ep_len}. gate={env.venv.cur_gate[0]}. rew={rew}\n\n")

    if render:
        env.disconnectUnity()