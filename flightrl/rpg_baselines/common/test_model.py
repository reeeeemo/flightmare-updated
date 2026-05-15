import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import cv2
from ultralytics import YOLO
import torch
import os
from pathlib import Path

def convert_to_yolo_labels(env):
    """Convert known gate positions if they are in camera view to YOLO-style
    pose estimation positions for a dataset."""
    
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
            u = 320 + 320 * p_cam[:, 0] / p_cam[:, 2]
            v = 180 + 320 * p_cam[:, 1] / p_cam[:, 2]
            # per corner bool mask, normalize & check visibility
            in_bounds = (u >= 0) & (u <= 640) & (v >= 0) & (v <= 360)
            
            if np.sum(in_bounds) >= 2:
                u_vis = u[in_bounds]
                v_vis = v[in_bounds]
                w = np.max(u_vis) - np.min(u_vis)
                h = np.max(v_vis) - np.min(v_vis)
                
                if (w > 20 and h > 20): # not too small
                    u /= 640
                    v /= 360
                    cx = (np.max(u[in_bounds]) + np.min(u[in_bounds])) / 2
                    cy = (np.max(v[in_bounds]) + np.min(v[in_bounds])) / 2
                    # yolo format for pose:
                    # <class> <xc> <yc> <w> <h> 
                    # <kp1_x> <kp1_y> <kp1_vis> <kp2_x> <kp2_y> <kp2_vis> 
                    # <kp3_x> <kp3_y> <kp3_vis> <kp4_x> <kp4_y> <kp4_vis>
                    temp_str = f"0 {cx} {cy} {w/640} {h/360}"
                    for j in range(len(u)):
                        vis = 2 if in_bounds[j] else 0
                        kp_x = float(u[j]) if in_bounds[j] else 0.0
                        kp_y = float(v[j]) if in_bounds[j] else 0.0
                        temp_str += f" {kp_x} {kp_y} {vis}"
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
    #
    fig = plt.figure(figsize=(18, 12), tight_layout=True)
    gs = gridspec.GridSpec(5, 12)
    #
    ax_x = fig.add_subplot(gs[0, 0:4])
    ax_y = fig.add_subplot(gs[0, 4:8])
    ax_z = fig.add_subplot(gs[0, 8:12])
    #
    ax_dx = fig.add_subplot(gs[1, 0:4])
    ax_dy = fig.add_subplot(gs[1, 4:8])
    ax_dz = fig.add_subplot(gs[1, 8:12])
    #
    ax_euler_x = fig.add_subplot(gs[2, 0:4])
    ax_euler_y = fig.add_subplot(gs[2, 4:8])
    ax_euler_z = fig.add_subplot(gs[2, 8:12])
    #
    ax_euler_vx = fig.add_subplot(gs[3, 0:4])
    ax_euler_vy = fig.add_subplot(gs[3, 4:8])
    ax_euler_vz = fig.add_subplot(gs[3, 8:12])
    #
    ax_action0 = fig.add_subplot(gs[4, 0:3])
    ax_action1 = fig.add_subplot(gs[4, 3:6])
    ax_action2 = fig.add_subplot(gs[4, 6:9])
    ax_action3 = fig.add_subplot(gs[4, 9:12])

    max_ep_length = env.max_episode_steps
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
        pos, euler, dpos, deuler = [], [], [], []
        actions = []
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
            obs, rew, done, infos = env.step(act)
            total_rew += rew
            ep_len += 1
            
            # capture video output and perform an action based on flags
            if vid:
                frame = np.array(env.venv.rgb_image[0], dtype=np.uint8)
                if vision_weights:
                    results = vis_model(frame, device=("cuda" if torch.cuda.is_available() else "cpu"))
                    out.write(results[0].plot())
                elif build_dataset and ep_len % 50 == 0:
                    labels = convert_to_yolo_labels(env)
                    if labels:
                        cv2.imwrite(str(images_dir / f"{dataset_iter}_{n_roll}_{ep_len:010d}.jpg"), frame)
                        with open(str(labels_dir / f"{dataset_iter}_{n_roll}_{ep_len:010d}.txt"), "w", encoding="utf-8") as f:
                            for line in labels:
                                f.write(f"{line}\n")
                elif not build_dataset:
                    out.write(frame)

            pos.append(obs[0, 0:3].tolist())
            dpos.append(obs[0, 12:15].tolist())
            euler.append(obs[0, 3:12].tolist())
            deuler.append(obs[0, 15:18].tolist())
            actions.append(act[0, :].tolist())
        if vid and not build_dataset:
            out.release()
        print(f"\n\nEpisode ended: step={ep_len}. gate={env.venv.cur_gate[0]}. rew={rew}\n\n")
        pos = np.asarray(pos)
        dpos = np.asarray(dpos)
        euler = np.asarray(euler)
        deuler = np.asarray(deuler)
        actions = np.asarray(actions)
        #
        t = np.arange(0, pos.shape[0])
        ax_x.step(t, pos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_y.step(t, pos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_z.step(t, pos[:, 2], color="C{0}".format(
            n_roll), label="pos [x, y, z] -- trail: {0}".format(n_roll))
        #
        ax_dx.step(t, dpos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dy.step(t, dpos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dz.step(t, dpos[:, 2], color="C{0}".format(
            n_roll), label="vel [x, y, z] -- trail: {0}".format(n_roll))
        #
        ax_euler_x.step(t, euler[:, -1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_y.step(t, euler[:, 0], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_z.step(t, euler[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        #
        ax_euler_vx.step(t, deuler[:, -1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_vy.step(t, deuler[:, 0], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_vz.step(t, deuler[:, 1], color="C{0}".format(
            n_roll), label=r"$\theta$ [x, y, z] -- trail: {0}".format(n_roll))
        #
        ax_action0.step(t, actions[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action1.step(t, actions[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action2.step(t, actions[:, 2], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action3.step(t, actions[:, 3], color="C{0}".format(
            n_roll), label="act [0, 1, 2, 3] -- trail: {0}".format(n_roll))
    #
    if render:
        env.disconnectUnity()
    ax_z.legend()
    ax_dz.legend()
    ax_euler_z.legend()
    ax_euler_vz.legend()
    ax_action3.legend()
    #
    plt.tight_layout()
    fig.savefig(weight_path.replace(".zip", ".png"))
    plt.show()
