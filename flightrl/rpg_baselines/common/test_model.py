import numpy as np
import cv2
from ultralytics import YOLO
import torch
import os
from pathlib import Path
from glob import glob
from rpg_baselines.common.plotting import Plotter
import uuid

EVAL_FOLDER = Path("./eval")
EVAL_FOLDER.mkdir(parents=True, exist_ok=True)

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
        if (np.all(p_cam[:, 2] > 2)):
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

def test_model(
        env, 
        model, 
        render: bool = False, 
        num_rollouts: int = 5, 
        weight_path: str = "", 
        vid: bool = False, 
        vision_weights: str = "",
        build_dataset: bool = False,
    ):
    """Inferences model for a number of rollouts.
    
    Can use/record camera with the option of a vision model inferencing.
    Additionally, can build a YOLO-style pose dataset using camera 
    and known absolute gate positions.
    Args:
        render : whether to render unity standalone
        num_rollouts : # of rollouts to inference on
        weight_path : weight path to use for inference
        vid : whether to record video via camera
        vision_weights : whether to use vision model inference (cam must be true) 
        build_dataset : whether to build dataset (cam must be true)
    """
    max_ep_length = env.max_episode_steps
    uuid_index = uuid.uuid4()
    obs_type = "vision" if vision_weights != "" else "gt"
    # Create a unique uuid folder for this run if not existing already
    while True:
        try:
            save_folder = EVAL_FOLDER / obs_type / str(uuid_index)
            save_folder.mkdir(parents=True)
        except FileExistsError:
            uuid_index = uuid.uuid4()
        else:
            break
    
    if render:
        env.connectUnity()
    if vision_weights != "":
        vis_model = YOLO(vision_weights)

    # --------------------
    # BUILD DATASET-SPECIFIC VARIABLES
    # --------------------
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
    
    # --------------------
    # ALL ROLLOUTS OF THE DRONE
    # --------------------
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        
        # --------------------
        # SETUP LOGGING VARIABLES PER ROLLOUT
        # --------------------
        traj = np.zeros((max_ep_length, 3), dtype=np.float32)
        residual = np.zeros((max_ep_length, 3), dtype=np.float32)
        detections = np.zeros(max_ep_length, dtype=bool)
        gt_dist = np.zeros(max_ep_length, dtype=np.float32)
        time_i = 0

        # --------------------
        # CREATE DATASET IF VIDEO/DATASET FLAGS
        # --------------------
        if vid and not build_dataset:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(
                weight_path.replace(".zip", f"_rollout_{n_roll}.avi"),
                fourcc, 30.0, (640, 360)
            )
        
        # --------------------
        # POLICY INFERENCE FOR ENTIRE EPISODE
        # --------------------
        total_rew = 0
        while not (done or (ep_len >= max_ep_length)):
            # policy input/output
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = env.step(act)
            ep_len += 1
            
            # save variables for data analysis
            drone_pos = env.venv.drone_pos[0]
            gt_rel_gate = env.venv.gates[env.venv.cur_gate[0]] - drone_pos

            traj[time_i] = drone_pos
            residual[time_i] = env.venv._full_obs[0, 0:3] - (gt_rel_gate)
            gt_dist[time_i] = np.linalg.norm(gt_rel_gate)
            detections[time_i] = env.venv.detected_gate[0]
            time_i += 1
            total_rew += rew

            # --------------------
            # CAPTURE VIDEO OUTPUT
            # --------------------
            if vid:
                frame = np.array(env.venv.rgb_image[0], dtype=np.uint8)
                if frame.size == 0:
                    continue

                if vision_weights:
                    # compute inference and save to video.
                    results = vis_model(
                        frame, 
                        device=("cuda" if torch.cuda.is_available() else "cpu"),
                        verbose=False,
                        conf=0.7
                    )
                    out.write(results[0].plot())
                elif build_dataset and ep_len % 25 == 0:
                    # build labels from ground truth and write to YOLO-style pose estimation ds
                    labels = convert_to_yolo_kp_labels(env)
                    if labels:
                        cv2.imwrite(str(images_dir / f"{dataset_iter}_{n_roll}_{ep_len:010d}.jpg"), frame)
                        with open(str(labels_dir / f"{dataset_iter}_{n_roll}_{ep_len:010d}.txt"), "w", encoding="utf-8") as f:
                            for line in labels:
                                f.write(f"{line}\n")
                elif not build_dataset:
                    # just write frame raw from camera
                    out.write(frame)

        # --------------------
        # SAVE ALL VALS FROM CUR ROLLOUT TO COMPRESSED NUMPY FILE
        # --------------------
        save_path = str(save_folder / f"rollout_{env.venv.cur_seed}_{n_roll:03d}.npz")
        np.savez_compressed(
            save_path,
            traj=traj[:time_i],
            hits=infos[0]["episode"]["gate_hits"],
            crosses=infos[0]["episode"]["gate_crosses"],
            residual=residual[:time_i],
            gt_dist=gt_dist[:time_i],
            detections=detections[:time_i],
            n_gates=env.venv.n_gates
        )

        if vid and not build_dataset:
            out.release()
        print(f"\n\nEpisode ended: step={ep_len}. gate={env.venv.cur_gate[0]}. rew={rew}\n\n")
        
    # --------------------
    # GRAPH ALL ROLLOUT DATA
    # --------------------
    gates = env.venv.gates.astype(np.float32)
    rots = env.venv.rot_mats.astype(np.float32)
    half_w, half_h = env.venv.half_w, env.venv.half_h
    data_keys = ["traj", "hits", "crosses", "residual", "gt_dist", "detections", "n_gates"]
    data = [fn for fn in glob(f"{str(save_folder)}/rollout_*")]
    
    plt = Plotter(gates, rots, env.venv.sim_dt, (half_w, half_h), str(save_folder))
    plt.load_data(npz=data, keys=data_keys)
    
    plt.plot_trajectory()
    plt.plot_side_trajectory()
    plt.plot_hits()
    plt.plot_residual()
    plt.plot_completion()
        
    if render:
        env.disconnectUnity()