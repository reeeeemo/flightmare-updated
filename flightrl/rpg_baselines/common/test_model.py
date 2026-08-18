import numpy as np
import cv2
from ultralytics import YOLO
import torch
import os
from pathlib import Path
from glob import glob
from rpg_baselines.common.plotting import Plotter

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
    os.makedirs("eval", exist_ok=True)

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
            residual[time_i] = env._full_obs[0, 0:3] - (gt_rel_gate)
            gt_dist[time_i] = np.linalg.norm(gt_rel_gate)
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
                        verbose=False
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
        np.savez_compressed(
            f"eval/rollout_{n_roll:03d}.npz",
            traj=traj[:time_i],
            hits=infos[0]["episode"]["gate_hits"],
            crosses=infos[0]["episode"]["gate_crosses"],
            residual=residual[:time_i],
            gt_dist=gt_dist[:time_i]
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
    data_keys = ["traj", "hits", "crosses", "residual", "gt_dist"]
    data = [f"eval/rollout_{n_roll:03d}.npz" for n_roll in range(num_rollouts)]
    
    plt = Plotter(gates, rots, env.venv.sim_dt, (half_w, half_h))
    plt.load_data(npz=data, keys=data_keys)
    plt.plot_trajectory()
    plt.plot_side_trajectory()
    plt.plot_hits()
    plt.plot_residual()
        
    if render:
        env.disconnectUnity()