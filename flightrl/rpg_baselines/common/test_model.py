import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import cv2
from ultralytics import YOLO
import torch
import os
from pathlib import Path
from glob import glob

def compute_speeds(trajectories: list, sim_dt: float):
    """Returns speeds of all trajectories.
    
    Computes magnitude of shifts in values per timestep.
    """
    all_speeds = [np.linalg.norm(np.diff(t, axis=0), axis=1) / sim_dt for t in trajectories]
    vmax = max((s.max() for s in all_speeds if len(s)), default=1.0)
    norm = Normalize(0, vmax)
    return all_speeds, norm

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
    
def plot_residual(data):
    """Plot difference between visual inference and GT gate values."""
    gt_dist = np.concatenate(data.get("gt_dist"))
    residual = np.concatenate(data.get("residual"))
    
    if gt_dist is None or residual is None:
        return
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10,12))
    
    ### create bins for each plotted distance/inferenced model ###
    # bins are built off of ground truth distance
    edges = np.arange(0, gt_dist.max() + 1, 1.0)
    bin_id = np.digitize(gt_dist, edges)
    
    ### compute means and standard deviations of each bin ###
    for axis, axis_n in zip([0, 1, 2], ['x', 'y', 'z']):
        means, stds, centers = [], [], []
        for b in range(1, len(edges)):
            m = (bin_id == b)
            if m.sum() < 20: # no zero or sparse pts
                continue
            centers.append((edges[b-1] + edges[b]) / 2)
            means.append(residual[m, axis].mean())
            stds.append(residual[m, axis].std())
            
        centers, means, stds = np.array(centers), np.array(means), np.array(stds)
        
        ### plot bias line, error variance band ###
        ax[axis].plot(centers, means, label=f"mean {axis_n} error")
        ax[axis].fill_between(centers, means - stds, means + stds, alpha=0.3)
        ax[axis].axhline(0, ls='--')
        ax[axis].set_ylabel(f"vision - GT, {axis_n} (m)")
        ax[axis].legend()

    ax[2].set_xlabel("true gate distance (m)")
    fig.suptitle("Vision error vs distance")
    fig.savefig(f"eval/residual_all.png")

def plot_trajectory(half_w: float, 
                    half_h: float, 
                    sim_dt: float,
                    gates: list,
                    rots: list,
                    data,):
    """Plot the overall trajectory and gate positions/rotations."""
    traj = data.get("traj", None)
    if traj is None:
        return
    fig, ax = plt.subplots(figsize=(10,8))
    
    ### COMPUTE SPEEDS OF TRAJECTORY ###
    all_speeds, norm = compute_speeds(traj, sim_dt)

    ### PIN TRAJECTORY ###
    # cannot stack since diff lengths :p
    for i, (t, speed) in enumerate(zip(traj, all_speeds)):
        pts = t[:, [0, 1]].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=norm, array=speed, linewidth=2)
        ax.add_collection(lc)

        ax.scatter(
            t[0, 0], t[0, 1], c="green", s=60,
            marker="o", label="start" if i==0 else None, zorder=60
        )
        ax.scatter(
            t[-1, 0], t[-1, 1], c="red", s=60,
            marker="o", label="end" if i==0 else None, zorder=6
        )
    
    ### PIN GATES ###
    # take the rotmat and find rotation as well on (x,y)
    for i, (center, R) in enumerate(zip(gates, rots)):
        right = R[:, 0] * half_w
        up = R[:, 2] * half_h
            
        corners_3d = np.stack([
            center - right + up,  # TL
            center + right + up,  # TR
            center + right - up,  # BR
            center - right - up,  # BL  
        ])
        corners_2d = corners_3d[:, :2]
        poly = Polygon(
            corners_2d, closed=True,
            facecolor="orange", alpha=1.0,
            edgecolor="darkorange", linewidth=3.5,
            label="gate" if i == 0 else None
        )
        ax.add_patch(poly)
    
    ### SET LABELS / FIGURE SETTINGS THEN SAVE ###
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    fig.colorbar(lc, ax=ax, label="speed (m/s)")
    ax.autoscale()
        
    plt.tight_layout()
    plt.savefig(f"eval/traj.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
def plot_side_trajectory(half_h: float,
                         sim_dt: float,
                         gates: list,
                         rots: list,
                         data):
    """Plot overall attitude-based trajectory of policy + gates."""
    traj = data.get("traj")
    if traj is None:
        return
    
    fig, ax = plt.subplots(figsize=(24,3))
    
    ### COMPUTE SPEEDS OF TRAJECTORY ###
    all_speeds, norm = compute_speeds(traj, sim_dt)
    
    ### PIN TRAJECTORY ###
    # cannot stack since diff lengths :p
    for i, (t, speed) in enumerate(zip(traj, all_speeds)):
        pts = t[:, [1, 2]].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=norm, array=speed, linewidth=2)
        ax.add_collection(lc)

        ax.scatter(
            t[0, 1], t[0, 2], c="green", s=60,
            marker="o", label="start" if i==0 else None, zorder=60
        )
        ax.scatter(
            t[-1, 1], t[-1, 2], c="red", s=60,
            marker="o", label="end" if i==0 else None, zorder=6
        )
    
    ### PIN GATES ###
    # take the rotmat and find rotation as well on (x,y)
    for i, (center, R) in enumerate(zip(gates, rots)):
        up = R[:, 2] * half_h
        
        ax.plot([center[1], center[1]],
                [center[2]-up[2], center[2]+up[2]],
                color="darkorange", lw=4, label="gate" if i==0 else None)

    ### SET LABELS / FIGURE SETTINGS THEN SAVE ###
    ax.set_xlabel("Y Foward (m)")
    ax.set_ylabel("Z up (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_aspect("equal")
    
    fig.colorbar(lc, ax=ax, label="speed (m/s)")
    ax.autoscale()

    plt.tight_layout()
    plt.savefig(f"eval/attitude_traj.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_hits(half_w: float, 
              half_h: float,
              gates: list,
              rots: list, 
              data):
    """Plot all gate crosses/hits by the trajectories the policy took."""
    ### GET ALL DATA ###
    hits = data.get("hits")
    crosses = data.get("crosses")
    if hits is None or crosses is None:
        return
    hits = np.stack(hits)
    crosses = np.stack(crosses)

    cols = min(4, len(gates))
    rows = (len(gates) + cols - 1) // cols
    
    ### PLOT (X,Y) WHERE TRAJ HITS/THREADS GATE ###
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)
    for i, (center, R) in enumerate(zip(gates, rots)):
        ax = axes[i // cols, i % cols]
        ax.add_patch(plt.Rectangle(
            (-half_w, -half_h), 2*half_w, 2*half_h,
            fill=False, edgecolor="black", linewidth=2)
        )
        
        hit = hits[:, i, :]
        hit = hit[~np.isnan(hit).any(axis=1)]
        if len(hit):
            local = (hit - center) @ R
            ax.scatter(local[:,0], local[:,2], c="red", s=30, alpha=0.7)
        
        cross = crosses[:, i, :]
        cross = cross[~np.isnan(cross).any(axis=1)]
        if len(cross):
            local = (cross - center) @ R
            ax.scatter(local[:,0], local[:,2], c="lime", s=25, alpha=0.7)
        
        ax.axhline(0, color="gray", ls="--", alpha=0.4)
        ax.axvline(0, color="gray", ls="--", alpha=0.4)
        ax.set_aspect("equal")
        ax.set_title(f"Gate {i}")
        ax.set_xlim(-half_w*1.3, half_w*1.3)
        ax.set_ylim(-half_h*1.3, half_h*1.3)
    
    for j in range(i+1, rows*cols):
        axes[j // cols, j % cols].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"eval/gate_spread.png", dpi=150, bbox_inches="tight")
    plt.close()

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
    all_data = {"traj": [], "hits": [], "crosses": [], "residual": [], "gt_dist": []}
    for n_roll in range(num_rollouts):
        data = np.load(f"eval/rollout_{n_roll:03d}.npz")
        for k in all_data:
            all_data[k].append(data[k])

    plot_trajectory(half_w, half_h, env.venv.sim_dt, gates, rots, all_data)
    plot_side_trajectory(half_h, env.venv.sim_dt, gates, rots, all_data)
    plot_hits(half_w, half_h, gates, rots, all_data)
    plot_residual(all_data)
        
    if render:
        env.disconnectUnity()