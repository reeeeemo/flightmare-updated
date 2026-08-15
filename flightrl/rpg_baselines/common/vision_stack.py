import numpy as np
import torch
import cv2
from ultralytics import YOLO

class VisionStack:
    def __init__(self, n_envs: int, vision_weights: str):
        self.vision_model = YOLO(vision_weights) if vision_weights else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = 0.5
        self.verbose = False
        self.n_envs = n_envs
        self.enabled = bool(self.vision_model != None)
        
        self.object_points = np.array([  # inner depth: 0.75x0.75
            [-0.75, 0.75, 0], # top left
            [0.75, 0.75, 0], # top right
            [0.75, -0.75, 0], # bottom right
            [-0.75, -0.75, 0] # bottom left
        ], dtype=np.float32)
        
        # fx = 180 = fy since square pixels
        self.camera_matrix = np.array([
            [180, 0, 320],
            [0, 180, 180],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 20 degree tilt upwards
        theta = np.radians(20)
        self.R_body_cam = np.array([
            [1, 0, 0],
            [0, np.sin(theta), np.cos(theta)],
            [0, -np.cos(theta), np.sin(theta)],
        ])
        
        self.prev_gate_crossed = np.zeros((n_envs, 3), dtype=np.float32)
        
    def set_prev_gate(self, env_idxs: np.ndarray, prev_gates: np.ndarray):
        """Sets any environment's previous gate to a vector XYZ."""
        self.prev_gate_crossed[env_idxs] = prev_gates

    def get_gate(self, 
                 frames: np.ndarray, 
                 rotation: np.ndarray, 
                 velocity: np.ndarray,
                 sim_dt: float,
                 cur_gate: np.ndarray,
                 cur_corners: np.ndarray):
        """Computes PnP on a pose estimated frame to get current XYZ + 4 corner XYZs.
        
        Args:
            frames: all frames gathered from camera
            R_world: 9D drone rotation matrix
        Returns:
            n_dets: num envs, num detections
            n_filtered_corners: num envs, num (xyz, 4) corners
            n_filtered_xyz: num envs, num relative xyz to gate
        """
        # propagate prev gate that was crossed by velocity.
        mask = np.any(self.prev_gate_crossed, axis=1)
        self.prev_gate_crossed[mask] -= velocity[mask] * sim_dt

        n_filtered_corners = np.zeros((self.n_envs, 12), dtype=np.float32)
        n_filtered_xyz = np.zeros((self.n_envs, 3), dtype=np.float32)
        
        results = self.vision_model(
            list(frames),
            device=self.device,
            verbose=self.verbose,
            conf=self.confidence_threshold
        )
        
        n_filtered_xyz = cur_gate - (velocity * sim_dt)
        n_filtered_corners = cur_corners - (np.tile(velocity, 4) * sim_dt)
                
        # get most confident gate segmentation mask to transform
        for env_idx, result in enumerate(results):
            if len(result.boxes) == 0:
                continue
                
            best_2 = (result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]).argsort(descending=True)[:2]
            cur_success = False
            for i, best in enumerate(best_2):
                # --------------------
                # GET KEYPOINTS OR DERIVE FROM SEGMENTATION MASKS
                # MUST BE IN FORM [TL, TR, BR, BL]
                # --------------------
                if self.vision_model.task == "pose":
                    kp_2d = result.keypoints.xy[best].cpu().numpy().astype(np.float32)
                else:
                    pts = result.masks.xy[best].astype(np.float32)
                    rect = cv2.minAreaRect(pts)
                    
                    box = cv2.boxPoints(rect)
                    box = box[np.argsort(box[:, 1])]
                    top = box[:2]
                    bot = box[2:]
                    TL, TR = top[np.argsort(top[:, 0])]
                    BL, BR = bot[np.argsort(bot[:, 0])]
                    kp_2d = np.array([TL, TR, BR, BL], dtype=np.float32)
                        
                if len(kp_2d) != 4:
                    continue
                    
                # --------------------
                # PNP TRANSLATION FROM 2D OBJECT POSE TO 3D CAM POS
                # --------------------
                try:
                    success, rot_vec, trans_vec = cv2.solvePnP(
                        self.object_points.reshape(4,1,3), 
                        kp_2d.reshape(4,1,2) if self.vision_model.task == "pose" else kp_2d, 
                        self.camera_matrix, 
                        None,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                except cv2.error:
                    continue

                # check for detection being in front of drone
                if success and trans_vec[2] <= 2:
                    success = False
                    continue
                if success:
                    # transform from 3d camera local pos to drone local pos
                    # first convert points to camera space
                    rot_mat, _ = cv2.Rodrigues(rot_vec)
                    corners_cam = (rot_mat @ self.object_points.T + trans_vec).T
                        
                    # 20 deg camera tilt + 0.3 z axis translations
                    corners_body = (self.R_body_cam @ corners_cam.T).T + np.array([0, 0, 0.3])
                    R_world = rotation[env_idx].reshape(3,3)
                    corners_world = (R_world @ corners_body.T).T
                    
                    cand = corners_world.mean(axis=0)
                        
                    # --------------------
                    # DON'T USE PREV GATE THAT WAS CROSSED
                    # --------------------
                    if (np.any(self.prev_gate_crossed[env_idx]) and 
                        np.linalg.norm(cand - self.prev_gate_crossed[env_idx]) <= 2.5):
                        continue
                        
                    # on success set cur gate
                    if not cur_success:
                        # ----------
                        # SET CUR GATE
                        # ----------
                        n_filtered_xyz[env_idx] = cand.copy()
                        # -----------
                        # SET CUR CORNERS
                        # ----------
                        n_filtered_corners[env_idx] = corners_world[[1, 0, 2, 3]].flatten()
                        cur_success=True

        return n_filtered_corners, n_filtered_xyz
    
    def is_enabled(self):
        return self.enabled