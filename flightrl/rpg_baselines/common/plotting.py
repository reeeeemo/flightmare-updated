import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

class Plotter:
    """Plotter class that allows for quick plotting given environemnet and rollout data.
    
    Attributes:
        data : dictionary of all rollout data
        half_w, half_h : gate dimensions
        sim_dt : simulator's deltatime
        gates : gate positions used for every rollout
        rots : gate rotations used for every rollout
    """
    def __init__(self, gates: list, rotations: list, sim_dt: float, gate_dims: tuple):
        self.data = {}
        self.half_w, self.half_h = gate_dims
        self.gates = gates
        self.rotations = rotations
        self.sim_dt = sim_dt

    def load_data(self, npz: list[str], keys: list[str]):
        """Load data for plotting."""
        if len(npz) <= 0:
            return
        
        ### default keys then load all data into dict
        self.data = {k: [] for k in keys}
        for n in npz:
            n_data = np.load(n)
            for k in self.data:
                self.data[k].append(n_data[k])
        
    def _compute_speeds(self, trajectories: list):
        """Returns speeds of all trajectories.
        
        Computes magnitude of shifts in values per timestep.
        """
        all_speeds = [np.linalg.norm(np.diff(t, axis=0), axis=1) / self.sim_dt for t in trajectories]
        vmax = max((s.max() for s in all_speeds if len(s)), default=1.0)
        norm = Normalize(0, vmax)
        return all_speeds, norm

    def plot_residual(self):
        """Plot difference between visual inference and GT gate values."""
        if "gt_dist" not in self.data or "residual" not in self.data:
            continue

        gt_dist = np.concatenate(self.data.get("gt_dist"))
        residual = np.concatenate(self.data.get("residual"))
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
    
    def plot_trajectory(self):
        """Plot the overall trajectory and gate positions/rotations."""
        traj = self.data.get("traj", None)
        if traj is None:
            return
        fig, ax = plt.subplots(figsize=(10,8))
        
        ### COMPUTE SPEEDS OF TRAJECTORY ###
        all_speeds, norm = self._compute_speeds(traj)

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
        for i, (center, R) in enumerate(zip(self.gates, self.rotations)):
            right = R[:, 0] * self.half_w
            up = R[:, 2] * self.half_h
                
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
        
    def plot_side_trajectory(self):
        """Plot overall attitude-based trajectory of policy + gates."""
        traj = self.data.get("traj")
        if traj is None:
            return
        
        fig, ax = plt.subplots(figsize=(24,3))
        
        ### COMPUTE SPEEDS OF TRAJECTORY ###
        all_speeds, norm = self._compute_speeds(traj)
        
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
        for i, (center, R) in enumerate(zip(self.gates, self.rotations)):
            up = R[:, 2] * self.half_h
            
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

    def plot_hits(self):
        """Plot all gate crosses/hits by the trajectories the policy took."""
        ### GET ALL DATA ###
        hits = self.data.get("hits")
        crosses = self.data.get("crosses")
        if hits is None or crosses is None:
            return
        hits = np.stack(hits)
        crosses = np.stack(crosses)

        cols = min(4, len(self.gates))
        rows = (len(self.gates) + cols - 1) // cols
        
        ### PLOT (X,Y) WHERE TRAJ HITS/THREADS GATE ###
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)
        for i, (center, R) in enumerate(zip(self.gates, self.rotations)):
            ax = axes[i // cols, i % cols]
            ax.add_patch(plt.Rectangle(
                (-self.half_w, -self.half_h), 2*self.half_w, 2*self.half_h,
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
            ax.set_xlim(-self.half_w*1.3, self.half_w*1.3)
            ax.set_ylim(-self.half_h*1.3, self.half_h*1.3)
        
        for j in range(i+1, rows*cols):
            axes[j // cols, j % cols].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"eval/gate_spread.png", dpi=150, bbox_inches="tight")
        plt.close()