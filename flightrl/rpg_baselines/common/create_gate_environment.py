import numpy as np

RANDOM_X_RANGE = (-5, 5)
RANDOM_Y_RANGE = [(6, 7), (8, 10), (12, 25)]
RANDOM_Z_RANGE = [(1, 2), (2, 3), (2, 3)]

def create_gates(n_gates: int = 1, phase: int = 1):
    """Create an environment with 3D gates.
    
    Utilizes a course layout in an S shape, turns capped at 40 degs.
    If phase is 1, lays out forward facing gates with 
    minimum dist between.
    
    Args:
        n_gates: number of gates to place
        phase: complexity of course (1, 2, 3)
    Returns:
        (positions, rotations): pos/rot of all gates placed
    """
    positions = np.zeros((n_gates, 3), dtype=np.float32)
    rotations = np.zeros((n_gates, 4), dtype=np.float32)
    rotations[:, 0] = 1  # quaternion w - identity default :p
    if phase not in (1, 2, 3):
        return positions, rotations

    # ---------------------
    # GATE CREATION VARIABLES
    # ---------------------
    flat_probability = [0.8, 0.6, 0.4][phase-1]  # z = prev_z or not
    random_x_range = RANDOM_X_RANGE
    random_y_range = RANDOM_Y_RANGE[phase-1]
    random_z_range = RANDOM_Z_RANGE[phase-1]
    
    # ---------------------
    # GATE ROTATION VARIABLES 
    #---------------------
    max_gate_turn = np.radians(40)
    dev_frac = 0.85  # shrinks leg angle and course width for gentler courses
    max_cross_w = np.random.uniform(45.0, 60.0) * dev_frac
    max_bound = max_cross_w / 2 * np.random.uniform(0.85, 1.0)

    rand_heading = np.radians(np.random.uniform(55.0, 62.0)) * dev_frac
    cur_heading = rand_heading * (1.0 if np.random.random() < 0.5 else -1.0)
    original_heading = cur_heading
    headings = np.zeros(n_gates)
    
    # ---------------------
    # RANDOMIZE POSITIONS
    # ---------------------
    for i in range(n_gates):
        old_pos_x = positions[i-1, 0] if i-1 >= 0 else 0
        old_pos_y = positions[i-1, 1] if i-1 >= 0 else 0'
        if phase == 1:
            # ---------- RANDOM X, Y RANGE ----------
            prev_dx = positions[i-1, 0] - positions[i-2, 0] if i >= 2 else 0
            # x: keep x close to old position and smooth so no insane angles drone has to cross
            # y: always close but not intersecting/too close
            positions[i, 0] = old_pos_x + prev_dx * 0.4 + np.random.uniform(*random_x_range)
            positions[i, 1] = old_pos_y + np.random.uniform(*random_y_range)
        else:
            # ---------- RANDOM X, Y RANGE IN S SHAPE ----------
            cur_heading += np.clip(original_heading-cur_heading, -max_gate_turn, max_gate_turn)
            rand_noise = np.random.uniform(6.5, 8.0)
            positions[i, 0] = old_pos_x + (rand_noise * np.sin(cur_heading))
            positions[i, 1] = old_pos_y + (rand_noise * np.cos(cur_heading))
            headings[i] = cur_heading  # hold for future yaw rot
            
            # walks diagonally until max bound, then flips and cur head ramps down
            if (
                abs(cur_heading - original_heading) < 1e-3 and 
                 (
                    (original_heading > 0 and positions[i, 0] >= max_bound) or 
                    (original_heading < 0 and positions[i, 0] <= -max_bound)
                 )
                ):
                original_heading = -original_heading
                max_bound = max_cross_w / 2 * np.random.uniform(0.85, 1.0)

        # ----------- RANDOM Z RANGE (flat or not) ------------
        old_pos_z = positions[i-1, 2] if i-1 >= 0 else 2
        random_z = np.random.uniform(*random_z_range)
        
        if np.random.random() < flat_probability:
            positions[i, 2] = old_pos_z
        else:
            positions[i, 2] = np.random.uniform(old_pos_z-random_z, old_pos_z+random_z)
    
    # ---------------------
    # RANDOMIZE ROTATIONS
    # ---------------------
    # only rotates yaw, leaning each gate partway towards the nxt gate 
    if phase != 1:
        for i in range(n_gates):
            next_rot = headings[i+1] if i+1 < n_gates else headings[i]
            rot_coef = headings[i] + np.random.uniform(0.35, 0.7) * (next_rot - headings[i])
            
            half = (-rot_coef + np.radians(np.random.uniform(-6, 6))) / 2
            rotations[i, 0] = np.cos(half)  # w
            rotations[i, 3] = np.sin(half)  # yaw
    
    return positions, rotations