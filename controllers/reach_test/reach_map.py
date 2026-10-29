"""Run in a normal terminal (no Webots) from the folder containing tiago.urdf:
    python reach_map.py
Prints, for several torso heights, which pregrasp points allow a straight-down gripper.
Each cell = best position error in cm; 'OK' = error < 1.5 cm and tilt < 15 deg."""
import numpy as np, os, warnings
from ikpy.chain import Chain
warnings.filterwarnings("ignore")

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiago.urdf")
BASE_ELEMENTS = [
    "base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
    "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm",
]
ARM = [f"arm_{i}_joint" for i in range(1, 8)]
TOOL_OFFSET = [0, 0, -0.1741]

Y = -0.105                                  # lateral position of your can (base frame)
XS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]  # forward distances to test
ZS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05]  # pregrasp heights to test (base frame)
TORSOS = [0.0, 0.15, 0.35]                 # torso lift heights to test (m)
RESTARTS = 8

chain = Chain.from_urdf_file(URDF, base_elements=BASE_ELEMENTS, last_link_vector=TOOL_OFFSET)
for i, l in enumerate(chain.links):
    chain.active_links_mask[i] = l.name in ARM
idx = {l.name: i for i, l in enumerate(chain.links)}
rng = np.random.default_rng(0)

def bounds(n):
    lo, hi = chain.links[idx[n]].bounds
    lo = -np.pi if lo is None or not np.isfinite(lo) else lo
    hi = np.pi if hi is None or not np.isfinite(hi) else hi
    return lo, hi

def best_topdown(p, torso):
    best = (1e9, 0.0)
    for k in range(RESTARTS):
        s = np.zeros(len(chain.links))
        s[idx["torso_lift_joint"]] = torso
        if k > 0:
            for n in ARM:
                s[idx[n]] = rng.uniform(*bounds(n))
        q = chain.inverse_kinematics(p, target_orientation=[0, 0, 1],
                                     orientation_mode="Z", initial_position=s)
        T = chain.forward_kinematics(q)
        err = np.linalg.norm(T[:3, 3] - p)
        tool = T[:3, :3] @ np.array([0, 0, -1.0])
        tilt = np.degrees(np.arccos(np.clip(-tool[2], -1, 1)))
        if err + 0.002 * tilt < best[0] + 0.002 * best[1]:
            best = (err, tilt)
        if err < 0.005 and tilt < 5:
            break
    return best

for t in TORSOS:
    print(f"\n=== torso = {t:.2f} m   (y = {Y}) ===")
    print("  z \\ x " + "".join(f"{x:>8.2f}" for x in XS))
    for z in ZS:
        row = []
        for x in XS:
            err, tilt = best_topdown(np.array([x, Y, z]), t)
            row.append("OK" if err < 0.015 and tilt < 15 else f"{err*100:.0f}cm")
        print(f"  {z:5.2f} " + "".join(f"{c:>8}" for c in row))