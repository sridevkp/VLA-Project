from controller import Supervisor
from ikpy.chain import Chain
import numpy as np, os

robot = Supervisor()
ts = int(robot.getBasicTimeStep())
URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiago.urdf")

BASE_ELEMENTS = [
    "base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
    "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm",
]
ARM = [f"arm_{i}_joint" for i in range(1, 8)]
TOOL_OFFSET = [0, 0, -0.1741]              # confirmed by the marker test

# Target in the robot base frame (x forward, y left, z up). Adjust after the first run.
TARGET_BASE = np.array([0.55, -0.20, 0.95])

chain = Chain.from_urdf_file(URDF, base_elements=BASE_ELEMENTS,
                             last_link_vector=TOOL_OFFSET)
for i, l in enumerate(chain.links):
    chain.active_links_mask[i] = l.name in ARM
idx = {l.name: i for i, l in enumerate(chain.links)}

motors, sensors = {}, {}
for n in ARM + ["torso_lift_joint"]:
    m = robot.getDevice(n)
    s = m.getPositionSensor()
    s.enable(ts)
    motors[n], sensors[n] = m, s
for n in ARM:
    motors[n].setVelocity(0.5)             # slow and safe for the first test
robot.step(ts)

me = robot.getSelf()

def base_to_world(p_base):
    R = np.array(me.getOrientation()).reshape(3, 3)
    p = np.array(me.getPosition())
    return R @ np.asarray(p_base) + p

def current_q():
    return [sensors[l.name].getValue() if l.name in sensors else 0.0
            for l in chain.links]

def marker(name, rgb):
    node = robot.getFromDef(name)
    if node is None:
        robot.getRoot().getField("children").importMFNodeFromString(
            -1,
            f"DEF {name} Solid {{ children [ Shape {{ appearance PBRAppearance "
            f"{{ baseColor {rgb} roughness 1 metalness 0 }} "
            f"geometry Sphere {{ radius 0.02 }} }} ] }}")
        node = robot.getFromDef(name)
    return node.getField("translation")

target_marker = marker("T_TARGET", "1 0 0")    # red: where we want the tool
tool_marker   = marker("T_TOOL",   "0 1 0")    # green: where FK says the tool is

def update_markers():
    target_marker.setSFVec3f(base_to_world(TARGET_BASE).tolist())
    tool_marker.setSFVec3f(
        base_to_world(chain.forward_kinematics(current_q())[:3, 3]).tolist())

def move_to_q(q, tol=0.01, max_steps=800):
    for n in ARM:
        motors[n].setPosition(float(q[idx[n]]))
    for _ in range(max_steps):
        if robot.step(ts) == -1:
            return False
        update_markers()
        if max(abs(sensors[n].getValue() - q[idx[n]]) for n in ARM) < tol:
            return True
    return False

# ---- Solve IK once ----
q_ik = chain.inverse_kinematics(TARGET_BASE, initial_position=current_q())
fk_pred = chain.forward_kinematics(q_ik)[:3, 3]
print("Target (base frame):   ", np.round(TARGET_BASE, 4))
print("IK predicted position: ", np.round(fk_pred, 4))
print("IK error (m):          ", round(float(np.linalg.norm(fk_pred - TARGET_BASE)), 4))
print("IK joint solution:")
for n in ARM:
    print(f"  {n}: {q_ik[idx[n]]: .3f}")

# ---- Execute ----
reached = move_to_q(q_ik)
fk_real = chain.forward_kinematics(current_q())[:3, 3]
print("\nReached joint targets:", reached)
print("Actual tool position (from sensors):", np.round(fk_real, 4))
print("Execution error (m):", round(float(np.linalg.norm(fk_real - TARGET_BASE)), 4))

# Keep the markers updating
while robot.step(ts) != -1:
    update_markers()