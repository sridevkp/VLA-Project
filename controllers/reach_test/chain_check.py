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

# Zero offset: FK now gives the frame of the last real link (the finger link)
chain = Chain.from_urdf_file(URDF, base_elements=BASE_ELEMENTS,
                             last_link_vector=[0, 0, 0])
for i, l in enumerate(chain.links):
    chain.active_links_mask[i] = l.name in ARM

motors, sensors = {}, {}
for n in ARM + ["torso_lift_joint"]:
    m = robot.getDevice(n)
    s = m.getPositionSensor()
    s.enable(ts)
    motors[n], sensors[n] = m, s

# Open the gripper so the finger gap is visible
for n in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
    robot.getDevice(n).setPosition(0.045)

# Three candidate tool points, expressed in the finger link frame
CANDIDATES = {
    "M_RED":   ([0, 0, 0],        "1 0 0"),   # finger link origin
    "M_GREEN": ([0, 0, -0.1741],  "0 1 0"),   # URDF offset as-is
    "M_BLUE":  ([0, 0,  0.1741],  "0 0 1"),   # opposite sign
}

def get_marker(name, rgb):
    node = robot.getFromDef(name)
    if node is None:
        robot.getRoot().getField("children").importMFNodeFromString(
            -1,
            f"DEF {name} Solid {{ children [ Shape {{ appearance PBRAppearance "
            f"{{ baseColor {rgb} roughness 1 metalness 0 }} "
            f"geometry Sphere {{ radius 0.015 }} }} ] }}")
        node = robot.getFromDef(name)
    return node.getField("translation")

fields = {n: get_marker(n, rgb) for n, (_, rgb) in CANDIDATES.items()}
me = robot.getSelf()

def current_q():
    return [sensors[l.name].getValue() if l.name in sensors else 0.0
            for l in chain.links]

def update_markers():
    T = chain.forward_kinematics(current_q())
    R_b, p_b = T[:3, :3], T[:3, 3]
    R_w = np.array(me.getOrientation()).reshape(3, 3)
    p_w = np.array(me.getPosition())
    for name, (vec, _) in CANDIDATES.items():
        p_base = R_b @ np.array(vec) + p_b
        fields[name].setSFVec3f((R_w @ p_base + p_w).tolist())

step = 0
while robot.step(ts) != -1:
    if step == 150:                      # move the elbow a little
        motors["arm_4_joint"].setPosition(0.8)
    update_markers()
    step += 1