"""Quick FK / gripper position check for TIAGo in Webots (no arm motion, only torso + fingers).

Set this as the robot's controller (next to tiago.urdf) and read the console while looking at
the 3D view. Markers (all spheres, updated every step):
    blue    = arm_7 frame origin          (FK)
    yellow  = wrist_ft_tool_link frame    (FK)
    magenta = left finger link origin     (FK)
    green   = tool point incl. TOOL_OFFSET (FK, same as T_TOOL in the main controller)
    white   = robot base frame origin     (from Webots, sanity check of the base frame)
"""
from controller import Supervisor, Field
from ikpy.chain import Chain
import numpy as np, os, warnings
warnings.filterwarnings("ignore", message=".*is of type 'fixed'.*")

TOOL_OFFSET      = [0, 0, -0.1741]     # keep equal to the main controller
TORSO_HEIGHT     = 0.35
FINGERS_INVERTED = True                # True: 0.0 = open, 0.045 = closed
FULL_STROKE      = 0.045



robot = Supervisor()
ts = int(robot.getBasicTimeStep())
URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiago.urdf")
BASE_ELEMENTS = [
    "base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
    "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm",
]
ARM = [f"arm_{i}_joint" for i in range(1, 8)]
chain = Chain.from_urdf_file(URDF, base_elements=BASE_ELEMENTS, last_link_vector=TOOL_OFFSET)
for i, l in enumerate(chain.links):
    chain.active_links_mask[i] = l.name in ARM
idx = {l.name: i for i, l in enumerate(chain.links)}
print("Chain links:", [l.name for l in chain.links])

OPEN_POS, CLOSED_POS = (0.0, FULL_STROKE) if FINGERS_INVERTED else (FULL_STROKE, 0.0)

# ---------- devices ----------
sensors, motors = {}, {}
for n in ARM + ["torso_lift_joint"]:
    m = robot.getDevice(n); s = m.getPositionSensor(); s.enable(ts)
    motors[n], sensors[n] = m, s
fingers = []
for n in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
    m = robot.getDevice(n)
    if m is None:
        continue
    s = m.getPositionSensor(); s.enable(ts); m.setVelocity(0.03)
    fingers.append((m, s))
    sensors[n] = s                      # so the FK also sees the finger joint
me = robot.getSelf()
robot.step(ts)
_R0 = np.array(me.getOrientation()).reshape(3, 3)
print("Robot world position:", np.round(me.getPosition(), 3))
print("Robot yaw (deg):", round(float(np.degrees(np.arctan2(_R0[1, 0], _R0[0, 0]))), 1),
      "| up-vector (expect ~[0 0 1]):", np.round(_R0[:, 2], 3))
print("Joint sensors at startup:", {n: round(s.getValue(), 3) for n, s in sensors.items()})

def base_to_world(p):
    R = np.array(me.getOrientation()).reshape(3, 3)
    return R @ np.asarray(p) + np.array(me.getPosition())

def world_to_base(p):
    R = np.array(me.getOrientation()).reshape(3, 3)
    return R.T @ (np.asarray(p) - np.array(me.getPosition()))

def current_q():
    return [sensors[l.name].getValue() if l.name in sensors else 0.0 for l in chain.links]

def fk_all():
    return chain.forward_kinematics(current_q(), full_kinematics=True)

# ---------- markers ----------
def marker(name, rgb, radius):
    node = robot.getFromDef(name)
    if node is None:
        robot.getRoot().getField("children").importMFNodeFromString(
            -1, f"DEF {name} Solid {{ children [ Shape {{ appearance PBRAppearance "
                f"{{ baseColor {rgb} roughness 1 metalness 0 }} geometry Sphere {{ radius {radius} }} }} ] }}")
        node = robot.getFromDef(name)
    return node.getField("translation")

# find the FK links we care about (ikpy names links after joints)
def find_link(*cands):
    for c in cands:
        if c in idx:
            return c
    return None

L_ARM7   = find_link("arm_7_joint")
L_WRIST  = find_link("arm_7_link_wrist_ft_tool_link_joint")
L_FINGER = find_link("gripper_left_finger_joint", "gripper_right_finger_joint")
print("Using links -> arm7:", L_ARM7, "| wrist:", L_WRIST, "| finger:", L_FINGER)

mk = {}
if L_ARM7:   mk["arm7"] = (marker("C_ARM7", "0 0 1", 0.012), lambda T: T[idx[L_ARM7]][:3, 3])
if L_WRIST:  mk["wrist"] = (marker("C_WRIST", "1 1 0", 0.012), lambda T: T[idx[L_WRIST]][:3, 3])
if L_FINGER: mk["finger"] = (marker("C_FINGER", "1 0 1", 0.012), lambda T: T[idx[L_FINGER]][:3, 3])
mk["tool"] = (marker("T_TOOL", "0 1 0", 0.015), lambda T: T[-1][:3, 3])
base_marker = marker("C_BASE", "1 1 1", 0.02)

def update_markers():
    T = fk_all()
    for key, (field, getp) in mk.items():
        field.setSFVec3f(base_to_world(getp(T)).tolist())
    base_marker.setSFVec3f(list(me.getPosition()))

def wait(n):
    for _ in range(n):
        if robot.step(ts) == -1:
            return False
        update_markers()
    return True

# ---------- optional: real solids (only works if the PROTO internals are reachable) ----------
def find_solid_by_name(node, name, depth=0):
    if node is None or depth > 40:
        return None
    nf = node.getField("name")
    if nf is not None and nf.getTypeName() == "SFString" and nf.getSFString() == name:
        return node
    for i in range(node.getNumberOfFields()):
        f = node.getFieldByIndex(i)
        if f.getType() == Field.SF_NODE:
            r = find_solid_by_name(f.getSFNode(), name, depth + 1)
        elif f.getType() == Field.MF_NODE:
            r = None
            for k in range(f.getCount()):
                r = find_solid_by_name(f.getMFNode(k), name, depth + 1)
                if r is not None:
                    break
        else:
            continue
        if r is not None:
            return r
    return None

# ---------- 1. torso + report ----------
motors["torso_lift_joint"].setPosition(TORSO_HEIGHT)
for _ in range(1000):
    if robot.step(ts) == -1:
        break
    if abs(sensors["torso_lift_joint"].getValue() - TORSO_HEIGHT) < 0.002:
        break
wait(10)

def report(title):
    T = fk_all()
    print(f"\n===== {title} =====")
    print("finger sensors:", [round(s.getValue(), 4) for _, s in fingers])
    print(f"{'link':45s} {'pos (base frame)':28s} z-axis (base)")
    for l in chain.links:
        M = T[idx[l.name]]
        print(f"{l.name[:44]:45s} {str(np.round(M[:3, 3], 3)):28s} {np.round(M[:3, 2], 2)}")
    if L_ARM7 and L_FINGER:
        za = T[idx[L_ARM7]][:3, 2]; zf = T[idx[L_FINGER]][:3, 2]
        ang = np.degrees(np.arccos(np.clip(za @ zf, -1, 1)))
        print(f"angle between arm_7 z-axis and finger-link z-axis: {ang:.1f} deg "
              "(~0 => TOOL_OFFSET z is along the approach axis; otherwise the offset points SIDEWAYS)")
        print("tool-axis (-Z of finger frame) in base frame:", np.round(-zf, 3),
              "| (-Z of arm_7 frame):", np.round(-za, 3))
    return T

T_open = None
for m, _ in fingers:
    m.setPosition(OPEN_POS)
wait(int(3000 / ts))
T_open = report("FINGERS OPEN (commanded %.3f)" % OPEN_POS)

# ---------- 2. real vs FK (if reachable) ----------
pairs = [(L_ARM7, "arm_7_link"), (L_WRIST, "wrist_ft_tool_link"),
         (L_FINGER, L_FINGER.replace("_joint", "_link") if L_FINGER else None)]
_front = find_link("wrist_ft_tool_link_front_joint")
if _front:
    _d = T_open[-1][:3, 3] - T_open[idx[_front]][:3, 3]
    print(f"\nTool point is {np.linalg.norm(_d):.3f} m from the 'front' frame, direction {np.round(_d / (np.linalg.norm(_d) + 1e-9), 3)} "
          "(this is the approach axis the controller assumes; the gripper should physically point that way)")
print("\n===== REAL (Webots) vs FK =====")
found_any = False
for link, solid_name in pairs:
    if link is None:
        continue
    node = find_solid_by_name(robot.getRoot(), solid_name)
    if node is None:
        print(f"{solid_name}: solid not reachable from the Supervisor")
        continue
    found_any = True
    real_b = world_to_base(node.getPosition())
    fk_b = T_open[idx[link]][:3, 3]
    print(f"{solid_name}: real {np.round(real_b, 3)}  fk {np.round(fk_b, 3)}  diff {np.round(real_b - fk_b, 3)}")
if not found_any:
    print("None reachable (TIAGo is a PROTO). To enable: scene tree -> right-click the robot -> "
          "'Convert Root to Base Nodes', save the world, rerun. Until then compare the markers by eye.")

# ---------- 3. finger travel -> lateral TOOL_OFFSET correction ----------
for m, _ in fingers:
    m.setPosition(CLOSED_POS)
wait(int(3000 / ts))
T_closed = report("FINGERS CLOSED (commanded %.3f)" % CLOSED_POS)

if L_FINGER:
    p_open = T_open[idx[L_FINGER]][:3, 3]
    p_closed = T_closed[idx[L_FINGER]][:3, 3]
    d_base = p_closed - p_open
    R_open = T_open[idx[L_FINGER]][:3, :3]
    d_local = R_open.T @ d_base
    print("\n===== FINGER TRAVEL (left finger link origin, closed - open) =====")
    print("in base frame :", np.round(d_base, 4), "| length", round(float(np.linalg.norm(d_base)), 4), "m")
    print("in finger frame:", np.round(d_local, 4))
    if np.linalg.norm(d_base) < 0.01:
        print("[!] the finger link barely moves in the FK: its joint is probably not in the chain "
              "(treated as fixed). Then the stroke direction must be measured by eye.")
    else:
        sug = np.array(TOOL_OFFSET) + d_local
        print("TOOL_OFFSET that puts the tool point where the left finger sits when CLOSED "
              "(~ the gripper centre plane):")
        print("   TOOL_OFFSET =", np.round(sug, 4).tolist())
        print("   (the finger-frame direction of the stroke is where the lateral error lives; "
              "check the green ball against the fingertips afterwards)")

print("\nMarkers: blue=arm_7, yellow=wrist_ft, magenta=left finger link, green=tool point, white=base.")
print("Check whether the green ball sits between the fingertips. Idle loop running...")
while robot.step(ts) != -1:
    update_markers()