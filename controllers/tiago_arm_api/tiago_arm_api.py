from controller import Supervisor, Field
from ikpy.chain import Chain
import numpy as np, os, warnings
warnings.filterwarnings("ignore", message=".*is of type 'fixed'.*")

# =====================  CONFIG  =====================
TIME_STEP = 32
CAN_DEF         = "CAN"            # add `DEF CAN ...` to your can in the world file
PLACE_BASE      = np.array([-1.07, -3.9, 0.8])    # where the CAN CENTRE should end up (base frame)
GRASP_Z         = 0.02             # tool point height above can centre when grasping (m)
APPROACH_H      = 0.12             # hover height above grasp/place point (m)
PLACE_CLEAR     = 0.01             # extra gap when setting the can down (m)
IK_TOL          = 0.015            # IK tolerance for large/free moves (m)
IK_TOL_GRASP    = 0.005            # tight IK tolerance for descend/lift/place moves (m)
ALIGN_TOL       = 0.005            # max can-vs-tool x,y error allowed before closing (m)
SKIP_ALIGN_CHECK = True            # True: only warn about misalignment and close anyway
MAX_TILT_DEG    = 15               # max allowed deviation of tool from straight-down
GRIP_FORCE      = None             # N, force limit used ONLY while closing (None = default)
MIN_GRASP_OPEN  = 0.004            # fingers stalled wider than this => something is held
LIFT_MIN        = 0.04             # can must rise at least this much to count as grasped (m)
MAX_RETRIES     = 2
STOP_AFTER      = None          # halt after this state while debugging, e.g. "INIT", "PREGRASP",
                                   # "DESCEND", "CLOSE". Use None for a full run.
TORSO_HEIGHT    = 0.35             # raise the torso first (m); None = leave as is
FINGER_SPEED    = 0.03             # m/s
CALIBRATE_TOOL  = False            # needs finger solids reachable from the Supervisor (not the case
                                   # while TIAGo is still a PROTO), so it is off by default
FINGERS_INVERTED = False            # True: joint value 0.0 = fingers OPEN, 0.045 = CLOSED.
                                   # False: 0.0 = closed, 0.045 = open. Verify in the 3D view.
GRIPPER_TEST    = False            # after the torso lift, cycle the fingers close/open/close/open
                                   # (3 s each) inside THIS controller and then stop. Set False after.


# ====================================================
robot = Supervisor()
ts = int(robot.getBasicTimeStep())
URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiago.urdf")

BASE_ELEMENTS = [
    "base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
    "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm",
]
ARM = [f"arm_{i}_joint" for i in range(1, 8)]
TOOL_OFFSET = [0, 0, -0.1741]

chain = Chain.from_urdf_file(URDF, base_elements=BASE_ELEMENTS,
                             last_link_vector=TOOL_OFFSET)
for i, l in enumerate(chain.links):
    chain.active_links_mask[i] = l.name in ARM
idx = {l.name: i for i, l in enumerate(chain.links)}

# ---------- devices ----------
motors, sensors = {}, {}
for n in ARM + ["torso_lift_joint"]:
    m = robot.getDevice(n)
    s = m.getPositionSensor()
    s.enable(ts)
    motors[n], sensors[n] = m, s
for n in ARM:
    motors[n].setVelocity(0.5)

fingers = []                                   # list of (motor, sensor)
for n in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
    m = robot.getDevice(n)
    if m is None:
        print(f"[warn] finger motor '{n}' not found")
        continue
    s = m.getPositionSensor()
    s.enable(ts)
    m.setVelocity(FINGER_SPEED)
    fingers.append((m, s))
print("Fingers found:", len(fingers))
_maxes = [m.getMaxPosition() for m, _ in fingers]
OPEN_POS = min([x for x in _maxes if x > 0] + [0.045])   # max==0 means "no limit set"
print("Finger limits:", [(round(m.getMinPosition(), 3), round(m.getMaxPosition(), 3))
                         for m, _ in fingers], "-> OPEN_POS =", OPEN_POS)

FULL_STROKE = OPEN_POS
if FINGERS_INVERTED:
    OPEN_POS, CLOSED_POS = 0.0, FULL_STROKE
else:
    CLOSED_POS = 0.0
print(f"Gripper mapping: OPEN_POS={OPEN_POS}, CLOSED_POS={CLOSED_POS} "
      f"(FINGERS_INVERTED={FINGERS_INVERTED})")

me = robot.getSelf()

can_node = robot.getFromDef(CAN_DEF)
if can_node is None:
    raise RuntimeError(f"No node with DEF {CAN_DEF} in the world.")
robot.step(ts)


# Cameras
rgb_camera = robot.getDevice("Astra rgb")
depth_camera = robot.getDevice("Astra depth")

rgb_camera.enable(TIME_STEP)
depth_camera.enable(TIME_STEP)

# ---------- frames ----------
def base_to_world(p_base):
    R = np.array(me.getOrientation()).reshape(3, 3)
    return R @ np.asarray(p_base) + np.array(me.getPosition())

def world_to_base(p_world):
    R = np.array(me.getOrientation()).reshape(3, 3)
    return R.T @ (np.asarray(p_world) - np.array(me.getPosition()))

def current_q():
    return [sensors[l.name].getValue() if l.name in sensors else 0.0
            for l in chain.links]

def tool_pos():
    return chain.forward_kinematics(current_q())[:3, 3]

# ---------- debug markers ----------
def marker(name, rgb):
    node = robot.getFromDef(name)
    if node is None:
        robot.getRoot().getField("children").importMFNodeFromString(
            -1,
            f"DEF {name} Solid {{ children [ Shape {{ appearance PBRAppearance "
            f"{{ baseColor {rgb} roughness 1 metalness 0 }} "
            f"geometry Sphere {{ radius 0.015 }} }} ] }}")
        node = robot.getFromDef(name)
    return node.getField("translation")

target_marker = marker("T_TARGET", "1 0 0")
tool_marker = marker("T_TOOL", "0 1 0")
current_target = [None]

def update_markers():
    if current_target[0] is not None:
        target_marker.setSFVec3f(base_to_world(current_target[0]).tolist())
    tool_marker.setSFVec3f(base_to_world(tool_pos()).tolist())

def wait(n):
    for _ in range(n):
        if robot.step(ts) == -1:
            return False
        update_markers()
    return True

# ---------- tool offset calibration (print only) ----------
def find_solid_by_name(node, name, depth=0, max_depth=40):
    """Recursively search the scene tree for a node whose `name` field equals `name`."""
    if node is None or depth > max_depth:
        return None
    nf = node.getField("name")
    if nf is not None and nf.getTypeName() == "SFString" and nf.getSFString() == name:
        return node
    for i in range(node.getNumberOfFields()):
        f = node.getFieldByIndex(i)
        t = f.getType()
        if t == Field.SF_NODE:
            r = find_solid_by_name(f.getSFNode(), name, depth + 1, max_depth)
            if r is not None:
                return r
        elif t == Field.MF_NODE:
            for k in range(f.getCount()):
                r = find_solid_by_name(f.getMFNode(k), name, depth + 1, max_depth)
                if r is not None:
                    return r
    return None

def calibrate_tool_offset():
    try:
        root = robot.getRoot()
        left = find_solid_by_name(root, "gripper_left_finger_link")
        right = find_solid_by_name(root, "gripper_right_finger_link")
        if left is None or right is None:
            print("[calib] finger solids not found by name "
                  f"(left={left is not None}, right={right is not None}); skipping.")
            return
        p_true = world_to_base((np.array(left.getPosition()) +
                                np.array(right.getPosition())) / 2)
        T = chain.forward_kinematics(current_q())
        err_local = T[:3, :3].T @ (p_true - T[:3, 3])   # last_link_vector is in the last link's frame
        print("[calib] true finger centre (base):", np.round(p_true, 4))
        print("[calib] FK tool point (base):     ", np.round(T[:3, 3], 4))
        print("[calib] suggested TOOL_OFFSET:    ", np.round(np.array(TOOL_OFFSET) + err_local, 4),
              "(trust x,y; keep z by eye)")
    except Exception as e:
        print("[calib] failed:", e)

# ---------- arm motion ----------
def move_to_q(q, tol=0.01, max_steps=800):
    for n in ARM:
        motors[n].setPosition(float(q[idx[n]]))
    for _ in range(max_steps):
        if robot.step(ts) == -1:
            return False
        update_markers()
        if max(abs(sensors[n].getValue() - q[idx[n]]) for n in ARM) < tol:
            return True
    worst = max(ARM, key=lambda n: abs(sensors[n].getValue() - q[idx[n]]))
    print(f"  [!] move timed out (likely a collision). Worst joint: {worst}, "
          f"error {abs(sensors[worst].getValue() - q[idx[worst]]):.3f} rad")
    return False

def _bounds(n):
    lo, hi = chain.links[idx[n]].bounds
    lo = -np.pi if lo is None or not np.isfinite(lo) else lo
    hi = np.pi if hi is None or not np.isfinite(hi) else hi
    return lo, hi

def _clip(q):
    q = np.array(q, float)
    for n in ARM:
        lo, hi = _bounds(n)
        q[idx[n]] = np.clip(q[idx[n]], lo, hi)
    return q

def _score(p_base, q):
    T = chain.forward_kinematics(q)
    err = float(np.linalg.norm(T[:3, 3] - p_base))
    tool_dir = T[:3, :3] @ np.array([0, 0, -1.0])
    tilt = float(np.degrees(np.arccos(np.clip(-tool_dir[2], -1, 1))))
    return err, tilt

_rng = np.random.default_rng(1)

def solve(p_base, seed, restarts=False, n_random=20):
    """Position + tool-pointing-down IK. Tool direction = -Z of the arm_7 frame,
    so we ask the frame's +Z to point up. With restarts=True, tries the
    position-only solution and random in-limit seeds and keeps the best."""
    seed = _clip(seed)
    seeds = [seed]
    if restarts:
        seeds.append(_clip(chain.inverse_kinematics(p_base, initial_position=seed)))
        for _ in range(n_random):
            s = seed.copy()
            for n in ARM:
                lo, hi = _bounds(n)
                s[idx[n]] = _rng.uniform(lo, hi)
            seeds.append(s)
    best = None
    for s in seeds:
        q = chain.inverse_kinematics(p_base, target_orientation=[0, 0, 1],
                                     orientation_mode="Z", initial_position=s)
        err, tilt = _score(p_base, q)
        cost = err + 0.002 * tilt
        if best is None or cost < best[0]:
            best = (cost, q, err, tilt)
        if err < 0.005 and tilt < 5:
            break
    return best[1], best[2], best[3]

def go_to(p_base, tol=0.01, ik_tol=IK_TOL):
    """Joint-space move (use for big moves where the orientation must change)."""
    current_target[0] = np.asarray(p_base)
    update_markers()
    q, err, tilt = solve(p_base, current_q(), restarts=True)
    print(f"  IK -> {np.round(p_base, 3)}  err={err:.4f} m  tilt={tilt:.1f} deg")
    if err > ik_tol or tilt > MAX_TILT_DEG:
        print("  [!] IK rejected (unreachable or tilted)")
        return False
    return move_to_q(q, tol=tol)

def move_linear(p_to, step=0.03, ik_tol=IK_TOL_GRASP):
    """Straight-line Cartesian move with IK seeded from the previous waypoint."""
    p_from = tool_pos()
    p_to = np.asarray(p_to)
    current_target[0] = p_to
    update_markers()
    n = max(2, int(np.linalg.norm(p_to - p_from) / step))
    q = current_q()
    for k, a in enumerate(np.linspace(0, 1, n + 1)[1:], 1):
        p = (1 - a) * p_from + a * p_to
        q, err, tilt = solve(p, q)
        if err > ik_tol or tilt > MAX_TILT_DEG:
            print(f"  [!] linear move failed at waypoint {k}/{n}: err={err:.4f} tilt={tilt:.1f}")
            return False
        last = (k == n)
        if not move_to_q(q, tol=0.01 if last else 0.03, max_steps=600):
            return False
    return True

# ---------- gripper ----------
def finger_values():
    return [round(s.getValue(), 4) for _, s in fingers]

def finger_opening():
    """Aperture: distance of the fingers from the CLOSED position (0 = closed)."""
    return float(np.mean([abs(s.getValue() - CLOSED_POS) for _, s in fingers]))

def open_gripper(timeout_s=4.0, cycle=False):
    """Open both fingers; wait (time-based) until EACH finger reaches OPEN_POS.
    cycle=True first closes the fingers, then opens them. At startup the sensors can
    already read OPEN_POS while the fingers are physically closed, so a plain
    setPosition(OPEN_POS) commands no motion at all."""
    if cycle:
        for m, _ in fingers:
            m.setAvailableForce(m.getMaxForce())
            m.setVelocity(FINGER_SPEED)
            m.setPosition(CLOSED_POS)
        for _ in range(int(timeout_s * 1000 / ts)):
            if robot.step(ts) == -1:
                return False
            update_markers()
            if all(abs(s.getValue() - CLOSED_POS) < 0.001 for _, s in fingers):
                break
    for m, _ in fingers:
        m.setAvailableForce(m.getMaxForce())      # restore default force (undo grip limit)
        m.setVelocity(FINGER_SPEED)
        m.setPosition(OPEN_POS)
    for _ in range(int(timeout_s * 1000 / ts)):
        if robot.step(ts) == -1:
            return False
        update_markers()
        if all(abs(s.getValue() - OPEN_POS) < 0.001 for _, s in fingers):
            break
    vals = finger_values()
    ok = all(abs(v - OPEN_POS) < 0.002 for v in vals)
    print(f"  fingers {vals} (target {OPEN_POS}) -> {'OPEN' if ok else 'NOT OPEN'}")
    return ok

def close_gripper():
    """Close until the fingers stall. Returns (grasped, opening)."""
    for m, _ in fingers:
        if GRIP_FORCE is not None:
            m.setAvailableForce(GRIP_FORCE)       # limit force only while gripping
        m.setPosition(CLOSED_POS)
    prev, still = finger_opening(), 0
    for i in range(300):
        if robot.step(ts) == -1:
            return False, 0.0
        update_markers()
        cur = finger_opening()
        still = still + 1 if abs(cur - prev) < 2e-5 else 0
        prev = cur
        if i > 10 and still >= 15:
            break
    opening = finger_opening()
    print(f"  fingers stalled at opening={opening:.4f} m")
    return opening > MIN_GRASP_OPEN, opening

# ---------- helpers ----------
def can_base_pos():
    return world_to_base(can_node.getPosition())

if TORSO_HEIGHT is not None:
    motors["torso_lift_joint"].setPosition(TORSO_HEIGHT)
    for _ in range(1000):
        if robot.step(ts) == -1:
            break
        if abs(sensors["torso_lift_joint"].getValue() - TORSO_HEIGHT) < 0.002:
            break
    print("Torso at", round(sensors["torso_lift_joint"].getValue(), 3))

HEAD_PAN, HEAD_TILT = 0.0, -0.77      # rad: pan left/right, tilt up/down (tune to taste)

head_pan = robot.getDevice("head_1_joint")
head_tilt = robot.getDevice("head_2_joint")
for m, target in ((head_pan, HEAD_PAN), (head_tilt, HEAD_TILT)):
    s = m.getPositionSensor(); s.enable(ts)
    lo, hi = m.getMinPosition(), m.getMaxPosition()
    if hi > lo:                          # min == max == 0 means no limits set
        target = min(max(target, lo), hi)
    m.setVelocity(0.5)
    m.setPosition(target)
wait(int(2000 / ts))                     # let the head settle before the arm moves

print("Finger sensors before any finger command:", finger_values())

if GRIPPER_TEST:
    for label, target in (("CLOSE", CLOSED_POS), ("OPEN", OPEN_POS), ("CLOSE", CLOSED_POS), ("OPEN", OPEN_POS)):
        for m, _ in fingers:
            m.setPosition(target)
        wait(int(3000 / ts))
        print(f"  [gripper test] {label}: sensors {finger_values()}  <- check the 3D view now")
    print("Gripper test finished. Set GRIPPER_TEST = False to run the pick-and-place.")
    while robot.step(ts) != -1:
        update_markers()

if CALIBRATE_TOOL:
    open_gripper()
    calibrate_tool_offset()

HOME_Q = current_q()
retries = 0
gripper_primed = False
state = "INIT"
can_z0_world = None
print("Starting pick-and-place")

while robot.step(ts) != -1 and state not in ("DONE", "FAILED"):
    finished = state

    if state == "INIT":
        can_b = can_base_pos()
        grasp_tip = can_b + np.array([0, 0, GRASP_Z])
        pregrasp = grasp_tip + np.array([0, 0, APPROACH_H])
        place_tip = PLACE_BASE + np.array([0, 0, GRASP_Z + PLACE_CLEAR])
        place_above = place_tip + np.array([0, 0, APPROACH_H])
        can_z0_world = can_node.getPosition()[2]
        print("Can (base frame):", np.round(can_b, 3))
        if not open_gripper(cycle=not gripper_primed):
            print("[!] gripper failed to open; aborting")
            state = "FAILED"
            continue
        gripper_primed = True
        state = "PREGRASP"

    elif state == "PREGRASP":
        print("[PREGRASP]")
        state = "DESCEND" if go_to(pregrasp) else "FAILED"

    elif state == "DESCEND":
        print("[DESCEND]")
        print("  finger sensors:", finger_values())
        can_before = np.array(can_node.getPosition())
        ok = move_linear(grasp_tip, ik_tol=IK_TOL_GRASP)
        print("  can moved by", np.round(np.array(can_node.getPosition()) - can_before, 4))
        print("  finger sensors:", finger_values())
        state = "CLOSE" if ok else "FAILED"

    elif state == "CLOSE":
        print("[CLOSE]")
        T_now = chain.forward_kinematics(current_q())
        e_base = can_base_pos() - T_now[:3, 3]
        e_base[2] = 0.0                      # horizontal error only; height is tuned separately
        err_local = T_now[:3, :3].T @ e_base  # all 3 components: the finger frame need not be axis-aligned
        print("  suggested TOOL_OFFSET (valid only if the can is visibly between the fingers now):",
              np.round(np.array(TOOL_OFFSET) + err_local, 4).tolist())
        xy_err = np.linalg.norm((can_base_pos() - tool_pos())[:2])
        print(f"  x,y misalignment can vs tool: {xy_err * 1000:.1f} mm")
        if xy_err > ALIGN_TOL and not SKIP_ALIGN_CHECK:
            print("  [!] misaligned beyond ALIGN_TOL; will re-plan from the can's measured pose")
            state = "RETRY"
        else:
            if xy_err > ALIGN_TOL:
                print("  [skip] alignment check disabled (SKIP_ALIGN_CHECK); closing anyway")
            wait(10)
            grasped, _ = close_gripper()
            wait(20)
            state = "LIFT" if grasped else "RETRY"

    elif state == "LIFT":
        print("[LIFT]")
        ok = move_linear(pregrasp)
        wait(20)
        rose = can_node.getPosition()[2] - can_z0_world
        dist = np.linalg.norm(can_base_pos() - tool_pos())
        print(f"  can rose {rose:.3f} m, can-to-tool {dist:.3f} m")
        if ok and rose > LIFT_MIN and dist < 0.10:
            # The tool point may not be at the can centre (uncalibrated TOOL_OFFSET), so
            # measure where the can really sits relative to the tool and place accordingly.
            held = tool_pos() - can_base_pos()
            place_tip = PLACE_BASE + np.array([0, 0, PLACE_CLEAR]) + held
            place_above = place_tip + np.array([0, 0, APPROACH_H])
            print("  held offset tool-can:", np.round(held, 4))
            state = "TRANSIT"
        else:
            state = "RETRY"

    elif state == "TRANSIT":
        print("[TRANSIT]")
        state = "PLACE_DESCEND" if move_linear(place_above) else "FAILED"

    elif state == "PLACE_DESCEND":
        print("[PLACE_DESCEND]")
        state = "RELEASE" if move_linear(place_tip) else "FAILED"

    elif state == "RELEASE":
        print("[RELEASE]")
        open_gripper()
        wait(30)
        state = "RETREAT"

    elif state == "RETREAT":
        print("[RETREAT]")
        state = "HOME" if move_linear(place_above) else "FAILED"

    elif state == "HOME":
        print("[HOME]")
        current_target[0] = None
        move_to_q(HOME_Q, tol=0.02)
        state = "DONE"

    elif state == "RETRY":
        retries += 1
        print(f"[RETRY] attempt {retries}/{MAX_RETRIES}")
        if retries > MAX_RETRIES:
            state = "FAILED"
        else:
            open_gripper()
            move_linear(tool_pos() + np.array([0, 0, APPROACH_H]))
            state = "INIT"           # re-read the can pose and try again

    if STOP_AFTER and finished == STOP_AFTER:
        print(f"Stopped after {finished}; next state would be {state}.")
        break

print("Final state:", state)
while robot.step(ts) != -1:
    update_markers()