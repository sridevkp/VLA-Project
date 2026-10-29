"""Supervisor-based tool-offset diagnostic. Prints a suggestion; does not mutate IKPy."""
from controller import Field
import numpy as np
from config import TOOL_OFFSET

def find_solid_by_name(node, name, depth=0, max_depth=40):
    if node is None or depth > max_depth:
        return None
    name_field = node.getField("name")
    if (name_field is not None and name_field.getTypeName() == "SFString"
            and name_field.getSFString() == name):
        return node
    for i in range(node.getNumberOfFields()):
        field = node.getFieldByIndex(i)
        field_type = field.getType()
        if field_type == Field.SF_NODE:
            found = find_solid_by_name(field.getSFNode(), name, depth + 1, max_depth)
            if found is not None:
                return found
        elif field_type == Field.MF_NODE:
            for j in range(field.getCount()):
                found = find_solid_by_name(field.getMFNode(j), name, depth + 1, max_depth)
                if found is not None:
                    return found
    return None

def calibrate_tool_offset(robot_io, frames, kinematics):
    try:
        root = robot_io.robot.getRoot()
        left = find_solid_by_name(root, "gripper_left_finger_link")
        right = find_solid_by_name(root, "gripper_right_finger_link")
        if left is None or right is None:
            print(f"[calib] finger solids not found (left={left is not None}, right={right is not None}); skipping.")
            return
        true_center = frames.world_to_base(
            (np.array(left.getPosition()) + np.array(right.getPosition())) / 2
        )
        transform = kinematics.forward(frames.current_q())
        local_error = transform[:3, :3].T @ (true_center - transform[:3, 3])
        print("[calib] true finger centre (base):", np.round(true_center, 4))
        print("[calib] FK tool point (base):     ", np.round(transform[:3, 3], 4))
        print("[calib] suggested TOOL_OFFSET:    ",
              np.round(np.array(TOOL_OFFSET) + local_error, 4),
              "(diagnostic only; verify against the 3D view)")
    except Exception as exc:
        print("[calib] failed:", exc)
