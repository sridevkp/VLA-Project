# inspect_urdf.py  ->  python inspect_urdf.py tiago.urdf
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()

parent_of = {}   # child_link -> (joint_name, parent_link, joint_type)
origins = {}
for j in root.iter("joint"):
    name, jtype = j.get("name"), j.get("type")
    p = j.find("parent").get("link")
    c = j.find("child").get("link")
    parent_of[c] = (name, p, jtype)
    o = j.find("origin")
    origins[name] = o.get("xyz") if o is not None else None

print("=== Joints with 'arm', 'torso', 'gripper' in the name ===")
for c, (jn, p, jt) in parent_of.items():
    if any(k in jn.lower() for k in ("arm", "torso", "gripper")):
        print(f"{jn:45s} {jt:10s} {p} -> {c}")

print("\n=== Gripper/finger links with visual origins ===")
for l in root.iter("link"):
    if "finger" in l.get("name", "").lower() or "gripper" in l.get("name", "").lower():
        v = l.find("visual/origin")
        print(l.get("name"), "visual origin:", v.get("xyz") if v is not None else None)

# Walk from a chosen end link up to the root
end = input("\nEnter the end link name (e.g. gripper_right_finger_link): ").strip()
path = [end]
cur = end
while cur in parent_of:
    jn, p, _ = parent_of[cur]
    path += [jn, p]
    cur = p
path.reverse()
print("\nFull path root -> end:")
print(path)