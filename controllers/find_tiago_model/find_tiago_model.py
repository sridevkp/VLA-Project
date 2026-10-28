
import os
from pathlib import Path

home = os.environ.get("WEBOTS_HOME")

print("WEBOTS_HOME =", home)

roots = []
if home:
    roots.append(Path(home) / "projects" / "robots" / "pal_robotics")

# Common Windows installation fallback.
roots.append(Path(r"C:\Program Files\Webots\projects\robots\pal_robotics"))

seen = set()

for root in roots:
    if not root.is_dir() or str(root).lower() in seen:
        continue
    seen.add(str(root).lower())

    print("\nSearching:", root)

    for pattern in ("*tiago*.proto", "*tiago*.urdf", "*tiago*.xacro"):
        matches = list(root.rglob(pattern))
        for path in matches:
            print(path)

if not seen:
    print("Could not find the standard installation path.")
    print("Check Webots Help > About for its installation directory.")
