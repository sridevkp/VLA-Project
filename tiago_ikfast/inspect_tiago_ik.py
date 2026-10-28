
from pathlib import Path
from ikpy.chain import Chain

urdf_path = Path(__file__).with_name("only_arm.urdf")

if not urdf_path.exists():
    raise FileNotFoundError(urdf_path)

chain = Chain.from_urdf_file(str(urdf_path))

print("URDF:", urdf_path)
print("Number of chain links:", len(chain.links))
print("Link names and active status:")

for i, link in enumerate(chain.links):
    print(
        f"{i:2d}: {link.name:25s} "
        f"active={chain.active_links_mask[i]} "
        f"bounds={link.bounds}"
    )
