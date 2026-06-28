"""Version info for all dependencies.

Usage:
    python 02_version_info.py
"""

import sys
from amverge import get_versions

versions = get_versions()
print(f"AMVerge CLI  {versions['amverge']}")
print(f"Python       {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print(f"Platform     {sys.platform}\n")

for name in ["av", "numpy", "pillow", "rich", "typer"]:
    ver = versions.get(name)
    print(f"  {name:20s} {f'v{ver}' if ver else 'not installed'}")

print()
for name in ["torch", "transnetv2_pytorch", "tqdm", "cv2", "pypresence"]:
    ver = versions.get(name)
    print(f"  {name:20s} {f'v{ver}' if ver else 'not installed'}")
