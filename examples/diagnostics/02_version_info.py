"""Version info for all dependencies.

Equivalent to `amverge version`.

Usage:
    python 02_version_info.py
"""

import sys
import amverge

print(f"AMVerge CLI  {amverge.__version__}")
print(f"Python       {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print(f"Platform     {sys.platform}")
print()

deps = ["av", "numpy", "pillow", "rich", "typer"]
for name in deps:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{name:12s} v{ver}")
    except ImportError:
        print(f"{name:12s} not installed")

optional = ["torch", "transnetv2_pytorch", "tqdm", "cv2", "pypresence"]
print()
for name in optional:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "installed")
        print(f"{name:12s} v{ver}")
    except ImportError:
        print(f"{name:12s} not installed")
