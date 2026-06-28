"""GPU and CUDA status check.

Shows PyTorch version, CUDA availability, GPU name and VRAM.
Equivalent to `amverge gpu`.

Usage:
    python 01_gpu_check.py
"""

import sys
from amverge import RPC_AVAILABLE

try:
    import torch
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:   {torch.version.cuda}")
        print(f"GPU count:      {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram = props.total_mem / (1024 ** 3)
            print(f"GPU {i}: {name}  ({vram:.1f} GB VRAM)")
    else:
        print("No CUDA GPU detected. TransNetV2 will use CPU.")
except ImportError:
    print("PyTorch not installed. Run: pip install amverge[ml]")

print()
try:
    import cv2
    print(f"OpenCV {cv2.__version__}  (edge detection)")
except ImportError:
    print("OpenCV not installed. Run: pip install amverge[edge]")

print(f"pypresence (RPC): {'available' if RPC_AVAILABLE else 'not installed'}")
print(f"  Run: pip install amverge[discord]")

try:
    from amverge.core.scene_detection import TRANSNET_AVAILABLE
    print(f"TransNetV2: {'installed' if TRANSNET_AVAILABLE else 'not installed'}")
except ImportError:
    print("TransNetV2: not installed")
