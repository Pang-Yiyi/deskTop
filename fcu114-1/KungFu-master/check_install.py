"""Check package installation"""
import sys

print("=" * 50)
print("Checking package installation")
print("=" * 50)

# PyTorch
try:
    import torch
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"     CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     CUDA version: {torch.version.cuda}")
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"[X] PyTorch: {e}")

# scikit-learn
try:
    import sklearn
    print(f"[OK] scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"[X] scikit-learn: {e}")

# OpenCV
try:
    import cv2
    print(f"[OK] OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"[X] OpenCV: {e}")

# Ultralytics
try:
    import ultralytics
    print(f"[OK] Ultralytics: {ultralytics.__version__}")
except ImportError as e:
    print(f"[X] Ultralytics: {e}")

# PyQt6
try:
    from PyQt6.QtWidgets import QApplication
    print("[OK] PyQt6")
except ImportError as e:
    print(f"[X] PyQt6: {e}")

# Seaborn
try:
    import seaborn
    print(f"[OK] Seaborn: {seaborn.__version__}")
except ImportError as e:
    print(f"[X] Seaborn: {e}")

# fastdtw
try:
    import fastdtw
    print("[OK] fastdtw")
except ImportError as e:
    print(f"[X] fastdtw: {e}")

# scipy
try:
    import scipy
    print(f"[OK] SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"[X] SciPy: {e}")

# pandas
try:
    import pandas
    print(f"[OK] Pandas: {pandas.__version__}")
except ImportError as e:
    print(f"[X] Pandas: {e}")

# matplotlib
try:
    import matplotlib
    print(f"[OK] Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"[X] Matplotlib: {e}")

# numpy
try:
    import numpy
    print(f"[OK] NumPy: {numpy.__version__}")
except ImportError as e:
    print(f"[X] NumPy: {e}")

# joblib
try:
    import joblib
    print(f"[OK] Joblib: {joblib.__version__}")
except ImportError as e:
    print(f"[X] Joblib: {e}")

print("=" * 50)
print("Check completed!")
print("=" * 50)
