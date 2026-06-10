## এনভায়রনমেন্ট সেটআপ

এই সেকশনে আমরা কম্পিউটার ভিশন প্রজেক্ট এর জন্য complete environment setup দেখবো — Conda environment, Google Colab, এবং local GPU configuration। সঠিক environment ছাড়া code run করতে গিয়ে অনেক সমস্যায় পড়বে — তাই এই section ভালো করে follow করো!

### Conda Environment Setup

Conda হলো Python এর সবচেয়ে popular environment manager। এটা দিয়ে isolated environment তৈরি করা যায় — এক machine এ একাধিক project এর আলাদা আলাদা dependency থাকতে পারে, কোনো conflict হবে না।

#### Step 1: Conda Install

যদি Conda install না থাকে, Miniconda download করো (Anaconda এর lightweight version):

```bash
# Linux/Mac
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows
# https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
# Download and run the installer
```

Miniconda তে Conda + Python + minimal packages থাকে। Full Anaconda install করতে গেলে 3GB+ download হবে — অনেক unnecessary packages আসবে। Miniconda তে শুধু যা দরকার তাই install করো — clean ও efficient!

#### Step 2: Environment Create ও Activate

```bash
# নতুন environment তৈরি করো (Python 3.11)
conda create -n cvdev python=3.11 -y

# Environment activate করো
conda activate cvdev
```

`-n cvdev` দিয়ে environment এর নাম `cvdev` দেওয়া হয়েছে — Computer Vision Development এর abbreviation। `python=3.11` দিয়ে specific Python version lock করা হয়েছে। TensorFlow 2.19 Python 3.11 এর সাথে best compatibility দেয় — তাই এই version।

Activate করলে terminal এ `(cvdev)` prefix দেখাবে — বুঝবে সঠিক environment এ আছো। Deactivate করতে `conda deactivate` লিখো।

#### Step 3: Core Requirements Install

```bash
# pip upgrade করো
pip install --upgrade pip

# সব dependencies install করো
pip install -r requirements.txt
```

#### Complete requirements.txt

```
# Core Computer Vision
opencv-python==4.9.0.80
Pillow==10.2.0

# Data Science & Visualization
numpy==1.26.4
matplotlib==3.8.3

# Deep Learning
tensorflow==2.19.0
keras==3.3.3

# Web Application
Flask==3.0.0
gdown==5.0.0

# Dataset
kaggle==1.6.6
```

Package গুলোর ভূমিকা:

| Package | কাজ |
|---------|------|
| `opencv-python` | Image read, write, process, filter |
| `Pillow` | Image manipulation, format conversion |
| `numpy` | Numerical computing, array operations |
| `matplotlib` | Plotting, visualization |
| `tensorflow` | Deep learning framework |
| `keras` | High-level neural network API (now part of TensorFlow) |
| `Flask` | Web application framework |
| `gdown` | Google Drive file download |
| `kaggle` | Kaggle dataset download via API |

**কেন pip দিয়ে install, conda দিয়ে নয়?** Conda ও pip দুটোই package installer, কিন্তু PyPI (pip) তে TensorFlow এর latest version আগে পাওয়া যায়। Conda channel এ version update কিছুটা delayed হয়। তাই best practice: Conda দিয়ে environment create, pip দিয়ে package install।

### Google Colab Setup

Google Colab হলো free cloud-based Jupyter notebook — GPU access সহ! Local machine এ GPU না থাকলে Colab এ model train করো — একদম free!

#### Colab GPU Enable করা

1. Google Colab এ যাও: [colab.research.google.com](https://colab.research.google.com)
2. New notebook create করো
3. **Runtime → Change runtime type → T4 GPU** select করো
4. Save করো

Free tier এ **NVIDIA T4 GPU** পাবে — 16GB VRAM। এটা training এর জন্য sufficient — VGG16, ResNet50 সব train হবে।

#### Colab এ Packages Pre-installed

Colab এ অনেক package আগেই install থাকে:

```python
# Colab এ pre-installed — আলাদা install করা লাগবে না
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
```

#### Colab এ Google Drive Mount

Model weights ও dataset persist করতে Google Drive mount করো:

```python
from google.colab import drive
drive.mount('/content/drive')

# Drive তে save করো
model.save('/content/drive/MyDrive/models/model_vgg16.h5')

# Drive থেকে load করো
model = tf.keras.models.load_model('/content/drive/MyDrive/models/model_vgg16.h5')
```

**গুরুত্বপূর্ণ:** Colab session close হলে সব data delete হয়ে যায়! Model weights, training curves — সব Google Drive তে save করো। এটা Colab এর সবচেয়ে common mistake — কয়েক ঘণ্টা train করে save করতে ভুলে গেলে সব শেষ!

#### Colab এ Dataset Upload

```python
# Kaggle API দিয়ে dataset download
!pip install kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d dataset-name -p ./data
```

অথবা Google Drive থেকে:

```python
# Drive থেকে dataset unzip
!unzip '/content/drive/MyDrive/datasets/fracture_dataset.zip' -d ./data
```

### Local GPU Setup

Local machine এ NVIDIA GPU থাকলে training অনেক fast হবে। Setup steps:

#### Step 1: NVIDIA GPU Driver

```bash
# GPU driver install check
nvidia-smi
```

যদি output এ GPU information দেখায়, driver install আছে। না দেখালে [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx) থেকে install করো।

#### Step 2: CUDA Toolkit

```bash
# CUDA version check
nvcc --version
```

TensorFlow 2.19 CUDA 12.x support করে। [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) থেকে compatible version download করো।

#### Step 3: cuDNN

[CUDA Deep Neural Network (cuDNN)](https://developer.nvidia.com/cudnn) download করে install করো — এটা TensorFlow এর GPU acceleration এর জন্য দরকার।

#### Step 4: GPU Verification

```python
import tensorflow as tf

# GPU detect হচ্ছে কিনা check
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ GPU detected: {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"   - {gpu.name}")
    
    # Memory growth enable (recommended)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("❌ No GPU detected — CPU mode only")

# Additional GPU info
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

**Memory Growth:** `set_memory_growth(gpu, True)` দিলে TensorFlow শুধু যতটুকু memory দরকার ততটুকু allocate করবে। না দিলে পুরো VRAM একসাথে allocate হয়ে যাবে — অন্য process কোনো GPU memory পাবে না।

#### GPU vs CPU Training Speed Comparison

```python
import time

# CPU training time
with tf.device('/CPU:0'):
    start = time.time()
    model.fit(train_generator, epochs=1, verbose=0)
    cpu_time = time.time() - start

# GPU training time
with tf.device('/GPU:0'):
    start = time.time()
    model.fit(train_generator, epochs=1, verbose=0)
    gpu_time = time.time() - start

print(f"CPU: {cpu_time:.1f}s per epoch")
print(f"GPU: {gpu_time:.1f}s per epoch")
print(f"Speedup: {cpu_time/gpu_time:.1f}x faster with GPU")
```

সাধারণত GPU training CPU থেকে 5-20x fast হয় — model size ও dataset size এর উপর depend করে। VGG16 training এ GPU তে 5-10x speedup পাওয়া যায়।

### Troubleshooting Common Issues

| সমস্যা | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python` |
| `ImportError: DLL load failed` (Windows) | Visual C++ Redistributable install করো |
| `CUDA out of memory` | Batch size কমাও, বা `set_memory_growth` enable করো |
| `tensorflow` import error | Python version check — 3.9-3.11 supported |
| Colab GPU not available | Runtime type change করো → T4 GPU |
| `gdown` download fail | File sharing permission check — "Anyone with link" |
| `kaggle` API error | `kaggle.json` ফাইল `~/.kaggle/` তে আছে কিনা check |

### সারসংক্ষেপ

এই সেকশনে আমরা complete environment setup করলাম — Conda দিয়ে isolated Python 3.11 environment, requirements.txt দিয়ে সব dependency install, Google Colab দিয়ে free GPU access, এবং local GPU setup (NVIDIA + CUDA + cuDNN)। সবচেয়ে গুরুত্বপূর্ণ: Python version (3.11) TensorFlow এর সাথে compatible রাখা, Colab এ model weights Google Drive তে save করা, এবং GPU verification code run করে confirm করা। এই setup complete হলে তুমি বই এর সব code smoothly run করতে পারবে!
