## এনভায়রনমেন্ট সেটআপ

এই সেকশনে কম্পিউটার ভিশনের জন্য দুইভাবে environment সেটআপ করা শেখানো হবে: **Google Colab** (ক্লাউড-based, কোনো সেটআপ লাগে না) এবং **Conda** (লোকাল, সম্পূর্ণ নিয়ন্ত্রণ)।

---

### Google Colab

Google Colab একটি ফ্রি ক্লাউড-based Jupyter notebook environment, যেখানে GPU access পাওয়া যায়। কোনো লোকাল সেটআপের প্রয়োজন নেই।

1. [colab.research.google.com](https://colab.research.google.com) এ যাও
2. **File → New Notebook** এ ক্লিক করো
3. **Runtime → Change runtime type → T4 GPU** সিলেক্ট করো
4. **Save** করো

বেশিরভাগ প্যাকেজ (TensorFlow, OpenCV, NumPy, Matplotlib) pre-installed থাকে:

```python
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")
```

ফাইল persist করতে Google Drive mount করো:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save model to Drive
model.save('/content/drive/MyDrive/models/my_model.h5')

# Load from Drive
model = tf.keras.models.load_model('/content/drive/MyDrive/models/my_model.h5')
```

---

### Conda Environment (লোকাল)

#### Step 1: Miniconda Install

[miniconda](https://docs.anaconda.com/miniconda/) থেকে download করো অথবা:

```bash
# Linux/Mac
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows
# Download from: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
```

#### Step 2: Environment তৈরি ও Activate

```bash
# Python 3.11 দিয়ে নতুন environment তৈরি
conda create -n cvdev python=3.11 -y

# Activate
conda activate cvdev
```

Activate করলে terminal এ `(cvdev)` prefix দেখাবে। বন্ধ করতে `conda deactivate`।

#### Step 3: Dependencies Install

```bash
pip install --upgrade pip
pip install opencv-python==4.9.0.80
pip install numpy==1.26.4
pip install matplotlib==3.8.3
pip install tensorflow==2.19.0
pip install keras==3.3.3
pip install Pillow==10.2.0
pip install kaggle==1.6.6
pip install gdown==5.0.0
```

অথবা requirements.txt ফাইল থেকে একসাথে install:

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
opencv-python==4.9.0.80
Pillow==10.2.0
numpy==1.26.4
matplotlib==3.8.3
tensorflow==2.19.0
keras==3.3.3
gdown==5.0.0
kaggle==1.6.6
```

#### Step 4: Install সফল হয়েছে কিনা যাচাই

```python
import tensorflow as tf
import cv2
import numpy as np

print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"  - {gpu.name}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected - running in CPU mode")
```
