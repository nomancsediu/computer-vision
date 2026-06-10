## ডেটা ডাউনলোড ও প্রিপ্রসেসিং

মডেল ট্রেইন করার আগে সবচেয়ে গুরুত্বপূর্ণ কাজ হলো ডেটা প্রস্তুত করা। ডিপ লার্নিং এর একটি জনপ্রিয় প্রবাদ আছে: **"Garbage in, garbage out"** — ডেটা ভালো না হলে মডেল কখনো ভালো performance দেবে না। তাই ডেটা preparation পর্যায়ে আমরা বিশেষভাবে সতর্ক থাকবো। এই সেকশনে আমরা Kaggle থেকে ডেটা ডাউনলোড করা থেকে শুরু করে `ImageDataGenerator` দিয়ে augmentation পর্যন্ত সব শিখবো।

### Kaggle API সেটআপ

Kaggle হলো ডেটা সায়েন্স community এর সবচেয়ে বড় platform — এখানে হাজার হাজার dataset ফ্রি পাওয়া যায়। আমরা Kaggle API ব্যবহার করে সরাসরি Python থেকে ডেটা ডাউনলোড করবো। প্রথমে Kaggle API token বানাতে হবে:

1. Kaggle এ লগইন করো (কোনো account না থাকলে sign up করো)
2. Account settings এ যাও → "Create New API Token" ক্লিক করো
3. একটি `kaggle.json` ফাইল ডাউনলোড হবে
4. এই ফাইলটি `~/.kaggle/` directory তে রাখো

```python
import os
import zipfile

# Kaggle API সেটআপ
# kaggle.json ফাইল ~/.kaggle/ এ রাখো
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

# Kaggle ইনস্টল (যদি না থাকে)
# !pip install kaggle

# apples-or-tomatoes ডেটাসেট ডাউনলোড
!kaggle datasets download -d samuelcortas/apples-or-tomatoes

# ZIP ফাইল extract করা
with zipfile.ZipFile('apples-or-tomatoes.zip', 'r') as zip_ref:
    zip_ref.extractall('apples_or_tomatoes')

print("ডেটাসেট ডাউনলোড সম্পন্ন!")
```

### ডেটাসেট ডিরেক্টরি স্ট্রাকচার

Keras এর `ImageDataGenerator` কাজ করার জন্য ডেটাকে একটি নির্দিষ্ট directory structure এ সাজাতে হয়। প্রতিটি class এর জন্য আলাদা sub-folder থাকতে হয়। এই structure `flow_from_directory()` method ব্যবহার করতে অত্যন্ত গুরুত্বপূর্ণ — কারণ এটি folder এর নাম থেকে automatically class label বুঝে নেয়।

সঠিক directory structure এমন হওয়া উচিত:

```
apples_or_tomatoes/
├── train/
│   ├── apple/          ← আপেলের ট্রেইনিং ছবি
│   │   ├── apple1.jpg
│   │   ├── apple2.jpg
│   │   └── ...
│   └── tomato/         ← টমেটোর ট্রেইনিং ছবি
│       ├── tomato1.jpg
│       ├── tomato2.jpg
│       └── ...
├── val/
│   ├── apple/          ← আপেলের validation ছবি
│   └── tomato/         ← টমেটোর validation ছবি
└── test/
    ├── apple/          ← আপেলের test ছবি
    └── tomato/         ← টমেটোর test ছবি
```

যদি ডাউনলোড করা dataset এ আলাদা train/val/test split না থাকে, তাহলে আমরা manually split করবো:

```python
import os
import shutil
import random

# মূল ডেটা ফোল্ডার
base_dir = 'apples_or_tomatoes'

# Train/Val/Test split ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# নতুন ডিরেক্টরি তৈরি
for split in ['train', 'val', 'test']:
    for cls in ['apple', 'tomato']:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# প্রতিটি class এর ছবি split করা
for cls in ['apple', 'tomato']:
    src_folder = os.path.join(base_dir, cls)  # মূল class ফোল্ডার
    images = os.listdir(src_folder)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    # ছবি কপি করা
    for img in train_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(base_dir, 'train', cls, img)
        )
    for img in val_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(base_dir, 'val', cls, img)
        )
    for img in test_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(base_dir, 'test', cls, img)
        )

# পরিসংখ্যান দেখা
for split in ['train', 'val', 'test']:
    for cls in ['apple', 'tomato']:
        path = os.path.join(base_dir, split, cls)
        print(f"{split}/{cls}: {len(os.listdir(path))} images")
```

কেন আমাদের তিনটি আলাদা split দরকার? **Training set** দিয়ে মডেলের weight update হয়। **Validation set** দিয়ে প্রতি epoch এ মডেলের performance মনিটর করি — overfitting ধরার জন্য এটি essential। **Test set** দিয়ে final evaluation করি — এটি মডেল কখনো training এ দেখেনি, তাই এটি unbiased estimate দেয়। একটি সাধারণ ভুল হলো validation ও test set একই করে ফেলা — এতে মডেল indirectly validation set এ optimize হয়ে যায় এবং test accuracy overly optimistic হয়।

### ImageDataGenerator দিয়ে ডেটা লোডিং ও Augmentation

`ImageDataGenerator` হলো Keras এর একটি অত্যন্ত শক্তিশালী utility যা disk থেকে batch-by-batch ইমেজ লোড করে, real-time augmentation করে, এবং মডেলে feed করে। এর সবচেয়ে বড় সুবিধা হলো — পুরো dataset একসাথে memory তে লোড করতে হয় না! হাজার হাজার বড় ইমেজ থাকলে RAM এ সব রাখা সম্ভব না — `ImageDataGenerator` এই problem solve করে।

**Training data তে Augmentation:**

Training data তে আমরা augmentation apply করবো। Augmentation মানে হলো — existing ইমেজ থেকে random transformation করে নতুন variation তৈরি করা। এতে মডেল বিভিন্ন orientation, scale, position এর ইমেজ দেখে — ফলে overfitting কম হয় এবং generalization বাড়ে। যেমন, একটি আপেলের ছবি একটু ঘুরিয়ে (rotation), একটু zoom করে, বা horizontally flip করে দিলে সেটি আপেলই থাকে — কিন্তু মডেল একটি "নতুন" ইমেজ দেখে।

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data এর জন্য augmentation সহ ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,              # Pixel value 0-255 → 0-1 normalize
    rotation_range=20,            # -20° থেকে +20° random rotation
    shear_range=0.2,              # Shear transformation (0.2 radians)
    zoom_range=0.2,               # ±20% random zoom
    horizontal_flip=True,         # অনুভূমিক flip
    vertical_flip=True,           # উল্লম্ব flip
    width_shift_range=0.2,        # ±20% horizontal shift
    height_shift_range=0.2,       # ±20% vertical shift
    validation_split=0.2          # 20% validation split (যদি আলাদা val folder না থাকে)
)

# Validation data এর জন্য — শুধু rescale, কোনো augmentation নয়!
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Test data এর জন্য — শুধু rescale, কোনো augmentation নয়!
test_datagen = ImageDataGenerator(rescale=1.0/255)
```

**কেন test/validation data তে augmentation করবো না?**

এটি একটি অত্যন্ত গুরুত্বপূর্ণ প্রশ্ন! Test ও validation data হলো আমাদের মডেলের "পরীক্ষা" — এখানে আমরা মডেলের প্রকৃত performance জানতে চাই। যদি test ইমেজে augmentation করি, তাহলে মডেল এমন একটি modified ইমেজ evaluate করবে যা real-world এ সে দেখবে না। উপরন্তু, augmentation random — তাই একই test set এ বারবার evaluate করলে বিভিন্ন result আসবে, যা evaluation কে unreliable করে। Training data তে augmentation করি কারণ সেখানে আমরা artificial diversity তৈরি করতে চাই — কিন্তু evaluation এ আমরা consistent, representative result চাই। তাই validation ও test set এ শুধু `rescale` করবো — কোনো geometric transformation নয়।

### flow_from_directory() দিয়ে ডেটা জেনারেট করা

এখন আমরা `flow_from_directory()` দিয়ে directory থেকে batch-by-batch ইমেজ লোড করবো। এই method automatically class name গুলো folder এর নাম থেকে detect করে এবং label assign করে:

```python
# Training generator
train_generator = train_datagen.flow_from_directory(
    directory='apples_or_tomatoes/train',   # Training folder path
    target_size=(224, 224),                 # সব ইমেজ 224×224 তে resize
    batch_size=32,                          # প্রতি batch এ 32টি ইমেজ
    class_mode='binary',                    # Binary classification (2 class)
    shuffle=True                            # Training এ shuffle করা জরুরি
)

# Validation generator
val_generator = val_datagen.flow_from_directory(
    directory='apples_or_tomatoes/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False                           # Validation এ shuffle দরকার নেই
)

# Test generator
test_generator = test_datagen.flow_from_directory(
    directory='apples_or_tomatoes/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False                           # Test এ কখনো shuffle করবে না!
)

# Class mapping দেখা
print(f"Class indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
```

কিছু গুরুত্বপূর্ণ পয়েন্ট নোট করো:

- **`target_size=(224, 224)`:** সব ইমেজ 224×224 তে resize হবে। 224×224 মানে না যে এটিই একমাত্র সাইজ — তুমি আপনার প্রয়োজন অনুযায়ী যেকোনো সাইজ ব্যবহার করতে পারো (যেমন 150×150, 128×128)। কিন্তু 224×224 standard কারণ বড় pretrained model (VGG, ResNet) এই সাইজ ব্যবহার করে — পরবর্তী চ্যাপ্টারে transfer learning এর সময় এটি কাজে লাগবে।

- **`class_mode='binary'`:** যেহেতু আমাদের দুটি class (apple, tomato), তাই binary mode ব্যবহার করছি। এতে label হিসেবে 0 বা 1 পাওয়া যায়। যদি 3+ class থাকতো, তাহলে `class_mode='categorical'` ব্যবহার করতে হতো এবং one-hot encoding হতো।

- **`shuffle=True` (training):** Training এ shuffle করা অত্যন্ত গুরুত্বপূর্ণ। যদি সব apple আগে আর tomato পরে থাকে, মডেল sequentially learn করবে — প্রথমে "সব কিছু apple" শিখবে, তারপর "সব কিছু tomato" শিখবে — এতে training unstable হয়। Shuffle করলে প্রতি batch এ apple ও tomato মিশে থাকে, training smooth হয়।

- **`shuffle=False` (validation/test):** Evaluation এ shuffle করলে prediction ও true label এর alignment নষ্ট হয়। যখন আমরা confusion matrix বা per-class accuracy বের করবো, তখন predicted label ও true label এর order match করা জরুরি — তাই shuffle=False রাখতে হবে।

### validation_split ব্যবহার করে আলাদা val folder ছাড়া কাজ করা

যদি আলাদা validation folder না থাকে, তাহলে `validation_split` parameter ব্যবহার করে training data থেকেই validation split করা যায়:

```python
# যদি আলাদা val/ folder না থাকে
train_datagen_with_split = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2    # 20% validation এ রাখা
)

# Training split (80%)
train_gen = train_datagen_with_split.flow_from_directory(
    directory='apples_or_tomatoes/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',          # ← training portion
    shuffle=True
)

# Validation split (20%)
val_gen = train_datagen_with_split.flow_from_directory(
    directory='apples_or_tomatoes/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',        # ← validation portion
    shuffle=False
)

print(f"Training: {train_gen.samples}, Validation: {val_gen.samples}")
```

এই approach এ একটি সূক্ষ্ম বিষয় আছে: যেহেতু একই `ImageDataGenerator` ব্যবহার হচ্ছে, validation data তেও augmentation apply হবে — কিন্তু আমরা চাই validation data raw থাকুক! এই problem এর সমাধান হলো আলাদা `ImageDataGenerator` ব্যবহার করা (যেমন প্রথম example এ করেছি)। `validation_split` ব্যবহার করলে augmentation ও split এর interaction নিয়ে সাবধান থাকতে হবে।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম কিভাবে Kaggle API দিয়ে dataset ডাউনলোড করতে হয়, কিভাবে `ImageDataGenerator` এর জন্য সঠিক directory structure তৈরি করতে হয়, এবং কিভাবে train/val/test split করতে হয়। আমরা `ImageDataGenerator` দিয়ে data augmentation শিখলাম — `rotation_range`, `shear_range`, `zoom_range`, `horizontal_flip`, `vertical_flip`, `width_shift_range`, `height_shift_range` ইত্যাদি parameter। সবচেয়ে গুরুত্বপূর্ণ শিক্ষা: training data তে augmentation করবো, কিন্তু validation ও test data তে শুধু `rescale` করবো — কোনো augmentation নয়! এখন আমরা এই prepared data দিয়ে কাস্টম CNN মডেল বানাতে প্রস্তুত।
