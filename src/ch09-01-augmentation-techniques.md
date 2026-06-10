## ডেটা অগমেন্টেশন টেকনিক

Deep learning model গুলো data-hungry — যত বেশি data দেবে, তত ভালো শিখবে। কিন্তু রিয়েল-ওয়ার্ল্ডে labeled data collect করা expensive ও time-consuming। একটি medical imaging dataset prepare করতে লাগতে পারে মাসের পর মাস, আর প্রতিটি image label করার জন্য domain expert (যেমন doctor) দরকার। এই সমস্যার সমাধানে আসে **Data Augmentation** — existing image গুলোতে various transform apply করে নতুন training sample তৈরি করা।

### Data Augmentation কী এবং কেন দরকার?

Data augmentation হলো training এর সময় image গুলোতে random transformation apply করা, যাতে মডেল একই image এর বিভিন্ন version দেখে। এর মূল উদ্দেশ্য হলো:

- **Overfitting কমানো:** ছোট dataset এ model training data মুখস্থ করে ফেলে। Augmentation এ effectively training data এর diversity বাড়ায়, তাই model specific image মুখস্থ করতে পারে না, বরং general pattern শিখতে বাধ্য হয়।

- **Model robustness বাড়ানো:** রিয়েল-ওয়ার্ল্ডে image গুলো হুবহু training data এর মতো থাকে না — একটু tilted হতে পারে, zoomed in হতে পারে, brightness আলাদা হতে পারে। Augmentation দিয়ে model কে এসব variation handle করতে train করা হয়।

- **Data collection cost বাঁচানো:** 500 image collect করে augmentation দিয়ে 5000+ effective training sample তৈরি করা সম্ভব। নতুন 5000 image collect ও label করার চেয়ে এটা অনেক সস্তা!

### ImageDataGenerator — Augmentation এর মূল টুল

Keras এ `ImageDataGenerator` ক্লাস দিয়ে real-time augmentation করা হয়। Training এর সময় প্রতি batch এ image গুলোতে random transform apply হয় — তাই model প্রতি epoch এ একই image এর আলাদা version দেখে। চলো সব parameter বিস্তারিত দেখি:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data এর জন্য augmentation-enabled generator
train_datagen = ImageDataGenerator(
    rescale=1./255,                  # Pixel value 0-255 → 0-1 normalize
    rotation_range=20,               # -20° থেকে +20° পর্যন্ত random rotation
    shear_range=0.2,                 # 0.2 radian shear intensity
    zoom_range=0.2,                  # 80% থেকে 120% পর্যন্ত random zoom
    horizontal_flip=True,            # অনুভূমিক flip (ডান-বাম)
    vertical_flip=False,             # উল্লম্ব flip বন্ধ (task ভেদে decide)
    width_shift_range=0.2,           # প্রস্থের 20% পর্যন্ত horizontal shift
    height_shift_range=0.2,          # উচ্চতার 20% পর্যন্ত vertical shift
    fill_mode='nearest',             # Shift এর পর empty space ভরাট পদ্ধতি
    brightness_range=[0.8, 1.2]      # 80% থেকে 120% brightness variation
)

# Validation/Test data এর জন্য শুধু rescale — augmentation নয়!
val_datagen = ImageDataGenerator(rescale=1./255)
```

> **গুরুত্বপূর্ণ:** Validation ও test data তে কখনো augmentation apply করবে না! শুধু `rescale=1./255` করবে। Augmentation শুধু training data তে দরকার — validation/test এ আমরা original image তে model evaluate করতে চাই।

### Parameter বিস্তারিত ব্যাখ্যা

**rotation_range:** Image কে random angle এ rotate করে। `rotation_range=20` মানে -20° থেকে +20° পর্যন্ত যেকোনো angle এ image rotate হতে পারে। উদাহরণস্বরূপ, একটা cat এর image কিছুটা tilted থাকলেও model যেন cat হিসেবে recognize করে, সেজন্য এই augmentation সাহায্য করে। General object recognition এ 15-30° rotation range ভালো কাজ করে।

**shear_range:** Shear transformation image কে একপাশে "ঢলিয়ে" দেয়, যেন কোনো পিচ্ছিল তলায় দাঁড়িয়ে ছবি তোলার মতো দেখায়। `shear_range=0.2` মানে 0.2 radian (প্রায় 11.5°) পর্যন্ত shear intensity। এটি perspective variation simulate করে। সাধারণত 0.1-0.3 range ভালো কাজ করে।

**zoom_range:** Image কে random ভাবে zoom in বা zoom out করে। `zoom_range=0.2` মানে 80% থেকে 120% পর্যন্ত zoom। Zoom in হলে object বড় দেখায়, zoom out হলে ছোট দেখায় — model কে বিভিন্ন scale এ object recognize করতে শেখায়। 0.1-0.3 সাধারণত ভালো range।

**horizontal_flip ও vertical_flip:** Image কে অনুভূমিক (ডান-বাম) বা উল্লম্ব (উপর-নিচ) ভাবে flip করে। `horizontal_flip=True` করলে 50% সম্ভাবনায় image flip হবে। **সতর্কতা:** সব image তে horizontal/vertical flip logical নয়! যেমন, X-ray image flip করলে medical information বিকৃত হতে পারে। আবার digit "6" কে vertical flip করলে "9" হয়ে যায় — এটা ভুল! তাই task বুঝে flip enable/disable করতে হবে।

**width_shift_range ও height_shift_range:** Image কে horizontal বা vertical দিকে random shift করে। `width_shift_range=0.2` মানে image এর width এর 20% পর্যন্ত ডানে বা বামে shift হতে পারে। Shift করার পর কিছু empty space তৈরি হয় — সেটা `fill_mode` দিয়ে ভরাট করা হয়। সাধারণত 0.1-0.3 range ব্যবহার হয়।

**fill_mode:** Shift বা rotation এর পর যে empty pixel তৈরি হয়, তা কিভাবে ভরাট হবে তা `fill_mode` নির্ধারণ করে। `'nearest'` হলো সবচেয়ে সাধারণ — nearest pixel এর value দিয়ে fill করে। অন্য options: `'constant'` (একটি constant value দিয়ে fill), `'reflect'` (mirror reflection), `'wrap'` (wrap around)। বেশিরভাগ ক্ষেত্রে `'nearest'` ভালো কাজ করে।

**brightness_range:** Image এর brightness random ভাবে পরিবর্তন করে। `brightness_range=[0.8, 1.2]` মানে original brightness এর 80% থেকে 120% পর্যন্ত variation হবে। রিয়েল-ওয়ার্ল্ডে lighting condition সবসময় একই থাকে না — তাই model কে varying brightness handle করতে শেখানো দরকার।

### Augmented Image ভিজুয়ালাইজেশন

Parameter গুলো পড়ে বোঝা কঠিন, তাই চলো augmented image গুলো visualize করে দেখি। আমরা একটি sample image নিয়ে বিভিন্ন augmentation apply করে দেখবো কেমন দেখায়:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# একটি sample image লোড করা
img_path = 'sample_image.jpg'
img = load_img(img_path, target_size=(224, 224))     # Resize to 224x224
img_array = img_to_array(img)                          # Convert to numpy array
img_array = img_array.reshape((1,) + img_array.shape)  # (1, 224, 224, 3) — batch dimension

# Augmentation generator তৈরি (শুধু augmentation, rescale নয় — visualization এর জন্য)
aug_datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3]
)

# 9টি augmented image generate করে visualize
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

i = 0
for batch in aug_datagen.flow(img_array, batch_size=1):
    augmented_img = batch[0].astype('uint8')    # Convert back to 0-255 for display
    axes[i].imshow(augmented_img)
    axes[i].set_title(f'Augmented #{i+1}', fontsize=12)
    axes[i].axis('off')
    i += 1
    if i >= 9:    # 9টি image হলে break
        break

plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
plt.show()
```

এই code চালালে 9টি augmented image দেখবে — প্রতিটি আলাদা! একটি একটু rotated, আরেকটি zoomed in, আরেকটি flipped, আরেকটি shifted — সব random। এভাবে একটি image থেকে training এর সময় effectively অসীম variation তৈরি হয়।

`flow()` method batch এ batch এ augmented image generate করে। Training এর সময় আমরা `flow_from_directory()` ব্যবহার করবো যা directly folder থেকে image load ও augment করে। কিন্তু visualization এর জন্য `flow()` বেশি convenient — একটি image কে repeat করে বিভিন্ন version দেখানো যায়।

### কখন Augmentation সাহায্য করে — আর কখন করে না

Data augmentation সবসময় সাহায্য করে না। কিছু scenario তে এটি খুব effective, আর কিছু scenario তে বরং ক্ষতি করতে পারে:

**যখন augmentation সাহায্য করে:**
- **ছোট dataset:** যখন training data কম (শত বা কয়েক হাজার image), augmentation এ effectively data diversity বাড়ায় এবং overfitting উল্লেখযোগ্য ভাবে কমায়।
- **Overfitting হচ্ছে:** Training curve দেখে যদি দেখো training accuracy অনেক বেশি কিন্তু validation accuracy কম, augmentation add করে gap কমানো সম্ভব।
- **Limited variation এর dataset:** সব image যদি একই angle, একই lighting এ তোলা হয়, augmentation দিয়ে natural variation simulate করা যায়।

**যখন augmentation সাহায্য করে না বা ক্ষতি করে:**
- **বড় ও diverse dataset:** যদি লাখ লাখ diverse image থাকে (যেমন ImageNet), augmentation এর additional benefit খুব কম। Training naturally ভালো হবে।
- **Unrealistic transforms:** X-ray image কে horizontal flip করলে heart left side থেকে right side এ চলে যাবে — এটা medically impossible! এমন augmentation model কে confuse করবে।
- **Orientation-critical tasks:** Digit recognition (6 vs 9), text recognition — এগুলোতে flip/rotation ভুল label তৈরি করতে পারে।
- **Already high accuracy:** যদি ছাড়াই 95%+ accuracy পাও, augmentation দিয়ে বাড়ানো কঠিন — বরং model architecture change করা বেশি effective।

### সারসংক্ষেপ

Data augmentation হলো small-to-medium dataset এ deep learning model train করার সবচেয়ে effective technique গুলোর একটি। `ImageDataGenerator` দিয়ে real-time augmentation খুব সহজে করা যায় — rotation, shear, zoom, flip, shift, brightness সব parameter adjust করে নিজের task এর জন্য optimal augmentation pipeline বানানো সম্ভব। তবে মনে রাখবে, augmentation random magic নয় — task বুঝে logical transform apply করতে হবে। পরের সেকশনে আমরা এই augmented data দিয়ে একটি সম্পূর্ণ binary classification pipeline বানাবো!
