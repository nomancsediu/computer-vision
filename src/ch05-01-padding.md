## প্যাডিং: Valid vs Same

আগের চ্যাপ্টারে আমরা দেখেছি কনভোলিউশন অপারেশন কিভাবে কাজ করে। কিন্তু একটি গুরুত্বপূর্ণ সমস্যা সেখানে চোখে পড়েছিল  প্রতিটি convolution layer ইনপুটের spatial dimension কমিয়ে দেয়। এই সমস্যাটা কতটা গুরুতর হতে পারে, আর এটা সমাধান করার জন্য প্যাডিং কিভাবে কাজে লাগে  সেটাই এই সেকশনে বিস্তারিত শিখবো। আমরা দুই ধরনের প্যাডিং  Valid ও Same  দেখবো, এবং Keras দিয়ে MNIST ডেটাসেটে hands-on ডেমো করবো।

### শ্রিংকিং আউটপুট সমস্যা

ধরো তোমার কাছে MNIST ডেটাসেটের 28×28 ইমেজ আছে। তুমি একটি 3×3 kernel দিয়ে convolution করলে আউটপুট হবে 26×26। আরেকটি 3×3 convolution layer যোগ করলে 24×24, তারপর আরেকটি দিলে 22×22। এভাবে প্রতিটি layer এ spatial dimension 2 করে কমে যাচ্ছে (3×3 kernel এর জন্য)। একটু হিসাব করা যাক:

```
Layer 1: 28×28 → 26×26  (2 decreased)
Layer 2: 26×26 → 24×24  (2 decreased)
Layer 3: 24×24 → 22×22  (2 decreased)
Layer 4: 22×22 → 20×20  (2 decreased)
Layer 5: 20×20 → 18×18  (2 decreased)
...
Layer 10: 10×10 → 8×8
Layer 11: 8×8 → 6×6
Layer 12: 6×6 → 4×4
Layer 13: 4×4 → 2×2
```

মাত্র 13টি convolution layer পরে 28×28 ইমেজ 2×2 তে নেমে এসেছে! আর 2×2 তে আর কোনো convolution করা সম্ভব নয় (3×3 kernel এর জন্য)। অথচ আধুনিক CNN আর্কিটেকচার (ResNet, VGG) এ 50-100+ convolution layer থাকে! সুতরাং, dimension shrink হওয়া একটি বড় সমস্যা  ডিপ নেটওয়ার্ক বানানো প্রায় অসম্ভব হয়ে পড়ে যদি প্রতি layer এ dimension কমতে থাকে।

এই সমস্যা শুধু dimension এর নয়  এটি feature quality কেও প্রভাবিত করে। ইমেজের border pixel গুলো কমবার convolution window তে appear করে, আর center pixel গুলো বেশিবার। যেমন, একটি 4×4 ইমেজে 3×3 kernel দিয়ে convolution করলে corner pixel গুলো মাত্র 1টা window এ appear করে, কিন্তু center pixel গুলো 4টা window এ appear করে। এর মানে border এর feature গুলো underrepresented  নেটওয়ার্ক border এর তথ্য ঠিকমতো learn করতে পারে না। এই সমস্যাকে বলা হয় **border effect** বা **boundary effect**।

### Valid Padding (কোনো প্যাডিং নেই)

সবচেয়ে সহজ অপশন হলো Valid Padding  অর্থাৎ কোনো padding যোগ না করা। ইনপুট যেমন আছে তেমনভাবেই convolution করা হয়। Kernel শুধু সেই জায়গাগুলোতে slide করে যেখানে সম্পূর্ণ kernel ইনপুটের ভেতরে fit করে  বাইরে বের হয়ে যাওয়া অবস্থান এড়িয়ে যায়। ফলে আউটপুটের dimension ইনপুটের চেয়ে ছোট হয়।

**Valid Padding এর আউটপুট সাইজ ফর্মুলা:**

```
Output Size = (Input Size - Kernel Size + 1)
```

কিছু উদাহরণ:

```
Input: 32×32, Kernel: 5×5 → Output: (32 - 5 + 1) = 28×28
Input: 28×28, Kernel: 3×3 → Output: (28 - 3 + 1) = 26×26
Input: 64×64, Kernel: 7×7 → Output: (64 - 7 + 1) = 58×58
```

খেয়াল করো  kernel যত বড়, dimension তত বেশি কমে। 5×5 kernel এ প্রতি side এ 2 করে কমে, 7×7 kernel এ 3 করে কমে। এই কারণেই আধুনিক CNN এ সাধারণত 3×3 kernel ব্যবহার করা হয়  dimension কম কমে, আর multiple 3×3 layer stack করে বড় receptive field পাওয়া যায়।

### Same Padding (জিরো প্যাডিং)

Same Padding এ ইনপুটের চারপাশে zero যোগ করা হয় যাতে আউটপুটের spatial dimension ইনপুটের সমান থাকে (stride=1 হলে)। "Same" নামটা এখান থেকেই  আউটপুট সাইজ ইনপুটের "same" (সমান) থাকে। এটি কনভোলিউশন লেয়ারের সবচেয়ে জনপ্রিয় padding mode, কারণ এতে dimension কমে না এবং ডিপ নেটওয়ার্ক বানানো সহজ হয়।

**Same Padding এর প্যাডিং সাইজ ফর্মুলা (stride=1):**

```
Padding = (Kernel Size - 1) / 2
```

কিছু উদাহরণ:

```
Kernel 3×3 → Padding = (3 - 1) / 2 = 1  (1 row/column zero added per side)
Kernel 5×5 → Padding = (5 - 1) / 2 = 2  (2 rows/columns zero added per side)
Kernel 7×7 → Padding = (7 - 1) / 2 = 3  (3 rows/columns zero added per side)
```

Same padding কিভাবে কাজ করে একটি উদাহরণ দিয়ে বুঝি। ধরো ইনপুট 5×5 আর kernel 3×3। Valid padding এ আউটপুট হতো 3×3। কিন্তু same padding এ ইনপুটের চারপাশে 1 row ও 1 column zero যোগ করে ইনপুট 7×7 হয়ে যায়, তারপর 3×3 kernel দিয়ে convolution করলে আউটপুট 5×5 হয়  ইনপুটের সমান! এই zero যোগ করাকেই **zero padding** বলে।

Same padding এর আরেকটি বড় সুবিধা হলো  border pixel গুলোও center pixel এর মতো সমান সংখ্যক convolution window এ appear করে। কারণ padding যোগ করার ফলে kernel এখন border pixel গুলোকেও center এ রেখে convolution করতে পারে। ফলে border effect দূর হয় এবং নেটওয়ার্ক ইমেজের প্রান্তের তথ্যও ভালোভাবে learn করতে পারে।

### সাধারণ আউটপুট সাইজ ফর্মুলা

Padding আর stride দুটোকেই একসাথে বিবেচনা করে আউটপুট সাইজের সাধারণ ফর্মুলা হলো:

```
Output Size = floor((n + 2p - f) / s) + 1
```

এখানে:
- `n` = ইনপুট সাইজ (width বা height)
- `p` = padding (প্রতি side এ কত zero যোগ হচ্ছে)
- `f` = kernel (filter) সাইজ
- `s` = stride (কত pixel পরপর kernel সরছে)

কিছু হিসাব করা যাক:

```
n=32, p=0, f=5, s=1 → floor((32 + 0 - 5) / 1) + 1 = 28   (Valid padding)
n=32, p=2, f=5, s=1 → floor((32 + 4 - 5) / 1) + 1 = 32   (Same padding)
n=28, p=0, f=3, s=1 → floor((28 + 0 - 3) / 1) + 1 = 26   (Valid padding)
n=28, p=1, f=3, s=1 → floor((28 + 2 - 3) / 1) + 1 = 28   (Same padding)
```

এই ফর্মুলাটি মুখস্থ না করে বুঝে রাখো  এটি পুরো CNN জুড়ে বারবার ব্যবহার হবে। যখন তুমি নতুন আর্কিটেকচার ডিজাইন করবে বা existing আর্কিটেকচার বুঝবে, তখন এই ফর্মুলা দিয়ে প্রতিটি layer এর আউটপুট dimension হিসাব করতে হবে।

### Keras MNIST ডেমো: Valid vs Same Padding

এবার আমরা Keras দিয়ে MNIST ডেটাসেটে Valid ও Same padding এর প্রভাব সরাসরি দেখবো। দুটি আলাদা মডেল বানাবো  একটি Valid padding দিয়ে, আরেকটি Same padding দিয়ে। তারপর `model.summary()` দিয়ে প্রতিটি layer এর আউটপুট dimension তুলনা করবো।

**Valid Padding মডেল:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Valid padding model (no padding)
model_valid = Sequential([
    Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), padding='valid', activation='relu'),
    Conv2D(128, (3, 3), padding='valid', activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_valid.summary()
```

এই মডেলের output হবে:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320
 conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496
 conv2d_2 (Conv2D)           (None, 22, 22, 128)       73856
 flatten (Flatten)           (None, 61952)             0
 dense (Dense)               (None, 128)               7929984
 dense_1 (Dense)             (None, 10)                1290
=================================================================
```

দেখো কিভাবে spatial dimension কমছে: **28 → 26 → 24 → 22**। প্রতিটি layer এ 2 করে কমছে (কারণ 3×3 kernel, valid padding)। ফলে Flatten layer এ 22×22×128 = 61,952 টি value আসছে, আর Dense layer এ প্রচুর parameter লাগছে (7,929,984)!

**Same Padding মডেল:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Same padding model (zero padding is being added)
model_same = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_same.summary()
```

এই মডেলের output হবে:

```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_3 (Conv2D)           (None, 28, 28, 32)        320
 conv2d_4 (Conv2D)           (None, 28, 28, 64)        18496
 conv2d_5 (Conv2D)           (None, 28, 28, 128)       73856
 flatten_1 (Flatten)         (None, 100352)            0
 dense_2 (Dense)             (None, 128)               12845184
 dense_3 (Dense)             (None, 10)                1290
=================================================================
```

Same padding এ spatial dimension অপরিবর্তিত থাকছে: **28 → 28 → 28 → 28**! কিন্তু খেয়াল করো  Flatten layer এ 28×28×128 = 100,352 টি value আসছে, যা Valid padding এর চেয়ে অনেক বেশি। ফলে Dense layer এ parameter আরও বেশি (12,845,184)। এটি same padding এর একটি trade-off  dimension বজায় থাকে ঠিকই, কিন্তু computation আর parameter বেড়ে যায়।

### Valid vs Same Padding তুলনা

দুটি padding mode এর পার্থক্য একটি টেবিলে দেখা যাক:

| বৈশিষ্ট্য | Valid Padding | Same Padding |
|---|---|---|
| আউটপুট সাইজ | ইনপুটের চেয়ে ছোট | ইনপুটের সমান (stride=1) |
| Zero যোগ | হয় না | হয় |
| Border effect | থাকে | থাকে না |
| প্যারামিটার | কম (কারণ dimension কম) | বেশি (কারণ dimension বেশি) |
| Computation | কম | বেশি |
| ডিপ নেটওয়ার্ক | Dimension অনেক কমে যায় | Dimension বজায় থাকে |

### কখন কোনটা ব্যবহার করবে?

**Same Padding ব্যবহার করো যখন:**
- তুমি চাও আউটপুটের spatial dimension ইনপুটের সমান থাকুক
- ডিপ নেটওয়ার্ক বানাচ্ছো যেখানে অনেক convolution layer আছে
- Border feature গুলোর গুরুত্ব আছে (যেমন medical imaging এ টিউমার এর লোকেশন ইমেজের যেকোনো জায়গায় হতে পারে)
- U-Net এর মতো architecture এ যেখানে encoder আর decoder এর dimension match করাতে হয়

**Valid Padding ব্যবহার করো যখন:**
- তুমি চাও spatial dimension ক্রমান্বয়ে কমুক (natural downsampling)
- Computation আর memory বাঁচাতে চাও
- ইনপুটের border এর তথ্য খুব গুরুত্বপূর্ণ নয়
- ছোট নেটওয়ার্ক বানাচ্ছো যেখানে খুব বেশি layer নেই

বাস্তবে, বেশিরভাগ আধুনিক CNN আর্কিটেকচার (ResNet, VGG, Inception) Same padding ব্যবহার করে, কারণ ডিপ নেটওয়ার্ক এ dimension কমে যাওয়া একটি বড় সমস্যা। Dimension reduction এর জন্য আলাদা pooling layer বা strided convolution ব্যবহার করা হয়, padding নয়। তবে কিছু ক্ষেত্রে (যেমন GAN এর discriminator) Valid padding ও ব্যবহার হয়।

### পূর্ণাঙ্গ Keras কোড: MNIST ক্লাসিফিকেশন

এখন আমরা একটি পূর্ণাঙ্গ MNIST ক্লাসিফিকেশন মডেল বানাবো same padding দিয়ে, ট্রেইন করবো, আর ইভ্যালুয়েট করবো:

```python
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# MNIST data load
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocessing
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Same padding model
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.1)

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")
```

এই কোডে আমরা MNIST ডেটাসেট লোড করেছি, 0-1 এর মধ্যে normalize করেছি, channel dimension যোগ করেছি, এবং same padding দিয়ে 3টি convolution layer বানিয়েছি। প্রতিটি Conv2D layer এ `padding='same'` specify করা হয়েছে  এটি Keras কে বলে যে প্রয়োজনীয় zero padding স্বয়ংক্রিয়ভাবে যোগ করতে হবে।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম কনভোলিউশন লেয়ারের shrinking output সমস্যা এবং কিভাবে padding এটি সমাধান করে। Valid padding এ কোনো zero যোগ হয় না, আউটপুট ইনপুটের চেয়ে ছোট হয়  `Output = Input - Kernel + 1`। Same padding এ zero যোগ করে আউটপুট ইনপুটের সমান রাখা হয় (stride=1 তে)  `Padding = (Kernel - 1) / 2`। সাধারণ ফর্মুলা `floor((n + 2p - f) / s) + 1` দিয়ে যেকোনো padding, stride, আর kernel এর জন্য আউটপুট সাইজ হিসাব করা যায়। আধুনিক CNN এ সাধারণত same padding ব্যবহার হয় কারণ এতে ডিপ নেটওয়ার্ক বানানো সহজ হয় এবং border feature ভালোভাবে capture হয়। পরবর্তী সেকশনে আমরা stride সম্পর্কে বিস্তারিত শিখবো  কিভাবে stride ব্যবহার করে feature map downsample করা যায়।
