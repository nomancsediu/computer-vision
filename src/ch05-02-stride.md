## স্ট্রাইড ও আউটপুট সাইজ ফর্মুলা

আগের সেকশনে আমরা শিখেছি কিভাবে padding দিয়ে আউটপুটের spatial dimension নিয়ন্ত্রণ করা যায়। এবার আমরা শিখবো কনভোলিউশনের আরেকটি গুরুত্বপূর্ণ হাইপারপ্যারামিটার — **স্ট্রাইড (Stride)**। Stride নির্ধারণ করে kernel ইনপুটের উপর দিয়ে কত pixel পরপর সরবে। এটি শুধু kernel এর movement কন্ট্রোলই করে না, feature map এর resolution ও কমানোর একটি শক্তিশালী উপায় প্রদান করে। এই সেকশনে আমরা stride এর কাজ, আউটপুট সাইজ ফর্মুলা, Keras ডেমো, আর strided convolution vs pooling এর debate বিস্তারিত আলোচনা করবো।

### স্ট্রাইড কি?

Stride হলো kernel এর step size — অর্থাৎ convolution করার সময় kernel কত pixel পরপর ডানদিকে বা নিচে সরবে। আগে পর্যন্ত আমরা সব উদাহরণে stride=1 ব্যবহার করেছি, মানে kernel একটি করে pixel সরছিল। কিন্তু stride=2 হলে kernel 2 pixel করে jump করবে, stride=3 হলে 3 pixel করে।

**Stride=1:** Kernel প্রতিটি position এ slide করে — সবচেয়ে বেশি overlapping, সবচেয়ে বেশি আউটপুট resolution।

```
ইনপুট (8×8), Kernel (3×3), Stride=1:
Kernel positions: (0,0) → (0,1) → (0,2) → ... → (5,5)
আউটপুট সাইজ: 6×6
```

**Stride=2:** Kernel একটি position skip করে সরে — কম overlapping, আউটপুটের spatial dimension অর্ধেকের কাছাকাছি।

```
ইনপুট (8×8), Kernel (3×3), Stride=2:
Kernel positions: (0,0) → (0,2) → (0,4) → ... → (4,4)
আউটপুট সাইজ: 3×3
```

Stride বাড়ানোর সরাসরি প্রভাব হলো আউটপুট feature map এর spatial dimension কমে যাওয়া — এটি একটি downsampling effect তৈরি করে। যেমন stride=2 দিলে width ও height প্রায় অর্ধেক হয়ে যায়। এটি pooling layer (max pooling, average pooling) এর মতোই কাজ করে, কিন্তু একটি গুরুত্বপূর্ণ পার্থক্য আছে — strided convolution learnable! মানে নেটওয়ার্ক শিখতে পারে কিভাবে downsampling করতে হয়, যেখানে pooling নির্দিষ্ট নিয়মে (max বা average) কাজ করে।

### সাধারণ আউটপুট সাইজ ফর্মুলা (পুনরাবৃত্তি)

আগের সেকশনে আমরা আউটপুট সাইজের সাধারণ ফর্মুলা শিখেছি। এখন stride ও বুঝেছি, তাই ফর্মুলাটি আরও গভীরভাবে বোঝা যাক:

```
Output Size = floor((n + 2p - f) / s) + 1
```

এখানে:
- `n` = ইনপুট সাইজ (width বা height)
- `p` = padding (প্রতি side এ কত zero যোগ হচ্ছে)
- `f` = kernel (filter) সাইজ
- `s` = stride (kernel এর step size)

`floor` মানে ভগ্নাংশ বাদ দেওয়া — যেমন floor(3.7) = 3। এটি তখন জরুরি যখন `(n + 2p - f)` ঠিকভাবে `s` দিয়ে ভাগ যায় না। সেক্ষেত্রে ডানদিক ও নিচের কিছু pixel skip হয়ে যায়।

কিছু হিসাব করা যাক — এবার stride এর ভ্যারিয়েশন সহ:

```
n=28, p=0, f=3, s=1 → floor((28 + 0 - 3) / 1) + 1 = 26   (Valid, Stride 1)
n=28, p=1, f=3, s=1 → floor((28 + 2 - 3) / 1) + 1 = 28   (Same, Stride 1)
n=28, p=0, f=3, s=2 → floor((28 + 0 - 3) / 2) + 1 = 13   (Valid, Stride 2)
n=28, p=1, f=3, s=2 → floor((28 + 2 - 3) / 2) + 1 = 14   (Same, Stride 2)
```

খেয়াল করো — stride=2 তে dimension প্রায় অর্ধেক হয়ে যাচ্ছে! Same padding + stride=2 তে 28→14, আর Valid padding + stride=2 তে 28→13। এই অর্ধেক হওয়াটা খুবই useful — এটি pooling এর বিকল্প হিসেবে কাজ করে।

### Stride > 1: Learnable Downsampling

Stride=1 সবসময় ব্যবহার করলে feature map এর resolution কমে না (same padding এ), বা খুব সামান্য কমে (valid padding এ)। কিন্তু অনেক CNN আর্কিটেকচারে feature map এর resolution কমানো দরকার — এটি spatial information compress করে এবং higher-level feature extract করতে সাহায্য করে। এই downsampling দুইভাবে করা যায়:

1. **Pooling Layer (Max/Average):** নির্দিষ্ট নিয়মে (সবচেয়ে বড় মান বা গড়) value বেছে নেয়। Learnable নয় — training এর সময় কিছু শেখে না।

2. **Strided Convolution:** Kernel এর step বাড়িয়ে feature map downsample করে। Learnable — নেটওয়ার্ক শিখতে পারে কোন value গুলো রাখতে হবে আর কোনগুলো ফেলতে হবে।

Strided convolution এর সবচেয়ে বড় সুবিধা হলো এটি learnable। সাধারণ max pooling শুধু সবচেয়ে বড় value নেয় — কিন্তু সবসময় কি সবচেয়ে বড় value ই সবচেয়ে informative? হয়তো নয়! Strided convolution এ kernel এর weight training এর মাধ্যমে learn হয়, তাই নেটওয়ার্ক নিজেই বুঝতে পারে downsampling এর সময় কোন feature গুলো গুরুত্বপূর্ণ আর কোনগুলো নয়।

### Keras MNIST ডেমো: Stride=2

এবার আমরা Keras দিয়ে stride=2 এর প্রভাব দেখবো MNIST ডেটাসেটে। দুটি মডেল বানাবো — একটি same padding + stride=2, আরেকটি valid padding + stride=2।

**Same Padding + Stride=2:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Same padding, Stride=2
model_same_s2 = Sequential([
    Conv2D(32, (3, 3), padding='same', strides=2,
           activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), padding='same', strides=2,
           activation='relu'),
    Conv2D(128, (3, 3), padding='same', strides=2,
           activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_same_s2.summary()
```

এই মডেলের output:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 14, 14, 32)        320
 conv2d_1 (Conv2D)           (None, 7, 7, 64)          18496
 conv2d_2 (Conv2D)           (None, 4, 4, 128)         73856
 flatten (Flatten)           (None, 2048)              0
 dense (Dense)               (None, 128)               262272
 dense_1 (Dense)             (None, 10)                1290
=================================================================
```

দেখো dimension কিভাবে কমছে: **28 → 14 → 7 → 4**! প্রতি layer এ spatial dimension প্রায় অর্ধেক হচ্ছে। এবং Flatten layer এ মাত্র 4×4×128 = 2,048 value — আগের same+stride1 মডেলে ছিল 100,352! Dense layer এ parameter ও অনেক কম: 262,272 vs 12,845,184। Stride=2 ব্যবহার করে আমরা 49 গুণ কম parameter পেলাম Dense layer এ!

**Valid Padding + Stride=2:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Valid padding, Stride=2
model_valid_s2 = Sequential([
    Conv2D(32, (3, 3), padding='valid', strides=2,
           activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), padding='valid', strides=2,
           activation='relu'),
    Conv2D(128, (3, 3), padding='valid', strides=2,
           activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_valid_s2.summary()
```

এই মডেলের output:

```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_3 (Conv2D)           (None, 13, 13, 32)        320
 conv2d_4 (Conv2D)           (None, 6, 6, 64)          18496
 conv2d_5 (Conv2D)           (None, 2, 2, 128)         73856
 flatten_1 (Flatten)         (None, 512)               0
 dense_2 (Dense)             (None, 128)               65664
 dense_3 (Dense)             (None, 10)                1290
=================================================================
```

Valid padding + stride=2 তে dimension আরও দ্রুত কমছে: **28 → 13 → 6 → 2**। Flatten layer এ মাত্র 2×2×128 = 512 value! Dense layer parameter ও মাত্র 65,664 — এটি সবচেয়ে lightweight মডেল। কিন্তু 2×2 feature map এ কতটুকু information রয়ে গেছে সেটা ভাবার বিষয় — এত ছোট feature map তে কিছু গুরুত্বপূর্ণ feature হারিয়ে যেতে পারে।

### সম্পূর্ণ তুলনা টেবিল

চলো চারটি combination এর সম্পূর্ণ তুলনা করি — MNIST 28×28 ইমেজ, 3টি Conv2D layer (32→64→128 filters, 3×3 kernel):

| কনফিগারেশন | Layer 1 | Layer 2 | Layer 3 | Flatten Size | Dense Params |
|---|---|---|---|---|---|
| **Valid + Stride 1** | 26×26 | 24×24 | 22×22 | 61,952 | 7,929,984 |
| **Same + Stride 1** | 28×28 | 28×28 | 28×28 | 100,352 | 12,845,184 |
| **Valid + Stride 2** | 13×13 | 6×6 | 2×2 | 512 | 65,664 |
| **Same + Stride 2** | 14×14 | 7×7 | 4×4 | 2,048 | 262,272 |

এই টেবিল থেকে কিছু গুরুত্বপূর্ণ observation:

1. **Same + Stride 1** এ সবচেয়ে বেশি parameter (12.8M) — resolution সবচেয়ে বেশি, কিন্তু computation ও সবচেয়ে বেশি।
2. **Valid + Stride 2** এ সবচেয়ে কম parameter (65K) — resolution সবচেয়ে কম, দ্রুত training, কিন্তু information loss বেশি।
3. **Same + Stride 2** একটি ভালো balance — 262K parameter, 4×4 feature map, যথেষ্ট information সংরক্ষিত।
4. **Valid + Stride 1** moderate — 7.9M parameter, 22×22 feature map, gradual reduction।

বাস্তবে বেশিরভাগ CNN আর্কিটেকচার **Same + Stride 2** বা **Valid + Stride 1** এর combination ব্যবহার করে — প্রথম কয়েকটা layer এ stride=1 দিয়ে feature extract করে, তারপর stride=2 বা pooling দিয়ে downsample করে।

### কখন কোনটা ব্যবহার করবে?

**Same Padding + Stride 1 (Resolution সংরক্ষণ):**
- যখন তুমি চাও feature map এর resolution অপরিবর্তিত থাকুক
- Encoder-decoder architecture (U-Net) এর encoder পার্টে, যেখানে পরে skip connection দিয়ে decoder এ পাঠাতে হবে
- Semantic segmentation এ যেখানে pixel-level prediction দরকার
- যখন আর্কিটেকচারে আলাদা pooling layer আছে downsampling এর জন্য

**Valid Padding + Stride 1 (Gradual Reduction):**
- যখন তুমি চাও dimension ধীরে ধীরে কমুক, কিন্তু stride বাড়াতে চাও না
- ছোট নেটওয়ার্ক এ যেখানে pooling layer যোগ করলে অতিরিক্ত computation বাড়বে
- Classification task এ যেখানে exact spatial location গুরুত্বপূর্ণ নয়
- GAN এর discriminator এ spatial dimension কমানোর জন্য

**Stride > 1 (Pooling এর বিকল্প):**
- যখন তুমি feature map downsample করতে চাও কিন্তু pooling এর নির্দিষ্ট নিয়মে সীমাবদ্ধ থাকতে চাও না
- আধুনিক architecture (ResNet, DenseNet) এ প্রথম layer এ stride=2 দিয়ে initial downsampling
- যখন computation ও memory বাঁচাতে চাও — stride=2 তে half the positions এ convolution হয়
- যখন তুমি চাও নেটওয়ার্ক নিজে শিখুক কিভাবে downsampling করতে হয়

### Strided Convolution vs Max Pooling Debate

CNN তে downsampling এর দুটি প্রধান উপায় — strided convolution আর max pooling। কোনটা ভালো? এটি ডিপ লার্নিং কমিউনিটিতে একটি দীর্ঘদিনের debate। চলো দুটোকেই বিস্তারিত তুলনা করি:

**Max Pooling:**
- নির্দিষ্ট নিয়ম: প্রতিটি window এর সবচেয়ে বড় value নেয়
- Learnable নয় — কোনো parameter নেই, training এ কিছু learn হয় না
- Translation invariance বাড়ায় — কারণ max operation local shift কে ignore করে
- Implementation সহজ আর fast
- Classic CNN (LeNet, AlexNet, VGG) সব তে ব্যবহৃত

**Strided Convolution:**
- Kernel এর weight training এর মাধ্যমে learn হয়
- Learnable — নেটওয়ার্ক শিখতে পারে কোন feature গুরুত্বপূর্ণ downsampling এর সময়
- বেশি flexibility — শুধু max নয়, যেকোনো weighted combination হতে পারে
- বেশি parameter — kernel এর weight ও বাড়ে
- আধুনিক architecture (ResNet, DenseNet) এ pooling এর বিকল্প হিসেবে জনপ্রিয়

Keras এ দুটির কোড তুলনা করি:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# পদ্ধতি ১: Stride=2 দিয়ে downsampling (learnable)
model_strided = Sequential([
    Conv2D(32, (3, 3), padding='same', strides=2,
           activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), padding='same', strides=2,
           activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# পদ্ধতি ২: Max Pooling দিয়ে downsampling (non-learnable)
model_pooled = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu',
           input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),           # 28→14
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),           # 14→7
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print("Strided Convolution মডেল:")
model_strided.summary()

print("\nMax Pooling মডেল:")
model_pooled.summary()
```

Strided convolution মডেলে প্রতিটি Conv2D layer এ stride=2 specify করা হয়েছে — kernel এর weight এবং stride দুটোই downsampling কন্ট্রোল করে। Max pooling মডেলে প্রতিটি Conv2D layer এ stride=1 (default), আর আলাদা MaxPooling2D layer দিয়ে downsampling হয়। দুটির আউটপুট dimension প্রায় একই (28→14→7), কিন্তু কাজ করার পদ্ধতি আলাদা।

কোনটা ভালো? সাধারণত strided convolution বেশি parameter efficient আর flexible, কিন্তু max pooling সহজ আর tested। বাস্তবে, অনেক আধুনিক architecture দুটোরই combination ব্যবহার করে — কিছু জায়গায় strided convolution, কিছু জায়গায় max pooling। ResNet মূলত strided convolution ব্যবহার করে, VGG মূলত max pooling ব্যবহার করে, আর Inception দুটোই ব্যবহার করে। তোমার task আর data এর উপর ভিত্তি করে তুমি সিদ্ধান্ত নিতে পারো।

### পূর্ণাঙ্গ Keras কোড: Stride Comparison

সবশেষে একটি পূর্ণাঙ্গ কোড দেখি যেখানে আমরা চারটি combination তুলনা করবো MNIST তে:

```python
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# MNIST ডেটা লোড ও প্রিপ্রসেসিং
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# চারটি মডেল define করা
configs = {
    'Valid + Stride 1': {'padding': 'valid', 'strides': 1},
    'Same + Stride 1':  {'padding': 'same',  'strides': 1},
    'Valid + Stride 2': {'padding': 'valid', 'strides': 2},
    'Same + Stride 2':  {'padding': 'same',  'strides': 2},
}

for name, cfg in configs.items():
    print(f"\n{'='*50}")
    print(f"Configuration: {name}")
    print(f"{'='*50}")

    model = Sequential([
        Conv2D(32, (3, 3), padding=cfg['padding'], strides=cfg['strides'],
               activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), padding=cfg['padding'], strides=cfg['strides'],
               activation='relu'),
        Conv2D(128, (3, 3), padding=cfg['padding'], strides=cfg['strides'],
               activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # প্রতিটি layer এর output shape দেখা
    for layer in model.layers:
        if hasattr(layer, 'output_shape'):
            print(f"  {layer.name}: {layer.output_shape}")

    total_params = model.count_params()
    print(f"  Total Parameters: {total_params:,}")

    # দ্রুত training (শুধু 2 epoch)
    history = model.fit(x_train, y_train,
                        epochs=2,
                        batch_size=128,
                        validation_split=0.1,
                        verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Accuracy: {test_acc:.4f}")
```

এই কোড চালালে তুমি দেখবে প্রতিটি combination এর output shape, parameter count, আর test accuracy। সাধারণত Same + Stride 1 সবচেয়ে বেশি accuracy দেয় (সবচেয়ে বেশি capacity), আর Valid + Stride 2 সবচেয়ে কম। কিন্তু parameter efficiency এর দিক থেকে Same + Stride 2 সবচেয়ে ভালো balance দেয় — যথেষ্ট accuracy অনেক কম parameter এ।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম stride কিভাবে কনভোলিউশন লেয়ারের আউটপুট dimension নিয়ন্ত্রণ করে। Stride=1 তে kernel একটি করে pixel সরে আর আউটপুটের resolution বেশি থাকে; stride=2 তে kernel 2 pixel করে সরে আর feature map downsample হয়। সাধারণ আউটপুট সাইজ ফর্মুলা `floor((n + 2p - f) / s) + 1` দিয়ে padding, stride, আর kernel সাইজ মিলিয়ে আউটপুট dimension হিসাব করা যায়। Strided convolution আর max pooling দুটোই downsampling এর উপায় — strided convolution learnable আর flexible, max pooling simple আর proven। বাস্তবে দুটোরই combination ব্যবহার হয়, আর তোমার task অনুযায়ী সঠিক combination বেছে নিতে হবে। পরবর্তী চ্যাপ্টারে আমরা pooling layer সম্পর্কে বিস্তারিত শিখবো এবং ক্লাসিক CNN আর্কিটেকচার (LeNet-5, AlexNet) দেখবো।
