## LeNet-5 আর্কিটেকচার

LeNet-5 হলো কম্পিউটার ভিশন এর ইতিহাসে অন্যতম গুরুত্বপূর্ণ CNN আর্কিটেকচার। 1998 সালে Yann LeCun এই নেটওয়ার্ক তৈরি করেন handwritten digit recognition এর জন্য  বিশেষ করে MNIST ডেটাসেটে। এটি শুধু একটি গবেষণাপত্র ছিল না  LeNet-5 বাস্তবে ব্যাংকিং সিস্টেমে চেক পড়ার জন্য (check reading) ব্যবহৃত হয়েছিল! এটি ছিল প্রথম সফল commercial CNN application। যে সময়ে ডিপ লার্নিং কে "impossible" মনে করা হতো, সেই সময়ে LeNet-5 প্রমাণ করলো যে CNN রিয়েল-ওয়ার্ল্ড problem এ কাজ করতে পারে। আজকের আধুনিক CNN গুলো (ResNet, EfficientNet) এর মূল নীতি  convolution তে feature extract করা, pooling এ dimension কমানো, fully connected layer এ classification করা  এই নীতিগুলো LeNet-5 তেই প্রথম সফলভাবে প্রয়োগ হয়েছিল। এই সেকশনে আমরা LeNet-5 এর architecture বিস্তারিতভাবে বুঝবো এবং Keras দিয়ে স্ক্র্যাচ থেকে ইমপ্লিমেন্ট করবো।

### LeNet-5 এর ইতিহাস

1998 সালে Yann LeCun, Léon Bottou, Yoshua Bengio এবং Patrick Haffner যৌথভাবে "Gradient-Based Learning Applied to Document Recognition" শিরোনামের একটি historical paper publish করেন। এই paper এ LeNet-5 প্রস্তাব করা হয়। Yann LeCun এর গবেষণার মূল লক্ষ্য ছিল handwritten zip codes আর bank checks স্বয়ংক্রিয়ভাবে পড়া  যা USPS (United States Postal Service) আর ব্যাংকগুলোর জন্য অত্যন্ত গুরুত্বপূর্ণ ছিল। সেই সময়ে optical character recognition (OCR) সাধারণত hand-crafted feature extraction + traditional classifier (SVM, k-NN) দিয়ে করা হতো। LeCun এর innovation ছিল  feature extraction আর classification একসাথে end-to-end learn করা, manual feature design এর প্রয়োজন নেই। LeNet-5 MNIST ডেটাসেটে 99%+ accuracy অর্জন করে, যা সেই সময়ের traditional method গুলোকে ছাড়িয়ে যায়। এই সাফল্য CNN কে research community তে গুরুত্বপূর্ণ করে তোলে, যদিও বড় আকারের ইমেজ ক্লাসিফিকেশন এ CNN এর সফলতা আরও এক দশক অপেক্ষা করতে হয়  AlexNet (2012) পর্যন্ত।

### LeNet-5 আর্কিটেকচার ওভারভিউ

LeNet-5 এর architecture অত্যন্ত সহজ  মাত্র 7টি layer (2 conv + 2 pool + 3 dense)। আধুনিক standard এ এটি অত্যন্ত shallow, কিন্তু 1998 সালে এটি যথেষ্ট গভীর ছিল। আসল পেপারে LeNet-5 এর ইনপুট 32×32 grayscale ইমেজ (MNIST এর 28×28 ইমেজকে 32×32 তে zero-pad করা হতো)। Architecture টি এভাবে:

```
Input: 32×32×1 (grayscale image)
    ↓
Conv2D(6 filters, 5×5, tanh) → 28×28×6
    ↓
AveragePooling2D(2×2, stride=2) → 14×14×6
    ↓
Conv2D(16 filters, 5×5, tanh) → 10×10×16
    ↓
AveragePooling2D(2×2, stride=2) → 5×5×16
    ↓
Flatten → 400
    ↓
Dense(120, tanh)
    ↓
Dense(84, tanh)
    ↓
Dense(10, softmax)
    ↓
Output: 10 classes (digits 0-9)
```

এই architecture এ কিছু চমকপ্রদ বৈশিষ্ট্য আছে যা আধুনিক CNN থেকে আলাদা:

**tanh activation:** LeNet-5 এ Conv2D ও Dense layer এ tanh activation ব্যবহার হয়েছে, ReLU নয়। কারণ 1998 সালে ReLU এখনও জনপ্রিয় হয়নি  ReLU এর widespread adoption 2012 সালে AlexNet এর সাথে শুরু হয়। tanh এর output range [-1, 1], যা zero-centered  এটি training এ সাহায্য করে। কিন্তু tanh এ gradient vanishing problem বেশি হয়, কারণ positive ও negative উভয় দিকেই gradient saturate হয়।

**Average Pooling:** LeNet-5 তে max pooling এর বদলে average pooling ব্যবহার হয়েছে। সেই সময়ে average pooling বেশি প্রচলিত ছিল। পরে research দেখায় যে max pooling feature presence ধরে রাখতে বেশি effective, আর 2012 সালের AlexNet থেকে max pooling standard হয়ে যায়। তবে average pooling এখনও global average pooling (GAP) আকারে ব্যবহৃত হয়।

**5×5 kernel:** LeNet-5 তে 5×5 convolution kernel ব্যবহার হয়েছে। আধুনিক CNN এ সাধারণত 3×3 kernel ব্যবহার হয়  VGG (2014) থেকে এটি standard হয়। 5×5 kernel এ বেশি parameter লাগে, কিন্তু একটি layer এ বড় receptive field পাওয়া যায়। আধুনিক approach এ multiple 3×3 layer stack করে 5×5 receptive field পাওয়া হয়  এতে parameter কম লাগে এবং non-linearity বেশি যোগ হয়।

**খুব কম parameter:** মাত্র ~44,426 টি parameter! আধুনিক CNN (ResNet-50: ~25.6M, VGG-16: ~138M) এর তুলনায় এটি অবিশ্বাস্যরকম ছোট। কিন্তু MNIST এর মতো সহজ ডেটাসেটে এটিই যথেষ্ট।

### LeNet-5 আর্কিটেকচার টেবিল

প্রতিটি layer এর output shape ও parameter count এর বিস্তারিত টেবিল:

| Layer | Type | Output Shape | Parameter | Formula |
|-------|------|-------------|-----------|---------|
| 1 | Conv2D(6, 5×5, tanh) | 28×28×6 | 156 | 6×(5×5×1+1) |
| 2 | AvgPool2D(2×2) | 14×14×6 | 0 |  |
| 3 | Conv2D(16, 5×5, tanh) | 10×10×16 | 2,416 | 16×(5×5×6+1) |
| 4 | AvgPool2D(2×2) | 5×5×16 | 0 |  |
| 5 | Flatten | 400 | 0 |  |
| 6 | Dense(120, tanh) | 120 | 48,120 | 400×120+120 |
| 7 | Dense(84, tanh) | 84 | 10,164 | 120×84+84 |
| 8 | Dense(10, softmax) | 10 | 850 | 84×10+10 |
| | | | **61,706** | |

> **দ্রষ্টব্য:** আসল LeNet-5 পেপারে কিছু special connection pattern ছিল (যেমন layer 3 এ সব 6টি channel থেকে সব 16টি channel এ connection ছিল না, কিছু channel select করা হতো)। কিন্তু আমরা simplified version ইমপ্লিমেন্ট করবো যেখানে সব channel fully connected। Simplified version এ total parameter ≈ 61,706। আসল পেপারের selective connection দিয়ে parameter ≈ 44,426 হয়।

একটি গুরুত্বপূর্ণ পর্যবেক্ষণ: parameter এর সিংহভাগ Dense layer এ! Dense(120) তে একা 48,120 টি parameter  মোট parameter এর প্রায় 78%! এটি প্রমাণ করে যে fully connected layer CNN এ parameter এর সবচেয়ে বড় overhead। আধুনিক আর্কিটেকচারে (ResNet, EfficientNet) Global Average Pooling ব্যবহার করে Dense layer এর parameter অনেক কমানো হয়।

### Keras দিয়ে LeNet-5 ইমপ্লিমেন্টেশন

এখন আমরা LeNet-5 স্ক্র্যাচ থেকে Keras Sequential API দিয়ে ইমপ্লিমেন্ট করবো। আমরা আসল পেপারের simplified version বানাবো:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten

# Create LeNet-5 model
lenet5 = Sequential([
    # Layer 1: Conv2D - 6 filters, 5×5 kernel, tanh activation
    Conv2D(6, (5, 5), activation='tanh', padding='valid',
           input_shape=(32, 32, 1), name='conv1'),

    # Layer 2: Average Pooling - 2×2, stride 2
    AveragePooling2D((2, 2), strides=2, name='avgpool1'),

    # Layer 3: Conv2D - 16 filters, 5×5 kernel, tanh activation
    Conv2D(16, (5, 5), activation='tanh', padding='valid', name='conv2'),

    # Layer 4: Average Pooling - 2×2, stride 2
    AveragePooling2D((2, 2), strides=2, name='avgpool2'),

    # Flatten
    Flatten(name='flatten'),

    # Layer 5: Dense - 120 neurons, tanh activation
    Dense(120, activation='tanh', name='dense1'),

    # Layer 6: Dense - 84 neurons, tanh activation
    Dense(84, activation='tanh', name='dense2'),

    # Layer 7: Dense - 10 neurons, softmax activation (output)
    Dense(10, activation='softmax', name='output')
])

# View model summary
lenet5.summary()
```

এই মডেলের `summary()` output:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1 (Conv2D)              (None, 28, 28, 6)         156
 avgpool1 (AveragePooling2D) (None, 14, 14, 6)         0
 conv2 (Conv2D)              (None, 10, 10, 16)        2416
 avgpool2 (AveragePooling2D) (None, 5, 5, 16)          0
 flatten (Flatten)           (None, 400)               0
 dense1 (Dense)              (None, 120)               48120
 dense2 (Dense)              (None, 84)                10164
 output (Dense)              (None, 10)                850
=================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
_________________________________________________________________
```

### LeNet-5 দিয়ে MNIST ট্রেইনিং

এখন আমরা LeNet-5 কে MNIST ডেটাসেটে train করবো। MNIST এর ইমেজ 28×28, কিন্তু LeNet-5 এর input 32×32  তাই আমরা zero padding যোগ করবো:

```python
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# MNIST data load
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocessing
# 28×28 → 32×32 (LeNet-5 input size)
x_train_padded = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode='constant')
x_test_padded = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode='constant')

# Reshape and Normalize
x_train_padded = x_train_padded.reshape(-1, 32, 32, 1).astype('float32') / 255.0
x_test_padded = x_test_padded.reshape(-1, 32, 32, 1).astype('float32') / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Training data shape: {x_train_padded.shape}")
print(f"Test data shape: {x_test_padded.shape}")

# LeNet-5 model
lenet5 = Sequential([
    Conv2D(6, (5, 5), activation='tanh', padding='valid',
           input_shape=(32, 32, 1)),
    AveragePooling2D((2, 2), strides=2),
    Conv2D(16, (5, 5), activation='tanh', padding='valid'),
    AveragePooling2D((2, 2), strides=2),
    Flatten(),
    Dense(120, activation='tanh'),
    Dense(84, activation='tanh'),
    Dense(10, activation='softmax')
])

# Compile
lenet5.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Training
history = lenet5.fit(x_train_padded, y_train_cat,
                     epochs=10,
                     batch_size=128,
                     validation_split=0.1)

# Evaluation
test_loss, test_acc = lenet5.evaluate(x_test_padded, y_test_cat)
print(f"\nLeNet-5 Test Accuracy: {test_acc:.4f}")
```

LeNet-5 MNIST ডেটাসেটে সাধারণত **99%+ accuracy** দেয়  মাত্র 61,706 parameter দিয়ে! এটি প্রমাণ করে যে CNN এর architecture অত্যন্ত efficient  অল্প parameter দিয়েই handwritten digit recognize করা সম্ভব।

### LeNet-5 এর সীমাবদ্ধতা

LeNet-5 এর অনেক সাফল্য থাকলেও এর কিছু সীমাবদ্ধতা আছে যা আধুনিক standard এ clear:

**খুব শ্যালো নেটওয়ার্ক:** মাত্র 2টি convolution layer  আধুনিক CNN এ 50-100+ conv layer থাকে। কম layer মানে কম non-linearity, কম abstract feature hierarchy। LeNet-5 শুধু low-level ও mid-level feature extract করতে পারে, high-level semantic feature (যেমন "এটি একটি গাড়ি" বা "এটি একটি বিড়াল") extract করতে আরও depth দরকার।

**ছোট ইনপুট সাইজ:** 32×32 grayscale ইমেজ  আধুনিক CNN সাধারণত 224×224 RGB ইমেজ নেয়। বড় ইমেজে আরও detail থাকে, আর RGB তে color information থাকে  LeNet-5 এই দুটোই miss করে।

**tanh activation:** ReLU এর তুলনায় tanh এ gradient vanishing বেশি হয়  বিশেষ করে ডিপ নেটওয়ার্ক এ। tanh এর output range [-1, 1], যেখানে input magnitude বড় হলে gradient প্রায় zero হয়ে যায় (saturate)। এটি training কে slow করে এবং ডিপ নেটওয়ার্ক train করা কঠিন করে।

**No regularization:** LeNet-5 এ কোনো Dropout, BatchNormalization, বা data augmentation নেই  কারণ সেই সময়ে এই technique গুলো আবিষ্কৃত হয়নি। বড় ডেটাসেটে বা কমplex task এ overfitting হতে পারে।

**Dense layer overhead:** Parameter এর সিংহভাগ Dense layer এ  যা inefficient। আধুনিক আর্কিটেকচারে GAP ব্যবহার করে Dense layer এর parameter অনেক কমানো হয়।

এই সীমাবদ্ধতা সত্ত্বেও, LeNet-5 কে "অপ্রচলিত" মনে করা ভুল হবে। এটি CNN এর মূল নীতিগুলো প্রতিষ্ঠা করেছে  এবং MNIST এর মতো সহজ task এ এটি আজও effective। এটি learning এর জন্য একটি আদর্শ architecture  কারণ এটি ছোট, বোঝা সহজ, এবং দ্রুত train হয়।

### আধুনিক LeNet-5: ReLU ও MaxPooling দিয়ে

আমরা LeNet-5 এর একটি modernized version ও বানাতে পারি  tanh এর বদলে ReLU এবং average pooling এর বদলে max pooling ব্যবহার করে। এই পরিবর্তনগুলো training দ্রুত করে এবং accuracy কিছুটা বাড়ায়:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Modernized LeNet-5
lenet5_modern = Sequential([
    Conv2D(6, (5, 5), activation='relu', padding='valid',
           input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2), strides=2),

    Conv2D(16, (5, 5), activation='relu', padding='valid'),
    MaxPooling2D((2, 2), strides=2),

    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

lenet5_modern.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

lenet5_modern.summary()
```

এই modernized version এ parameter count একই (61,706), কিন্তু ReLU activation training দ্রুত করে এবং gradient flow ভালো রাখে। Max pooling strongest feature ধরে রাখে  যা average pooling এর চেয়ে empirically ভালো কাজ করে। সাধারণত modernized version আসল version এর তুলনায় 0.1-0.3% বেশি accuracy দেয় MNIST এ।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম LeNet-5  কম্পিউটার ভিশন এর ইতিহাসে প্রথম সফল CNN আর্কিটেকচার। 1998 সালে Yann LeCun এটি তৈরি করেন handwritten digit recognition এর জন্য, এবং এটি banking system এ check reading এ ব্যবহৃত হয়েছিল। LeNet-5 এর architecture: Conv2D(6,5×5,tanh) → AvgPool(2×2) → Conv2D(16,5×5,tanh) → AvgPool(2×2) → Flatten → Dense(120,tanh) → Dense(84,tanh) → Dense(10,softmax)। মাত্র ~61,706 parameter দিয়ে MNIST এ 99%+ accuracy! আধুনিক standard এ LeNet-5 shallow, কিন্তু এর মূল নীতি  convolution → pooling → fully connected  আজও সব CNN এ ব্যবহৃত হয়। পরবর্তী সেকশনে আমরা AlexNet দেখবো  যা 2012 সালে ডিপ লার্নিং বিপ্লব শুরু করেছিল।
