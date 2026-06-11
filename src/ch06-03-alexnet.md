## AlexNet আর্কিটেকচার

যদি LeNet-5 হলো CNN এর জন্ম, তাহলে AlexNet হলো CNN এর জাগরণ। 2012 সালে Alex Krizhevsky, Ilya Sutskever এবং Geoffrey Hinton এর তৈরি AlexNet ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 তে এতটাই চমৎকার performance দেখায় যে পুরো কম্পিউটার ভিশন কমিউনিটি অবাক হয়ে যায়। AlexNet এর top-5 error rate ছিল **15.3%**, যেখানে runner-up এর error rate ছিল **26.2%**  প্রায় 11 শতাংশ পয়েন্ট ব্যবধান! এই massive victory ডিপ লার্নিং বিপ্লবের সূচনা করে। এর আগে ডিপ লার্নিং কে অনেকেই "academic curiosity" বলে ভাবতো  AlexNet প্রমাণ করলো যে ডিপ CNN রিয়েল-ওয়ার্ল্ড ইমেজ ক্লাসিফিকেশন এ traditional method গুলোকে বড় ব্যবধানে হারাতে পারে। এই সেকশনে আমরা AlexNet এর architecture বিস্তারিতভাবে বুঝবো, এর key innovations শিখবো, এবং Keras দিয়ে স্ক্র্যাচ থেকে ইমপ্লিমেন্ট করবো।

### ILSVRC 2012: ডিপ লার্নিং বিপ্লবের সূচনা

ILSVRC (ImageNet Large Scale Visual Recognition Challenge) হলো ইমেজ ক্লাসিফিকেশন এর সবচেয়ে প্রতিযোগিতামূলক benchmark। এতে 1.2 মিলিয়ন training image, 50,000 validation image, এবং 1,000টি category থাকে  কুকুর, গাড়ি, ফুল, খাবার থেকে শুরু করে বিভিন্ন type এর object। 2012 সালের আগে এই competition এ traditional computer vision method গুলো dominate করতো  SIFT, HOG, Bag of Visual Words এর মতো hand-crafted feature extraction + SVM/Random Forest classifier। Error rate ধীরে ধীরে কমছিল  2010 এ 28.2%, 2011 এ 25.8%  মাত্র 2-3% improvement প্রতি বছর।

তারপর 2012 তে AlexNet এলো  এবং error rate এক লাফে 15.3% এ নেমে এলো! 10.5% improvement এক বছরে  যা আগের 2 বছরের total improvement এর চেয়েও বেশি। এটি শুধু accuracy এর improvement ছিল না  এটি একটি paradigm shift ছিল। AlexNet প্রমাণ করলো যে (1) বড় dataset এ ডিপ CNN traditional method কে massively outperform করে, (2) GPU দিয়ে বড় নেটওয়ার্ক train করা সম্ভব, (3) data augmentation ও regularization দিয়ে overfitting কন্ট্রোল করা যায়। এই ফলাফলের পর পুরো AI industry ডিপ লার্নিং এর দিকে ঘুরে যায়  Google, Facebook, Microsoft সবাই বিশাল investment শুরু করে।

### AlexNet আর্কিটেকচার ওভারভিউ

AlexNet এর architecture LeNet-5 এর তুলনায় অনেক বড় ও deep  8টি learnable layer (5 conv + 3 dense) এবং মোট ~62.4 মিলিয়ন parameter। এটি 224×224 RGB ইমেজ input হিসেবে নেয়। Architecture টি এভাবে:

```
Input: 224×224×3 (RGB image)
    ↓
Conv2D(96, 11×11, stride=4, ReLU) → 55×55×96
    ↓
MaxPooling2D(3×3, stride=2) → 27×27×96
    ↓
BatchNormalization
    ↓
Conv2D(256, 5×5, padding='same', ReLU) → 27×27×256
    ↓
MaxPooling2D(3×3, stride=2) → 13×13×256
    ↓
BatchNormalization
    ↓
Conv2D(384, 3×3, padding='same', ReLU) → 13×13×384
    ↓
Conv2D(384, 3×3, padding='same', ReLU) → 13×13×384
    ↓
Conv2D(256, 3×3, padding='same', ReLU) → 13×13×256
    ↓
MaxPooling2D(3×3, stride=2) → 6×6×256
    ↓
Flatten → 9,216
    ↓
Dense(4096, ReLU) → Dropout(0.5)
    ↓
Dense(4096, ReLU) → Dropout(0.5)
    ↓
Dense(1000, softmax)
    ↓
Output: 1000 classes (ImageNet)
```

### AlexNet এর Key Innovations

AlexNet শুধু বড় নেটওয়ার্ক ছিল না  এটি একাধিক innovation একসাথে প্রয়োগ করেছিল যা একে সফল করেছিল:

**১. ReLU Activation:** AlexNet সর্বপ্রথম বড় আকারে ReLU (Rectified Linear Unit) activation ব্যবহার করে। আগের CNN গুলো (LeNet-5 সহ) tanh বা sigmoid ব্যবহার করতো, যেগুলোতে gradient vanishing problem হয়  বিশেষ করে ডিপ নেটওয়ার্ক এ। ReLU এর gradient positive region এ সবসময় 1, তাই gradient efficiently backpropagate হয় এবং training অনেক দ্রুত হয়। AlexNet এর পেপারে দেখানো হয় যে ReLU দিয়ে tanh এর তুলনায় 6 গুণ দ্রুত training converge হয়! এই innovation এর পর থেকে ReLU CNN এর default activation হয়ে যায়।

**২. Dropout Regularization:** AlexNet এর Dense layer এ Dropout (rate=0.5) ব্যবহার করা হয়। Dropout training এর সময় র‍্যান্ডমলি 50% neuron deactivate করে দেয়  যা overfitting কমায়। বড় নেটওয়ার্ক এ (62M parameter!) overfitting এর ঝুঁকি অনেক বেশি, কারণ নেটওয়ার্ক training data memorize করে ফেলতে পারে। Dropout নেটওয়ার্ক কে এমনভাবে train করতে বাধ্য করে যে কোনো single neuron খুব বেশি depend না করে  এটি redundant representation তৈরি করে এবং generalization উন্নত করে। Srivastava এর 2014 সালের paper "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" এই technique formally establish করে।

**৩. GPU Training:** AlexNet প্রথম বড় CNN যা GPU তে train করা হয়। আসল paper এ two GTX 580 GPU ব্যবহার করা হয়েছিল  নেটওয়ার্ক কে দুইভাগে ভাগ করে প্রতিটি GPU তে অর্ধেক করে train করা হতো। GPU তে matrix multiplication অনেক দ্রুত হয় CPU এর তুলনায়  ফলে 62M parameter এর এই বিশাল নেটওয়ার্ক 5-6 দিনে train হয়েছিল, যা CPU তে হলে মাস লেগে যেতো। এই innovation প্রমাণ করে যে GPU computing ডিপ লার্নিং এর জন্য game-changer।

**৪. Data Augmentation:** AlexNet training data artificially বাড়ানোর জন্য data augmentation ব্যবহার করে  random crop (224×224 patches from 256×256 images), horizontal flip, এবং color jittering (PCA-based RGB shift)। Data augmentation training data এর diversity বাড়ায় এবং overfitting কমায়। 1.2M training image থেকে augmentation দিয়ে effectively অনেক বেশি training example তৈরি করা হয়। এই technique আজও প্রায় সব image classification model এ ব্যবহৃত হয়।

**৫. BatchNormalization (আধুনিক সংযোজন):** আসল AlexNet পেপারে BatchNormalization ছিল না (এটি 2015 সালে Ioffe ও Szegedy আবিষ্কার করেন)। কিন্তু আধুনিক implementation এ BN যোগ করা হয় কারণ এটি training stabilize করে, higher learning rate ব্যবহার করতে দেয়, এবং overall performance উন্নত করে। BN প্রতিটি mini-batch এর activation কে normalize করে (mean=0, variance=1), যা internal covariate shift কমায়।

### AlexNet আর্কিটেকচার টেবিল

প্রতিটি layer এর output shape ও parameter count:

| Layer | Type | Output Shape | Parameter |
|------|------|-------------|-----------|
| 1 | Conv2D(96, 11×11, stride=4) | 55×55×96 | 34,944 |
| 2 | MaxPool2D(3×3, stride=2) | 27×27×96 | 0 |
| 3 | BatchNormalization | 27×27×96 | 384 |
| 4 | Conv2D(256, 5×5, same) | 27×27×256 | 614,656 |
| 5 | MaxPool2D(3×3, stride=2) | 13×13×256 | 0 |
| 6 | BatchNormalization | 13×13×256 | 1,024 |
| 7 | Conv2D(384, 3×3, same) | 13×13×384 | 885,120 |
| 8 | Conv2D(384, 3×3, same) | 13×13×384 | 1,327,488 |
| 9 | Conv2D(256, 3×3, same) | 13×13×256 | 884,992 |
| 10 | MaxPool2D(3×3, stride=2) | 6×6×256 | 0 |
| 11 | Flatten | 9,216 | 0 |
| 12 | Dense(4096) + Dropout | 4,096 | 37,752,832 |
| 13 | Dense(4096) + Dropout | 4,096 | 16,781,312 |
| 14 | Dense(1000, softmax) | 1,000 | 4,097,000 |
| | | **Total** | **~62.4M** |

খেয়াল করো  parameter এর সিংহভাগ Dense layer এ! Dense(4096) এ একা 37.7M parameter  মোট parameter এর 60% এরও বেশি! এটি AlexNet এর একটি major inefficiency, যা পরবর্তী আর্কিটেকচারগুলো (VGG, ResNet) এ address করা হয়েছে। Convolution layer গুলোতে মাত্র ~3.7M parameter, কিন্তু Dense layer এ ~58.6M  এটি দেখায় যে fully connected layer কতটা parameter-heavy।

### Keras দিয়ে AlexNet ইমপ্লিমেন্টেশন

এখন আমরা AlexNet স্ক্র্যাচ থেকে Keras Sequential API দিয়ে ইমপ্লিমেন্ট করবো। আমরা BatchNormalization সহ modernized version বানাবো:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense,
                                      Flatten, Dropout, BatchNormalization)

# Create AlexNet model
alexnet = Sequential([
    # Layer 1: Conv - 96 filters, 11×11, stride 4
    Conv2D(96, (11, 11), strides=4, activation='relu',
           padding='valid', input_shape=(224, 224, 3), name='conv1'),
    MaxPooling2D((3, 3), strides=2, name='maxpool1'),
    BatchNormalization(name='bn1'),

    # Layer 2: Conv - 256 filters, 5×5, same padding
    Conv2D(256, (5, 5), activation='relu', padding='same', name='conv2'),
    MaxPooling2D((3, 3), strides=2, name='maxpool2'),
    BatchNormalization(name='bn2'),

    # Layer 3: Conv - 384 filters, 3×3, same padding
    Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3'),

    # Layer 4: Conv - 384 filters, 3×3, same padding
    Conv2D(384, (3, 3), activation='relu', padding='same', name='conv4'),

    # Layer 5: Conv - 256 filters, 3×3, same padding
    Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5'),
    MaxPooling2D((3, 3), strides=2, name='maxpool3'),

    # Flatten
    Flatten(name='flatten'),

    # Layer 6: Dense - 4096 + Dropout
    Dense(4096, activation='relu', name='dense1'),
    Dropout(0.5, name='dropout1'),

    # Layer 7: Dense - 4096 + Dropout
    Dense(4096, activation='relu', name='dense2'),
    Dropout(0.5, name='dropout2'),

    # Layer 8: Dense - 1000 (output)
    Dense(1000, activation='softmax', name='output')
])

# View model summary
alexnet.summary()
```

এই মডেলের `summary()` output:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1 (Conv2D)              (None, 55, 55, 96)        34944
 maxpool1 (MaxPooling2D)     (None, 27, 27, 96)        0
 bn1 (BatchNormalization)    (None, 27, 27, 96)        384
 conv2 (Conv2D)              (None, 27, 27, 256)       614656
 maxpool2 (MaxPooling2D)     (None, 13, 13, 256)       0
 bn2 (BatchNormalization)    (None, 13, 13, 256)       1024
 conv3 (Conv2D)              (None, 13, 13, 384)       885120
 conv4 (Conv2D)              (None, 13, 13, 384)       1327488
 conv5 (Conv2D)              (None, 13, 13, 256)       884992
 maxpool3 (MaxPooling2D)     (None, 6, 6, 256)         0
 flatten (Flatten)           (None, 9216)              0
 dense1 (Dense)              (None, 4096)              37752832
 dropout1 (Dropout)          (None, 4096)              0
 dense2 (Dense)              (None, 4096)              16781312
 dropout2 (Dropout)          (None, 4096)              0
 output (Dense)              (None, 1000)              4097000
=================================================================
Total params: 62,378,752
Trainable params: 62,378,048
Non-trainable params: 704
_________________________________________________________________
```

### AlexNet দিয়ে ইমেজ ক্লাসিফিকেশন

AlexNet কে একটি dummy input দিয়ে test করা যাক:

```python
import numpy as np

# Create dummy input (224×224×3 RGB image)
dummy_input = np.random.randn(1, 224, 224, 3).astype('float32')

# Prediction
predictions = alexnet.predict(dummy_input)
print(f"Output shape: {predictions.shape}")
print(f"Sum of probabilities: {predictions.sum():.4f}")
print(f"Top predicted class: {np.argmax(predictions)}")
```

বাস্তবে AlexNet train করতে হলে ImageNet dataset (1.2M training image) দরকার, যা download ও process করতে অনেক সময় ও resource লাগে। তাই আমরা এখানে শুধু architecture তৈরি করেছি। পরবর্তী চ্যাপ্টারে আমরা দেখবো কিভাবে pretrained model (Transfer Learning) ব্যবহার করে কোনো dataset এ classification করা যায়  সেখানে ImageNet pretrained AlexNet, VGG, ResNet এর মতো model ব্যবহার করা হবে।

### LeNet-5 বনাম AlexNet: তুলনা

LeNet-5 (1998) এবং AlexNet (2012)  এই দুটি আর্কিটেকচার 14 বছরের ব্যবধানে তৈরি, এবং এদের মধ্যে পার্থক্য ডিপ লার্নিং এর evolution এর প্রতিফলন:

| বৈশিষ্ট্য | LeNet-5 (1998) | AlexNet (2012) |
|-----------|----------------|----------------|
| প্রস্তাবক | Yann LeCun | Alex Krizhevsky et al. |
| কনভোলিউশন লেয়ার | 2 | 5 |
| মোট প্যারামিটার | ~61K | ~62.4M |
| Activation Function | tanh | ReLU |
| Pooling Type | Average Pooling | Max Pooling |
| Regularization | নেই | Dropout (0.5) |
| Batch Normalization | নেই | আধুনিক version এ আছে |
| ইনপুট সাইজ | 32×32 grayscale | 224×224 RGB |
| ডেটাসেট | MNIST (60K images) | ImageNet (1.2M images) |
| ক্লাস সংখ্যা | 10 | 1,000 |
| GPU Training | না | হ্যাঁ (2× GTX 580) |
| Data Augmentation | নেই | হ্যাঁ (crop, flip, color jitter) |
| Top-5 Error | ~1% (MNIST) | 15.3% (ImageNet) |
| Kernel Size | 5×5 | 11×11, 5×5, 3×3 |
| Training সময় | মিনিট | 5-6 দিন (GPU) |

এই তুলনা থেকে কিছু গুরুত্বপূর্ণ পর্যবেক্ষণ: AlexNet এ parameter LeNet-5 এর চেয়ে প্রায় **1,000 গুণ** বেশি! 61K থেকে 62.4M  এটি শুধু সাইজের বৃদ্ধি নয়, এটি capacity এর বিশাল বৃদ্ধি। বড় dataset (ImageNet) এ বড় model train করা সম্ভব হয়েছে GPU computing এর কারণে। tanh থেকে ReLU তে shift training কে অনেক দ্রুত করেছে। Dropout ও data augmentation overfitting কন্ট্রোল করেছে। এই innovations মিলে AlexNet কে সম্ভব করেছে।

### AlexNet পরবর্তী বিবর্তন

AlexNet এর সাফল্যের পর CNN আর্কিটেকচার এ দ্রুত advancement হতে থাকে। প্রতি বছর ILSVRC এ নতুন architecture আসতে থাকে, এবং error rate দ্রুত কমতে থাকে:

**VGGNet (2014):** Karen Simonyan ও Andrew Zisserman এর VGGNet এর মূল ধারণা  শুধু 3×3 convolution kernel ব্যবহার করা। AlexNet এ 11×11 ও 5×5 kernel ছিল, কিন্তু VGGNet দেখালো যে multiple 3×3 layer stack করে same receptive field পাওয়া যায়, অনেক কম parameter দিয়ে। যেমন, দুটি 3×3 conv layer একটি 5×5 conv layer এর সমান receptive field দেয় (3+3-1=5), কিন্তু parameter কম (2×9=18 vs 25) এবং non-linearity বেশি (2টা ReLU vs 1টা)। VGG-16 তে 16 weight layer ও 138M parameter আছে, এবং এটি architecture এর simplicity ও uniformity এর জন্য বিখ্যাত। VGGNet এর design philosophy আজও influential  অনেক transfer learning application এ VGG-16/VGG-19 ব্যবহৃত হয়।

**GoogLeNet / Inception (2014):** Christian Szegedy এর GoogLeNet এ "Inception module" প্রস্তাব করা হয়  যেখানে একই layer এ 1×1, 3×3, ও 5×5 convolution parallel ভাবে চালানো হয়, এবং result concatenate করা হয়। এটি network কে নিজেই decide করতে দেয় কোন kernel size best। 1×1 convolution দিয়ে dimensionality reduction করা হয়, যা computation কন্ট্রোল রাখে। GoogLeNet এ মাত্র 6.8M parameter  AlexNet এর তুলনায় 10 গুণ কম!  কিন্তু accuracy বেশি। এটি "wider" নেটওয়ার্ক এর concept introduce করে, শুধু "deeper" নয়।

**ResNet (2015):** Kaiming He এর ResNet এ **skip connection (residual connection)** introduce করা হয়  যেখানে input সরাসরি output এ যোগ করা হয়: `output = F(x) + x`। এটি gradient কে সরাসরি flow করতে দেয়, ফলে অত্যন্ত ডিপ নেটওয়ার্ক (152 layer!) train করা সম্ভব হয়। ResNet এর আগে 20-30 layer এর বেশি নেটওয়ার্ক train করা কঠিন ছিল  gradient vanishing/exploding এর কারণে। Skip connection এই সমস্যা solve করে। ResNet-152 তে 152 layer ও 60M parameter আছে, এবং ILSVRC 2015 এ top-5 error 3.57% অর্জন করে  যা human performance (5.1%) কেও ছাড়িয়ে যায়! ResNet এর skip connection আজকাল প্রায় সব আধুনিক আর্কিটেকচারে ব্যবহৃত হয়।

**EfficientNet (2019):** Mingxing Tan ও Quoc Le এর EfficientNet এ **compound scaling** প্রস্তাব করা হয়  network এর depth, width, এবং input resolution একসাথে systematically scale করা। আগের approach এ শুধু depth বা শুধু width বাড়ানো হতো, কিন্তু EfficientNet দেখালো balanced scaling সবচেয়ে effective। EfficientNet-B7 শুধু 66M parameter দিয়ে state-of-the-art accuracy অর্জন করে  আগের model গুলো এর চেয়ে অনেক বেশি parameter ব্যবহার করতো। এটি "efficiency" এর যুগের সূচনা করে  শুধু accuracy নয়, parameter efficiency ও important।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম AlexNet  2012 সালের যুগান্তকারী CNN আর্কিটেকচার যা ডিপ লার্নিং বিপ্লব শুরু করেছিল। AlexNet ILSVRC 2012 তে top-5 error 15.3% দিয়ে runner-up (26.2%) কে massive margin এ হারিয়েছিল। এর key innovations: ReLU activation (faster training), Dropout regularization (overfitting কন্ট্রোল), GPU training (বড় নেটওয়ার্ক train করা সম্ভব), এবং data augmentation (training data diversity বৃদ্ধি)। AlexNet এ ~62.4M parameter  LeNet-5 এর চেয়ে 1,000 গুণ বেশি  যা ImageNet এর মতো বড় dataset এ powerful feature extraction সম্ভব করে। AlexNet পরবর্তী বিবর্তনে VGGNet (3×3 kernel uniformity), GoogLeNet (Inception module), ResNet (skip connection), এবং EfficientNet (compound scaling) এর মতো milestone আর্কিটেকচার এসেছে। এই চ্যাপ্টারে আমরা pooling লেয়ার থেকে শুরু করে LeNet-5 ও AlexNet  দুটি historical CNN আর্কিটেকচার  স্ক্র্যাচ থেকে ইমপ্লিমেন্ট করেছি। পরবর্তী চ্যাপ্টারে আমরা নিজেদের কাস্টম CNN মডেল বানিয়ে একটি সম্পূর্ণ image classification pipeline তৈরি করবো  data preparation থেকে training আর evaluation পর্যন্ত।
