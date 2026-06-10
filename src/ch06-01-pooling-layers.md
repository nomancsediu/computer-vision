## ম্যাক্স পুলিং ও এভারেজ পুলিং

আগের চ্যাপ্টারে আমরা দেখেছি কিভাবে convolution layer feature map তৈরি করে। কিন্তু যত বেশি convolution layer যোগ করা হয়, feature map এর spatial dimension তত বেশি বড় হতে থাকে (নতুন ফিল্টার যোগ হওয়ার কারণে channel বাড়ে), আর computation ও memory এর চাপ বাড়তে থাকে। এছাড়া, feature map এর প্রতিটি pixel যে আলাদাভাবে গুরুত্বপূর্ণ তা নয় — অনেক সময় একটি ছোট অঞ্চলের সবচেয়ে শক্তিশালী activation-ই আমাদের জানার জন্য যথেষ্ট। এই সমস্যা সমাধানের জন্য CNN এ **পুলিং লেয়ার (Pooling Layer)** ব্যবহার করা হয়। পুলিং লেয়ার feature map এর spatial dimension কমায়, গুরুত্বপূর্ণ তথ্য ধরে রাখে, এবং নেটওয়ার্ককে আরও robust করে তোলে। এই সেকশনে আমরা পুলিং এর ধরন, কেন এটি দরকার, এবং Keras দিয়ে কিভাবে ইমপ্লিমেন্ট করতে হয় — সব বিস্তারিত শিখবো।

### পুলিং কি?

পুলিং হলো একটি downsampling অপারেশন যা feature map এর spatial dimension (width ও height) কমায়, কিন্তু channel সংখ্যা অপরিবর্তিত রাখে। সহজ কথায়, পুলিং feature map কে ছোট করে — ঠিক যেভাবে একটি বড় ছবিকে resize করে ছোট করা যায়, কিন্তু পুলিং এ একটি নির্দিষ্ট নিয়ম অনুসারে value বাছাই করা হয়। পুলিং লেয়ার একটি ছোট window (যাকে pooling window বা pooling kernel বলে) feature map এর উপর দিয়ে slide করে এবং প্রতিটি window থেকে একটি single value output করে। এই window এর size ও কত দূর পরপর সরবে তা (stride) আমরা specify করি।

পুলিং এর সবচেয়ে গুরুত্বপূর্ণ বৈশিষ্ট্য হলো — এটি **প্রতিটি channel এ স্বাধীনভাবে (independently)** কাজ করে। অর্থাৎ, যদি feature map এর dimension হয় H × W × C, তাহলে পুলিং প্রতিটি C টি channel আলাদাভাবে process করে, এবং আউটপুটের dimension হয় H' × W' × C — channel সংখ্যা একই থাকে। এটি convolution থেকে ভিন্ন, যেখানে সব channel একসাথে combine হয়। পুলিং শুধু spatial dimension নিয়ে কাজ করে, depth নিয়ে নয়।

পুলিং এর আরেকটি মূল বৈশিষ্ট্য — এতে **কোনো learnable parameter নেই**। Convolution layer এ kernel এর weight learn হয় training এর সময়, কিন্তু পুলিং তে নিয়মটা fixed — max pooling হলে window এর maximum নেওয়া, average pooling হলে average নেওয়া। কোনো weight বা bias learn হয় না। তাই pooling layer কে overhead হিসেবে না ভেবে, একটি smart downsampling strategy হিসেবে ভাবা উচিত।

### ম্যাক্স পুলিং (Max Pooling)

ম্যাক্স পুলিং হলো সবচেয়ে বেশি ব্যবহৃত pooling পদ্ধতি। এটি pooling window এর মধ্যে থাকা value গুলোর মধ্যে **সর্বোচ্চ (maximum)** value টি বেছে নেয়। যেমন, 2×2 max pooling window তে 4টি value থাকলে তাদের মধ্যে সবচেয়ে বড়টি output হবে। এটি খুবই intuitive — যদি একটি window এর মধ্যে কোনো feature "present" থাকে (অর্থাৎ high activation থাকে), তাহলে max value সেটি ধরে রাখবে।

ম্যাক্স পুলিং এর intuition টা এভাবে বোঝা যায়: convolution layer এর feature map এ high activation value মানে সেই spatial location এ কোনো feature detect হয়েছে। যেমন, একটি edge-detecting filter যদি কোনো এজ detect করে, তাহলে সেই এজ এর location এ activation high হবে। ম্যাক্স পুলিং সেই high activation value টি ধরে রাখে — অর্থাৎ "এই অঞ্চলে এই feature আছে" এই তথ্যটুকু preserve করে, আর বাকি less important value গুলো discard করে। এটি অনেকটা এমন — তুমি একটি room এ 4 জন ছাত্রের পরীক্ষার ফল জানতে চাও, আর room এর সেরা ফলটাই তোমার কাছে যথেষ্ট।

একটি 4×4 feature map এ 2×2 max pooling stride 2 দিয়ে কিভাবে কাজ করে তা দেখি:

```
ইনপুট Feature Map (4×4):
┌────┬────┬────┬────┐
│ 1  │ 3  │ 2  │ 1  │
├────┼────┼────┼────┤
│ 5  │ 6  │ 0  │ 4  │
├────┼────┼────┼────┤
│ 3  │ 2  │ 8  │ 7  │
├────┼────┼────┼────┤
│ 1  │ 0  │ 3  │ 4  │
└────┴────┴────┴────┘

2×2 Max Pooling (stride=2):

Window 1 (উপর-বাম):  max(1,3,5,6) = 6
Window 2 (উপর-ডান):  max(2,1,0,4) = 4
Window 3 (নিচ-বাম):  max(3,2,1,0) = 3
Window 4 (নিচ-ডান):  max(8,7,3,4) = 8

আউটপুট (2×2):
┌────┬────┐
│ 6  │ 4  │
├────┼────┤
│ 3  │ 8  │
└────┴────┘
```

দেখো, 4×4 feature map 2×2 তে সংকুচিত হয়েছে — dimension অর্ধেক হয়ে গেছে, কিন্তু প্রতিটি অঞ্চলের সবচেয়ে শক্তিশালী activation ধরে রাখা হয়েছে। এটিই max pooling এর মূল শক্তি — dimension কমানো সত্ত্বেও গুরুত্বপূর্ণ feature information preserve করা।

### এভারেজ পুলিং (Average Pooling)

এভারেজ পুলিং pooling window এর মধ্যে থাকা value গুলোর **গড় (average)** বের করে। যেমন, 2×2 average pooling window তে 4টি value এর average output হবে। এটি max pooling এর চেয়ে less aggressive — সব value এর contribution নেয়, শুধু সর্বোচ্চটা নয়।

একটি 4×4 feature map এ 2×2 average pooling stride 2 দিয়ে:

```
ইনপুট Feature Map (4×4):
┌────┬────┬────┬────┐
│ 1  │ 3  │ 2  │ 1  │
├────┼────┼────┼────┤
│ 5  │ 6  │ 0  │ 4  │
├────┼────┼────┼────┤
│ 3  │ 2  │ 8  │ 7  │
├────┼────┼────┼────┤
│ 1  │ 0  │ 3  │ 4  │
└────┴────┴────┴────┘

2×2 Average Pooling (stride=2):

Window 1:  avg(1,3,5,6) = 15/4 = 3.75
Window 2:  avg(2,1,0,4) = 7/4  = 1.75
Window 3:  avg(3,2,1,0) = 6/4  = 1.50
Window 4:  avg(8,7,3,4) = 22/4 = 5.50

আউটপুট (2×2):
┌──────┬──────┐
│ 3.75 │ 1.75 │
├──────┼──────┤
│ 1.50 │ 5.50 │
└──────┴──────┘
```

এভারেজ পুলিং সাধারণত CNN এর intermediate layer এ খুব বেশি ব্যবহৃত হয় না — কারণ এটি feature এর শক্তিশালী signal কে অন্যান্য weak signal দিয়ে dilute করে দেয়। কিন্তু এভারেজ পুলিং এর একটি অত্যন্ত গুরুত্বপূর্ণ ব্যবহার আছে: **Global Average Pooling (GAP)**। GAP তে pooling window এর size পুরো feature map এর সমান — অর্থাৎ পুরো spatial dimension collapse করে একটি single average value তে পরিণত হয়। যেমন, 7×7×512 feature map এ GAP প্রয়োগ করলে আউটপুট হবে 1×1×512 — মাত্র 512টি value। এটি Flatten + Dense layer এর alternative হিসেবে কাজ করে এবং অনেক কম parameter ব্যবহার করে। ResNet, GoogLeNet এর মতো আধুনিক আর্কিটেকচারে GAP ব্যবহার হয়।

### পুলিং কেন দরকার?

পুলিং লেয়ার CNN এ কেন এত গুরুত্বপূর্ণ তার তিনটি প্রধান কারণ আছে:

**১. Dimensionality Reduction (ডাইমেনশনালিটি রিডাকশন):** পুলিং feature map এর spatial dimension কমায়, যার ফলে পরবর্তী layer এ কম parameter লাগে এবং computation কম হয়। যেমন, 2×2 max pooling stride 2 দিলে spatial dimension অর্ধেক হয়ে যায় — 28×28 feature map 14×14 হয়ে যায়। এর মানে parameter ও computation প্রায় ৪ গুণ কমে যায় (কারণ 28×28 = 784, আর 14×14 = 196)। কম parameter মানে training দ্রুত হবে, memory কম লাগবে, এবং overfitting এর সম্ভাবনা কমবে। ডিপ নেটওয়ার্ক এ একাধিক pooling layer থাকে, যা step by step dimension কমায় — 224→112→56→28→14→7 — এভাবে image-level representation তৈরি হয়।

**২. Translation Invariance (ট্রান্সলেশন ইনভ্যারিয়েন্স):** পুলিং নেটওয়ার্ককে কিছুটা translation invariance প্রদান করে — অর্থাৎ ইমেজে feature একটু সরে গেলেও (shift হলেও) pooling output অনেকটা একই থাকে। ধরো একটি 4×4 region এ একটি edge আছে, আর max pooling সেই region এর max নিচ্ছে। Edge টা একটু বামে বা ডানে সরলেও region এর max value প্রায় একই থাকবে। এটি অনেকটা এমন — তুমি একটি object কে একটু নড়াচড়া করলেও max pooling তার presence ধরে রাখবে। এই property classification task এ খুবই কাজে লাগে — বিড়াল ইমেজের যেকোনো কোণায় থাকুক না কেন, আমরা চাই নেটওয়ার্ক "বিড়াল" বলুক।

**৩. Increasing Receptive Field (রিসেপ্টিভ ফিল্ড বৃদ্ধি):** Receptive field হলো output feature map এর একটি pixel ইনপুট ইমেজের কতটুকু অংশ দেখছে। Pooling dimension কমানোর ফলে deeper layer এর একটি neuron ইনপুট ইমেজের আরও বড় অংশ cover করতে পারে। যেমন, প্রথম conv layer এর receptive field 3×3 হতে পারে, কিন্তু একটি pooling এর পরে পরবর্তী conv layer এর receptive field effectively বেড়ে যায়। এভাবে ডিপ নেটওয়ার্ক এর deeper layer গুলো ইমেজের বড় বড় structure (যেমন পুরো একটি object) detect করতে পারে, শুধু local edge বা texture নয়।

### 2×2 ম্যাক্স পুলিং: ডাইমেনশন অর্ধেক

CNN এ সবচেয়ে বেশি ব্যবহৃত pooling configuration হলো **2×2 max pooling, stride 2**। এটি feature map এর width ও height উভয়কে অর্ধেক করে দেয়। কিছু common example:

```
ইনপুট: 28×28×32  →  2×2 MaxPool  →  আউটপুট: 14×14×32
ইনপুট: 14×14×64  →  2×2 MaxPool  →  আউটপুট: 7×7×64
ইনপুট: 224×224×64  →  2×2 MaxPool  →  আউটপুট: 112×112×64
```

খেয়াল করো — channel সংখ্যা অপরিবর্তিত থাকে, শুধু spatial dimension অর্ধেক হয়। এই সহজ অপারেশনটি CNN এ অত্যন্ত শক্তিশালী — এটি without any learnable parameter feature map downsample করে, computation ৪ গুণ কমায়, এবং গুরুত্বপূর্ণ feature ধরে রাখে।

Keras এ pooling এর output shape হিসাব করার ফর্মুলা:

```
Output Size = floor((Input Size - Pool Size) / Stride) + 1
```

2×2 pooling, stride 2 এর জন্য:
```
Output Size = floor((28 - 2) / 2) + 1 = floor(13) + 1 = 14
```

### Keras দিয়ে পুলিং ইমপ্লিমেন্টেশন

Keras এ পুলিং লেয়ার ইমপ্লিমেন্ট করা অত্যন্ত সহজ। চলো দেখি কিভাবে MaxPooling2D ও AveragePooling2D ব্যবহার করতে হয়:

**MaxPooling2D:**

```python
from tensorflow.keras.layers import MaxPooling2D

# সবচেয়ে সাধারণ configuration
pool = MaxPooling2D(pool_size=(2, 2), strides=2)

# strides specify না করলে default = pool_size
# নিচের দুটি সমার্থক:
pool = MaxPooling2D(pool_size=(2, 2), strides=2)
pool = MaxPooling2D(pool_size=(2, 2))  # strides স্বয়ংক্রিয়ভাবে (2,2)

# padding specify করা (default='valid')
pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
pool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
```

**AveragePooling2D:**

```python
from tensorflow.keras.layers import AveragePooling2D

# Average pooling
pool = AveragePooling2D(pool_size=(2, 2), strides=2)

# Global Average Pooling
from tensorflow.keras.layers import GlobalAveragePooling2D
gap = GlobalAveragePooling2D()  # পুরো spatial dimension collapse করে 1D vector তৈরি করে
```

**GlobalAveragePooling2D vs Flatten:** Global Average Pooling এবং Flatten দুটোই spatial dimension শেষ করে 1D vector তৈরি করে, কিন্তু কাজের ধরন আলাদা। Flatten সব value রেখে দেয় — 7×7×512 feature map থেকে 7×7×512 = 25,088 দৈর্ঘ্যের vector তৈরি করে। কিন্তু GAP প্রতিটি channel এর spatial average নেয় — 7×7×512 থেকে মাত্র 512 দৈর্ঘ্যের vector তৈরি করে। GAP অনেক কম parameter ব্যবহার করে এবং overfitting কমায়।

### পুলিং সহ একটি সম্পূর্ণ CNN মডেল

এখন আমরা একটি সম্পূর্ণ CNN মডেল বানাবো যেখানে convolution ও pooling layer একসাথে কাজ করবে:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential([
    # Block 1: Conv + Pool
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2), strides=2),  # 28×28 → 14×14

    # Block 2: Conv + Pool
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),  # 14×14 → 7×7

    # Block 3: Conv + Pool
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),  # 7×7 → 3×3

    # Classifier
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()
```

এই মডেলের output:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320
 max_pooling2d (MaxPooling2D) (None, 14, 14, 32)       0
 conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496
 max_pooling2d_1 (MaxPooling2D) (None, 7, 7, 64)       0
 conv2d_2 (Conv2D)           (None, 7, 7, 128)         73856
 max_pooling2d_2 (MaxPooling2D) (None, 3, 3, 128)      0
 flatten (Flatten)           (None, 1152)              0
 dense (Dense)               (None, 128)               147584
 dense_1 (Dense)             (None, 10)                1290
=================================================================
Total params: 241,546
Trainable params: 241,546
Non-trainable params: 0
_________________________________________________________________
```

গুরুত্বপূর্ণ পর্যবেক্ষণ: MaxPooling2D লেয়ারে **Param # = 0** — কারণ pooling এ কোনো learnable parameter নেই! তারপরও এটি spatial dimension কমাচ্ছে: **28→14→7→3**। Flatten layer এ 3×3×128 = 1,152 টি value আসছে। যদি pooling না থাকতো, তাহলে Flatten এ আসতো 28×28×128 = 100,352 টি value — যা প্রায় 87 গুণ বেশি! এত বেশি parameter দিয়ে Dense layer train করা overfitting এর কারণ হতো।

### পুলিং বনাম স্ট্রাইডেড কনভোলিউশন: বিতর্ক

পুলিং লেয়ার কি সত্যিই দরকার? এটি নিয়ে ডিপ লার্নিং community তে যথেষ্ট বিতর্ক আছে। একটি মতবাদ হলো — pooling তে কোনো learnable parameter নেই, তাই এটি information loss করে। বিকল্প হিসেবে **strided convolution** ব্যবহার করা যায়, যেখানে stride > 1 দিয়ে downsampling করা হয়। Strided convolution এ learnable parameter আছে, তাই নেটওয়ার্ক নিজেই শিখতে পারে কিভাবে downsample করতে হয় — শুধু max বা average নেওয়া নয়।

```python
# পুলিং দিয়ে downsampling
model_pooling = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=2),  # কোনো learnable parameter নেই
])

# Strided convolution দিয়ে downsampling
model_strided = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),  # learnable parameter আছে
])
```

Strided convolution এর সুবিধা হলো এটি learnable — নেটওয়ার্ক নিজেই optimal downsampling strategy শিখতে পারে। কিন্তু এর অসুবিধা হলো বেশি parameter এবং overfitting এর ঝুঁকি। অন্যদিকে, max pooling simple, fast, এবং empirically অনেক ভালো কাজ করে। বাস্তবে, বেশিরভাগ জনপ্রিয় আর্কিটেকচার (VGG, ResNet, এমনকি অনেক আধুনিক model) এখনও max pooling বা strided convolution ব্যবহার করে। কিছু আর্কিটেকচার (যেমন ResNet) প্রথম layer এ strided convolution ব্যবহার করে, আর পরবর্তী downsampling এ আবার pooling ব্যবহার করে — মানে দুটোরই জায়গা আছে। সবচেয়ে গুরুত্বপূর্ণ বিষয় হলো — spatial dimension কমানো কোনোভাবেই দরকার, সেটা pooling দিয়েই হোক বা strided convolution দিয়েই হোক।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম pooling layer কি, কেন এটি দরকার, এবং কিভাবে ইমপ্লিমেন্ট করতে হয়। ম্যাক্স পুলিং window এর maximum value নেয় — এটি সবচেয়ে জনপ্রিয় কারণ এটি strongest activation ধরে রাখে, যা feature presence নির্দেশ করে। এভারেজ পুলিং window এর average নেয় — এটি intermediate layer এ কম ব্যবহৃত, কিন্তু Global Average Pooling (GAP) আধুনিক আর্কিটেকচারে অত্যন্ত গুরুত্বপূর্ণ। পুলিং এর তিনটি মূল সুবিধা: (1) dimensionality reduction — কম parameter, কম computation, কম overfitting, (2) translation invariance — feature shift হলেও output stable থাকে, (3) receptive field বৃদ্ধি — deeper layer ইমেজের বড় অংশ দেখতে পায়। Pooling vs strided convolution বিতর্ক এখনও চলমান, কিন্তু max pooling বাস্তবে সবচেয়ে বেশি ব্যবহৃত এবং empirically proven। পরবর্তী সেকশনে আমরা প্রথম historical CNN আর্কিটেকচার — LeNet-5 — স্ক্র্যাচ থেকে ইমপ্লিমেন্ট করবো।
