## কাস্টম CNN মডেল বিল্ডিং

ডেটা প্রস্তুত হয়ে গেলে পরবর্তী ধাপ হলো মডেল ডিজাইন করা। এই সেকশনে আমরা নিজেদের CNN architecture ডিজাইন করবো  কোন layer কেন ব্যবহার করবো, filter সংখ্যা কেন বাড়াবো, এবং binary classification এর জন্য output layer কিভাবে সাজাবো  সব বিস্তারিত আলোচনা করবো। মডেল ডিজাইন একটি empirical process  নির্দিষ্ট কোনো "সঠিক" architecture নেই, কিন্তু কিছু well-established principle আছে যা মেনে চললে ভালো result পাওয়া যায়।

### CNN Architecture Design Principles

একটি typical CNN architecture এ দুটি অংশ থাকে: **Feature Extraction** (Conv2D + Pooling layers) এবং **Classification** (Dense layers)। Feature extraction part ইমেজ থেকে hierarchical feature বের করে  shallow layers edges ও textures detect করে, deeper layers আরও complex pattern (shape, object part) detect করে। Classification part এই extracted feature গুলো ব্যবহার করে final prediction করে।

আমাদের architecture এর blueprint:

```
Input: 224x224x3 (RGB image)
    ↓
Conv2D(32, 3×3, relu) → 222×222×32
MaxPool2D(2×2)         → 111×111×32
    ↓
Conv2D(64, 3×3, relu) → 109×109×64
MaxPool2D(2×2)         → 54×54×64
    ↓
Conv2D(128, 3×3, relu) → 52×52×128
MaxPool2D(2×2)          → 26×26×128
    ↓
Flatten                 → 86,528
    ↓
Dense(128, relu)
    ↓
Dense(64, relu)
    ↓
Dense(1, sigmoid)      → Output: 0 or 1 (apple or tomato)
```

### কেন Filter সংখ্যা বাড়াই (32→64→128)?

এটি CNN design এর একটি অত্যন্ত গুরুত্বপূর্ণ principle। যেহেতু pooling layer spatial dimension (height, width) কমায়, আমরা feature map এর spatial resolution হারাই। এই loss compensate করার জন্য আমরা filter সংখ্যা বাড়াই  অর্থাৎ, ছোট feature map হলেও সেখানে বেশি channel থাকে, তাই মোট information content প্রায় সমান থাকে।

এটি একটি trade-off: spatial detail কমে, কিন্তু semantic richness বাড়ে। Shallow layers এ (32 filters) ইমেজ এর অনেক spatial detail থাকে  প্রতিটি filter একটি সহজ pattern detect করে (horizontal edge, vertical edge)। এগুলো অনেকগুলো হওয়ার দরকার নেই, কারণ edge type limited। Deep layers এ (128 filters) spatial resolution কম কিন্তু প্রতিটি location এ অনেক বেশি channel  প্রতিটি channel একটি complex, abstract feature represent করে (apple shape, tomato color pattern)। এই abstract feature গুলোর variety অনেক বেশি  তাই বেশি filter দরকার।

একটি analogy দিলে: একটি বই এর প্রতিটি পৃষ্ঠায় অনেক word থাকে (high spatial, low channel)  কিন্তু বই এর summary তে কম word এ অনেক বেশি condensed information থাকে (low spatial, high channel)। CNN ও ঠিক এভাবেই কাজ করে!

### কেন Spatial Dimension কমাই?

Pooling layer দিয়ে আমরা intentionally spatial dimension কমাই। এর কয়েকটি কারণ:

**Computation কমানো:** 224×224 feature map এ convolution করা অনেক expensive  26×26 feature map এ convolution অনেক দ্রুত। Spatial dimension অর্ধেক হলে computation প্রায় 4 গুণ কমে।

**Receptive field বাড়ানো:** Deep layers এর প্রতিটি neuron ইনপুট ইমেজ এর একটি বড় অংশ "দেখে"  একে receptive field বলে। Pooling ছাড়া অনেক deep layer যোগ করতে হতো receptive field বাড়াতে। Pooling effectively receptive field বাড়ায়  ফলে কম layer এই বড় pattern detect করা যায়।

**Translation invariance:** একটি object ইমেজ এর যেকোনো position এ থাকতে পারে। Pooling কিছুটা position invariance প্রদান করে  বাম দিকের apple ও ডান দিকের apple একই feature map activate করে।

### Binary Classification Output Layer

আমাদের task এ দুটি class আছে: apple ও tomato। Binary classification এ output layer তে একটি মাত্র neuron থাকে `sigmoid` activation সহ। Sigmoid function output 0 থেকে 1 এর মধ্যে  যা probability হিসেবে interpret করা যায়। যদি output > 0.5 হয়, তাহলে class 1 (ধরো tomato), আর ≤ 0.5 হলে class 0 (apple)।

কেন 2টি neuron + softmax এর বদলে 1টি neuron + sigmoid? Binary classification এ 2 neuron ব্যবহার করলেও কাজ হবে  কিন্তু 1 neuron sigmoid বেশি efficient: কম parameter, কম computation, আর `binary_crossentropy` loss numerically stable। যখন class দুটি, তখন sigmoid + binary_crossentropy ই industry standard।

### কাস্টম CNN মডেল কোড

এখন আমরা আমাদের architecture Keras Sequential API দিয়ে ইমপ্লিমেন্ট করবো:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Custom CNN Model
model = Sequential([
    # === Feature Extraction Part ===

    # Block 1: 32 filters
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(224, 224, 3), name='conv1'),
    MaxPooling2D((2, 2), name='pool1'),

    # Block 2: 64 filters
    Conv2D(64, (3, 3), activation='relu', name='conv2'),
    MaxPooling2D((2, 2), name='pool2'),

    # Block 3: 128 filters
    Conv2D(128, (3, 3), activation='relu', name='conv3'),
    MaxPooling2D((2, 2), name='pool3'),

    # === Classification Part ===

    # Flatten: 3D feature map → 1D vector
    Flatten(name='flatten'),

    # Dense layers
    Dense(128, activation='relu', name='dense1'),
    Dense(64, activation='relu', name='dense2'),

    # Output layer: 1 neuron + sigmoid (binary classification)
    Dense(1, activation='sigmoid', name='output')
])

# Compile model
model.compile(
    optimizer='adam',                   # Adam optimizer (adaptive learning rate)
    loss='binary_crossentropy',         # Binary classification loss
    metrics=['accuracy']                # Accuracy metric
)

# View model summary
model.summary()
```

### model.summary() Output বোঝা

`model.summary()` আউটপুটে প্রতিটি layer এর output shape ও parameter count দেখায়। আসুন বিস্তারিত বুঝি:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1 (Conv2D)              (None, 222, 222, 32)      896
 pool1 (MaxPooling2D)        (None, 111, 111, 32)      0
 conv2 (Conv2D)              (None, 109, 109, 64)      18496
 pool2 (MaxPooling2D)        (None, 54, 54, 64)        0
 conv3 (Conv2D)              (None, 52, 52, 128)       73856
 pool3 (MaxPooling2D)        (None, 26, 26, 128)       0
 flatten (Flatten)           (None, 86528)             0
 dense1 (Dense)              (None, 128)               11075712
 dense2 (Dense)              (None, 64)                8256
 output (Dense)              (None, 1)                 65
=================================================================
Total params: 11,177,281
Trainable params: 11,177,281
Non-trainable params: 0
_________________________________________________________________
```

Parameter count কিভাবে হিসাব করি? প্রতিটি Conv2D layer এর parameter = (kernel_h × kernel_w × input_channels + 1) × output_channels (এখানে +1 হলো bias)।

- **Conv2D(32, 3×3):** (3×3×3 + 1) × 32 = 28 × 32 = **896**
- **Conv2D(64, 3×3):** (3×3×32 + 1) × 64 = 289 × 64 = **18,496**
- **Conv2D(128, 3×3):** (3×3×64 + 1) × 128 = 577 × 128 = **73,856**
- **Dense(128):** 86,528 × 128 + 128 = **11,075,712**
- **Dense(64):** 128 × 64 + 64 = **8,256**
- **Dense(1):** 64 × 1 + 1 = **65**

একটি চমকপ্রদ পর্যবেক্ষণ: **মোট parameter এর 99% Dense layer এ!** Dense(128) এ একা 11,075,712 টি parameter  মোট parameter এর প্রায় 99.1%! এটি আমাদের মডেলের সবচেয়ে বড় weakness  Flatten এর পরে feature vector অনেক বড় (86,528) হয়ে যায়, যা Dense layer এ parameter explosion ঘটায়। এই problem solve করার জন্য আধুনিক architecture গুলো Global Average Pooling (GAP) ব্যবহার করে  যা spatial dimension কে 1×1 তে কমিয়ে Dense layer এর parameter অনেক কমায়।

### কম্পাইলিং: Adam ও Binary Crossentropy

মডেল compile করার সময় তিনটি জিনিস specify করতে হয়:

**Optimizer: `adam`**  Adam (Adaptive Moment Estimation) হলো সবচেয়ে জনপ্রিয় optimizer। এটি প্রতিটি parameter এর জন্য individual learning rate maintain করে  যে parameter এর gradient বড়, তার learning rate কম; যার gradient ছোট, তার learning rate বেশি। এতে training stable ও fast হয়। Adam এর default learning rate 0.001  বেশিরভাগ case এ এটি ভালো কাজ করে।

**Loss: `binary_crossentropy`**  Binary classification এর জন্য standard loss function। এটি predicted probability ও true label এর মধ্যে difference measure করে: loss = -(y×log(p) + (1-y)×log(1-p)), যেখানে y = true label, p = predicted probability। সঠিক prediction এ loss কাছাকাছি 0 হয়, ভুল prediction এ loss অনেক বড় হয়।

**Metrics: `accuracy`**  Training চলাকালীন performance track করার জন্য। Accuracy = (সঠিক prediction) / (মোট prediction)।

### সারসংক্ষেপ

এই সেকশনে আমরা একটি কাস্টম CNN architecture ডিজাইন করলাম: Conv2D(32)→MaxPool→Conv2D(64)→MaxPool→Conv2D(128)→MaxPool→Flatten→Dense(128)→Dense(64)→Dense(1,sigmoid)। আমরা শিখলাম কেন filter সংখ্যা বাড়াই (spatial resolution কমে → channel বাড়াই information compensate করি), কেন spatial dimension কমাই (computation কমানো, receptive field বাড়ানো, translation invariance), এবং কেন binary classification এ 1 neuron + sigmoid ব্যবহার করি। আমরা `model.summary()` output analyze করে দেখলাম যে Dense layer এ parameter এর সিংহভাগ  এটি একটি known issue যা Global Average Pooling দিয়ে solve করা যায়। মডেল compile করলাম Adam optimizer ও binary_crossentropy loss দিয়ে। পরবর্তী সেকশনে আমরা এই মডেল ট্রেইন করবো!
