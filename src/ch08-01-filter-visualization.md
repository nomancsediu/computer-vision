## ফিল্টার ভিজুয়ালাইজেশন

CNN এর প্রতিটি convolution layer এ অনেকগুলো ফিল্টার (বা kernel) থাকে  এই ফিল্টার গুলোই হলো নেটওয়ার্কের "চোখ"। Training এর সময় এই ফিল্টার গুলো learnable parameter হিসেবে update হয়, এবং training শেষে প্রতিটি ফিল্টার একটি নির্দিষ্ট visual pattern detect করতে বিশেষজ্ঞ হয়ে যায়। এই সেকশনে আমরা সেই লার্নড ফিল্টার গুলো directly extract করে visualize করবো  দেখবো নেটওয়ার্ক কী কী pattern শিখেছে।

### কেন CNN এর ভিতরে উঁকি দেবো?

ডিপ লার্নিং এর সবচেয়ে বড় criticism হলো এর interpretability problem  "ব্ল্যাক বক্স" সমস্যা। একটি CNN 95% accuracy দিতে পারে, কিন্তু আমরা জানি না এটি কীভাবে সিদ্ধান্ত নিচ্ছে। এটি কি সত্যিই object এর shape দেখে classify করছে, নাকি background color দেখে? নাকি image artifact দেখে cheating করছে? এই প্রশ্নগুলোর উত্তর পাওয়া অত্যন্ত জরুরি  বিশেষ করে safety-critical application এ (medical diagnosis, autonomous driving)।

CNN internal visualization আমাদের কয়েকটি জিনিস বুঝতে সাহায্য করে:

- **কোন pattern শিখেছে:** প্রথম layer এর filter গুলো কেমন? শুধু edge, নাকি কিছু complex?
- **Training সঠিক হয়েছে কিনা:** সব filter যদি random noise এর মতো দেখায়, training ঠিক হয়নি। Dead filter (সব zero) থাকলে সেটিও problem।
- **Architecture ঠিক আছে কিনা:** কোন layer এ কতগুলো active filter আছে, সেটি দেখে architecture adjust করা যায়।
- **Transfer learning decision:** Pretrained model এর কোন layer freeze করবো, কোন layer fine-tune করবো  সেটি filter visualization দেখে decide করা যায়।

### VGG16 মডেল লোড করা

আমরা ImageNet এ pretrained VGG16 মডেল ব্যবহার করবো। VGG16 হলো Oxford এর Visual Geometry Group (VGG) এর তৈরি একটি ক্লাসিক CNN যা 2014 সালে ImageNet challenge এ 2nd place পায়। এর architecture অত্যন্ত clean  শুধু 3×3 convolution আর 2×2 max pooling, কোনো fancy trick নেই। এই simplicity এর কারণে এটি visualization এর জন্য আদর্শ:

```python
from tensorflow.keras.applications import VGG16

# Load ImageNet pretrained VGG16 (convolutional part only)
model = VGG16(weights='imagenet', include_top=False)

# View model summary
model.summary()
```

`include_top=False` মানে আমরা fully connected (Dense) layer গুলো load করছি না  শুধু convolutional feature extraction part। এতে memory বাঁচে এবং আমাদের visualization এর জন্য শুধু conv layer-ই দরকার।

VGG16 এর summary দেখলে আমরা 13টি convolution layer দেখতে পাবো, 5টি block এ ভাগ করা:

```
Model: "vgg16"
_________________________________________________________________
 Block 1:
   block1_conv1 (Conv2D)    (None, ?, ?, 64)     1,792
   block1_conv2 (Conv2D)    (None, ?, ?, 64)     36,928
   block1_pool  (MaxPooling2D)
 Block 2:
   block2_conv1 (Conv2D)    (None, ?, ?, 128)    73,856
   block2_conv2 (Conv2D)    (None, ?, ?, 128)    147,584
   block2_pool  (MaxPooling2D)
 Block 3:
   block3_conv1 (Conv2D)    (None, ?, ?, 256)    295,168
   block3_conv2 (Conv2D)    (None, ?, ?, 256)    590,080
   block3_conv3 (Conv2D)    (None, ?, ?, 256)    590,080
   block3_pool  (MaxPooling2D)
 Block 4:
   block4_conv1 (Conv2D)    (None, ?, ?, 512)    1,180,160
   block4_conv2 (Conv2D)    (None, ?, ?, 512)    2,359,808
   block4_conv3 (Conv2D)    (None, ?, ?, 512)    2,359,808
   block4_pool  (MaxPooling2D)
 Block 5:
   block5_conv1 (Conv2D)    (None, ?, ?, 512)    2,359,808
   block5_conv2 (Conv2D)    (None, ?, ?, 512)    2,359,808
   block5_conv3 (Conv2D)    (None, ?, ?, 512)    2,359,808
   block5_pool  (MaxPooling2D)
=================================================================
```

### ফিল্টার Extract করা

প্রতিটি Conv2D layer এর filter গুলো `get_weights()` method দিয়ে extract করা যায়। এই method দুটি array return করে: weights (filters) ও biases। আমাদের শুধু weights দরকার:

```python
# Extract filters from a specific layer
layer = model.layers[1]  # block1_conv1
filters, biases = layer.get_weights()

print(f"Layer: {layer.name}")
print(f"Filter shape: {filters.shape}")
print(f"Bias shape: {biases.shape}")
```

Output আসবে:
```
Layer: block1_conv1
Filter shape: (3, 3, 3, 64)
Bias shape: (64,)
```

Filter shape `(3, 3, 3, 64)` এর মানে কী? এটি 4D array: `(kernel_height, kernel_width, input_channels, output_channels)`। অর্থাৎ:

- `3, 3`  kernel size 3×3
- `3`  input channels (RGB, তাই 3)
- `64`  এই layer এ 64টি filter আছে

প্রতিটি filter এর shape `(3, 3, 3)`  এটি একটি 3D volume যা RGB ইমেজের উপর convolution হয়। Visualization এর জন্য আমরা প্রতিটি filter কে RGB image হিসেবে plot করবো (কারণ input 3 channel)। Deep layer এ filter shape হবে `(3, 3, 64, 128)`  সেখানে 64 input channels, তাই সব channel plot করা সম্ভব না  সেক্ষেত্রে আমরা প্রতিটি filter এর কিছু channel দেখাবো বা average করে দেখাবো।

### ফিল্টার Normalize করা

Filter value গুলো সাধারণত একটি বড় range এ থাকে (যেমন -2.5 থেকে +3.1)। সরাসরি plot করলে বেশিরভাগ detail দেখা যাবে না। তাই normalize করে 0-1 range এ আনতে হয়:

```python
import numpy as np

def normalize_filters(filters):
    """Normalize filter values to 0-1 range."""
    f_min = filters.min()
    f_max = filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    return filters
```

Normalization এর পর প্রতিটি pixel value 0 (কালো) থেকে 1 (সাদা) এর মধ্যে থাকে  এতে সব detail clearly visible হয়। এটি min-max normalization  সবচেয়ে সহজ ও effective approach।

### প্রথম Conv Layer এর ফিল্টার Plot করা

প্রথম convolution layer (block1_conv1) এর filter গুলো সবচেয়ে বেশি interpretable  কারণ এদের input হলো RGB ইমেজ, তাই এদের সরাসরি color image হিসেবে visualize করা যায়। আমরা প্রথম 16টি filter plot করবো:

```python
import matplotlib.pyplot as plt

# Extract filters from first conv layer
layer = model.layers[1]  # block1_conv1
filters, biases = layer.get_weights()

# Normalize
filters_norm = normalize_filters(filters)

# Plot first 16 filters
n_filters = 16
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

for i in range(n_filters):
    ax = axes[i // 8, i % 8]
    # Each filter shape: (3, 3, 3)  RGB image
    f = filters_norm[:, :, :, i]
    ax.imshow(f)
    ax.set_title(f'Filter {i}', fontsize=9)
    ax.axis('off')

plt.suptitle('VGG16 Block1_Conv1  First 16 Filters', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('vgg16_block1_filters.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Total filters: {filters.shape[3]}")
```

### ফিল্টার দেখে কী বোঝা যায়?

যখন তুমি প্রথম layer এর filter গুলো plot করবে, কিছু চমকপ্রদ pattern দেখতে পাবে:

**Edge detectors:** কিছু filter দেখবে যেখানে একদিকে সাদা, অন্যদিকে কালো  এগুলো edge detector! Horizontal edge, vertical edge, diagonal edge  এরা ইমেজের boundary detect করে। যেমন, একটি filter যেখানে বাম দিক সাদা ও ডান দিক কালো, সেটি vertical edge detect করবে যেখানে left side bright আর right side dark।

**Color detectors:** কিছু filter এ একটি dominant color দেখা যাবে  সবুজ, লাল, নীল। এরা specific color channel এর pattern detect করে। যেমন, একটি "লাল" filter লাল object এর region এ strongly activate হবে।

**Gabor-like patterns:** কিছু filter oriented stripe pattern দেখাবে  এগুলো Gabor filter এর মতো, যা specific orientation ও frequency এর pattern detect করে। Gabor filter biological vision system এও পাওয়া যায়!

**Human V1 Visual Cortex এর সাথে মিল:** এটি অত্যন্ত চমকপ্রদ  CNN এর প্রথম layer এর learned filter গুলো human brain এর primary visual cortex (V1) এর neuron গুলোর response pattern এর সাথে প্রায় identical! V1 neuron ও edge, orientation, color detect করে। এটি প্রমাণ করে যে CNN natural visual feature hierarchy learn করে  এটি artificial trick না, বরং biological vision system এর computational analog।

**Deep layer এর ফিল্টার:** Deep layer এর filter গুলো visualize করলে তুমি দেখবে সেগুলো আর RGB image এর মতো interpretable না। সেগুলো abstract, complex  কারণ তাদের input আর RGB pixel না, বরং আগের layer এর feature map। Deep layer এর filter গুলো specific texture, shape part, বা object part detect করে  কিন্তু সেগুলো সরাসরি visualize করা কঠিন। এখানেই feature map visualization বেশি helpful  যা আমরা পরবর্তী সেকশনে দেখবো।

### সব ব্লক এর ফিল্টার একসাথে Plot করা

একটি comprehensive visualization যেখানে প্রতিটি block এর প্রথম layer এর কিছু filter দেখাবো:

```python
# First conv layer of each block
block_layers = {
    'Block 1': 'block1_conv1',
    'Block 2': 'block2_conv1',
    'Block 3': 'block3_conv1',
    'Block 4': 'block4_conv1',
    'Block 5': 'block5_conv1'
}

fig, axes = plt.subplots(5, 8, figsize=(16, 10))

for row, (block_name, layer_name) in enumerate(block_layers.items()):
    # Find layer
    layer = model.get_layer(layer_name)
    filters, _ = layer.get_weights()
    filters_norm = normalize_filters(filters)

    for col in range(8):
        ax = axes[row, col]
        f = filters_norm[:, :, :, col]

        # Deep layer filters have input channels > 3
        # So show only first 3 channels (pseudo-RGB)
        if f.shape[2] > 3:
            f = f[:, :, :3]
            f = normalize_filters(f)

        ax.imshow(f, interpolation='nearest')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(block_name, fontsize=11, rotation=0, labelpad=50)

plt.suptitle('VGG16  First 8 Filters of Each Block', fontsize=14)
plt.tight_layout()
plt.savefig('vgg16_all_blocks_filters.png', dpi=150, bbox_inches='tight')
plt.show()
```

এই plot এ তুমি স্পষ্টভাবে দেখতে পাবে কীভাবে filter গুলো shallow থেকে deep এ গেলে increasingly abstract হয়ে যায়। Block 1 এ clearly identifiable edge ও color pattern, Block 3 এ কিছুটা textured pattern, আর Block 5 এ প্রায় random noise এর মতো মনে হবে  কিন্তু এগুলো actually highly specialized filter যা specific high-level pattern detect করে।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম কিভাবে CNN এর learned filter গুলো extract ও visualize করতে হয়। আমরা VGG16 এর `get_weights()` method দিয়ে filter extract করলাম, filter shape `(kernel_h, kernel_w, input_channels, output_channels)` বুঝলাম, min-max normalization করে 0-1 range এ আনলাম, এবং matplotlib দিয়ে plot করলাম। সবচেয়ে চমকপ্রদ আবিষ্কার: প্রথম layer এর filter গুলো edge detector ও color detector এর মতো  যা human brain এর V1 visual cortex এর neuron response এর সাথে প্রায় identical! Deep layer এর filter গুলো increasingly abstract হয়, যা সরাসরি visualize করা কঠিন  তাই পরবর্তী সেকশনে আমরা feature map visualization শিখবো, যা deep layer এর behavior বোঝার আরও effective way।
