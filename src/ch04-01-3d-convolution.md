## 3D কনভোলিউশন ও মাল্টিপল ফিল্টার

আগের চ্যাপ্টারে আমরা 2D কনভোলিউশন শিখেছি  একটি grayscale ইমেজ (single channel) আর একটি 2D kernel দিয়ে convolution করে একটি 2D feature map পেতাম। কিন্তু বাস্তব দুনিয়ার ইমেজ সাধারণত RGB  তিনটি চ্যানেল (Red, Green, Blue) সহ। তাছাড়া CNN এর একটি লেয়ারে শুধু একটি ফিল্টার থাকে না, অনেকগুলো ফিল্টার থাকে  প্রতিটি আলাদা feature detect করে। এই দুটি বিষয়  3D কনভোলিউশন এবং মাল্টিপল ফিল্টার  একসাথে মিলে CNN এর আউটপুট volume তৈরি করে। এই সেকশনে আমরা step by step দেখবো কিভাবে 2D থেকে 3D তে যাওয়া হয় এবং কিভাবে মাল্টিপল ফিল্টার output volume তৈরি করে।

### 2D থেকে 3D: RGB ইমেজে কনভোলিউশন

2D কনভোলিউশনে আমাদের ইনপুট ছিল H×W (single channel), kernel ছিল k×k, আর আউটপুট ছিল (H-k+1)×(W-k+1)। কিন্তু RGB ইমেজের dimension হলো H×W×3  এখানে 3 হলো depth (চ্যানেলের সংখ্যা)। এই 3 channel সহ ইমেজে convolution করতে হলে আমাদের kernel ও 3D হতে হবে  অর্থাৎ k×k×3। মনে রাখবে, kernel এর depth সবসময় ইনপুটের depth এর সমান হতে হবে। এটি খুবই গুরুত্বপূর্ণ নিয়ম  kernel সব channel একসাথে cover করে, আলাদা আলাদা নয়।

3D কনভোলিউশনে কী হয়? Kernel টি ইমেজের উপর spatial dimension এ (width আর height) slide করে  ঠিক যেমন 2D তে করতো। কিন্তু প্রতিটি পজিশনে এটি সব 3টি চ্যানেলের সাথে একসাথে element-wise multiplication করে এবং সব মান যোগ করে। ফলাফল? একটি একক সংখ্যা (single number)  ঠিক যেমন 2D তে পেতাম। তার মানে, যতই channel থাকুক, একটি 3D kernel দিয়ে convolution করলে একটি 2D feature map পাওয়া যায়। এটি বুঝতে সবচেয়ে গুরুত্বপূর্ণ  kernel এর depth ইনপুটের depth সমান, আর প্রতিটি kernel একটি single feature map তৈরি করে।

```
2D Convolution:
  Input:    H × W         (1 channel)
  Kernel:   k × k         (2D)
  Output:   (H-k+1) × (W-k+1)   (1 feature map)

3D Convolution:
  Input:    H × W × C_in  (C_in channels, e.g. 3 for RGB)
  Kernel:   k × k × C_in  (3D kernel, depth matches input)
  Output:   (H-k+1) × (W-k+1)   (1 feature map  single number per position!)
```

একটি সহজ উদাহরণ দিয়ে বোঝাই। ধরো ইনপুট হলো 5×5×3 (RGB) আর kernel হলো 3×3×3। প্রতিটি spatial position এ kernel টি 3×3×3 = 27টি মানের সাথে multiply করে এবং সব যোগ করে 1টি মান পায়। এভাবে পুরো spatial grid এ slide করে 3×3×1 feature map পাওয়া যায়।

### মাল্টিপল ফিল্টার → Output Volume

একটি kernel দিয়ে একটি feature map পাওয়া যায়। কিন্তু একটি CNN লেয়ারে আমরা শুধু একটি feature বের করতে চাই না  আমরা একসাথে অনেকগুলো feature বের করতে চাই। যেমন, একটি ফিল্টার vertical edge detect করুক, আরেকটি horizontal edge detect করুক, আরেকটি blur করুক, এভাবে অনেকগুলো। তাই আমরা একাধিক kernel (ফিল্টার) ব্যবহার করি। প্রতিটি kernel আলাদা আলাদা feature map তৈরি করে। এই feature map গুলোকে একটি আরেকটির উপর stack করলে যে volume পাওয়া যায় তাকেই output volume বলে।

যদি আমরা `C_out` সংখ্যক kernel ব্যবহার করি, তাহলে আউটপুট volume এর dimension হবে:

**Output Volume = (H-k+1) × (W-k+1) × C_out**

এখানে C_out হলো kernel এর সংখ্যা, যাকে আমরা output depth ও বলি। প্রতিটি kernel আলাদা feature detect করে  একটি vertical edge, আরেকটি horizontal edge, আরেকটি corner, এভাবে। CNN training এর সময় এই kernel গুলো data থেকে automatically learn হয়, তাই নেটওয়ার্ক নিজেই decide করে কোন কোন feature গুরুত্বপূর্ণ।

```
Multiple Filters → Output Volume:

  Input:    H × W × C_in
  Filters:  C_out kernels, each k × k × C_in
  Output:   (H-k+1) × (W-k+1) × C_out

  Example:
  Input:    32 × 32 × 3    (RGB image)
  Filters:  16 kernels of 3×3×3
  Output:   30 × 30 × 16   (16 feature maps stacked)
```

### Output Depth = Number of Filters

এটি একটি মৌলিক নিয়ম  output volume এর depth সবসময় ফিল্টারের সংখ্যার সমান। এটি কেন গুরুত্বপূর্ণ? কারণ এটি CNN এর feature extraction এর ক্ষমতা নির্ধারণ করে। বেশি ফিল্টার মানে বেশি feature map, বেশি feature map মানে ইমেজ থেকে বেশি তথ্য বের করা। কিন্তু বেশি ফিল্টার মানে বেশি parameter, বেশি computation, আর বেশি memory  তাই এটি একটি trade-off। সাধারণত CNN এর early layer এ কম ফিল্টার (32-64) থাকে আর deeper layer এ বেশি ফিল্টার (128-512) থাকে।

### একটি সরল ConvNet এ দিয়ে Dimension Tracing

সবচেয়ে ভালোভাবে বোঝার উপায় হলো একটি সরল ConvNet এর মধ্য দিয়ে dimension ট্রেস করা। ধরো আমরা একটি 32×32 RGB ইমেজ নিই এবং দুটি convolution layer দিয়ে পাস করি:

```
Layer 0 (Input):
  Volume: 32 × 32 × 3

Layer 1 (Conv: 16 filters, 3×3):
  Each filter: 3 × 3 × 3
  Output feature map per filter: 30 × 30 × 1
  Total output: 30 × 30 × 16   (16 filters → depth=16)
  Parameters: 16 × (3×3×3 + 1) = 16 × 28 = 448

Layer 2 (Conv: 32 filters, 3×3):
  Each filter: 3 × 3 × 16     ← Notice! Now depth=16
  Output feature map per filter: 28 × 28 × 1
  Total output: 28 × 28 × 32   (32 filters → depth=32)
  Parameters: 32 × (3×3×16 + 1) = 32 × 145 = 4,640
```

খেয়াল করো কী হচ্ছে  spatial dimension কমছে (32→30→28) কিন্তু depth বাড়ছে (3→16→32)। এটি CNN এর একটি fundamental pattern:

- **Spatial dimensions shrink**: কনভোলিউশনের কারণে height আর width কমে যায় (padding ছাড়া)। এটি feature map কে progressively smaller করে।
- **Depth grows**: প্রতি লেয়ারে ফিল্টারের সংখ্যা বাড়ানো হয়, তাই depth বাড়ে। বেশি depth মানে বেশি abstract feature representation।

এই pattern টি CNN architecture এর সবখানে দেখা যায়  VGG, ResNet, AlexNet সবগুলোতেই। Early layer এ ইমেজ বড় কিন্তু চ্যানেল কম (low-level features like edges), deeper layer এ ইমেজ ছোট কিন্তু চ্যানেল বেশি (high-level features like object parts)।

### 3D কনভোলিউশন কোড উদাহরণ

এখন আমরা 3D কনভোলিউশন কোডে ইমপ্লিমেন্ট করে দেখবো। আমরা একটি RGB ইমেজে মাল্টিপল ফিল্টার apply করবো এবং output volume visualize করবো:

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

def conv3d(image, kernels):
    """
    3D convolution with multiple filters

    Parameters:
    -----------
    image : 3D numpy array (H × W × C_in)
        Input RGB image
    kernels : 4D numpy array (C_out × k × k × C_in)
        Multiple 3D kernels

    Returns:
    --------
    output : 3D numpy array (H' × W' × C_out)
        Output volume (stacked feature maps)
    """
    img_h, img_w, c_in = image.shape
    num_filters, k_h, k_w, k_c = kernels.shape

    # Check if kernel depth matches input depth
    assert k_c == c_in, f"Kernel depth ({k_c}) != Input depth ({c_in})"

    # Output size
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1

    # Output volume
    output = np.zeros((out_h, out_w, num_filters), dtype=np.float64)

    # Separate convolution for each kernel
    for f in range(num_filters):
        kernel = kernels[f]
        kernel_flipped = np.flipud(np.fliplr(kernel))  # spatial flip

        for i in range(out_h):
            for j in range(out_w):
                # 3D patch (all channels together)
                patch = image[i:i+k_h, j:j+k_w, :]
                # Multiply all values and sum → single number
                output[i, j, f] = np.sum(patch * kernel_flipped)

    return output

# Load image
img = cv2.imread("resources/lena.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
print(f"Input shape: {img_rgb.shape}")

# Create 4 different 3D kernels (C_out=4, k=3, C_in=3)
np.random.seed(42)
num_filters = 4
kernels = np.random.randn(num_filters, 3, 3, 3) * 0.1  # small random values

# 3D convolution apply
output_volume = conv3d(img_rgb, kernels)
print(f"Output shape: {output_volume.shape}")
print(f"Each feature map: {output_volume.shape[0]} × {output_volume.shape[1]}")

# Visualize each feature map
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title(f"Input RGB\n{img_rgb.shape}")

for f in range(num_filters):
    row = (f + 1) // 3
    col = (f + 1) % 3
    axes[row, col].imshow(output_volume[:, :, f], cmap="gray")
    axes[row, col].set_title(f"Feature Map {f+1}\n{output_volume[:,:,f].shape}")

axes[1, 2].axis("off")

for ax in axes.flat:
    if ax:
        ax.axis("off")

plt.tight_layout()
plt.show()
```

Random kernel ব্যবহার করায় feature map গুলো random pattern দেখাবে  কিন্তু এটি প্রমাণ করে যে প্রতিটি kernel একটি আলাদা feature map তৈরি করছে। Training এর পর এই kernel গুলো meaningful feature detect করতে শিখবে। মনে রাখবে, প্রতিটি feature map একটি 2D array, আর এগুলো stacked হয়ে output volume তৈরি করে।

### PyTorch দিয়ে 3D কনভোলিউশন

বাস্তবে আমরা স্ক্র্যাচ থেকে convolution লিখি না  PyTorch বা TensorFlow এর built-in layer ব্যবহার করি। চলো দেখি PyTorch তে কিভাবে একই কাজ করা যায়:

```python
import torch
import torch.nn as nn

# Define Conv2d layer
# Parameters: (in_channels=3, out_channels=16, kernel_size=3)
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)

# Dummy input: batch_size=1, channels=3, height=32, width=32
dummy_input = torch.randn(1, 3, 32, 32)

# Forward pass
output = conv_layer(dummy_input)

print(f"Input shape:  {dummy_input.shape}")
print(f"Weight shape: {conv_layer.weight.shape}")  # (16, 3, 3, 3)
print(f"Bias shape:   {conv_layer.bias.shape}")    # (16,)
print(f"Output shape: {output.shape}")              # (1, 16, 30, 30)
```

খেয়াল করো  `conv_layer.weight.shape` হলো (16, 3, 3, 3) অর্থাৎ (C_out, C_in, k_h, k_w)। এটি আমাদের আগের manual implementation এর kernels array এর সমান। `conv_layer.bias.shape` হলো (16,)  প্রতিটি filter এ একটি bias থাকে। Output shape হলো (1, 16, 30, 30) অর্থাৎ (batch, C_out, H', W')। PyTorch এ spatial dimension কমে 32→30 হয়েছে, ঠিক যেমন আমাদের formula অনুযায়ী (32-3+1=30)।

এই সেকশনে আমরা শিখলাম কিভাবে 2D convolution থেকে 3D convolution এ যেতে হয়, কিভাবে মাল্টিপল ফিল্টার output volume তৈরি করে, এবং CNN তে spatial dimension shrink হওয়া আর depth বাড়ার pattern কেমন। পরের সেকশনে আমরা ConvNet লেয়ারের পূর্ণাঙ্গ গঠন শিখবো  Convolution, Bias Addition, আর ReLU Activation একসাথে কিভাবে কাজ করে।
