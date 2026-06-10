## ConvNet লেয়ার ও প্যারামিটার কাউন্টিং

আগের সেকশনে আমরা শিখেছি কিভাবে 3D কনভোলিউশন কাজ করে এবং মাল্টিপল ফিল্টার কিভাবে output volume তৈরি করে। কিন্তু একটি পূর্ণাঙ্গ ConvNet লেয়ার শুধু convolution-ই নয় — এর সাথে আরও দুটি অপারেশন যুক্ত থাকে: Bias Addition এবং Activation Function (ReLU)। এই তিনটি অপারেশন মিলে একটি complete ConvNet layer তৈরি করে। এই সেকশনে আমরা একটি ConvNet লেয়ারের পূর্ণাঙ্গ গঠন শিখবো, parameter counting শিখবো, এবং বুঝবো কেন CNN fully connected layer এর তুলনায় অনেক কম parameter ব্যবহার করে। এছাড়া ReLU activation কেন জরুরি এবং non-linearity ছাড়া কেন ডিপ নেটওয়ার্ক কাজ করে না — এই গুরুত্বপূর্ণ বিষয়টিও আলোচনা করবো।

### ConvNet লেয়ারের তিনটি অপারেশন

একটি ConvNet লেয়ারে তিনটি ক্রমানুসারে অপারেশন হয়:

**Convolution → Bias Addition → Activation (ReLU)**

প্রতিটি অপারেশনের কাজ আলাদা:

1. **Convolution**: ইনপুট volume এর উপর মাল্টিপল kernel slide করে feature map তৈরি করে। এটি ইমেজ থেকে local pattern (এজ, কর্নার, টেক্সচার) extract করে। আউটপুট হলো একটি set of feature maps।

2. **Bias Addition**: প্রতিটি feature map এ একটি learnable bias value যোগ করা হয়। Bias হলো একটি scalar সংখ্যা যা feature map কে উপরে বা নিচে shift করে। এটি linear transformation কে আরও flexible করে — না থাকলে convolution এর আউটপুট সবসময় zero-centered হতো না। প্রতিটি filter এর নিজস্ব bias থাকে।

3. **Activation (ReLU)**: সব negative মানকে 0 তে convert করে, positive মান অপরিবর্তিত রাখে। এটি non-linearity introduce করে, যা ছাড়া ডিপ নেটওয়ার্ক কাজ করতে পারে না। ReLU ছাড়াও অন্যান্য activation function আছে (Sigmoid, Tanh), কিন্তু ReLU CNN এ সবচেয়ে বেশি ব্যবহৃত কারণ এটি simple, fast, এবং gradient vanishing problem কম করে।

গাণিতিকভাবে, একটি ConvNet লেয়ারের অপারেশন হলো:

```
Output = ReLU(Convolution(Input, Kernel) + Bias)
```

অথবা আরও স্পষ্টভাবে:

```
Z = Σ(W * X) + b     ← Convolution + Bias (linear operation)
A = max(0, Z)         ← ReLU (non-linear operation)
```

### ReLU: f(x) = max(0, x)

ReLU (Rectified Linear Unit) হলো সবচেয়ে সহজ activation function — ইনপুট negative হলে 0 রিটার্ন করে, positive হলে তাই রিটার্ন করে। এটি এত সহজ মনে হতে পারে যে এর কোনো effect আছে কিনা! কিন্তু এই সরল non-linearity ই ডিপ লার্নিং কে সম্ভব করেছে।

```python
import numpy as np
import matplotlib.pyplot as plt

# ReLU function
def relu(x):
    return np.maximum(0, x)

# ReLU visualize করা
x = np.linspace(-10, 10, 100)
y = relu(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=3, color="blue")
plt.axhline(y=0, color="black", linewidth=0.5)
plt.axvline(x=0, color="black", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.title("ReLU Activation Function: f(x) = max(0, x)")
plt.grid(True, alpha=0.3)
plt.show()

# উদাহরণ
values = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])
print(f"Input:  {values}")
print(f"ReLU:   {relu(values)}")
```

ReLU এর কিছু গুরুত্বপূর্ণ বৈশিষ্ট্য আছে। প্রথমত, এটি computation এ খুবই সস্তা — শুধু একটি comparison (`x > 0`) আর একটি assignment। Sigmoid বা Tanh এর মতো exponentiation লাগে না। দ্বিতীয়ত, positive region এ gradient সবসময় 1 — তাই gradient vanishing problem (যেখানে gradient এত ছোট হয়ে যায় যে learning বন্ধ হয়ে যায়) কম হয়। তৃতীয়ত, যখন আউটপুট 0 হয়, সেই neuron "dead" হয়ে যায় — এটি sparse activation তৈরি করে, যা computation efficient এবং overfitting কমায়।

### কেন Non-linearity জরুরি?

এটি বোঝা সবচেয়ে গুরুত্বপূর্ণ — কেন activation function লাগে? উত্তর হলো: non-linearity ছাড়া ডিপ নেটওয়ার্ক কাজ করে না। চলো প্রমাণ করি।

ধরো আমাদের দুটি linear layer আছে (activation ছাড়া):

```
Layer 1: Z₁ = W₁ · X + b₁    (linear)
Layer 2: Z₂ = W₂ · Z₁ + b₂   (linear)
```

এখন Z₂ কে expand করি:

```
Z₂ = W₂ · (W₁ · X + b₁) + b₂
   = W₂ · W₁ · X + W₂ · b₁ + b₂
   = W' · X + b'
```

যেখানে W' = W₂ · W₁ আর b' = W₂ · b₁ + b₂। অর্থাৎ, দুটি linear layer stack করলে ফলাফল আরেকটি single linear transformation! 100টি linear layer stack করলেও ফলাফল একটিই linear layer এর সমান। মানে, non-linearity ছাড়া depth এর কোনো মানে হয় না — আমরা যতই layer যোগ করি না কেন, শেষ পর্যন্ত এটি একটি single linear model ই থাকবে। Linear model দিয়ে complex pattern (যেমন ইমেজের মধ্যে বিড়াল চেনা) learn করা সম্ভব নয়।

কিন্তু ReLU ব্যবহার করলে গল্প বদলে যায়:

```
Layer 1: A₁ = ReLU(W₁ · X + b₁)     (non-linear!)
Layer 2: A₂ = ReLU(W₂ · A₁ + b₂)    (non-linear!)
```

এখন এই দুটি layer কে আর একটি single linear transformation এ reduce করা যাবে না, কারণ ReLU non-linear। প্রতিটি layer নতুন non-linear transformation যোগ করে, যা মডেলকে increasingly complex function learn করতে দেয়। এটিই universal approximation theorem এর মূল ধারণা — non-linearity সহ যথেষ্ট গভীর নেটওয়ার্ক যেকোনো continuous function approximate করতে পারে।

### প্যারামিটার কাউন্টিং ফর্মুলা

একটি ConvNet লেয়ারে কতগুলো learnable parameter থাকে? এটি বোঝা খুবই গুরুত্বপূর্ণ — model এর size, memory requirement, আর training time সব এর উপর নির্ভর করে। একটি convolution layer এ দুই ধরনের parameter থাকে — kernel weights আর bias:

**Total Parameters = C_out × (k × k × C_in + 1)**

এখানে:
- `C_out` = output channels (ফিল্টারের সংখ্যা)
- `k × k` = kernel এর spatial size
- `C_in` = input channels
- `1` = প্রতিটি filter এর bias

খেয়াল করো — প্রতিটি filter এর parameter সংখ্যা হলো `k × k × C_in + 1` (kernel weights + bias)। আর মোট `C_out` টি filter আছে, তাই মোট parameter = `C_out × (k × k × C_in + 1)`।

### প্যারামিটার কাউন্টিং উদাহরণ

চলো কিছু concrete উদাহরণ দেখি:

**উদাহরণ ১: প্রথম Conv লেয়ার (RGB ইনপুট)**
```
Input:  3 channels (RGB)
Filters: 64 filters, 3×3 kernel
Parameters = 64 × (3 × 3 × 3 + 1)
           = 64 × (27 + 1)
           = 64 × 28
           = 1,792
```

মাত্র 1,792 টি parameter দিয়ে এই লেয়ার 64টি feature map তৈরি করছে! এটি অবিশ্বাস্যরকম efficient।

**উদাহরণ ২: দ্বিতীয় Conv লেয়ার**
```
Input:  64 channels (প্রথম লেয়ারের আউটপুট)
Filters: 128 filters, 3×3 kernel
Parameters = 128 × (3 × 3 × 64 + 1)
           = 128 × (576 + 1)
           = 128 × 577
           = 73,856
```

দ্বিতীয় লেয়ারে parameter বেড়েছে কারণ input channel বেড়েছে (3→64)। kernel এর spatial size একই (3×3), কিন্তু depth বেড়েছে বলে প্রতিটি kernel এ weight বেশি।

**উদাহরণ ৩: Fully Connected Layer এর সাথে তুলনা**

ধরো একটি 32×32×3 ইমেজ কে fully connected layer এ পাস করতে চাই 128 neurons দিয়ে:
```
Input neurons: 32 × 32 × 3 = 3,072
Output neurons: 128
Parameters = 3,072 × 128 + 128 = 393,344
```

একই ইমেজে একটি Conv layer (64 filters, 3×3):
```
Parameters = 1,792 (উপরে হিসাব করা হয়েছে)
```

তুলনা করো — Conv layer এ মাত্র 1,792 parameter, আর fully connected layer এ 393,344 parameter! অর্থাৎ fully connected layer এ parameter প্রায় 220 গুণ বেশি! এটিই CNN এর সবচেয়ে বড় সুবিধা — অনেক কম parameter দিয়ে অনেক বেশি effective feature extraction।

### CNN এর মূল বৈশিষ্ট্য

CNN কেন এত কম parameter দিয়ে এত ভালো কাজ করে? এর পেছনে তিনটি মূল নীতি কাজ করে:

**1. Parameter Sharing (প্যারামিটার শেয়ারিং):** একটি kernel পুরো ইমেজের উপর দিয়ে slide করে — অর্থাৎ একই kernel এর মান সব spatial position এ ব্যবহার হয়। একটি vertical edge detect করার kernel ইমেজের যেকোনো জায়গায় vertical edge detect করতে পারে — তাই আলাদা আলাদা kernel লাগে না। Fully connected layer এ প্রতিটি connection এ আলাদা weight থাকে, তাই parameter অনেক বেশি হয়।

**2. Local Connectivity (লোকাল কানেকটিভিটি):** প্রতিটি output neuron শুধু ইমেজের একটি ছোট local region (receptive field) এর সাথে connected। 3×3 kernel মানে প্রতিটি output value শুধু 9টি input pixel এর উপর নির্ভর করে, পুরো ইমেজের উপর নয়। এটি biological vision এর মতো — আমাদের চোখের visual neuron ও শুধু local region দেখে। Fully connected layer এ প্রতিটি neuron সব input এর সাথে connected, যা parameter বাড়ায় আর overfitting এর সম্ভাবনা বাড়ায়।

**3. Translation Invariance (ট্রান্সলেশন ইনভ্যারিয়েন্স):** একটি object ইমেজে যেখানেই থাকুক না কেন, একই kernel দিয়ে detect করা যাবে। একটি বিড়াল ইমেজের উপরে থাকুক বা নিচে, একই vertical edge kernel তার এজ detect করবে। এটি parameter sharing এর সরাসরি ফলাফল। Fully connected layer এ এই property naturally থাকে না — object shift হলে আলাদা weight learn করতে হয়।

### পূর্ণাঙ্গ ConvNet লেয়ার কোড উদাহরণ

এখন আমরা একটি পূর্ণাঙ্গ ConvNet লেয়ার স্ক্র্যাচ থেকে ইমপ্লিমেন্ট করবো — Convolution + Bias + ReLU:

```python
import numpy as np

def convnet_layer(input_volume, kernels, biases):
    """
    পূর্ণাঙ্গ ConvNet লেয়ার: Convolution + Bias + ReLU

    Parameters:
    -----------
    input_volume : 3D numpy array (H × W × C_in)
    kernels : 4D numpy array (C_out × k × k × C_in)
    biases : 1D numpy array (C_out,)

    Returns:
    --------
    output_volume : 3D numpy array (H' × W' × C_out)
    """
    img_h, img_w, c_in = input_volume.shape
    num_filters, k_h, k_w, k_c = kernels.shape

    assert k_c == c_in, "Kernel depth must match input depth!"
    assert num_filters == len(biases), "Number of filters must match biases!"

    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1

    # Step 1: Convolution + Bias
    output = np.zeros((out_h, out_w, num_filters), dtype=np.float64)

    for f in range(num_filters):
        kernel = kernels[f]
        kernel_flipped = np.flipud(np.fliplr(kernel))

        for i in range(out_h):
            for j in range(out_w):
                patch = input_volume[i:i+k_h, j:j+k_w, :]
                # Convolution result + bias
                output[i, j, f] = np.sum(patch * kernel_flipped) + biases[f]

    # Step 2: ReLU Activation
    output = np.maximum(0, output)  # ReLU: max(0, x)

    return output

# প্যারামিটার কাউন্টিং ফাংশন
def count_parameters(c_in, c_out, kernel_size):
    """
    ConvNet লেয়ারের parameter সংখ্যা হিসাব
    """
    kernel_params = kernel_size * kernel_size * c_in
    bias_params = 1
    total_per_filter = kernel_params + bias_params
    total = c_out * total_per_filter
    return total

# উদাহরণ: 3 input channels, 64 filters, 3×3 kernel
params = count_parameters(c_in=3, c_out=64, kernel_size=3)
print(f"Layer 1 parameters: {params}")
# Output: 1,792

params2 = count_parameters(c_in=64, c_out=128, kernel_size=3)
print(f"Layer 2 parameters: {params2}")
# Output: 73,856

# একটি ছোট example run করা
np.random.seed(42)
input_vol = np.random.randn(8, 8, 3)      # 8×8×3 input
kernels = np.random.randn(4, 3, 3, 3) * 0.1  # 4 filters, 3×3×3
biases = np.zeros(4)                         # 4 biases

output_vol = convnet_layer(input_vol, kernels, biases)
print(f"\nInput volume:  {input_vol.shape}")
print(f"Output volume: {output_vol.shape}")
print(f"Parameters: {count_parameters(3, 4, 3)}")

# ReLU এর effect দেখা
pre_relu_values = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])
post_relu = np.maximum(0, pre_relu_values)
print(f"\nPre-ReLU:  {pre_relu_values}")
print(f"Post-ReLU: {post_relu}")
```

### PyTorch দিয়ে ConvNet লেয়ার ও Parameter Counting

বাস্তবে আমরা PyTorch ব্যবহার করে খুব সহজেই ConvNet লেয়ার তৈরি করতে পারি এবং parameter count করতে পারি:

```python
import torch
import torch.nn as nn

# ConvNet লেয়ার তৈরি
conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

# Parameter count করা
def count_params(layer):
    return sum(p.numel() for p in layer.parameters())

print(f"Conv1 parameters: {count_params(conv1)}")
# Output: 1,792 (= 64 × (3×3×3 + 1))

print(f"Conv2 parameters: {count_params(conv2)}")
# Output: 73,856 (= 128 × (3×3×64 + 1))

# বিস্তারিত parameter breakdown
print(f"\nConv1 weight shape: {conv1.weight.shape}")  # (64, 3, 3, 3)
print(f"Conv1 bias shape:   {conv1.bias.shape}")      # (64,)
print(f"Conv1 weight params: {conv1.weight.numel()}") # 1,728
print(f"Conv1 bias params:   {conv1.bias.numel()}")   # 64

# একটি সরল ConvNet তৈরি
simple_cnn = nn.Sequential(
    nn.Conv2d(3, 16, 3),    # Layer 1: 3→16, 3×3
    nn.ReLU(),               # Activation
    nn.Conv2d(16, 32, 3),   # Layer 2: 16→32, 3×3
    nn.ReLU(),               # Activation
    nn.Conv2d(32, 64, 3),   # Layer 3: 32→64, 3×3
    nn.ReLU(),               # Activation
)

total_params = sum(p.numel() for p in simple_cnn.parameters())
print(f"\nTotal CNN parameters: {total_params}")

# প্রতিটি লেয়ারের parameter
for i, layer in enumerate(simple_cnn):
    if isinstance(layer, nn.Conv2d):
        params = sum(p.numel() for p in layer.parameters())
        print(f"  Layer {i} (Conv2d): {params} parameters")
```

এই কোডটি দেখায় যে একটি সরল 3-layer CNN এ মোট parameter কত আছে এবং কিভাবে প্রতিটি লেয়ারে parameter বাড়তে থাকে। প্রথম লেয়ারে সবচেয়ে কম parameter (কারণ input channel কম), আর deeper layer এ বেশি (কারণ input channel বেশি)। কিন্তু সব মিলিয়েও parameter অনেক কম হয় fully connected network এর তুলনায়।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম একটি ConvNet লেয়ারের পূর্ণাঙ্গ গঠন — Convolution + Bias + ReLU। ReLU non-linearity ছাড়া ডিপ নেটওয়ার্ক কাজ করে না কারণ linear layer stack করলে শেষ পর্যন্ত একটি linear transformation ই থাকে। Parameter counting formula `C_out × (k × k × C_in + 1)` ব্যবহার করে আমরা যেকোনো convolution layer এর parameter সংখ্যা হিসাব করতে পারি। CNN fully connected layer এর তুলনায় অনেক কম parameter ব্যবহার করে parameter sharing, local connectivity, আর translation invariance এর কারণে। এই তিনটি property ই CNN কে image data তে এত effective করে তোলে। পরবর্তী চ্যাপ্টারে আমরা padding আর stride সম্পর্কে শিখবো, যা output dimension নিয়ন্ত্রণ করার জন্য জরুরি।
