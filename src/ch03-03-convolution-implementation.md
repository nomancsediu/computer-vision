## কনভোলিউশন স্ক্র্যাচ থেকে ইমপ্লিমেন্টেশন

আগের সেকশনগুলোতে আমরা ফিল্টার কী এবং কিভাবে কাজ করে তা শিখেছি, আর `cv2.filter2D()` দিয়ে ফিল্টার apply ও করেছি। কিন্তু ভেতরে ভেতরে কনভোলিউশন আসলে কিভাবে কাজ করে? এই প্রশ্নের উত্তর জানতে হলে সবচেয়ে ভালো উপায় হলো নিজে স্ক্র্যাচ থেকে কনভোলিউশন ইমপ্লিমেন্ট করা। তাই এই সেকশনে আমরা কোনো লাইব্রেরির convolution ফাংশন ব্যবহার না করে, শুধু NumPy এর basic array operation ব্যবহার করে আমাদের নিজস্ব `simple_conv()` ফাংশন লিখবো। এই হাতে-কলমে ইমপ্লিমেন্টেশন তোমাকে CNN এর ভেতরের কাজ গভীরভাবে বুঝতে সাহায্য করবে, কারণ CNN ও ঠিক এই একই অপারেশন করে — শুধু অনেক বেশি স্পিডে এবং GPU তে।

### কনভোলিউশন কিভাবে কাজ করে — ধাপে ধাপে

কনভোলিউশন অপারেশনকে একটি সহজ উদাহরণ দিয়ে বোঝাই। ধরো আমাদের কাছে একটি 5×5 ইমেজ আছে এবং একটি 3×3 kernel আছে। কনভোলিউশন নিচের ধাপগুলো অনুসরণ করে কাজ করে:

**ধাপ ১:** Kernel কে ইমেজের উপর-বাম কোণে (top-left corner) রাখো।

**ধাপ ২:** Kernel এর প্রতিটি মান কে ইমেজের corresponding (overlapping) পিক্সেলের সাথে গুণ করো (element-wise multiplication)।

**ধাপ ৩:** সব গুণফল যোগ করো — এটিই আউটপুট ইমেজের প্রথম পিক্সেলের মান।

**ধাপ ৪:** Kernel কে এক পিক্সেল ডানে সরাও (slide) এবং ধাপ ২-৩ আবার করো।

**ধাপ ৫:** একটি সারি শেষ হলে নিচের সারিতে যাও এবং আবার ধাপ ২-৪ করো।

**ধাপ ৬:** পুরো ইমেজের উপর দিয়ে slide করা শেষ হলে আউটপুট ইমেজ তৈরি!

চলো একটি ছোট সংখ্যার উদাহরণ দেখি:

```
ইমেজ (5×5):              Kernel (3×3):
[10, 10, 10, 20, 20]      [1, 0, -1]
[10, 10, 10, 20, 20]      [2, 0, -2]
[10, 10, 10, 20, 20]      [1, 0, -1]
[30, 30, 30, 40, 40]
[30, 30, 30, 40, 40]

প্রথম পজিশনে (0,0):
10×1 + 10×0 + 10×(-1) +
10×2 + 10×0 + 10×(-2) +
10×1 + 10×0 + 10×(-1)
= 10 + 0 - 10 + 20 + 0 - 20 + 10 + 0 - 10 = 0
```

এখানে ফলাফল 0 কারণ প্রথম 3×3 এলাকায় সব পিক্সেলের মান সমান (10), তাই কোনো এজ নেই। কিন্তু যখন kernel টি সেই জায়গায় যাবে যেখানে 10 থেকে 20 তে পরিবর্তন হচ্ছে, সেখানে ফলাফল 0 হবে না — সেখানে একটি বড় মান পাওয়া যাবে, কারণ intensity পরিবর্তন হচ্ছে। এটিই এজ ডিটেকশন এর মূল নীতি!

### আউটপুট সাইজ ফর্মুলা

কনভোলিউশনের একটি গুরুত্বপূর্ণ বিষয় হলো আউটপুট ইমেজের সাইজ। Stride=1 (অর্থাৎ প্রতি ধাপে 1 পিক্সেল করে সরছি) হলে আউটপুট সাইজ হবে:

**Output Size = (H - k + 1) × (W - k + 1)**

যেখানে H = ইমেজের height, W = ইমেজের width, k = kernel এর সাইজ। যেমন, 5×5 ইমেজ আর 3×3 kernel হলে আউটপুট হবে (5-3+1) × (5-3+1) = 3×3। মানে আউটপুট ইমেজ মূল ইমেজের চেয়ে ছোট হয় — এটি কনভোলিউশন এর একটি স্বাভাবিক বৈশিষ্ট্য। পরবর্তী চ্যাপ্টারে আমরা দেখবো কিভাবে padding ব্যবহার করে আউটপুট সাইজ মূল ইমেজের সমান রাখা যায়।

Stride যদি 1 এর বেশি হয়, তাহলে ফর্মুলা হয়:

**Output Size = ⌊(H - k) / stride⌋ + 1 × ⌊(W - k) / stride⌋ + 1**

Stride=2 হলে kernel দুই পিক্সেল করে জাম্প করে, তাই আউটপুট আরও ছোট হয়। এটি CNN তে downsampling এর একটি উপায়।

### simple_conv() ফাংশন ইমপ্লিমেন্টেশন

এখন আমরা আমাদের `simple_conv()` ফাংশন লিখবো। এই ফাংশনটি একটি 2D grayscale ইমেজ এবং একটি kernel নেবে, আর একটি convolved আউটপুট ইমেজ রিটার্ন করবে। ফাংশনের প্রতিটি লাইনে detailed comment দেওয়া আছে যাতে তুমি সহজে বুঝতে পারো:

```python
import numpy as np

def simple_conv(image, kernel):
    """
    2D convolution স্ক্র্যাচ থেকে ইমপ্লিমেন্টেশন

    Parameters:
    -----------
    image : 2D numpy array
        ইনপুট grayscale ইমেজ (H × W)
    kernel : 2D numpy array
        Convolution kernel (k × k)

    Returns:
    --------
    output : 2D numpy array
        Convolved আউটপুট ইমেজ (H-k+1 × W-k+1)
    """

    # ধাপ ১: ইমেজ ও kernel এর dimension বের করা
    img_h, img_w = image.shape    # ইমেজের height ও width
    k_h, k_w = kernel.shape       # Kernel এর height ও width

    # ধাপ ২: আউটপুট ইমেজের সাইজ হিসাব করা
    # ফর্মুলা: (H - k + 1) × (W - k + 1)  (stride=1 এর জন্য)
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1

    # ধাপ ৩: আউটপুট array তৈরি (শুরুতে সব 0 দিয়ে initialize)
    output = np.zeros((out_h, out_w), dtype=np.float64)

    # ধাপ ৪: Kernel কে 180° rotate করা
    # গাণিতিক convolution এ kernel flip করতে হয়
    # (cross-correlation এ flip করা লাগে না, কিন্তু
    #  ডিপ লার্নিং ফ্রেমওয়ার্কগুলো সাধারণত cross-correlation ব্যবহার করে)
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # ধাপ ৫: Sliding window — kernel কে পুরো ইমেজের উপর দিয়ে slide করানো
    for i in range(out_h):          # প্রতিটি row এর জন্য
        for j in range(out_w):      # প্রতিটি column এর জন্য

            # ইমেজের বর্তমান window (patch) বের করা
            # kernel যে জায়গাটা cover করছে সেটাই patch
            patch = image[i:i+k_h, j:j+k_w]

            # Element-wise multiplication এবং sum
            # এটিই convolution এর মূল কম্পিউটেশন
            output[i, j] = np.sum(patch * kernel_flipped)

    return output
```

এই ফাংশনে কিছু গুরুত্বপূর্ণ বিষয় লক্ষ্য করো। প্রথমত, আমরা kernel কে 180° rotate (flip) করেছি — এটি গাণিতিক convolution এর সংজ্ঞা অনুযায়ী প্রয়োজন। তবে ডিপ লার্নিং ফ্রেমওয়ার্কগুলো (PyTorch, TensorFlow) সাধারণত cross-correlation ব্যবহার করে যেখানে flip করা লাগে না — কারণ learnable kernel এর ক্ষেত্রে flip করা আর না করা একই ফলাফল দেয়, শুধু kernel এর মান flipped হয়ে learn হয়। দ্বিতীয়ত, nested for loop ব্যবহার করা হয়েছে — এটি slow কিন্তু বোঝার জন্য সবচেয়ে স্পষ্ট। বাস্তব ইমপ্লিমেন্টেশনে (যেমন OpenCV বা PyTorch) optimized C++/CUDA code ব্যবহার হয় যা অনেক ফাস্ট কাজ করে।

### বিভিন্ন ফিল্টার দিয়ে simple_conv() টেস্ট করা

এখন আমরা আমাদের `simple_conv()` ফাংশন বিভিন্ন ফিল্টার দিয়ে টেস্ট করবো এবং ফলাফল ভিজুয়ালাইজ করবো:

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ইমেজ লোড ও grayscale convert
img = cv2.imread("resources/lena.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

# বিভিন্ন ফিল্টারের kernel ডিফাইন
filters = {
    "Identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]], dtype=np.float64),

    "Box Blur": np.ones((3, 3), dtype=np.float64) / 9,

    "Sharpen": np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float64),

    "Sobel-X": np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64),

    "Sobel-Y": np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64),

    "Laplacian": np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=np.float64)
}

# প্রতিটি ফিল্টার apply করা
results = {}
for name, kernel in filters.items():
    results[name] = simple_conv(img_gray, kernel)
    print(f"{name}: input={img_gray.shape} → output={results[name].shape}")

# ভিজুয়ালাইজেশন
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(img_gray, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

idx = 1
for name, result in results.items():
    row = idx // 4
    col = idx % 4

    # Sobel ফিল্টারের absolute value দেখানো
    if "Sobel" in name:
        axes[row, col].imshow(np.abs(result), cmap="gray")
    else:
        # মান clamp করা [0, 255] range এ
        display = np.clip(result, 0, 255)
        axes[row, col].imshow(display, cmap="gray")

    axes[row, col].set_title(f"{name}\n{result.shape}")
    axes[row, col].axis("off")
    idx += 1

# খালি subplot hide
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()
```

Identity ফিল্টার ইমেজে কোনো পরিবর্তন করবে না — এটি আমাদের convolution ফাংশন সঠিকভাবে কাজ করছে কিনা verify করার উপায়। Box Blur ইমেজকে smooth করবে। Sharpen এজগুলো স্পষ্ট করবে। Sobel-X আর Sobel-Y যথাক্রমে vertical আর horizontal এজ বের করবে। Laplacian সব দিকের এজ বের করবে। আউটপুট সাইজ প্রতিটি ক্ষেত্রেই (H-2) × (W-2) হবে কারণ kernel সাইজ 3×3, তাই উপর-নিচ আর বাম-ডানে 1 পিক্সেল করে কমে যায়।

### Stride কনসেপ্ট

Stride মানে কনভোলিউশনের সময় kernel কত পিক্সেল করে জাম্প করে সরছে। পর্যন্ত আমরা stride=1 ব্যবহার করেছি — অর্থাৎ প্রতি ধাপে 1 পিক্সেল করে ডানে বা নিচে সরছি। কিন্তু stride=2 হলে kernel 2 পিক্সেল করে জাম্প করবে, ফলে আউটপুট ইমেজ আরও ছোট হবে। এটি একটি downsampling technique — বাস্তবে pooling layer এর মতো কাজ করে।

```python
def simple_conv_with_stride(image, kernel, stride=1):
    """
    Stride সহ 2D convolution ইমপ্লিমেন্টেশন
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # Stride সহ আউটপুট সাইজ
    out_h = (img_h - k_h) // stride + 1
    out_w = (img_w - k_w) // stride + 1

    output = np.zeros((out_h, out_w), dtype=np.float64)
    kernel_flipped = np.flipud(np.fliplr(kernel))

    for i in range(out_h):
        for j in range(out_w):
            # Stride অনুযায়ী পজিশন হিসাব
            r = i * stride
            c = j * stride
            patch = image[r:r+k_h, c:c+k_w]
            output[i, j] = np.sum(patch * kernel_flipped)

    return output

# Stride এর effect দেখা
img_small = img_gray[:100, :100]  # ছোট crop

out_s1 = simple_conv_with_stride(img_small, filters["Box Blur"], stride=1)
out_s2 = simple_conv_with_stride(img_small, filters["Box Blur"], stride=2)

print(f"Input: {img_small.shape}")
print(f"Stride=1 output: {out_s1.shape}")
print(f"Stride=2 output: {out_s2.shape}")
```

Stride=2 তে আউটপুট সাইজ অর্ধেকের কিছু বেশি হবে — এটি spatial resolution কমিয়ে দেয় কিন্তু computation ও মেমরি বাঁচায়। CNN তে stride ব্যবহার করে feature map এর spatial dimension কমানো হয়, যাতে deeper layer এ larger receptive field পাওয়া যায়।

### কেন ম্যানুয়াল কনভোলিউশন বোঝা জরুরি?

তুমি হয়তো ভাবতে পারো — "যখন `cv2.filter2D()` আছে, তখন নিজে কনভোলিউশন লিখব কেন?" এটি একটি যৌক্তিক প্রশ্ন। কিন্তু ম্যানুয়াল কনভোলিউশন বোঝা কয়েকটি কারণে অত্যন্ত জরুরি:

**প্রথমত**, CNN এর ভেতরে ঠিক এই একই অপারেশন হচ্ছে — শুধু অনেকগুলো ফিল্টার একসাথে, অনেকগুলো চ্যানেলে। তুমি যদি 2D convolution ভালো না বোঝো, 3D convolution বা CNN architecture বোঝা অসম্ভব হয়ে যাবে। দ্বিতীয়ত, debugging এর সময় এই বোঝা কাজে লাগে — যখন CNN model ভুল আউটপুট দেবে, তখন তোমাকে বুঝতে হবে ভেতরে কী হচ্ছে। তৃতীয়ত, custom layer বা custom operation লিখতে হলে তোমাকে নিজেই convolution ইমপ্লিমেন্ট করতে হবে — ফ্রেমওয়ার্ক সবসময় সব feature support করে না। সবশেষে, interview তে বা research paper পড়ার সময় এই fundamental understanding অপরিহার্য।

CNN এর কনভোলিউশন আর আমাদের `simple_conv()` এর মধ্যে প্রধান পার্থক্য হলো — CNN তে kernel এর মান আমরা দিই না, training এর সময় backpropagation দিয়ে learn হয়। কিন্তু অপারেশন একই — slide, multiply, sum। এই সরল অপারেশনটি যখন লক্ষ লক্ষ ফিল্টারে আর ডজনখানেক লেয়ারে ব্যবহার হয়, তখনই তৈরি হয় সেই শক্তিশালী feature extraction যা ইমেজ ক্লাসিফিকেশন, অবজেক্ট ডিটেকশন ইত্যাদি টাস্কে অবিশ্বাস্য accuracy দেয়।
