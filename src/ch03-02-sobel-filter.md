## সোবেল ফিল্টার ও এজ ডিটেকশন

এজ ডিটেকশন হলো কম্পিউটার ভিশন এর অন্যতম গুরুত্বপূর্ণ টাস্ক। ইমেজে এজ (edge) হলো সেই জায়গা যেখানে pixel intensity হঠাৎ করে বদলায় — অর্থাৎ দুটি ভিন্ন রং বা brightness এর মধ্যে সীমানা। মানুষের চোখ অনায়াসেই এজ চিনতে পারে, কিন্তু কম্পিউটারকে গাণিতিকভাবে এজ বের করতে হয়। সোবেল ফিল্টার (Sobel filter) হলো এজ ডিটেকশন এর সবচেয়ে জনপ্রিয় এবং ক্লাসিক পদ্ধতিগুলোর একটি। এটি ১৯৬৮ সালে Irwin Sobel এবং Gary Feldman তৈরি করেছিলেন, এবং আজও এটি ব্যাপকভাবে ব্যবহৃত। সোবেল ফিল্টার আসলে দুটি আলাদা ফিল্টার — Sobel-X এবং Sobel-Y, যা যথাক্রমে vertical এবং horizontal এজ বের করে। এই সেকশনে আমরা সোবেল ফিল্টার কিভাবে কাজ করে তা বিস্তারিত দেখবো এবং নিজেদের হাতে ইমপ্লিমেন্ট করবো।

### Sobel-X Kernel — Vertical Edge Detection

Sobel-X kernel হলো একটি 3×3 ম্যাট্রিক্স যা vertical edge বের করতে ব্যবহৃত হয়। Vertical edge মানে এমন এজ যেখানে বাম থেকে ডানে pixel intensity হঠাৎ বদলায় — অর্থাৎ বাম দিকে একটা রং আর ডান দিকে আরেকটা রং। Sobel-X kernel টি হলো:

```
Sobel-X Kernel:
[[-1,  0,  1],
 [-2,  0,  2],
 [-1,  0,  1]]
```

এই kernel টি খেয়াল করো — মাঝের কলামে সব 0, বামের কলামে negative মান, আর ডানের কলামে positive মান। মাঝের সারির weight (-2 আর 2) কোনার সারির weight (-1 আর 1) থেকে দ্বিগুণ — এটি central difference approximation যা noise এর প্রভাব কমায়। যখন এই kernel টি এমন একটি জায়গায় slide করে যেখানে বাম দিকে intensity বেশি আর ডান দিকে intensity কম (অর্থাৎ একটি vertical edge), তখন convolution এর ফলাফল একটি বড় positive বা negative মান হয়। আর যেখানে intensity সমান, সেখানে ফলাফল প্রায় 0 হয় — কারণ negative আর positive মানগুলো cancel out হয়ে যায়।

### Sobel-Y Kernel — Horizontal Edge Detection

Sobel-Y kernel হলো Sobel-X এর transposed রূপ, যা horizontal edge বের করে। Horizontal edge মানে এমন এজ যেখানে উপর থেকে নিচে pixel intensity হঠাৎ বদলায়। Sobel-Y kernel টি হলো:

```
Sobel-Y Kernel:
[[-1, -2, -1],
 [ 0,  0,  0],
 [ 1,  2,  1]]
```

এখানেও একই নীতি — উপরের সারিতে negative মান, নিচের সারিতে positive মান, আর মাঝের সারিতে 0। মাঝের কলামের weight বেশি কারণ central row বেশি গুরুত্বপূর্ণ। যখন এই kernel টি এমন জায়গায় slide করে যেখানে উপরে intensity বেশি আর নিচে intensity কম (অর্থাৎ একটি horizontal edge), তখন convolution এর ফলাফল বড় মান হয়। সমান intensity এলাকায় ফলাফল প্রায় 0।

সহজ কথায় — Sobel-X দেখে বাম-ডান পরিবর্তন, Sobel-Y দেখে উপর-নিচ পরিবর্তন। দুটি মিলিয়েই আমরা সব দিকের এজ পাই।

### Combined Edge Magnitude

Sobel-X দিয়ে শুধু vertical edge পাই, Sobel-Y দিয়ে শুধু horizontal edge পাই। কিন্তু বাস্তব ইমেজে এজ সবসময় সোজা vertical বা horizontal হয় না — তির্যক (diagonal) দিকেও এজ থাকে। সব দিকের এজ একসাথে পেতে আমরা দুটি Sobel ফিল্টারের ফলাফল মিলিয়ে একটি combined edge magnitude বের করি। এর ফর্মুলা হলো Pythagorean theorem এর মতো:

**Magnitude = √(SobelX² + SobelY²)**

এটি আসলে একটি gradient vector এর magnitude। Sobel-X হলো x-দিকের gradient component (∂I/∂x), Sobel-Y হলো y-দিকের gradient component (∂I/∂y)। এই দুটি component থেকে gradient এর মোট শক্তি (magnitude) বের করাই হলো edge magnitude। যেখানে magnitude বেশি, সেখানে এজ আছে — এটাই আমাদের কাঙ্ক্ষিত ফলাফল।

### সোবেল ফিল্টার ইমপ্লিমেন্টেশন

এখন আমরা সোবেল ফিল্টার আমাদের কাস্টম convolution ফাংশন দিয়ে apply করবো। আগের সেকশনে শেখা `simple_conv()` ফাংশন ব্যবহার করে আমরা Sobel-X এবং Sobel-Y আলাদাভাবে apply করবো, তারপর দুটো মিলিয়ে combined edge map বের করবো। এটি করতে হলে প্রথমে ইমেজটিকে grayscale এ convert করতে হবে, কারণ সোবেল ফিল্টার single channel ইমেজে কাজ করে।

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# simple_conv ফাংশন (আগের সেকশন থেকে)
def simple_conv(image, kernel):
    """
    2D convolution স্ক্র্যাচ থেকে ইমপ্লিমেন্টেশন
    image: 2D numpy array (grayscale)
    kernel: 2D numpy array (filter)
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # আউটপুট সাইজ হিসাব
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1

    # আউটপুট array তৈরি
    output = np.zeros((out_h, out_w), dtype=np.float64)

    # kernel কে 180° rotate করা (cross-correlation vs convolution)
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # sliding window দিয়ে convolution
    for i in range(out_h):
        for j in range(out_w):
            # ইমেজের বর্তমান window
            patch = image[i:i+k_h, j:j+k_w]
            # element-wise multiplication ও sum
            output[i, j] = np.sum(patch * kernel_flipped)

    return output

# ইমেজ লোড ও grayscale convert
img = cv2.imread("resources/lena.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Sobel kernel ডিফাইন
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

sobel_y_kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64)

# সোবেল ফিল্টার apply করা
sobel_x = simple_conv(img_gray, sobel_x_kernel)
sobel_y = simple_conv(img_gray, sobel_y_kernel)

# Combined edge magnitude
edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

print(f"Original shape: {img_gray.shape}")
print(f"Sobel-X shape: {sobel_x.shape}")
print(f"Sobel-Y shape: {sobel_y.shape}")
print(f"Edge magnitude shape: {edge_magnitude.shape}")
print(f"Edge magnitude range: [{edge_magnitude.min():.2f}, {edge_magnitude.max():.2f}]")
```

### ভিজুয়ালাইজেশন

এখন আমরা তিনটি ফলাফল — Sobel-X, Sobel-Y, এবং Combined Edge Magnitude — পাশাপাশি দেখবো। এতে করে পরিষ্কার বোঝা যাবে প্রতিটি ফিল্টার কোন ধরনের এজ বের করছে।

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Original grayscale
axes[0, 0].imshow(img_gray, cmap="gray")
axes[0, 0].set_title("Original Grayscale")

# Sobel-X — Vertical edges
axes[0, 1].imshow(np.abs(sobel_x), cmap="gray")
axes[0, 1].set_title("Sobel-X (Vertical Edges)")

# Sobel-Y — Horizontal edges
axes[1, 0].imshow(np.abs(sobel_y), cmap="gray")
axes[1, 0].set_title("Sobel-Y (Horizontal Edges)")

# Combined edge magnitude
axes[1, 1].imshow(edge_magnitude, cmap="gray")
axes[1, 1].set_title("Combined Edge Magnitude\n√(SobelX² + SobelY²)")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

Sobel-X এর ফলাফলে তুমি দেখবে vertical এজ গুলো স্পষ্ট — যেমন ইমেজের বস্তুর বাম-ডান সীমানা। Sobel-Y তে horizontal এজ গুলো স্পষ্ট — যেমন উপর-নিচ সীমানা। Combined magnitude তে সব দিকের এজ একসাথে দেখা যাচ্ছে, যা সবচেয়ে complete edge map। মনে রাখবে, `np.abs()` ব্যবহার করা হয়েছে কারণ Sobel এর ফলাফল negative মানও হতে পারে, কিন্তু ইমেজ ডিসপ্লে করতে হলে positive মান লাগে।

### OpenCV এর Built-in Sobel ফাংশন

আমাদের কাস্টম ফাংশন ছাড়াও OpenCV তে built-in `cv2.Sobel()` ফাংশন আছে যা অনেক ফাস্ট কাজ করে (কারণ এটি C++ এ optimized)। তুমি দৈনন্দিন কাজে এটি ব্যবহার করতে পারো:

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("resources/lena.png", cv2.IMREAD_GRAYSCALE)

# OpenCV এর built-in Sobel
sobel_x_cv = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # dx=1, dy=0 → Sobel-X
sobel_y_cv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # dx=0, dy=1 → Sobel-Y

# Combined magnitude
edge_mag_cv = np.sqrt(sobel_x_cv ** 2 + sobel_y_cv ** 2)

# uint8 তে convert (ডিসপ্লে করার জন্য)
edge_mag_cv_uint8 = np.uint8(np.clip(edge_mag_cv, 0, 255))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(np.abs(sobel_x_cv), cmap="gray")
axes[0].set_title("Sobel-X (OpenCV)")

axes[1].imshow(np.abs(sobel_y_cv), cmap="gray")
axes[1].set_title("Sobel-Y (OpenCV)")

axes[2].imshow(edge_mag_cv_uint8, cmap="gray")
axes[2].set_title("Combined Edge Magnitude (OpenCV)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

`cv2.Sobel()` ফাংশনে `cv2.CV_64F` ব্যবহার করা হয়েছে যাতে negative মানও সংরক্ষণ হয়। `1, 0` মানে x-দিকে derivative নিতে হবে (Sobel-X), `0, 1` মানে y-দিকে derivative নিতে হবে (Sobel-Y)। `ksize=3` মানে 3×3 kernel ব্যবহার হবে। আমাদের কাস্টম ইমপ্লিমেন্টেশন আর OpenCV এর built-in ফলাফল প্রায় একই হবে — এটি নিশ্চিত করে যে আমাদের বোঝা সঠিক।

সোবেল ফিল্টার শুধু এজ ডিটেকশন এ সীমাবদ্ধ নয় — এটি অবজেক্ট ডিটেকশন, ইমেজ সেগমেন্টেশন, feature extraction এর মতো অনেক জায়গায় ব্যবহৃত হয়। আর সবচেয়ে গুরুত্বপূর্ণ ব্যাপার — CNN এর প্রথম লেয়ারের ফিল্টারগুলো training এর পর অনেক সময় Sobel ফিল্টারের মতো দেখতে হয়। মানে নেটওয়ার্ক নিজে থেকেই শিখে নেয় যে এজ ডিটেকশন ইমেজ বোঝার জন্য গুরুত্বপূর্ণ — ঠিক যেমনটা আমরা মানুষের তৈরি সোবেল ফিল্টার দিয়ে করি!
