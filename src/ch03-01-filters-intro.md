## ইমেজ ফিল্টার পরিচিতি

ইমেজ ফিল্টার হলো কম্পিউটার ভিশন এর অন্যতম গুরুত্বপূর্ণ ধারণা। ফিল্টার ব্যবহার করে আমরা ইমেজের বিভিন্ন বৈশিষ্ট্য বের করতে পারি  যেমন এজ, কর্নার, টেক্সচার ইত্যাদি। ফিল্টারকে আরেক নামেও ডাকা হয়  kernel। একটি kernel হলো ছোট একটি সংখ্যার ম্যাট্রিক্স, যা ইমেজের উপর দিয়ে slide করে এবং প্রতিটি পজিশনে element-wise multiplication ও sum করে একটি নতুন মান তৈরি করে। এই পুরো প্রক্রিয়াটিকেই বলে convolution। এই সেকশনে আমরা ফিল্টার কী, কিভাবে কাজ করে, এবং কোন কোন ধরনের ফিল্টার আছে  সেসব বিস্তারিত শিখবো।

### ফিল্টার বা কারনেল কী?

ফিল্টার বা kernel হলো একটি ছোট সংখ্যার ম্যাট্রিক্স, সাধারণত 3×3 বা 5×5 সাইজের। এই ম্যাট্রিক্সের মানগুলো নির্ধারণ করে যে ফিল্টারটি কী কাজ করবে। যেমন, সব মান সমান হলে সেটি blur ফিল্টার হবে, আর মাঝের মান বড় আর চারপাশের মান ছোট হলে সেটি sharpen ফিল্টার হবে। কনভোলিউশন অপারেশনের সময় এই kernel টি ইমেজের প্রতিটি পিক্সেলের উপর দিয়ে slide করে, প্রতিটি overlapping পজিশনে element-wise multiplication করে এবং সব গুণফল যোগ করে একটি আউটপুট মান পাওয়া যায়। এভাবে পুরো ইমেজের উপর দিয়ে slide করে একটি নতুন আউটপুট ইমেজ তৈরি হয়। এই নতুন ইমেজকে feature map বা activation map বলা হয়, কারণ এটি মূল ইমেজ থেকে কোনো একটি নির্দিষ্ট feature (যেমন এজ, কর্নার) বের করে আনে।

ফিল্টার দিয়ে ইমেজকে বিভিন্নভাবে transform করা যায়। সবচেয়ে সাধারণ transformation গুলো হলো:

- **Sharpen**  ইমেজের এজ ও ডিটেইল বেশি স্পষ্ট করে তোলে। মাঝের পিক্সেলের মান বড় আর পার্শ্ববর্তী পিক্সেলের মান ছোট রাখা হয়, যাতে লোকাল কন্ট্রাস্ট বাড়ে।
- **Blur**  ইমেজের ছোট ছোট noise দূর করে ইমেজকে smooth করে। সব মান সমান রাখা হয় (box blur), অথবা মাঝের দিকে বড় আর কোনায় ছোট মান দেওয়া হয় (Gaussian blur)।
- **Edge Detect**  ইমেজে যেখানে intensity হঠাৎ করে বদলায়, সেই boundary গুলো বের করে। Sobel, Laplacian এগুলো edge detection ফিল্টার।
- **Emboss**  ইমেজকে এমনভাবে transform করে যেন মনে হয় ইমেজটি কাগজে উঠিয়ে লেখা (embossed)। এটি এজের একপাশে highlight আর অন্যপাশে shadow তৈরি করে।

### ইমেজ রিডিং  matplotlib vs OpenCV

ইমেজ রিড করার জন্য আমরা দুটি জনপ্রিয় লাইব্রেরি ব্যবহার করতে পারি  OpenCV এবং matplotlib। তবে এই দুটি লাইব্রেরির মধ্যে একটি গুরুত্বপূর্ণ পার্থক্য আছে যেটা মনে রাখা জরুরি। OpenCV ইমেজ রিড করে BGR (Blue-Green-Red) ফরম্যাটে, আর matplotlib এর `mpimg` মডিউল রিড করে RGB (Red-Green-Blue) ফরম্যাটে। যদি তুমি OpenCV দিয়ে রিড করে matplotlib দিয়ে ডিসপ্লে করো, তাহলে রং ভুল দেখাবে  লাল জায়গায় নীল, নীল জায়গায় লাল দেখাবে। তাই যখন দুটি লাইব্রেরি একসাথে ব্যবহার করবে, তখন অবশ্যই BGR থেকে RGB তে convert করতে হবে।

```python
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read with matplotlib - comes in RGB format
img_mpl = mpimg.imread("resources/lena.png")
print(f"matplotlib shape: {img_mpl.shape}")

# Read with OpenCV - comes in BGR format
img_cv2 = cv2.imread("resources/lena.png")
print(f"OpenCV shape: {img_cv2.shape}")

# Convert from BGR to RGB
img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_mpl)
axes[0].set_title("matplotlib (RGB)")

axes[1].imshow(img_cv2)
axes[1].set_title("OpenCV BGR (Wrong color!)")

axes[2].imshow(img_cv2_rgb)
axes[2].set_title("OpenCV → RGB (Correct)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

এই কোডটি রান করলে তুমি দেখবে মাঝের ইমেজটিতে রং ভুল দেখাচ্ছে  লাল আর নীল জায়গা বদলে গেছে। তৃতীয় ইমেজটিতে `cv2.cvtColor()` ব্যবহার করে BGR কে RGB তে convert করা হয়েছে, তাই সেটি সঠিক দেখাচ্ছে। এই বিষয়টি মনে রাখা অত্যন্ত জরুরি  যখনই OpenCV আর matplotlib একসাথে ব্যবহার করবে, অবশ্যই conversion করবে।

### RGB চ্যানেল ভিজুয়ালাইজেশন

RGB ইমেজ আসলে তিনটি আলাদা চ্যানেলের সমষ্টি  Red, Green, এবং Blue। `cv2.split()` ফাংশন ব্যবহার করে আমরা এই তিনটি চ্যানেলকে আলাদা করতে পারি। প্রতিটি চ্যানেল আসলে একটি 2D grayscale ইমেজ, যেখানে প্রতিটি পিক্সেলের মান 0 থেকে 255 এর মধ্যে থাকে। 0 মানে সেই রং একদম নেই, 255 মানে সেই রং সবচেয়ে বেশি। তিনটি চ্যানেল একসাথে mix হয়ে আমরা যে রঙিন ইমেজ দেখি সেটি তৈরি হয়।

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("resources/lena.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Split the three channels
R, G, B = cv2.split(img_rgb)

print(f"Red channel shape: {R.shape}")
print(f"Green channel shape: {G.shape}")
print(f"Blue channel shape: {B.shape}")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original RGB")

axes[0, 1].imshow(R, cmap="Reds")
axes[0, 1].set_title("Red Channel")

axes[1, 0].imshow(G, cmap="Greens")
axes[1, 0].set_title("Green Channel")

axes[1, 1].imshow(B, cmap="Blues")
axes[1, 1].set_title("Blue Channel")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

প্রতিটি চ্যানেলের ভিজুয়ালাইজেশনে উজ্জ্বল এলাকা মানে সেই রং বেশি আছে, আর অন্ধকার এলাকা মানে সেই রং কম আছে। যেমন, Red channel এ যেসব জায়গায় উজ্জ্বল দেখাচ্ছে, সেখানে মূল ইমেজে লাল রং বেশি আছে। এভাবে চ্যানেল আলাদা করে দেখলে বোঝা যায় প্রতিটি রং কোথায় কতটুকু অবদান রাখছে।

### Grayscale কনভার্শন

অনেক সময় আমাদের রঙিন ইমেজের বদলে grayscale (সাদা-কালো) ইমেজ নিয়ে কাজ করতে হয়। Grayscale ইমেজে শুধু একটি চ্যানেল থাকে, তাই computation অনেক কম হয় এবং অনেক অপারেশন সহজ হয়ে যায়। RGB থেকে grayscale এ convert করার একটি নির্দিষ্ট ফর্মুলা আছে  এটি সরল গড় (average) নয়, বরং মানুষের চোখ কিভাবে বিভিন্ন রং দেখে তার উপর ভিত্তি করে তৈরি:

**Gray = 0.299 × R + 0.587 × G + 0.114 × B**

এখানে Green এর weight সবচেয়ে বেশি (0.587), কারণ মানুষের চোখ সবুজ রং সবচেয়ে বেশি সংবেদনশীল। Red এর weight (0.299) মাঝারি, আর Blue এর weight (0.114) সবচেয়ে কম। এই weighted sum ব্যবহার করে grayscale ইমেজ তৈরি করলে মানুষের চোখে মূল ইমেজের সাথে সবচেয়ে কাছের brightness পাওয়া যায়।

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("resources/lena.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale with OpenCV
img_gray_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Manually convert to grayscale (using formula)
R, G, B = cv2.split(img_rgb)
img_gray_manual = np.uint8(0.299 * R + 0.587 * G + 0.114 * B)

print(f"OpenCV grayscale shape: {img_gray_cv2.shape}")
print(f"Manual grayscale shape: {img_gray_manual.shape}")
print(f"Difference between the two (max): {np.max(np.abs(img_gray_cv2.astype(float) - img_gray_manual.astype(float)))}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original RGB")

axes[1].imshow(img_gray_cv2, cmap="gray")
axes[1].set_title("Grayscale (OpenCV)")

axes[2].imshow(img_gray_manual, cmap="gray")
axes[2].set_title("Grayscale (Manual Formula)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

দুটি grayscale ইমেজ প্রায় একই দেখাবে, কারণ OpenCV ও একই ফর্মুলা ব্যবহার করে (শুধু rounding এ সামান্য পার্থক্য হতে পারে)। এই ফর্মুলাটি মনে রাখা জরুরি  গাণিতিকভাবে কনভোলিউশন বোঝার জন্য grayscale ইমেজ নিয়ে কাজ করা অনেক সহজ।

### সাধারণ ফিল্টার টাইপ ও তাদের Kernel

এখন আসি সবচেয়ে মজার পার্টে  কোন ফিল্টার কোন কাজ করে এবং তাদের kernel দেখতে কেমন। নিচের টেবিলে সবচেয়ে সাধারণ ফিল্টার গুলো দেওয়া হলো:

| ফিল্টার নাম | 3×3 Kernel | কাজ |
|---|---|---|
| Identity | [[0,0,0],[0,1,0],[0,0,0]] | ইমেজে কোনো পরিবর্তন করে না |
| Box Blur | [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]] | ইমেজকে smooth করে, noise কমায় |
| Gaussian Blur | [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]] | Smooth blur, কেন্দ্রে বেশি weight |
| Sharpen | [[0,-1,0],[-1,5,-1],[0,-1,0]] | এজ ও ডিটেইল স্পষ্ট করে |
| Laplacian | [[0,1,0],[1,-4,1],[0,1,0]] | এজ ডিটেকশন (সব দিকে) |
| Emboss | [[-2,-1,0],[-1,1,1],[0,1,2]] | 3D embossed ইফেক্ট তৈরি |

Identity kernel হলো সবচেয়ে সহজ ফিল্টার  মাঝে 1 আর বাকি সব 0। এটি ইমেজে কোনো পরিবর্তন করে না, convolution এর সবচেয়ে বেসিক উদাহরণ। Box Blur kernel এ সব মান সমান (1/9), তাই এটি একটি সাধারণ average নেয়  প্রতিটি পিক্সেলকে তার পাশের 8টি পিক্সেলের সাথে average করে দেয়। Gaussian Blur একটু ভিন্ন  মাঝের পিক্সেলে সবচেয়ে বেশি weight (4/16) আর কোনায় সবচেয়ে কম weight (1/16), যার ফলে blur বেশি natural দেখায়। Sharpen kernel এ মাঝে 5 আর চারপাশে -1, যা মাঝের পিক্সেলকে বাড়িয়ে দেয় আর পাশের পিক্সেলকে কমিয়ে দেয়  ফলে লোকাল কন্ট্রাস্ট বেড়ে গিয়ে এজ শার্প হয়। Laplacian kernel সব দিকের এজ একসাথে বের করে  মাঝে -4 আর চারদিকে 1, যা দ্বিতীয় ডেরিভেটিভ (second derivative) হিসেবে কাজ করে। Emboss kernel একপাশে negative আর অন্যপাশে positive মান রাখে, যা এজের একদিকে highlight আর অন্যদিকে shadow তৈরি করে।

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("resources/lena.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define kernels for different filters
kernels = {
    "Identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]),

    "Box Blur": np.ones((3, 3), dtype=np.float32) / 9,

    "Gaussian Blur": np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=np.float32) / 16,

    "Sharpen": np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32),

    "Laplacian": np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=np.float32),

    "Emboss": np.array([[-2, -1, 0],
                        [-1,  1, 1],
                        [ 0,  1, 2]], dtype=np.float32)
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Original image
axes[0, 0].imshow(img_gray, cmap="gray")
axes[0, 0].set_title("Original Grayscale")
axes[0, 0].axis("off")

# Apply and show each filter
idx = 1
for name, kernel in kernels.items():
    row = idx // 4
    col = idx % 4
    filtered = cv2.filter2D(img_gray, -1, kernel)
    axes[row, col].imshow(filtered, cmap="gray")
    axes[row, col].set_title(name)
    axes[row, col].axis("off")
    idx += 1

# Hide empty subplot
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()
```

এই কোডটি রান করলে তুমি দেখতে পাবে Identity ফিল্টার ইমেজে কোনো পরিবর্তন করে না, Box Blur আর Gaussian Blur ইমেজকে smooth করে, Sharpen এজগুলো স্পষ্ট করে, Laplacian শুধু এজ বের করে, আর Emboss একটি 3D ইফেক্ট তৈরি করে। `cv2.filter2D()` ফাংশনটি OpenCV এর built-in convolution ফাংশন  পরে আমরা নিজের convolution ফাংশন লিখবো।

### Hand-crafted ফিল্টার থেকে CNN Learned ফিল্টার

এখন পর্যন্ত আমরা যেসব ফিল্টার দেখলাম সেগুলো মানুষ ডিজাইন করেছে  এদের বলে hand-crafted filters বা traditional filters। এই ফিল্টারগুলোর kernel মান নির্দিষ্ট, আমরা আগে থেকেই জানি কোন kernel দিয়ে কী কাজ হবে। কিন্তু Convolutional Neural Network (CNN) এ ফিল্টারের kernel মান আমরা ডিজাইন করি না  নেটওয়ার্ক training এর সময় data থেকে নিজেই learn করে। এটিই CNN এর সবচেয়ে বড় শক্তি।

CNN এর প্রথম লেয়ারের ফিল্টারগুলো অনেক সময় Sobel বা Gabor ফিল্টারের মতো দেখতে হয়  কারণ এজ ডিটেকশন অনেক ভিশন টাস্কের জন্য একটি মৌলিক feature। কিন্তু ডিপ লেয়ারের ফিল্টারগুলো আরও complex pattern detect করে  texture, shape, এমনকি object এর অংশও। Traditional ফিল্টার দিয়ে এই complex pattern detect করা সম্ভব না, কিন্তু CNN data থেকে স্বয়ংক্রিয়ভাবে সেরা ফিল্টার learn করে নেয়। তাই CNN শুধু edge নয়, যেকোনো visual feature automatically learn করতে পারে  এটিই deep learning এর মূল শক্তি। পরের চ্যাপ্টারগুলোতে আমরা দেখবো কিভাবে CNN এই ফিল্টারগুলো learn করে এবং কিভাবে 3D ইমেজে convolution কাজ করে।
