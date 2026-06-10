## বেসিক ইমেজ অপারেশনস

ইমেজ রিড করা শিখলেই হবে না, সেই ইমেজের উপর বিভিন্ন অপারেশন করতে হবে — রং পরিবর্তন, ব্লার করা, এজ ডিটেকশন ইত্যাদি। এই সেকশনে আমরা এই বেসিক অপারেশনগুলো শিখবো যগুলো প্রায় প্রতিটি কম্পিউটার ভিশন প্রজেক্টে ব্যবহার হয়। এগুলো হলো ইমেজ প্রসেসিং এর বিল্ডিং ব্লক — যেকোনো অ্যাডভান্সড টেকনিক শিখতে গেলে এই বেসিকগুলো জানা অপরিহার্য।

### কালার কনভার্শন — cv2.cvtColor()

ইমেজ প্রসেসিং এ কালার স্পেস conversion সবচেয়ে বেশি ব্যবহৃত অপারেশনগুলোর একটি। OpenCV তে `cv2.cvtColor()` ফাংশন দিয়ে এক কালার স্পেস থেকে অন্য কালার স্পেসে রূপান্তর করা যায়। সবচেয়ে বেশি ব্যবহৃত conversion গুলো হলো — BGR to RGB, BGR to Grayscale, এবং BGR to HSV। প্রতিটি conversion এর নিজস্ব ব্যবহার আছে। যেমন, Grayscale ব্যবহার করা হয় edge detection এবং thresholding এর আগে, আর HSV ব্যবহার করা হয় কালার-বেসড অবজেক্ট ডিটেকশনে।

**BGR to RGB:** আগের সেকশনে আমরা দেখেছি যে OpenCV ইমেজ BGR ফরম্যাটে রিড করে। যখন আমরা matplotlib বা অন্য RGB-based টুল ব্যবহার করবো, তখন BGR কে RGB তে convert করতে হবে। এটি করা খুবই সহজ:

**BGR to Grayscale:** Grayscale ইমেজ হলো সাদা-কালো ইমেজ যেখানে পিক্সেলের মান ০ (কালো) থেকে ২৫৫ (সাদা) পর্যন্ত হয়। Grayscale conversion করলে ইমেজের ডেটা তিনগুণ কমে যায় (৩ চ্যানেল থেকে ১ চ্যানেল), ফলে প্রসেসিং অনেক দ্রুত হয়। Edge detection, thresholding, face detection ইত্যাদি অ্যালগরিদম সাধারণত grayscale ইমেজে ভালো কাজ করে।

**BGR to HSV:** HSV (Hue, Saturation, Value) কালার স্পেস কালার-বেসড অবজেক্ট ডিটেকশনের জন্য অত্যন্ত কার্যকর। RGB/BGR এ কালার detect করা কঠিন কারণ সেখানে লাইটিং এর প্রভাব বেশি পড়ে। কিন্তু HSV তে Hue channel লাইটিং এর থেকে অনেকটাই স্বাধীন, তাই নির্দিষ্ট রং এর অবজেক্ট detect করা অনেক সহজ হয়। যেমন হলুদ বল detect করতে HSV range ব্যবহার করা হয়।

```python
import cv2

img = cv2.imread("resources/lena.png")

# BGR to RGB (matplotlib দিয়ে দেখানোর জন্য)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR to Grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BGR to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# সব ইমেজ একসাথে দেখানো
cv2.imshow("Original (BGR)", img)
cv2.imshow("Grayscale", img_gray)
cv2.imshow("HSV", img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Shape চেক করা
print(f"BGR shape: {img.shape}")        # (height, width, 3)
print(f"Grayscale shape: {img_gray.shape}")  # (height, width) — চ্যানেল নেই!
print(f"HSV shape: {img_hsv.shape}")        # (height, width, 3)
```

খেয়াল করো, grayscale ইমেজের shape এ চ্যানেল নম্বর থাকে না — কারণ সেখানে শুধু একটি চ্যানেল আছে। এটি পরবর্তী অপারেশনগুলোতে গুরুত্বপূর্ণ ভূমিকা রাখে।

### ব্লারিং — cv2.GaussianBlur()

ইমেজ ব্লার করা অনেক ক্ষেত্রেই প্রয়োজন হয় — noise কমানো, প্রাইভেসি সুরক্ষা, বা edge detection এর আগে smoothing করার জন্য। OpenCV তে বিভিন্ন ধরনের blur আছে — Average Blur, Gaussian Blur, Median Blur, Bilateral Filter ইত্যাদি। এর মধ্যে Gaussian Blur সবচেয়ে বেশি ব্যবহৃত কারণ এটি natural-looking blur তৈরি করে এবং গাণিতিকভাবে well-defined।

`cv2.GaussianBlur()` ফাংশনের প্রধান প্যারামিটার হলো kernel size — এটি একটি tuple যেমন `(5, 5)` বা `(15, 15)`। Kernel size যত বড় হবে, blur তত বেশি হবে। Kernel size সবসময় odd number হতে হবে (3, 5, 7, 9...)। একটি ছোট kernel (3, 3) সামান্য blur দেবে, আর বড় kernel (21, 21) ইমেজটাকে প্রায় অস্পষ্ট করে দেবে।

```python
import cv2

img = cv2.imread("resources/lena.png")

# বিভিন্ন kernel size দিয়ে Gaussian Blur
blur_light = cv2.GaussianBlur(img, (5, 5), 0)
blur_medium = cv2.GaussianBlur(img, (15, 15), 0)
blur_heavy = cv2.GaussianBlur(img, (31, 31), 0)

cv2.imshow("Original", img)
cv2.imshow("Light Blur (5x5)", blur_light)
cv2.imshow("Medium Blur (15x15)", blur_medium)
cv2.imshow("Heavy Blur (31x31)", blur_heavy)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

তৃতীয় প্যারামিটার `0` হলো sigmaX — এটি Gaussian kernel এর standard deviation। `0` দিলে OpenCV নিজেই kernel size থেকে সঠিক sigma ক্যালকুলেট করে নেয়। সাধারণত এটি `0` রাখাই ভালো।

### এজ ডিটেকশন — cv2.Canny()

Edge detection হলো ইমেজের মধ্যে সীমানা বা প্রান্ত খুঁজে বের করার প্রক্রিয়া। এটি object detection, shape recognition, feature extraction ইত্যাদি কাজের জন্য অপরিহার্য। সবচেয়ে জনপ্রিয় edge detection algorithm হলো Canny Edge Detector, যা John Canny ১৯৮৬ সালে তৈরি করেছিলেন। `cv2.Canny()` ফাংশন দুটি threshold মান নেয় — lower threshold এবং upper threshold। এই threshold গুলো কিভাবে কাজ করে তা বোঝা অত্যন্ত জরুরি।

**Threshold কিভাবে কাজ করে:** ইমেজের প্রতিটি পিক্সেলের gradient magnitude ক্যালকুলেট করা হয়। যেসব পিক্সেলের gradient upper threshold এর চেয়ে বেশি, সেগুলো অবশ্যই edge হিসেবে ধরা হয় (strong edge)। যেসব পিক্সেলের gradient lower threshold এর চেয়ে কম, সেগুলো edge হিসেবে বাদ দেওয়া হয় (non-edge)। আর যেগুলো দুটি threshold এর মাঝে পড়ে, সেগুলো তখনই edge হিসেবে ধরা হয় যখন সেগুলো কোনো strong edge এর সাথে connected থাকে (weak edge)। একে hysteresis thresholding বলে।

Threshold এর মান কম হলে অনেক বেশি edge পাওয়া যাবে (noise সহ), আর বেশি হলে শুধু স্পষ্ট edge গুলো পাওয়া যাবে। সাধারণত 1:2 বা 1:3 অনুপাতে lower এবং upper threshold সেট করা হয়।

```python
import cv2

img = cv2.imread("resources/lena.png")

# প্রথমে blur করা ভালো — noise কমানোর জন্য
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# Canny Edge Detection
edges_sensitive = cv2.Canny(img_blur, 50, 150)   # বেশি edge
edges_moderate = cv2.Canny(img_blur, 100, 200)   # মাঝারি edge
edges_strict = cv2.Canny(img_blur, 150, 250)     # কম edge (শুধু স্পষ্ট edges)

cv2.imshow("Original", img)
cv2.imshow("Edges (50, 150)", edges_sensitive)
cv2.imshow("Edges (100, 200)", edges_moderate)
cv2.imshow("Edges (150, 250)", edges_strict)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### ইমেজ Shape এবং Dimensions

ইমেজ প্রসেসিং এ ইমেজের dimensions জানা অনেক গুরুত্বপূর্ণ। `img.shape` দিয়ে ইমেজের height, width এবং number of channels জানা যায়। Color ইমেজে এটি `(height, width, 3)` রিটার্ন করে, আর grayscale ইমেজে `(height, width)` রিটার্ন করে। `img.size` দিয়ে total number of pixels এবং `img.dtype` দিয়ে data type জানা যায়।

```python
import cv2

img = cv2.imread("resources/lena.png")

print(f"Shape: {img.shape}")     # (height, width, channels)
print(f"Height: {img.shape[0]}")
print(f"Width: {img.shape[1]}")
print(f"Channels: {img.shape[2]}")
print(f"Total pixels: {img.size}")
print(f"Data type: {img.dtype}")  # সাধারণত uint8
```

এই তথ্যগুলো বিশেষ করে resizing, cropping, এবং matrix operations এর সময় দরকার হয়। মনে রাখবে, OpenCV তে shape এ height আগে আসে, width পরে — কিন্তু `cv2.resize()` তে width আগে দিতে হয়! এটি নিয়ে পরের সেকশনে বিস্তারিত আলোচনা করবো।
