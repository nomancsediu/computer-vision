## ইমেজ রিসাইজ এবং ক্রপ

ইমেজ প্রসেসিং এ সবচেয়ে বেশি ব্যবহৃত দুটি অপারেশন হলো resizing এবং cropping। রিয়েল-ওয়ার্ল্ড প্রজেক্টে আমরা বিভিন্ন সোর্স থেকে বিভিন্ন সাইজের ইমেজ পাই — কোনোটা 4K, কোনোটা 720p, কোনোটা আবার ছোট thumbnail। মেশিন লার্নিং মডেলগুলো সাধারণত নির্দিষ্ট সাইজের ইনপুট এক্সপেক্ট করে (যেমন 224×224), তাই resizing প্রায় সবসময়ই দরকার হয়। আর cropping দিয়ে আমরা ইমেজের নির্দিষ্ট অংশ কেটে নিতে পারি — যেমন শুধু মুখের অংশ, বা নির্দিষ্ট কোনো object।

### ইমেজ রিসাইজ — cv2.resize()

OpenCV তে ইমেজ resize করার জন্য `cv2.resize()` ফাংশন ব্যবহার করা হয়। এই ফাংশনে দুটি উপায়ে নতুন সাইজ নির্দিষ্ট করা যায় — একটি হলো সরাসরি specific dimensions দেওয়া, আর অন্যটি হলো scale factor ব্যবহার করা। প্রতিটি পদ্ধতির নিজস্ব সুবিধা আছে এবং বিভিন্ন পরিস্থিতিতে বিভিন্নটা ব্যবহার করা হয়।

**⚠️ একটি অত্যন্ত গুরুত্বপূর্ণ বিষয়:** `img.shape` যখন ইমেজের dimensions রিটার্ন করে, তখন সেটি হয় `(height, width, channels)` ফরম্যাটে। কিন্তু `cv2.resize()` ফাংশনে নতুন সাইজ দিতে হয় `(width, height)` ফরম্যাটে! এটি একটি অত্যন্ত সাধারণ ভুলের জায়গা — অনেকেই shape থেকে সরাসরি মান নিয়ে resize করতে গিয়ে width এবং height উল্টে দেয়। সবসময় মনে রাখবে: shape = (H, W, C), resize = (W, H)।

**পদ্ধতি ১ — Specific Dimensions:** তুমি সরাসরি নতুন width এবং height দিতে পারো। এটি তখন কাজে লাগে যখন তুমি জানো ঠিক কোন সাইজ দরকার, যেমন মডেল ইনপুট 224×224 বা থাম্বনেইল 150×150। এক্ষেত্রে aspect ratio preserved নাও হতে পারে, অর্থাৎ ইমেজ stretch বা compress হতে পারে।

```python
import cv2

img = cv2.imread("resources/lena.png")
print(f"Original shape: {img.shape}")  # (height, width, channels)

# নির্দিষ্ট dimension এ resize করা — (width, height) ফরম্যাট!
img_resized = cv2.resize(img, (300, 200))  # width=300, height=200
print(f"Resized shape: {img_resized.shape}")

# সঠিক উপায় — shape থেকে মান নিলে width আর height উল্টে দিতে হবে
h, w = img.shape[:2]
img_resized2 = cv2.resize(img, (w // 2, h // 2))  # অর্ধেক সাইজ
print(f"Half size shape: {img_resized2.shape}")

cv2.imshow("Original", img)
cv2.imshow("Resized 300x200", img_resized)
cv2.imshow("Half Size", img_resized2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

**পদ্ধতি ২ — Scale Factor:** Scale factor ব্যবহার করলে aspect ratio স্বয়ংক্রিয়ভাবে preserved হয়। এটি তখন ব্যবহার করো যখন তুমি চাও ইমেজের অনুপাত ঠিক রেখে বড় বা ছোট করতে। `fx` দিয়ে horizontal scale এবং `fy` দিয়ে vertical scale নির্দিষ্ট করা হয়। `fx=2.0, fy=2.0` মানে ইমেজটি দ্বিগুণ বড় হবে, `fx=0.5, fy=0.5` মানে অর্ধেক হবে।

```python
import cv2

img = cv2.imread("resources/lena.png")

# Scale factor দিয়ে resize — aspect ratio ঠিক থাকে
img_double = cv2.resize(img, None, fx=2.0, fy=2.0)   # ২গুণ বড়
img_half = cv2.resize(img, None, fx=0.5, fy=0.5)     # অর্ধেক
img_wide = cv2.resize(img, None, fx=1.5, fy=1.0)     # শুধু width বড়

cv2.imshow("Original", img)
cv2.imshow("Double Size", img_double)
cv2.imshow("Half Size", img_half)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

দ্বিতীয় প্যারামিটার `None` দেওয়ার কারণ হলো — আমরা specific dimensions দিচ্ছি না, বরং scale factor ব্যবহার করছি। OpenCV স্বয়ংক্রিয়ভাবে scale factor থেকে নতুন dimensions ক্যালকুলেট করে নেয়।

### ইমেজ ক্রপ করা — NumPy Slicing

OpenCV তে ইমেজ crop করার জন্য কোনো আলাদা ফাংশন নেই — এটি সরাসরি NumPy array slicing দিয়ে করা হয়। ইমেজ হলো একটি NumPy array, তাই আমরা Python এর slicing syntax ব্যবহার করে যেকোনো অংশ কেটে নিতে পারি। সিনট্যাক্স হলো: `img[y_start:y_end, x_start:x_end]`। খেয়াল করো, প্রথমে y (row/height) আসে, তারপর x (column/width) — এটি স্বাভাবিক NumPy convention।

ক্রপ করার আগে ইমেজের dimensions জেনে নেওয়া ভালো, কারণ তাহলে তুমি সঠিক slicing range দিতে পারবে। একটি সহজ উপায় হলো প্রথমে ইমেজ ডিসপ্লে করা এবং mouse position দেখে কোঅর্ডিনেট নির্ণয় করা, তারপর সেই অনুযায়ী crop করা।

```python
import cv2

img = cv2.imread("resources/lena.png")
h, w = img.shape[:2]
print(f"Image dimensions: {w}x{h}")

# নির্দিষ্ট অংশ ক্রপ করা
# img[y_start:y_end, x_start:x_end]
cropped = img[100:400, 200:500]

cv2.imshow("Original", img)
cv2.imshow("Cropped Region", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

তুমি চাইলে ইমেজের center portion ক্রপ করতে পারো — এটি অনেক প্র্যাকটিক্যাল যখন তুমি image এর মাঝের অংশে focus করতে চাও:

```python
import cv2

img = cv2.imread("resources/lena.png")
h, w = img.shape[:2]

# Center crop — মাঝের 50% অংশ নেওয়া
center_x, center_y = w // 2, h // 2
crop_w, crop_h = w // 4, h // 4

cropped_center = img[center_y - crop_h:center_y + crop_h,
                     center_x - crop_w:center_x + crop_w]

cv2.imshow("Original", img)
cv2.imshow("Center Crop", cropped_center)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### রিসাইজ এবং ক্রপ একসাথে — প্র্যাকটিক্যাল উদাহরণ

রিয়েল প্রজেক্টে অনেক সময় resize এবং crop দুটোই লাগে। যেমন, একটি বড় ইমেজ থেকে নির্দিষ্ট অংশ crop করে সেটাকে model input এর জন্য resize করা। নিচে একটি প্র্যাকটিক্যাল উদাহরণ দেওয়া হলো:

```python
import cv2

img = cv2.imread("resources/large_photo.jpg")

# ধরো আমরা ইমেজের উপরের-বাম কোনা থেকে 500x500 অংশ ক্রপ করতে চাই
cropped = img[0:500, 0:500]

# তারপর সেটাকে 224x224 তে resize করতে চাই (model input)
resized = cv2.resize(cropped, (224, 224))

cv2.imshow("Original", img)
cv2.imshow("Cropped 500x500", cropped)
cv2.imshow("Resized 224x224", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

মনে রাখবে, cropping করার সময় slicing range ইমেজের actual dimensions এর বাইরে গেলে error হবে না, কিন্তু তুমি যতটুকু চাইবে ততটুকু পাবে না। তাই সবসময় shape চেক করে নিশ্চিত হও যে তোমার slicing range valid কিনা।
