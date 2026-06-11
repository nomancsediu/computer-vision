## ইমেজ রিসাইজ এবং ক্রপ

ইমেজ প্রসেসিংয়ে সবচেয়ে বেশি ব্যবহৃত দুটি অপারেশন হলো resizing এবং cropping। বাস্তব প্রজেক্টে আমরা বিভিন্ন সোর্স থেকে বিভিন্ন সাইজের ইমেজ পাই, কোনোটা বড়, কোনোটা ছোট, আবার কোনোটা ভিন্ন aspect ratio এর। কিন্তু মেশিন লার্নিং মডেল সাধারণত নির্দিষ্ট সাইজ ইনপুট চায় (যেমন 224×224), তাই resizing প্রায় সব ক্ষেত্রেই দরকার হয়। অন্যদিকে cropping ব্যবহার করে ইমেজের নির্দিষ্ট অংশ আলাদা করা হয়, যেমন শুধু মুখ, নির্দিষ্ট object বা region of interest।

---

### ইমেজ রিসাইজ  cv2.resize()

OpenCV তে `cv2.resize()` ফাংশন দিয়ে ইমেজ resize করা হয়। এখানে দুইভাবে কাজ করা যায়, নতুন dimensions দিয়ে অথবা scale factor দিয়ে।

একটি খুব গুরুত্বপূর্ণ বিষয় হলো, `img.shape` রিটার্ন করে `(height, width, channels)` কিন্তু `cv2.resize()` এ দিতে হয় `(width, height)`। এটি অনেক সময় ভুলের কারণ হয়।

#### Specific Dimensions ব্যবহার করে resize

এক্ষেত্রে তুমি সরাসরি নতুন width এবং height নির্দিষ্ট করে দাও। এটি তখন ব্যবহার করা হয় যখন তোমার ইনপুট সাইজ আগে থেকেই fixed (যেমন neural network input size)।

```python id="resize_dim"
import cv2

img = cv2.imread("resources/sample_image.png")

print(f"Original shape: {img.shape}")

img_resized = cv2.resize(img, (300, 200))
print(f"Resized shape: {img_resized.shape}")

h, w = img.shape[:2]
img_half = cv2.resize(img, (w // 2, h // 2))
print(f"Half size shape: {img_half.shape}")

cv2.imshow("Original", img)
cv2.imshow("Resized 300x200", img_resized)
cv2.imshow("Half Size", img_half)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

#### Scale Factor ব্যবহার করে resize

Scale factor ব্যবহার করলে aspect ratio maintain করা সহজ হয়। এখানে fx এবং fy ব্যবহার করা হয়।

```python id="resize_scale"
import cv2

img = cv2.imread("resources/sample_image.png")

img_double = cv2.resize(img, None, fx=2.0, fy=2.0)
img_half = cv2.resize(img, None, fx=0.5, fy=0.5)
img_wide = cv2.resize(img, None, fx=1.5, fy=1.0)

cv2.imshow("Original", img)
cv2.imshow("Double Size", img_double)
cv2.imshow("Half Size", img_half)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### ইমেজ ক্রপ করা  NumPy Slicing

OpenCV তে cropping করার জন্য আলাদা কোনো ফাংশন নেই। কারণ ইমেজ আসলে একটি NumPy array, তাই slicing দিয়েই crop করা হয়।

Syntax:
`img[y_start:y_end, x_start:x_end]`

খেয়াল রাখবে, প্রথমে height (y-axis), পরে width (x-axis)।

```python id="crop_basic"
import cv2

img = cv2.imread("resources/sample_image.png")
h, w = img.shape[:2]

print(f"Image size: {w}x{h}")

cropped = img[100:400, 200:500]

cv2.imshow("Original", img)
cv2.imshow("Cropped", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Center Crop (প্র্যাকটিক্যাল)

অনেক ক্ষেত্রে আমরা ইমেজের মাঝের অংশ focus করতে চাই। নিচে center crop এর উদাহরণ:

```python id="crop_center"
import cv2

img = cv2.imread("resources/sample_image.png")
h, w = img.shape[:2]

center_x = w // 2
center_y = h // 2

crop_w = w // 4
crop_h = h // 4

cropped_center = img[
    center_y - crop_h:center_y + crop_h,
    center_x - crop_w:center_x + crop_w
]

cv2.imshow("Original", img)
cv2.imshow("Center Crop", cropped_center)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Resize + Crop একসাথে (Real-world Example)

বাস্তব প্রজেক্টে প্রায়ই cropping করার পর resize করা হয়, বিশেষ করে model input তৈরি করার সময়।

```python id="crop_resize"
import cv2

img = cv2.imread("resources/sample_image.png")

cropped = img[0:500, 0:500]
resized = cv2.resize(cropped, (224, 224))

cv2.imshow("Original", img)
cv2.imshow("Cropped", cropped)
cv2.imshow("Resized", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

