## ইমেজ জয়েনিং (Stacking)

কম্পিউটার ভিশন প্রজেক্টে অনেক সময় একাধিক ইমেজ একসাথে দেখানোর প্রয়োজন হয়। যেমন — original ইমেজ এবং processed ইমেজ side by side compare করা, বা বিভিন্ন threshold value তে edge detection এর ফলাফল একসাথে দেখা। আলাদা আলাদা উইন্ডোতে দেখানোর চেয়ে একটি উইন্ডোতে সব ইমেজ জয়েন করে দেখানো অনেক বেশি সুবিধাজনক। এই সেকশনে আমরা শিখবো কিভাবে NumPy এর stacking functions ব্যবহার করে ইমেজ জয়েন করা যায়।

### Horizontal Stacking — np.hstack()

`np.hstack()` (horizontal stack) ব্যবহার করে দুটি বা তার বেশি ইমেজ পাশাপাশি (side by side) জয়েন করা যায়। এটি NumPy এর একটি ফাংশন যা arrays কে horizontally concatenate করে। এটি ব্যবহার করার জন্য সবচেয়ে গুরুত্বপূর্ণ শর্ত হলো — জয়েন করা ইমেজগুলোর height এবং number of channels একই হতে হবে। Width ভিন্ন হতে পারে, কিন্তু height এবং channels অবশ্যই same হতে হবে। যদি height ভিন্ন হয়, তাহলে NumPy error দেবে।

```python
import cv2
import numpy as np

img = cv2.imread("resources/lena.png")

# বিভিন্ন প্রসেসিং করা ইমেজ
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img, (15, 15), 0)
img_canny = cv2.Canny(img, 100, 200)

# Horizontal stack — পাশাপাশি জয়েন
# সব ইমেজের height একই হতে হবে!
hor_stack = np.hstack((img, img_blur))

cv2.imshow("Horizontal Stack", hor_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Vertical Stacking — np.vstack()

`np.vstack()` (vertical stack) ব্যবহার করে ইমেজগুলো উপর-নিচ (top-bottom) করে জয়েন করা যায়। Horizontal stack এর উল্টো — এখানে width এবং channels একই হতে হবে, height ভিন্ন হতে পারে। এটি তখন কাজে লাগে যখন তুমি একই width এর কয়েকটি ইমেজ উপর-নিচে দেখাতে চাও।

```python
import cv2
import numpy as np

img = cv2.imread("resources/lena.png")

img_blur = cv2.GaussianBlur(img, (15, 15), 0)
img_canny = cv2.Canny(img, 100, 200)

# Canny ইমেজটি grayscale (1 channel), তাই 3 channel করতে হবে
img_canny_3ch = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

# Vertical stack — উপর-নিচ জয়েন
# সব ইমেজের width একই হতে হবে!
ver_stack = np.vstack((img, img_blur, img_canny_3ch))

cv2.imshow("Vertical Stack", ver_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### সমান Dimension এর প্রয়োজনীয়তা

Stacking এর সবচেয়ে বড় সীমাবদ্ধতা হলো dimension এর সমতা। Horizontal stack এর জন্য height same হতে হবে, vertical stack এর জন্য width same হতে হবে। এছাড়া সব ইমেজের channel number একই হতে হবে — অর্থাৎ সব ইমেজ যেন BGR (3 channel) হয় বা সব যেন grayscale (1 channel) হয়। Grayscale এবং color ইমেজ একসাথে stack করলে error হবে। এই সমস্যা সমাধানের জন্য আমরা grayscale ইমেজকে BGR তে convert করে নিতে পারি — যদিও ইমেজটি সাদা-কালোই থাকবে, কিন্তু technically সেটি 3 channel হয়ে যাবে:

```python
import cv2
import numpy as np

img = cv2.imread("resources/lena.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# এটা error দেবে — channel সংখ্যা ভিন্ন!
# hor = np.hstack((img, img_gray))  # ❌ ValueError!

# সমাধান — grayscale কে BGR তে convert করা
img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# এখন stack করা যাবে
hor = np.hstack((img, img_gray_bgr))  # ✅ কাজ করবে

cv2.imshow("Color + Grayscale Stack", hor)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### বিভিন্ন সাইজের ইমেজ Stack করা

বাস্তবে অনেক সময় আমাদের বিভিন্ন সাইজের ইমেজ stack করতে হয়। সরাসরি stack করলে dimension mismatch error হবে। সমাধান হলো — প্রথমে সব ইমেজকে একই সাইজে resize করে নেওয়া, তারপর stack করা। আরেকটি উপায় হলো ছোট ইমেজের চারপাশে padding যোগ করা বড় ইমেজের dimension এর সাথে মেলানোর জন্য। নিচে resize করে stack করার পদ্ধতি দেখানো হলো:

```python
import cv2
import numpy as np

img1 = cv2.imread("resources/lena.png")       # ধরো 512x512
img2 = cv2.imread("resources/photo.jpg")      # ধরে নাও এটা আলাদা সাইজের

# দুটোকে একই সাইজে resize করা
target_h, target_w = 300, 300
img1_resized = cv2.resize(img1, (target_w, target_h))
img2_resized = cv2.resize(img2, (target_w, target_h))

# এখন সহজে stack করা যায়
hor = np.hstack((img1_resized, img2_resized))
ver = np.vstack((img1_resized, img2_resized))

cv2.imshow("Horizontal", hor)
cv2.imshow("Vertical", ver)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### একটি কমপ্লিট Image Grid তৈরি

বাস্তব প্রজেক্টে আমরা প্রায়ই 2×2 বা 3×3 grid তে ইমেজ দেখাতে চাই। এটি করার জন্য horizontal এবং vertical stacking দুটোই ব্যবহার করতে হয়। প্রথমে প্রতিটি row horizontal ভাবে stack করো, তারপর row গুলো vertical ভাবে stack করো:

```python
import cv2
import numpy as np

img = cv2.imread("resources/lena.png")

# বিভিন্ন processing
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_blur = cv2.GaussianBlur(img, (15, 15), 0)
img_canny = cv2.Canny(img, 100, 200)
img_canny_bgr = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

# প্রতিটি row horizontal stack
row1 = np.hstack((img, img_blur))
row2 = np.hstack((img_gray_bgr, img_canny_bgr))

# Row গুলো vertical stack
grid = np.vstack((row1, row2))

cv2.imshow("2x2 Image Grid", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### লেবেল সহ Stacking

শুধু ইমেজ stack করলে কোনটা original আর কোনটা processed তা বোঝা কঠিন হয়। তাই প্রতিটি ইমেজের উপর label যোগ করে নেওয়া ভালো। আমরা `cv2.putText()` ব্যবহার করে প্রতিটি ইমেজে label লিখে তারপর stack করবো:

```python
import cv2
import numpy as np

img = cv2.imread("resources/lena.png")

img_blur = cv2.GaussianBlur(img, (15, 15), 0)
img_canny = cv2.Canny(img, 100, 200)
img_canny_bgr = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

# প্রতিটি ইমেজে label যোগ করা
img_labeled = img.copy()
cv2.putText(img_labeled, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

img_blur_labeled = img_blur.copy()
cv2.putText(img_blur_labeled, "Blur", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

img_canny_labeled = img_canny_bgr.copy()
cv2.putText(img_canny_labeled, "Canny Edge", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Stack করা
stacked = np.hstack((img_labeled, img_blur_labeled, img_canny_labeled))

cv2.imshow("Labeled Stack", stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Stacking হলো একটি simple কিন্তু অত্যন্ত useful technique। যেকোনো image processing pipeline এ মাঝে মাঝে intermediate results দেখতে হয় — তখন stacking দিয়ে সব একসাথে visualize করা অনেক সুবিধার। মনে রাখবে, dimension match করা সবচেয়ে গুরুত্বপূর্ণ — আর grayscale এবং color ইমেজ একসাথে stack করতে গেলে অবশ্যই `cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)` করে channel সমান করে নিতে হবে।
