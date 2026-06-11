## ইমেজ জয়েনিং (Stacking)

কম্পিউটার ভিশন প্রজেক্টে অনেক সময় একাধিক ইমেজ একসাথে দেখানোর প্রয়োজন হয়। যেমন original ইমেজ এবং processed ইমেজ side by side compare করা, বা বিভিন্ন threshold value তে edge detection এর ফলাফল একসাথে দেখা। আলাদা আলাদা উইন্ডোতে দেখানোর চেয়ে একটি উইন্ডোতে সব ইমেজ জয়েন করে দেখানো অনেক বেশি সুবিধাজনক।

এই সেকশনে আমরা শিখবো কিভাবে NumPy এর stacking functions ব্যবহার করে ইমেজ জয়েন করা যায়।

---

### Horizontal Stacking  np.hstack()

`np.hstack()` (horizontal stack) ব্যবহার করে দুটি বা তার বেশি ইমেজ পাশাপাশি (side by side) জয়েন করা যায়। এটি arrays কে horizontally concatenate করে।

সবচেয়ে গুরুত্বপূর্ণ শর্ত হলো:

* সব ইমেজের **height একই হতে হবে**
* সব ইমেজের **channels একই হতে হবে**

```python id="hstack_basic"
import cv2
import numpy as np

img = cv2.imread("resources/image.png")

img_blur = cv2.GaussianBlur(img, (15, 15), 0)

hor_stack = np.hstack((img, img_blur))

cv2.imshow("Horizontal Stack", hor_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Vertical Stacking  np.vstack()

`np.vstack()` ব্যবহার করে ইমেজগুলো উপর-নিচ (top-bottom) করে জয়েন করা যায়।

এখানে শর্ত:

* সব ইমেজের **width একই হতে হবে**
* সব ইমেজের **channels একই হতে হবে**

```python id="vstack_basic"
import cv2
import numpy as np

img = cv2.imread("resources/image.png")

img_blur = cv2.GaussianBlur(img, (15, 15), 0)
img_canny = cv2.Canny(img, 100, 200)

img_canny_bgr = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

ver_stack = np.vstack((img, img_blur, img_canny_bgr))

cv2.imshow("Vertical Stack", ver_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Channel Problem এবং Solution

Grayscale (1 channel) এবং BGR (3 channel) একসাথে stack করলে error হবে।

```python
# ❌ Wrong (will give error)
# np.hstack((img, img_gray))
```

সমাধান হলো grayscale কে BGR এ convert করা:

```python id="channel_fix"
import cv2
import numpy as np

img = cv2.imread("resources/image.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

hor = np.hstack((img, img_gray_bgr))

cv2.imshow("Fixed Stack", hor)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Different Size Images Stack করা

Stack করার আগে সব ইমেজকে same size এ resize করতে হয়।

```python id="resize_stack"
import cv2
import numpy as np

img1 = cv2.imread("resources/image1.jpg")
img2 = cv2.imread("resources/image2.jpg")

target_w, target_h = 300, 300

img1 = cv2.resize(img1, (target_w, target_h))
img2 = cv2.resize(img2, (target_w, target_h))

hor = np.hstack((img1, img2))
ver = np.vstack((img1, img2))

cv2.imshow("Horizontal", hor)
cv2.imshow("Vertical", ver)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Image Grid (2×2 Example)

Horizontal + Vertical stacking combine করে grid তৈরি করা যায়।

```python id="grid_2x2"
import cv2
import numpy as np

img = cv2.imread("resources/image.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

img_blur = cv2.GaussianBlur(img, (15, 15), 0)
img_canny = cv2.Canny(img, 100, 200)
img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

row1 = np.hstack((img, img_blur))
row2 = np.hstack((img_gray, img_canny))

grid = np.vstack((row1, row2))

cv2.imshow("2x2 Grid", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Labeled Stacking

ইমেজ বুঝতে label যোগ করা ভালো practice।

```python id="labeled_stack"
import cv2
import numpy as np

img = cv2.imread("resources/image.png")

img_blur = cv2.GaussianBlur(img, (15, 15), 0)
img_canny = cv2.Canny(img, 100, 200)
img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

img1 = img.copy()
cv2.putText(img1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

img2 = img_blur.copy()
cv2.putText(img2, "Blur", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

img3 = img_canny.copy()
cv2.putText(img3, "Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

stacked = np.hstack((img1, img2, img3))

cv2.imshow("Labeled Stack", stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

Stacking খুব simple কিন্তু খুব powerful technique। যেকোনো image processing pipeline এ intermediate result compare করার জন্য এটি খুব বেশি ব্যবহার হয়।
