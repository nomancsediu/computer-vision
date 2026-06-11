## পারস্পেক্টিভ ট্রান্সফরমেশন

পারস্পেক্টিভ ট্রান্সফরমেশন হলো এমন একটি technique যার মাধ্যমে একটি ইমেজের দৃষ্টিভঙ্গি (perspective) পরিবর্তন করা যায়। সহজভাবে বললে, এটি tilted বা angled ইমেজকে সোজা করার একটি উপায়। রিয়েল লাইফে এটি document scanning, ID card scanning, license plate reading ইত্যাদিতে খুব বেশি ব্যবহার হয়।

---

### Bird’s-Eye View কী এবং কেন দরকার?

Bird’s-eye view মানে হলো একেবারে ওপর থেকে নিচের দিকে দেখা দৃশ্য। বাস্তবে আমরা অনেক সময় কোনো object (যেমন বই, কার্ড, ডকুমেন্ট) তির্যক কোণ থেকে ছবি তুলি। তখন সেটি trapezoid আকারে দেখা যায়কাছের অংশ বড়, দূরের অংশ ছোট।

কিন্তু অনেক ক্ষেত্রে আমরা চাই সেটিকে rectangle আকারে, সোজাভাবে দেখতে। বিশেষ করে OCR (Optical Character Recognition) এর ক্ষেত্রে tilted image এ accuracy কমে যায়।

পারস্পেক্টিভ ট্রান্সফরমেশন এই সমস্যার সমাধান করে tilted view কে bird’s-eye view এ রূপান্তর করে।

---

### cv2.getPerspectiveTransform() এবং cv2.warpPerspective()

পারস্পেক্টিভ ট্রান্সফরমেশন দুই ধাপে করা হয়:

ধাপ ১: চারটি source point এবং চারটি destination point নির্ধারণ করে transformation matrix তৈরি করা হয়। এই কাজটি করে `cv2.getPerspectiveTransform()`।

ধাপ ২: সেই matrix ব্যবহার করে ইমেজ warp করা হয় `cv2.warpPerspective()` দিয়ে।

Point গুলো অবশ্যই `np.float32()` ফরম্যাটে দিতে হয় এবং সাধারণত order থাকে: top-left → top-right → bottom-right → bottom-left

```python id="perspective_transform"
import cv2
import numpy as np

img = cv2.imread("resources/card.jpg")

pts1 = np.float32([
    [111, 219],
    [287, 188],
    [154, 482],
    [352, 440]
])

width, height = 250, 350

pts2 = np.float32([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Original", img)
cv2.imshow("Warped", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Point নির্বাচন করার প্র্যাকটিক্যাল পদ্ধতি

সবচেয়ে কঠিন অংশ হলো source points বের করা। সাধারণত ইমেজ দেখে manually coordinate নেওয়া হয়। তবে mouse click ব্যবহার করে সহজে points collect করা যায়।

```python id="mouse_points"
import cv2

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 0, 255), cv2.FILLED)
        cv2.imshow("Select Points", param)
        print(x, y)

img = cv2.imread("resources/card.jpg")
cv2.imshow("Select Points", img)

cv2.setMouseCallback("Select Points", mouse_callback, img)

print("Click 4 corners: top-left, top-right, bottom-right, bottom-left")
cv2.waitKey(0)
cv2.destroyAllWindows()

print(points)
```

---

### Document Scanning Example (Complete Pipeline)

এই উদাহরণে আমরা tilted document কে straight bird’s-eye view এ convert করছি।

```python id="doc_scan"
import cv2
import numpy as np

img = cv2.imread("resources/card.jpg")

pts1 = np.float32([
    [111, 219],
    [287, 188],
    [154, 482],
    [352, 440]
])

width = int(max(
    np.linalg.norm(pts1[0] - pts1[1]),
    np.linalg.norm(pts1[2] - pts1[3])
))

height = int(max(
    np.linalg.norm(pts1[0] - pts1[2]),
    np.linalg.norm(pts1[1] - pts1[3])
))

pts2 = np.float32([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Original", img)
cv2.imshow("Bird Eye View", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### গুরুত্বপূর্ণ টিপস

* Source points ভুল হলে output distorted হবে
* Point order অবশ্যই consistent রাখতে হবে
* `np.float32()` ব্যবহার করা বাধ্যতামূলক
* Output size নিজে ঠিক করে নিতে হয়
* Aspect ratio ঠিক রাখা ভালো results দেয়
* Auto detection পরে শেখা যায় (contours + edge detection ব্যবহার করে)
