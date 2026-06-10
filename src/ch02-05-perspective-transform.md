## পারস্পেক্টিভ ট্রান্সফরমেশন

পারস্পেক্টিভ ট্রান্সফরমেশন হলো এমন একটি technique যার মাধ্যমে আমরা একটি ইমেজের দৃষ্টিভঙ্গি (perspective) পরিবর্তন করতে পারি। সহজ ভাষায় বললে — এটি একটি angled বা tilted ইমেজকে সোজা করার উপায়। এটি অনেকটা আয়নায় দাঁড়িয়ে ক্যামেরা দিয়ে ছবি তোলার মতো — ছবিটি একটু তির্যক হয়ে যায়, আর পারস্পেক্টিভ ট্রান্সফরমেশন দিয়ে সেটাকে সোজা করা যায়। রিয়েল-ওয়ার্ল্ডে এটি document scanning, নম্বর প্লেট পড়া, ID card scanning ইত্যাদি কাজে ব্যাপকভাবে ব্যবহৃত হয়।

### Bird's-Eye View কী এবং কেন দরকার?

Bird's-eye view হলো ওপর থেকে নিচের দিকে সোজা তাকানোর দৃষ্টিভঙ্গি — ঠিক যেমন পাখি ওপর থেকে নিচের দিকে তাকায়। অনেক সময় আমরা একটি তির্যক কোণ থেকে ছবি তুলি — যেমন টেবিলে রাখা একটি বই বা কার্ড। সেই ছবিতে object টি trapezoid আকারে দেখায় — কাছের দিকটা বড়, দূরের দিকটা ছোট। কিন্তু আমরা চাই সেটি rectangle আকারে দেখতে — ঠিক ওপর থেকে তোলা ছবির মতো। পারস্পেক্টিভ ট্রান্সফরমেশন ঠিক এই কাজটিই করে।

এটি বিশেষ করে কাজে লাগে যখন আমরা কোনো document বা card এর ছবি থেকে text extract করতে চাই। Tilted ইমেজ থেকে OCR (Optical Character Recognition) ভালো কাজ করে না, কিন্তু bird's-eye view তে রূপান্তর করলে OCR এর accuracy অনেক বেড়ে যায়।

### cv2.getPerspectiveTransform() এবং cv2.warpPerspective()

পারস্পেক্টিভ ট্রান্সফরমেশন করতে দুটি ফাংশন লাগে। প্রথমে `cv2.getPerspectiveTransform()` দিয়ে transformation matrix তৈরি করতে হয়, তারপর `cv2.warpPerspective()` দিয়ে সেই matrix ব্যবহার করে ইমেজটি transform করতে হয়। পুরো প্রক্রিয়াটি দুটি ধাপে কাজ করে:

**ধাপ ১:** ইমেজের মধ্যে চারটি source point (যেগুলো বাস্তব ইমেজে আছে) এবং চারটি destination point (যেখানে আমরা সেগুলোকে ম্যাপ করতে চাই) নির্ধারণ করতে হয়। `cv2.getPerspectiveTransform(src_points, dst_points)` এই দুই সেট point থেকে একটি 3×3 transformation matrix ক্যালকুলেট করে।

**ধাপ ২:** সেই transformation matrix ব্যবহার করে `cv2.warpPerspective(img, matrix, (width, height))` দিয়ে ইমেজটি warp করা হয়। এখানে third parameter হলো output ইমেজের size — (width, height) ফরম্যাটে।

Point গুলো `np.float32()` array আকারে দিতে হয়, যেখানে প্রতিটি point একটি (x, y) coordinate। Point গুলোর ক্রম গুরুত্বপূর্ণ — সাধারণত top-left, top-right, bottom-right, bottom-left ক্রমে দেওয়া হয়।

```python
import cv2
import numpy as np

img = cv2.imread("resources/cards.jpg")

# ইমেজের কার্ডের চার কোণার পয়েন্ট (source points)
# এই পয়েন্ট গুলো তুমি নিজে ইমেজ দেখে নির্ধারণ করবে
pts1 = np.float32([
    [111, 219],     # top-left
    [287, 188],     # top-right
    [154, 482],     # bottom-left
    [352, 440]      # bottom-right
])

# আউটপুট ইমেজে এই পয়েন্ট গুলো কোথায় যাবে (destination points)
# এখানে আমরা একটি সোজা rectangle চাই
width, height = 250, 350
pts2 = np.float32([
    [0, 0],          # top-left
    [width, 0],      # top-right
    [0, height],     # bottom-left
    [width, height]  # bottom-right
])

# Transformation matrix তৈরি
matrix = cv2.getPerspectiveTransform(pts1, pts2)
print("Transformation Matrix:")
print(matrix)

# ইমেজ warp করা
img_output = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Original", img)
cv2.imshow("Warped (Bird's-Eye View)", img_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Point নির্ধারণ করা — ব্যবহারিক পদ্ধতি

Source point গুলো কীভাবে নির্ধারণ করবে সেটিই সবচেয়ে বড় চ্যালেঞ্জ। ম্যানুয়ালি করার একটি উপায় হলো — ইমেজটি প্রথমে ডিসপ্লে করো, mouse দিয়ে click করে coordinate গুলো নোট করো, তারপর সেগুলো ব্যবহার করো। নিচে একটি helper function দেওয়া হলো যা mouse click এর coordinate track করে:

```python
import cv2
import numpy as np

# Mouse click এর coordinate সংরক্ষণের জন্য
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Click করা জায়গায় একটি ছোট circle আঁকা
        cv2.circle(param, (x, y), 5, (0, 0, 255), cv2.FILLED)
        cv2.imshow("Select Points", param)
        print(f"Point {len(points)}: ({x}, {y})")

img = cv2.imread("resources/cards.jpg")
cv2.imshow("Select Points", img)

# Mouse callback সেট করা
cv2.setMouseCallback("Select Points", mouse_callback, img)

print("চারটি কোণায় ক্লিক করো: top-left, top-right, bottom-right, bottom-left")
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"সংগৃহীত পয়েন্ট: {points}")
```

### কার্ড স্ক্যানিং উদাহরণ — কমপ্লিট পাইপলাইন

এখন আমরা একটি বাস্তব উদাহরণ দেখবো — একটি তির্যক কোণ থেকে তোলা কার্ডের ছবিকে bird's-eye view তে রূপান্তর করা। এটি document scanner app গুলো যেভাবে কাজ করে তার মূল নীতি:

```python
import cv2
import numpy as np

img = cv2.imread("resources/cards.jpg")

# কার্ডের চার কোণা (তির্যক ইমেজে)
pts1 = np.float32([
    [111, 219],   # top-left corner of card
    [287, 188],   # top-right corner of card
    [154, 482],   # bottom-left corner of card
    [352, 440]    # bottom-right corner of card
])

# আউটপুট সাইজ নির্ধারণ
# কার্ডের width এবং height অনুযায়ী output সাইজ ক্যালকুলেট
width = int(max(
    np.sqrt((pts1[0][0]-pts1[1][0])**2 + (pts1[0][1]-pts1[1][1])**2),
    np.sqrt((pts1[2][0]-pts1[3][0])**2 + (pts1[2][1]-pts1[3][1])**2)
))
height = int(max(
    np.sqrt((pts1[0][0]-pts1[2][0])**2 + (pts1[0][1]-pts1[2][1])**2),
    np.sqrt((pts1[1][0]-pts1[3][0])**2 + (pts1[1][1]-pts1[3][1])**2)
))

print(f"Output size: {width}x{height}")

# Destination points — একটি সোজা rectangle
pts2 = np.float32([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
])

# Transform
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_warped = cv2.warpPerspective(img, matrix, (width, height))

# মূল ইমেজে কার্ডের কোণাগুলো circle দিয়ে mark করা
img_marked = img.copy()
for pt in pts1:
    pt_int = (int(pt[0]), int(pt[1]))
    cv2.circle(img_marked, pt_int, 8, (0, 0, 255), cv2.FILLED)

cv2.imshow("Original with Points", img_marked)
cv2.imshow("Bird's-Eye View", img_warped)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### গুরুত্বপূর্ণ টিপস

- Source point গুলো সঠিকভাবে নির্বাচন করা অত্যন্ত গুরুত্বপূর্ণ — একটু ভুল হলেও output ইমেজ বিকৃত হয়ে যাবে।
- Point গুলোর ক্রম সবসময় same রাখো — top-left, top-right, bottom-right, bottom-left বা clockwise order।
- `np.float32()` data type ব্যবহার করা বাধ্যতামূলক — integer দিলে `cv2.getPerspectiveTransform()` error দেবে।
- Output ইমেজের size তুমি নিজেই নির্ধারণ করো — সাধারণত source ইমেজের বা object এর actual aspect ratio maintain করাই ভালো।
- যদি তুমি স্বয়ংক্রিয়ভাবে point detect করতে চাও, তাহলে edge detection + contour finding ব্যবহার করতে পারো — সেটি আমরা পরের চ্যাপ্টারে শিখবো।
