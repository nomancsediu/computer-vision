## শেইপ এবং টেক্সট ড্রয়িং

কম্পিউটার ভিশন প্রজেক্টে অনেক সময় ইমেজের উপর বিভিন্ন shape এবং text আঁকতে হয়। যেমন — object detection এর ফলাফল দেখানোর জন্য bounding box আঁকা, কোনো নির্দিষ্ট পয়েন্ট mark করার জন্য circle আঁকা, বা লেবেল লেখার জন্য text যোগ করা। OpenCV তে এই কাজগুলো করার জন্য বিভিন্ন drawing ফাংশন আছে যা অত্যন্ত সহজে ব্যবহার করা যায়। এই সেকশনে আমরা লাইন, আয়তক্ষেত্র, বৃত্ত এবং টেক্সট আঁকা শিখবো।

### ফাঁকা ইমেজ তৈরি — np.zeros()

Shape বা text আঁকার আগে আমাদের একটি canvas দরকার। NumPy এর `np.zeros()` ফাংশন দিয়ে একটি সম্পূর্ণ কালো ফাঁকা ইমেজ তৈরি করা যায়। এই ফাংশনটি একটি tuple নেয় যেখানে (height, width, channels) দিতে হয়, আর data type সাধারণত `np.uint8` হয়। সব পিক্সেলের মান ০ দিয়ে initialize হয় বলে ইমেজটি কালো দেখায়। এটি drawing practice করার জন্য দারুণ একটি উপায়, কারণ তুমি বারবার নতুন করে তৈরি করতে পারবে।

```python
import numpy as np
import cv2

# 500x500 কালো ইমেজ তৈরি (3 channels — BGR)
img = np.zeros((500, 500, 3), np.uint8)

cv2.imshow("Black Canvas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

তুমি চাইলে `np.ones()` ব্যবহার করে সাদা ইমেজও তৈরি করতে পারো, তবে সেটিকে 255 দিয়ে multiply করতে হবে কারণ সাদা মানে সব চ্যানেলে মান 255:

```python
# সাদা ইমেজ তৈরি
white_img = np.ones((500, 500, 3), np.uint8) * 255

cv2.imshow("White Canvas", white_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### লাইন আঁকা — cv2.line()

লাইন আঁকার জন্য `cv2.line()` ফাংশন ব্যবহার করা হয়। এই ফাংশনে ইমেজ, starting point (x1, y1), ending point (x2, y2), color (BGR format), এবং thickness দিতে হয়। Point গুলো tuple আকারে দিতে হয়, যেমন `(100, 200)` মানে x=100, y=200। মনে রাখবে, OpenCV তে coordinate system এর origin (0,0) হলো উপরের-বাম কোণা, x ডানদিকে বাড়ে, আর y নিচে বাড়ে।

```python
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

# একটি নীল লাইন — BGR format এ রং দিতে হয়!
cv2.line(img, (50, 50), (450, 50), (255, 0, 0), 3)      # নীল (Blue=255)
cv2.line(img, (50, 100), (450, 100), (0, 255, 0), 3)     # সবুজ (Green=255)
cv2.line(img, (50, 150), (450, 150), (0, 0, 255), 3)     # লাল (Red=255)
cv2.line(img, (50, 200), (450, 200), (255, 255, 0), 5)   # হলুদ (Blue+Green)

cv2.imshow("Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### আয়তক্ষেত্র আঁকা — cv2.rectangle()

আয়তক্ষেত্র আঁকার জন্য `cv2.rectangle()` ব্যবহার করা হয়। এটি object detection এ bounding box আঁকার জন্য সবচেয়ে বেশি ব্যবহৃত ফাংশন। এখানে top-left corner এবং bottom-right corner এর coordinate দিতে হয়। Thickness এর জায়গায় `cv2.FILLED` বা `-1` দিলে আয়তক্ষেত্রটি filled হবে, অর্থাৎ ভেতরটা রং দিয়ে ভরে যাবে। এটি overlay বা mask তৈরি করার সময় খুব কাজে লাগে।

```python
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

# খোলা আয়তক্ষেত্র (outlined)
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)

# ভরাট আয়তক্ষেত্র (filled)
cv2.rectangle(img, (250, 50), (450, 200), (0, 0, 255), cv2.FILLED)

# পাতলা বর্ডার
cv2.rectangle(img, (50, 250), (200, 450), (255, 0, 0), 1)

cv2.imshow("Rectangles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### বৃত্ত আঁকা — cv2.circle()

বৃত্ত আঁকার জন্য `cv2.circle()` ফাংশন ব্যবহার করা হয়। এখানে center point, radius, color, এবং thickness দিতে হয়। rectangle এর মতো এখানেও `cv2.FILLED` ব্যবহার করে ভেতরটা ভরাট করা যায়। Circle আঁকা বিশেষ করে feature point visualization, eye detection result দেখানো, বা ROI mark করার জন্য কাজে লাগে।

```python
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

# খোলা বৃত্ত
cv2.circle(img, (150, 150), 80, (0, 255, 0), 3)

# ভরাট বৃত্ত
cv2.circle(img, (350, 150), 50, (0, 0, 255), cv2.FILLED)

# পাতলা বৃত্ত
cv2.circle(img, (250, 350), 100, (255, 0, 0), 1)

cv2.imshow("Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### টেক্সট লেখা — cv2.putText()

ইমেজের উপর text লেখার জন্য `cv2.putText()` ব্যবহার করা হয়। এটি object detection এ class name এবং confidence score দেখানোর জন্য অপরিহার্য। ফাংশনের প্যারামিটার গুলো হলো — ইমেজ, text string, position (bottom-left corner of text), font type, font scale, color, এবং thickness। OpenCV তে বেশ কয়েকটি built-in font আছে যেমন `cv2.FONT_HERSHEY_SIMPLEX`, `cv2.FONT_HERSHEY_PLAIN`, `cv2.FONT_HERSHEY_COMPLEX` ইত্যাদি। তবে OpenCV তে Bengali বা অন্য Unicode text সরাসরি লেখা যায় না — সেটি করতে হলে PIL/Pillow ব্যবহার করতে হবে।

```python
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

# বিভিন্ন font এ টেক্সট
cv2.putText(img, "Hello OpenCV!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(img, "Computer Vision", (50, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
cv2.putText(img, "Bangla Text Nao!", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# ছোট এবং বড় size
cv2.putText(img, "Small", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.putText(img, "BIG", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 3)

cv2.imshow("Text on Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### BGR কালার ফরম্যাট মনে রাখা

OpenCV তে রং নির্দিষ্ট করার সময় সবসময় BGR (Blue, Green, Red) ফরম্যাট ব্যবহার করতে হয়। এটি RGB এর উল্টো — প্রথমে Blue এর মান, তারপর Green, তারপর Red। কিছু সাধারণ রং এর BGR value মনে রাখা সুবিধাজনক:

| রং | BGR Value |
|------|-----------|
| সাদা | (255, 255, 255) |
| কালো | (0, 0, 0) |
| লাল | (0, 0, 255) |
| সবুজ | (0, 255, 0) |
| নীল | (255, 0, 0) |
| হলুদ | (0, 255, 255) |
| কমলা | (0, 165, 255) |
| বেগুনি | (255, 0, 255) |

### সব একসাথে — একটি কমপ্লিট উদাহরণ

এখন আমরা যা যা শিখলাম তা একসাথে ব্যবহার করে একটি কমপ্লিট উদাহরণ তৈরি করবো। এই উদাহরণে আমরা একটি ফাঁকা canvas এ বিভিন্ন shape এবং text আঁকবো:

```python
import numpy as np
import cv2

# কালো canvas তৈরি
img = np.zeros((600, 800, 3), np.uint8)

# আয়তক্ষেত্র — বাড়ির দেয়াল
cv2.rectangle(img, (200, 200), (600, 500), (0, 200, 0), cv2.FILLED)

# ত্রিভুজ আকৃতির ছাদ — লাইন দিয়ে
cv2.line(img, (200, 200), (400, 50), (0, 0, 200), 5)
cv2.line(img, (400, 50), (600, 200), (0, 0, 200), 5)

# দরজা
cv2.rectangle(img, (350, 350), (450, 500), (200, 200, 200), cv2.FILLED)

# জানালা
cv2.rectangle(img, (240, 260), (320, 340), (255, 255, 0), cv2.FILLED)
cv2.rectangle(img, (480, 260), (560, 340), (255, 255, 0), cv2.FILLED)

# সূর্য
cv2.circle(img, (680, 80), 50, (0, 200, 255), cv2.FILLED)

# টেক্সট
cv2.putText(img, "My OpenCV House", (220, 560), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Complete Drawing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Drawing ফাংশনগুলো in-place কাজ করে — অর্থাৎ এগুলো নতুন ইমেজ রিটার্ন করে না, বরং মূল ইমেজটিকেই modify করে। তাই যদি মূল ইমেজটি সংরক্ষণ করতে চাও, তাহলে আগে `img.copy()` করে নিয়ে কাজ করবে। এটি একটি সাধারণ ভুল — অনেকেই original ইমেজে সরাসরি drawing করে ফেলে, আর পরে মূল ইমেজটি আর পায় না।
