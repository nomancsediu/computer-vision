## ইমেজ রিডিং এবং ডিসপ্লে করা

কম্পিউটার ভিশনের সবচেয়ে মৌলিক এবং প্রথম ধাপ হলো ইমেজ রিড করা এবং সেটিকে স্ক্রিনে প্রদর্শন করা। যেকোনো ইমেজ প্রসেসিং পাইপলাইন সাধারণত ইমেজ লোড করার মাধ্যমেই শুরু হয়। তাই এই অংশটি অত্যন্ত গুরুত্বপূর্ণ, কারণ এটি ভালোভাবে বোঝা গেলে পরবর্তী ধাপগুলো সহজ হয়ে যায়। এখানে আমরা শিখব কীভাবে স্ট্যাটিক ইমেজ রিড করতে হয়, ভিডিও ফাইল ও ওয়েবক্যাম থেকে ফ্রেম নিতে হয়, এবং BGR ও RGB ফরম্যাটের পার্থক্য।

### ইমেজ রিড করা  cv2.imread()

OpenCV-তে ইমেজ রিড করার জন্য `cv2.imread()` ব্যবহার করা হয়। এটি একটি ইমেজ ফাইলের পাথ ইনপুট হিসেবে নিয়ে একটি NumPy array রিটার্ন করে। ডিফল্টভাবে এটি ইমেজকে color mode-এ লোড করে, অর্থাৎ তিনটি চ্যানেল (Blue, Green, Red) সহ।

তুমি চাইলে দ্বিতীয় প্যারামিটার দিয়ে রিডিং মোড নির্ধারণ করতে পারো। যেমন `cv2.IMREAD_GRAYSCALE` ব্যবহার করলে ইমেজ গ্রেস্কেলে লোড হবে, এবং `cv2.IMREAD_UNCHANGED` ব্যবহার করলে alpha channel সহ লোড হবে।

```python id="q9w1ka"
import cv2

# Load color image (default)
img = cv2.imread("resources/sample_image.png")

# Load grayscale image
img_gray = cv2.imread("resources/sample_image.png", cv2.IMREAD_GRAYSCALE)

# Load image with alpha channel if available
img_unchanged = cv2.imread("resources/sample_image.png", cv2.IMREAD_UNCHANGED)

print(f"Color image shape: {img.shape}")
print(f"Grayscale image shape: {img_gray.shape}")
```

ইমেজ লোড করার পর অবশ্যই যাচাই করা উচিত এটি সফলভাবে লোড হয়েছে কিনা। ভুল পাথ দিলে `cv2.imread()` কোনো error না দিয়ে `None` রিটার্ন করে, যা পরবর্তীতে বড় সমস্যা তৈরি করতে পারে।

```python id="2xv9ld"
import cv2

img = cv2.imread("resources/sample_image.png")

if img is None:
    print("Error: Image could not be loaded. Check file path.")
else:
    print(f"Image loaded successfully. Shape: {img.shape}")
```

### ইমেজ ডিসপ্লে করা  cv2.imshow(), cv2.waitKey(), cv2.destroyAllWindows()

ইমেজ দেখানোর জন্য `cv2.imshow()` ব্যবহার করা হয়। এর প্রথম প্যারামিটার উইন্ডোর নাম এবং দ্বিতীয়টি ইমেজ array।

`cv2.waitKey()` না দিলে উইন্ডো সাথে সাথে বন্ধ হয়ে যায়। `cv2.waitKey(0)` মানে key press না করা পর্যন্ত অপেক্ষা করবে। কাজ শেষে `cv2.destroyAllWindows()` দিয়ে সব উইন্ডো বন্ধ করা হয়।

```python id="9k3d8p"
import cv2

img = cv2.imread("resources/sample_image.png")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

নির্দিষ্ট key press করলে উইন্ডো বন্ধ করার উদাহরণ:

```python id="v1m2op"
import cv2

img = cv2.imread("resources/sample_image.png")

cv2.imshow("Press q to close", img)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### BGR vs RGB

OpenCV ইমেজকে BGR ফরম্যাটে লোড করে, যেখানে বেশিরভাগ লাইব্রেরি RGB ব্যবহার করে। তাই সরাসরি matplotlib দিয়ে OpenCV image দেখালে রং ভুল দেখা যায়।

```python id="c8n2la"
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("resources/sample_image.png")

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("BGR (Incorrect Colors)")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(1, 2, 2)
plt.imshow(img_rgb)
plt.title("RGB (Correct Colors)")

plt.show()
```

### ভিডিও রিড করা  cv2.VideoCapture()

ভিডিও মূলত ফ্রেমের একটি ধারাবাহিকতা। `cv2.VideoCapture()` দিয়ে ভিডিও ফাইল পড়া হয়। `cap.read()` একটি boolean success flag এবং একটি frame return করে।

```python id="f3k9zz"
import cv2

cap = cv2.VideoCapture("resources/sample_video.mp4")

while True:
    success, frame = cap.read()

    if not success:
        print("Video ended or cannot be read.")
        break

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

`cap.release()` ব্যবহার করা জরুরি, কারণ এটি system resource মুক্ত করে।

### ওয়েবক্যাম থেকে ভিডিও ক্যাপচার

ওয়েবক্যাম ব্যবহার করতে ফাইল পাথের বদলে camera index দিতে হয়। সাধারণত built-in camera হলো 0।

```python id="w7t1mn"
import cv2

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Cannot read from camera.")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### ওয়েবক্যাম Resolution সেট করা

OpenCV দিয়ে camera properties যেমন resolution সেট করা যায়। তবে সব ক্যামেরা সব value support করে না।

```python id="p2x8qq"
import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Resolution: {int(width)}x{int(height)}")

while True:
    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("HD Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
