## ইমেজ রিডিং এবং ডিসপ্লে করা

কম্পিউটার ভিশন এর সবচেয়ে প্রথম এবং মৌলিক কাজ হলো ইমেজ রিড করা এবং স্ক্রিনে ডিসপ্লে করা। যেকোনো ইমেজ প্রসেসিং পাইপলাইন শুরু হয় ইমেজ লোড করার মাধ্যমে। তাই এই সেকশনটি খুবই গুরুত্বপূর্ণ — এখানে ভালো ভাবে শিখলে পরবর্তী সব কাজ সহজ হয়ে যাবে। আমরা শিখবো কিভাবে স্ট্যাটিক ইমেজ রিড করতে হয়, কিভাবে ভিডিও ফাইল এবং ওয়েবক্যাম থেকে ফ্রেম পড়তে হয়, এবং সবচেয়ে গুরুত্বপূর্ণ — BGR এবং RGB ফরম্যাটের পার্থক্য কী।

### ইমেজ রিড করা — cv2.imread()

OpenCV তে ইমেজ রিড করার জন্য `cv2.imread()` ফাংশন ব্যবহার করা হয়। এই ফাংশনটি ইমেজ ফাইলের পাথ নেয় এবং একটি NumPy array রিটার্ন করে। ইমেজ রিড করার সময় তুমি দুটি প্যারামিটার পাস করতে পারো — প্রথমটি হলো ফাইলের পাথ, আর দ্বিতীয়টি হলো রিডিং মোড। ডিফল্টভাবে ইমেজটি color mode এ লোড হয়, অর্থাৎ তিনটি চ্যানেল (Blue, Green, Red) সহ আসে। তুমি চাইলে `cv2.IMREAD_GRAYSCALE` ব্যবহার করে সরাসরি grayscale এ লোড করতে পারো, কিংবা `cv2.IMREAD_UNCHANGED` দিয়ে alpha channel সহ লোড করতে পারো।

```python
import cv2

# Color ইমেজ রিড করা (ডিফল্ট)
img = cv2.imread("resources/lena.png")

# Grayscale ইমেজ রিড করা
img_gray = cv2.imread("resources/lena.png", cv2.IMREAD_GRAYSCALE)

# Alpha channel সহ রিড করা (PNG ফাইলের জন্য)
img_unchanged = cv2.imread("resources/lena.png", cv2.IMREAD_UNCHANGED)

print(f"Color image shape: {img.shape}")
print(f"Grayscale image shape: {img_gray.shape}")
```

ইমেজ রিড করার পর অবশ্যই চেক করে নিতে হবে যে ইমেজটি সঠিকভাবে লোড হয়েছে কিনা। যদি ফাইল পাথ ভুল হয়, তাহলে `cv2.imread()` কোনো error দেবে না, বরং `None` রিটার্ন করবে। এটি অনেক ব্যাগের কারণ হতে পারে, তাই সবসময় চেক করো:

```python
import cv2

img = cv2.imread("resources/my_photo.jpg")

if img is None:
    print("Error: ইমেজ লোড করা যায়নি! ফাইল পাথ চেক করো।")
else:
    print(f"ইমেজ সফলভাবে লোড হয়েছে! Shape: {img.shape}")
```

### ইমেজ ডিসপ্লে করা — cv2.imshow(), cv2.waitKey(), cv2.destroyAllWindows()

ইমেজ রিড করার পর সেটাকে স্ক্রিনে দেখানোর জন্য `cv2.imshow()` ব্যবহার করা হয়। এই ফাংশনের প্রথম প্যারামিটার হলো উইন্ডোর নাম (যেকোনো string দিতে পারো), আর দ্বিতীয়টি হলো ইমেজের NumPy array। ইমেজ ডিসপ্লে করার পর `cv2.waitKey()` কল করা অত্যন্ত জরুরি — এটি ছাড়া উইন্ডোটি এক মুহূর্তেই বন্ধ হয়ে যাবে। `cv2.waitKey(0)` মানে অসীমকাল অপেক্ষা করবে যতক্ষণ না তুমি কোনো key press করো। আর `cv2.waitKey(1000)` মানে ১০০০ মিলিসেকেন্ড অর্থাৎ ১ সেকেন্ড অপেক্ষা করবে। কাজ শেষে `cv2.destroyAllWindows()` দিয়ে সব উইন্ডো বন্ধ করে দিতে হয়, নাহলে memory leak হতে পারে।

```python
import cv2

img = cv2.imread("resources/lena.png")

# ইমেজ ডিসপ্লে করা
cv2.imshow("Lena Image", img)

# যেকোনো key press করার জন্য অপেক্ষা করা
cv2.waitKey(0)

# সব উইন্ডো বন্ধ করা
cv2.destroyAllWindows()
```

তুমি চাইলে নির্দিষ্ট key press করলেই উইন্ডো বন্ধ করতে পারো। যেমন 'q' key press করলে উইন্ডো বন্ধ হবে — এটি অনেক প্র্যাকটিক্যাল:

```python
import cv2

img = cv2.imread("resources/lena.png")
cv2.imshow("Press Q to close", img)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### BGR vs RGB — একটি ক্রিটিক্যাল পার্থক্য!

OpenCV নিয়ে কাজ করার সময় সবচেয়ে বেশি যে বিভ্রান্তি হয় সেটি হলো BGR এবং RGB এর পার্থক্য। OpenCV ইমেজ রিড করার সময় BGR (Blue-Green-Red) ফরম্যাট ব্যবহার করে, অন্যদিকে বেশিরভাগ অন্যান্য লাইব্রেরি — যেমন matplotlib, PIL/Pillow — RGB (Red-Green-Red) ফরম্যাট ব্যবহার করে। এর মানে হলো, যদি তুমি OpenCV দিয়ে ইমেজ রিড করে সরাসরি matplotlib দিয়ে ডিসপ্লে করো, তাহলে রং ভুল দেখাবে — নীল হয়ে লাল দেখাবে, লাল হয়ে নীল দেখাবে! এটি নিয়ে অনেক ব্যাগ হয়, তাই সবসময় মনে রাখবে — OpenCV = BGR, অন্যান্য = RGB।

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("resources/lena.png")

# ভুল উপায় — রং ভুল দেখাবে (BGR কে RGB ছাড়া দেখানো)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("BGR (ভুল রং)")

# সঠিক উপায় — BGR কে RGB তে convert করে দেখানো
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 2)
plt.imshow(img_rgb)
plt.title("RGB (সঠিক রং)")

plt.show()
```

তবে `cv2.imshow()` ব্যবহার করলে এই conversion এর দরকার নেই, কারণ OpenCV নিজেই BGR ফরম্যাটে ডিসপ্লে করে এবং সঠিকভাবেই দেখায়। শুধু matplotlib বা অন্য RGB-based লাইব্রেরি ব্যবহার করলে conversion করতে হবে।

### ভিডিও রিড করা — cv2.VideoCapture()

ভিডিও ফাইল রিড করার জন্য `cv2.VideoCapture()` ব্যবহার করা হয়। ভিডিও আসলে অনেকগুলো ফ্রেম (image) এর সমষ্টি — সাধারণত প্রতি সেকেন্ডে ৩০টি ফ্রেম (30 FPS)। তাই ভিডিও রিড করার সময় আমরা একটি loop এর মধ্যে ফ্রেম বাফ্রেম রিড করি। `cap.read()` মেথড দুটি মান রিটার্ন করে — প্রথমটি `success` (বুলিয়ান, ফ্রেম সফলভাবে পড়া হয়েছে কিনা), আর দ্বিতীয়টি `frame` (ফ্রেমের NumPy array)। যখন ভিডিও শেষ হয়ে যায়, `success` এর মান `False` হয়ে যায়।

```python
import cv2

# ভিডিও ফাইল রিড করা
cap = cv2.VideoCapture("resources/video.mp4")

while True:
    success, frame = cap.read()

    if not success:
        print("ভিডিও শেষ হয়ে গেছে বা ফাইল পাওয়া যায়নি।")
        break

    cv2.imshow("Video Frame", frame)

    # 'q' key press করলে বন্ধ হবে
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

`cap.release()` কল করা অত্যন্ত গুরুত্বপূর্ণ — এটি video capture object কে release করে এবং ক্যামেরা/ফাইল resource মুক্ত করে। এটি না করলে ক্যামেরা ব্যস্ত থেকে যেতে পারে এবং অন্য প্রোগ্রাম ক্যামেরা অ্যাক্সেস করতে পারবে না।

### ওয়েবক্যাম থেকে ভিডিও ক্যাপচার

ওয়েবক্যাম থেকে লাইভ ভিডিও ক্যাপচার করাও `cv2.VideoCapture()` দিয়েই করা হয়, তবে ফাইল পাথের বদলে ক্যামেরার index number দিতে হয়। সাধারণত ল্যাপটপের built-in ওয়েবক্যাম এর index হলো `0`, আর বাইরে যুক্ত করা USB ক্যামেরা এর index হতে পারে `1`, `2` ইত্যাদি। ওয়েবক্যাম ব্যবহারের pattern ভিডিও ফাইলের মতোই, শুধু পাথের জায়গায় index number বসে।

```python
import cv2

# ওয়েবক্যাম ক্যাপচার (index 0 = ডিফল্ট ক্যামেরা)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    
    if not success:
        print("ক্যামেরা থেকে ফ্রেম পড়া যাচ্ছে না।")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### ওয়েবক্যাম Resolution সেট করা

ডিফল্টভাবে ওয়েবক্যাম কম resolution এ ক্যাপচার করে, যেমন 640×480। কিন্তু অনেক সময় আমাদের বেশি resolution দরকার হয়। OpenCV তে `cap.set()` মেথড ব্যবহার করে ক্যামেরার resolution, FPS, brightness ইত্যাদি সেট করা যায়। `cv2.CAP_PROP_FRAME_WIDTH` দিয়ে width এবং `cv2.CAP_PROP_FRAME_HEIGHT` দিয়ে height সেট করা হয়। তবে মনে রাখবে, সব ওয়েবক্যাম সব resolution support করে না — তুমি যা চাইবে তা নাও পেতে পারো। সেট করার পর `cap.get()` দিয়ে আসল resolution চেক করে দেখো।

```python
import cv2

cap = cv2.VideoCapture(0)

# Resolution সেট করা
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# আসল resolution চেক করা
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"ক্যামেরার Resolution: {int(width)}x{int(height)}")

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

এইভাবে তুমি OpenCV দিয়ে ইমেজ এবং ভিডিও রিড করতে পারবে, ডিসপ্লে করতে পারবে, এবং ওয়েবক্যাম থেকে লাইভ ফ্রেম ক্যাপচার করতে পারবে। BGR vs RGB এর বিষয়টি মনে রাখা অত্যন্ত জরুরি — এটি নিয়ে ভুল করলে পুরো প্রোজেক্টে রং ভুল হতে পারে!
