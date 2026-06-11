## বেসিক ইমেজ অপারেশনস

ইমেজ রিড করা শিখলেই হবে না, সেই ইমেজের উপর বিভিন্ন অপারেশনও করতে হবে যেমন রং পরিবর্তন, ব্লার করা, এজ ডিটেকশন ইত্যাদি। এই সেকশনে আমরা এমন কিছু বেসিক অপারেশন শিখব যেগুলো প্রায় প্রতিটি কম্পিউটার ভিশন প্রজেক্টে ব্যবহার হয়। এগুলো ইমেজ প্রসেসিং এর মূল ভিত্তি এবং এগুলো ভালোভাবে বোঝা ছাড়া অ্যাডভান্সড টপিক শেখা কঠিন।

---

### কালার কনভার্শন  cv2.cvtColor()

কালার স্পেস কনভার্শন ইমেজ প্রসেসিং এর একটি গুরুত্বপূর্ণ অপারেশন। OpenCV তে `cv2.cvtColor()` ব্যবহার করে এক কালার স্পেস থেকে অন্য কালার স্পেসে রূপান্তর করা যায়। সাধারণত ব্যবহৃত কনভার্শনগুলো হলো BGR থেকে RGB, BGR থেকে Grayscale এবং BGR থেকে HSV।

BGR থেকে RGB কনভার্শন মূলত matplotlib বা অন্যান্য RGB based লাইব্রেরিতে সঠিক রং দেখানোর জন্য ব্যবহার করা হয়।

Grayscale ইমেজে প্রতিটি পিক্সেল 0 থেকে 255 এর মধ্যে একটি মান ধারণ করে। এটি 3 channel থেকে 1 channel এ রূপান্তরিত হয়, ফলে ডেটা কমে যায় এবং প্রসেসিং দ্রুত হয়। Edge detection, thresholding এবং অনেক classical computer vision অ্যালগরিদম grayscale এ বেশি কার্যকর।

HSV (Hue, Saturation, Value) কালার স্পেস color based segmentation এর জন্য ব্যবহৃত হয়। RGB/BGR এ lighting পরিবর্তনের কারণে সমস্যা হয়, কিন্তু HSV তে Hue component অনেকটা lighting independent হওয়ায় নির্দিষ্ট রং আলাদা করা সহজ হয়।

```python id="cv_color"
import cv2

img = cv2.imread("resources/sample_image.png")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Original BGR", img)
cv2.imshow("Grayscale", img_gray)
cv2.imshow("HSV", img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"BGR shape: {img.shape}")
print(f"Grayscale shape: {img_gray.shape}")
print(f"HSV shape: {img_hsv.shape}")
```

---

### ব্লারিং  cv2.GaussianBlur()

Blurring ব্যবহার করা হয় noise কমাতে এবং image smooth করতে। এটি edge detection এর আগে একটি গুরুত্বপূর্ণ preprocessing step।

Gaussian blur সবচেয়ে বেশি ব্যবহৃত কারণ এটি natural looking smooth effect দেয়।

Kernel size যত বড় হবে blur তত বেশি হবে। Kernel অবশ্যই odd number হতে হবে।

```python id="cv_blur"
import cv2

img = cv2.imread("resources/sample_image.png")

blur_1 = cv2.GaussianBlur(img, (5, 5), 0)
blur_2 = cv2.GaussianBlur(img, (15, 15), 0)
blur_3 = cv2.GaussianBlur(img, (31, 31), 0)

cv2.imshow("Original", img)
cv2.imshow("Blur 5x5", blur_1)
cv2.imshow("Blur 15x15", blur_2)
cv2.imshow("Blur 31x31", blur_3)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

Sigma value 0 দিলে OpenCV automatically kernel size অনুযায়ী sigma calculate করে।

---

### এজ ডিটেকশন  cv2.Canny()

Edge detection ব্যবহার করা হয় image থেকে object boundary বের করার জন্য। Canny Edge Detector একটি widely used algorithm।

এটি দুইটি threshold ব্যবহার করে:

* Lower threshold: weak edges filter বা connect করার জন্য
* Upper threshold: strong edges নিশ্চিত করার জন্য

```python id="cv_canny"
import cv2

img = cv2.imread("resources/sample_image.png")

img_blur = cv2.GaussianBlur(img, (5, 5), 0)

edges_1 = cv2.Canny(img_blur, 50, 150)
edges_2 = cv2.Canny(img_blur, 100, 200)
edges_3 = cv2.Canny(img_blur, 150, 250)

cv2.imshow("Original", img)
cv2.imshow("Edges 50-150", edges_1)
cv2.imshow("Edges 100-200", edges_2)
cv2.imshow("Edges 150-250", edges_3)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

Threshold কম হলে বেশি edges পাওয়া যায়, কিন্তু noise বেশি থাকে। Threshold বেশি হলে clean but fewer edges পাওয়া যায়।

---

### ইমেজ Shape এবং Dimensions

ইমেজের structure বোঝা খুব গুরুত্বপূর্ণ।

* shape → height, width, channels
* size → total number of pixels
* dtype → data type (usually uint8)

```python id="cv_shape"
import cv2

img = cv2.imread("resources/sample_image.png")

print(f"Shape: {img.shape}")
print(f"Height: {img.shape[0]}")
print(f"Width: {img.shape[1]}")
print(f"Channels: {img.shape[2]}")
print(f"Total pixels: {img.size}")
print(f"Data type: {img.dtype}")
```

OpenCV তে একটি গুরুত্বপূর্ণ বিষয় হলো shape এ height আগে আসে, কিন্তু অনেক operation এ width আগে দিতে হয়।
