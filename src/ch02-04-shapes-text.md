## শেইপ এবং টেক্সট ড্রয়িং

কম্পিউটার ভিশন প্রজেক্টে অনেক সময় ইমেজের উপর বিভিন্ন shape এবং text আঁকতে হয়। যেমন object detection এর ফলাফল দেখানোর জন্য bounding box আঁকা, কোনো নির্দিষ্ট পয়েন্ট mark করার জন্য circle আঁকা, বা লেবেল লেখার জন্য text যোগ করা। OpenCV তে এই কাজগুলো করার জন্য বিভিন্ন drawing ফাংশন আছে, যেগুলো খুব সহজে ব্যবহার করা যায়।

---

### ফাঁকা ইমেজ তৈরি  np.zeros()

Shape বা text আঁকার আগে একটি canvas দরকার হয়। NumPy এর `np.zeros()` ব্যবহার করে একটি সম্পূর্ণ কালো ইমেজ তৈরি করা যায়। এখানে (height, width, channels) দিতে হয় এবং সাধারণত data type থাকে `np.uint8`। সব পিক্সেল 0 হওয়ার কারণে ইমেজটি কালো দেখায়।

```python id="canvas_black"
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

cv2.imshow("Black Canvas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

সাদা canvas তৈরি করতে `np.ones()` ব্যবহার করে 255 দিয়ে multiply করা হয়।

```python id="canvas_white"
import numpy as np
import cv2

white_img = np.ones((500, 500, 3), np.uint8) * 255

cv2.imshow("White Canvas", white_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### লাইন আঁকা  cv2.line()

`cv2.line()` দিয়ে ইমেজে line আঁকা যায়। এখানে starting point, ending point, color (BGR format), এবং thickness দিতে হয়।

OpenCV তে coordinate system এ (0,0) থাকে top-left corner এ, x ডানদিকে এবং y নিচের দিকে বাড়ে।

```python id="draw_line"
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

cv2.line(img, (50, 50), (450, 50), (255, 0, 0), 3)
cv2.line(img, (50, 100), (450, 100), (0, 255, 0), 3)
cv2.line(img, (50, 150), (450, 150), (0, 0, 255), 3)
cv2.line(img, (50, 200), (450, 200), (255, 255, 0), 5)

cv2.imshow("Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### আয়তক্ষেত্র আঁকা  cv2.rectangle()

Bounding box এবং region highlight করার জন্য `cv2.rectangle()` ব্যবহার করা হয়। এখানে top-left এবং bottom-right coordinate দিতে হয়।

Thickness এর জায়গায় `cv2.FILLED` দিলে সম্পূর্ণ fill হয়ে যায়।

```python id="draw_rect"
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)
cv2.rectangle(img, (250, 50), (450, 200), (0, 0, 255), cv2.FILLED)
cv2.rectangle(img, (50, 250), (200, 450), (255, 0, 0), 1)

cv2.imshow("Rectangles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### বৃত্ত আঁকা  cv2.circle()

Circle আঁকার জন্য center point, radius, color এবং thickness দিতে হয়।

```python id="draw_circle"
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

cv2.circle(img, (150, 150), 80, (0, 255, 0), 3)
cv2.circle(img, (350, 150), 50, (0, 0, 255), cv2.FILLED)
cv2.circle(img, (250, 350), 100, (255, 0, 0), 1)

cv2.imshow("Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### টেক্সট লেখা  cv2.putText()

ইমেজের উপর text লেখার জন্য `cv2.putText()` ব্যবহার করা হয়। এখানে text string, position, font, font scale, color এবং thickness দিতে হয়।

OpenCV Unicode text (যেমন বাংলা) সরাসরি support করে না।

```python id="put_text"
import numpy as np
import cv2

img = np.zeros((500, 500, 3), np.uint8)

cv2.putText(img, "Hello OpenCV", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(img, "Computer Vision", (50, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
cv2.putText(img, "Text Example", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(img, "Small", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.putText(img, "BIG", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 3)

cv2.imshow("Text", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### BGR কালার ফরম্যাট

OpenCV তে সব রং BGR format এ দিতে হয়।

| Color  | BGR             |
| ------ | --------------- |
| White  | (255, 255, 255) |
| Black  | (0, 0, 0)       |
| Red    | (0, 0, 255)     |
| Green  | (0, 255, 0)     |
| Blue   | (255, 0, 0)     |
| Yellow | (0, 255, 255)   |
| Orange | (0, 165, 255)   |
| Purple | (255, 0, 255)   |

---

### কমপ্লিট উদাহরণ

```python id="complete_draw"
import numpy as np
import cv2

img = np.zeros((600, 800, 3), np.uint8)

cv2.rectangle(img, (200, 200), (600, 500), (0, 200, 0), cv2.FILLED)
cv2.line(img, (200, 200), (400, 50), (0, 0, 200), 5)
cv2.line(img, (400, 50), (600, 200), (0, 0, 200), 5)

cv2.rectangle(img, (350, 350), (450, 500), (200, 200, 200), cv2.FILLED)
cv2.rectangle(img, (240, 260), (320, 340), (255, 255, 0), cv2.FILLED)
cv2.rectangle(img, (480, 260), (560, 340), (255, 255, 0), cv2.FILLED)

cv2.circle(img, (680, 80), 50, (0, 200, 255), cv2.FILLED)

cv2.putText(img, "My OpenCV House", (220, 560), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Drawing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

Drawing functions in-place কাজ করে, তাই এগুলো original image modify করে। যদি original image preserve করতে চাও তাহলে `img.copy()` ব্যবহার করা উচিত।
