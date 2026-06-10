## Flask ওয়েব অ্যাপ্লিকেশন

এই চ্যাপ্টারে আমরা আমাদের বোন ফ্র্যাকচার ক্লাসিফিকেশন মডেলটাকে Flask ওয়েব অ্যাপ হিসেবে ডিপ্লয় করবো। ইউজার একটা X-ray ইমেজ আপলোড করবে এবং রিয়েল-টাইমে প্রেডিকশন পাবে।

গত চ্যাপ্টারে আমরা VGG16 দিয়ে একটি শক্তিশালী bone fracture classification model তৈরি করেছি — যেটা X-ray image দেখে বলতে পারে ফ্র্যাকচারটি Oblique নাকি Spiral। কিন্তু model টা Jupyter Notebook এ পড়ে থাকলে তো আর কারো কাজে আসবে না! একটা ML model এর সত্যিকারের value তখনই যখন সেটা end-user এর হাতে পৌঁছায় — আর সেজন্য দরকার একটি web application।

### কেন Flask?

Flask হলো Python এর একটি lightweight micro web framework। Django এর মতো heavy framework এর তুলনায় Flask অনেক সহজ, minimal এবং flexible। ML model deployment এর জন্য Flask একদম perfect কারণ:

- **Minimal boilerplate:** মাত্র কয়েক লাইন কোডে একটি web server চালু করা যায়
- **Flexible:** আমাদের যেভাবে দরকার সেভাবে structure করা যায় — কোনো enforced pattern নেই
- **Python ecosystem:** TensorFlow/Keras model সরাসরি Python এ load করে predict করা যায় — কোনো language bridging লাগবে না
- **Large community:** সমস্যায় পড়লে solution খুঁজে পাওয়া সহজ

### আমরা কী বানাবো?

আমরা এমন একটি web app বানাবো যেখানে:

1. **ইউজার একটি X-ray image আপলোড করবে** — একটি simple upload form এর মাধ্যমে
2. **Server এ model load থাকবে** — VGG16 transfer learning model, Google Drive থেকে download হবে
3. **Image preprocess হবে** — model এর expected format এ convert হবে (224×224, normalized)
4. **Prediction হবে** — model.predict() দিয়ে রিয়েল-টাইমে
5. **Result দেখাবে** — কোন class (Oblique/Spiral) এবং কত confidence — সুন্দর HTML page এ

এই চ্যাপ্টারের শেষে তোমার একটি fully functional web application থাকবে যেটা browser এ চলবে এবং যেকোনো X-ray image তে fracture type predict করতে পারবে। পরবর্তী চ্যাপ্টারে আমরা এই app টাকে Docker দিয়ে containerize করবো এবং cloud এ deploy করবো — কিন্তু আপাতত চলো Flask app টা বানাই!
