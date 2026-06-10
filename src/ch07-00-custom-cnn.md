# কাস্টম CNN ট্রেনিং

এই চ্যাপ্টারে আমরা নিজেদের ডেটাসেটে কাস্টম CNN মডেল ট্রেইন করবো। Kaggle থেকে ডেটা ডাউনলোড, প্রিপ্রসেসিং, মডেল বিল্ডিং, ট্রেনিং ও ইভ্যালুয়েশন — সব কভার করবো।

আগের চ্যাপ্টারগুলোতে আমরা CNN এর তাত্ত্বিক দিক শিখেছি — convolution কিভাবে কাজ করে, pooling কেন দরকার, LeNet-5 ও AlexNet এর মতো ক্লাসিক architecture দেখেছি। কিন্তু একটি গুরুত্বপূর্ণ প্রশ্ন বাকি থেকে গেছে: **নিজের ডেটাসেটে কিভাবে CNN ট্রেইন করবো?** এই চ্যাপ্টারে ঠিক এটাই করবো।

রিয়েল-ওয়ার্ল্ডে তুমি কখনো MNIST বা CIFAR-10 এর মতো ready-made ডেটাসেট পাবে না। তুমি নিজের ডেটা collect করবে, organize করবে, preprocess করবে, তারপর মডেল বানাবে ও ট্রেইন করবে। এই পুরো pipeline টাই আমরা এই চ্যাপ্টারে হাতে-কলমে শিখবো। আমরা Kaggle থেকে "apples-or-tomatoes" ডেটাসেট ব্যবহার করবো — এটি একটি binary classification problem যেখানে আমাদের ছবি দেখে বলতে হবে এটি আপেল না টমেটো।

এই চ্যাপ্টারের মূল বিষয়গুলো:

- **ডেটা প্রস্তুতি:** Kaggle API সেটআপ করে ডেটা ডাউনলোড করা, directory structure তৈরি করা, train/validation/test split করা, এবং `ImageDataGenerator` দিয়ে data augmentation করা। Data augmentation হলো এমন একটি technique যেখানে existing image থেকে নতুন variations তৈরি করা হয় — rotation, flip, zoom ইত্যাদি — যাতে মডেল বেশি robust হয় এবং overfitting কম হয়।

- **কাস্টম CNN মডেল বিল্ডিং:** নিজের architecture ডিজাইন করা। আমরা একটি Sequential model বানাবো — Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Dense → Output। কেন filter সংখ্যা বাড়াই (32→64→128), কেন spatial dimension কমাই, কেন binary classification এ 1টি neuron ও sigmoid activation ব্যবহার করি — সব বিস্তারিত আলোচনা করবো।

- **ট্রেনিং ও ইভ্যালুয়েশন:** `model.fit()` দিয়ে ট্রেইন করা, training curve plot করে overfitting/underfitting নির্ণয় করা, model save/load করা, এবং নতুন ইমেজে prediction করা।

এই চ্যাপ্টার শেষে তুমি যেকোনো custom dataset এ নিজে একটা CNN মডেল train করতে পারবে — end-to-end pipeline সম্পূর্ণভাবে বুঝতে পারবে। তাই চলো শুরু করা যাক!
