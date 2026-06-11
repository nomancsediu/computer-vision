# ডেটা অগমেন্টেশন ও ইমেজ ক্লাসিফিকেশন

এই চ্যাপ্টারে আমরা ডেটা অগমেন্টেশন টেকনিক শিখবো এবং একটা সম্পূর্ণ বাইনারি ইমেজ ক্লাসিফিকেশন পাইপলাইন বানাবো।

আগের চ্যাপ্টারে আমরা কাস্টম CNN মডেল train করতে শিখেছি। কিন্তু একটি সাধারণ সমস্যা আমরা সবাই ফেস করি  **ডেটা কম থাকলে কী হবে?** রিয়েল-ওয়ার্ল্ডে হাজার হাজার labeled image collect করা অনেক সময়সাপেক্ষ ও ব্যয়বহুল। একজন radiologist কে bone fracture এর হাজারো X-ray image label করতে বললে, সেটা সম্ভবত কয়েক মাসের কাজ। কিন্তু ডেটা কম থাকলে deep learning model overfit করে  training data মুখস্থ করে ফেলে, নতুন data তে ভালো কাজ করে না।

এখানেই **Data Augmentation** আমাদের বাঁচায়! Data augmentation হলো এমন একটি technique যেখানে আমাদের existing training image গুলো থেকে artificial variations তৈরি করা হয়  rotation, flip, zoom, shift ইত্যাদি transform apply করে। এভাবে 500 টা image থেকে effectively হাজারো training sample পাওয়া যায়, আর মডেল বেশি robust হয়।

এই চ্যাপ্টারে আমরা দুটি মূল বিষয় কভার করবো:

- **ডেটা অগমেন্টেশন টেকনিক:** `ImageDataGenerator` ক্লাসের সব গুরুত্বপূর্ণ parameter বিস্তারিত শিখবো  `rotation_range`, `shear_range`, `zoom_range`, `horizontal_flip`, `width_shift_range`, `height_shift_range`, `fill_mode`, `brightness_range` ইত্যাদি। প্রতিটি parameter কী করে, কোন value ব্যবহার করতে হয়, আর কখন augmentation সাহায্য করে আর কখন করে না  সব বোঝার চেষ্টা করবো। আমরা augmented image গুলো visualize ও করবো যাতে পরিষ্কার বোঝা যায় প্রতিটি transform দেখতে কেমন।

- **বাইনারি ইমেজ ক্লাসিফিকেশন:** Augmented data দিয়ে একটি সম্পূর্ণ binary classification pipeline বানাবো  CNN model architecture design, compile, train, evaluate, training curve plot, এবং নতুন image তে prediction। এটি হবে একটি end-to-end project যা তুমি যেকোনো binary classification problem এ adapt করতে পারবে।

Data augmentation শুধু image classification এ নয়, object detection, segmentation, যেকোনো computer vision task এ ব্যবহৃত হয়। এটি হলো একজন CV engineer এর toolkit এর সবচেয়ে গুরুত্বপূর্ণ weapon গুলোর একটি। তাই চলো, ভালো করে শিখে নিই!
