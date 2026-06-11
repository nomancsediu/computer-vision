# ট্রান্সফার লার্নিং ও প্রিট্রেইন্ড মডেল

এই চ্যাপ্টারে আমরা ট্রান্সফার লার্নিং শিখবো  কিভাবে ImageNet এ প্রিট্রেইন্ড মডেল ব্যবহার করে নতুন টাস্কে দ্রুত ভালো রেজাল্ট পাওয়া যায়। ResNet50 ও VGG16 দিয়ে হাতে-কলমে কাজ করবো।

আগের চ্যাপ্টারগুলোতে আমরা নিজেদের CNN model scratch থেকে train করেছি। কিন্তু একটি সত্য হলো  **scratch থেকে train করা সবসময় সম্ভব হয় না।** কারণ তিনটি:

1. **ডেটা কম:** Deep model train করতে লাখ লাখ labeled image দরকার। কিন্তু medical imaging, satellite imaging, industrial quality control  এসব domain এ এত data collect করা অসম্ভব।

2. **সময় ও রিসোর্স:** ImageNet (14M+ images) এ ResNet train করতে কয়েক সপ্তাহ লাগে, multiple GPU দরকার। সবার কাছে এত resource থাকে না।

3. **Accuracy:** Scratch থেকে ছোট dataset এ train করলে accuracy অনেক কম হয়  overfitting ও underfitting দুটোই হতে পারে।

এখানেই **Transfer Learning** জাদু দেখায়! Transfer learning এর core idea অনেক সহজ  **একজন যা শিখেছে, তা অন্য কাজে লাগানো।** ঠিক যেমন একজন পাইলট car drive করতে গেলে scratch থেকে শিখতে হয় না  steering, brake, traffic rule এর basic concept তো আগেই জানে! তেমনি ImageNet এ train করা model ইতিমধ্যে edges, textures, shapes, object parts  এসব general feature শিখে ফেলেছে। এই features নতুন task এও কাজে লাগানো যায়!

Transfer learning এ দুটি জনপ্রিয় approach আছে:

- **Feature Extraction:** Pretrained model এর convolutional base কে "feature extractor" হিসেবে ব্যবহার করা  শুধু নতুন fully connected layers train করা। এটি fast ও effective, বিশেষ করে ছোট dataset এ।

- **Fine-tuning:** Pretrained model এর কিছু last layer unfreeze করে নতুন task এর জন্য adjust করা। এটি feature extraction এর চেয়ে বেশি powerful, কিন্তু overfitting এর risk ও বেশি।

এই চ্যাপ্টারে আমরা:

- **ResNet50 দিয়ে prediction:** ImageNet এ pretrained ResNet50 directly use করে 1000 class এর image classification করবো। এতে বুঝবো pretrained model কতটা powerful  কোনো training ছাড়াই ভালো prediction!

- **VGG16 দিয়ে bone fracture classification:** Transfer learning এর real application দেখবো  VGG16 কে feature extractor হিসেবে ব্যবহার করে bone fracture X-ray image classify করবো (Oblique vs Spiral fracture)। Custom head add, training, evaluation, fine-tuning  সব step বিস্তারিত শিখবো।

Transfer learning modern deep learning এর সবচেয়ে important paradigm  কারণ এটি "standing on the shoulders of giants" এর principle। যারা ImageNet এ মাসের পর মাস গবেষণা করে model train করেছেন, তাদের work এর উপর দাঁড়িয়ে আমরা নতুন task এ দ্রুত ভালো result পাই। চলো শুরু করা যাক!
