## Docker কন্টেইনারাইজেশন ও ডিপ্লয়মেন্ট

এই চ্যাপ্টারে আমরা Docker দিয়ে আমাদের Flask অ্যাপকে কন্টেইনারাইজ করবো এবং প্রোডাকশন ডিপ্লয়মেন্ট এর বেস্ট প্র্যাকটিস শিখবো।

গত চ্যাপ্টারে আমরা একটি সুন্দর Flask web app বানালাম — যেটা bone fracture X-ray image classify করতে পারে। কিন্তু "আমার ল্যাপটপে কাজ করছে" আর "সবার কাছে কাজ করবে" এর মধ্যে অনেক ফারাক আছে! "It works on my machine" — এটা software engineering এর সবচেয়ে কুখ্যাত sentence। এই সমস্যা সমাধানের জন্যই Docker এসেছে।

### কেন Docker দরকার?

তোমার Flask app টা তোমার machine এ run হচ্ছে কারণ তোমার কাছে সব dependency correctly install আছে — TensorFlow 2.19.0, Flask 3.0.0, specific Python version, system libraries... কিন্তু যখন তুমি এটা অন্য কাউকে দিবে বা cloud server এ deploy করবে:

- **Dependency conflicts:** অন্য machine এ TensorFlow version আলাদা থাকতে পারে
- **OS differences:** Linux এ develop করে Windows এ run করতে গেলে problem হতে পারে
- **System library missing:** TensorFlow এর জন্য specific C++ libraries দরকার — নাও থাকতে পারে
- **Python version mismatch:** Python 3.11 তে develop, server এ Python 3.9 — compatibility issue

Docker এই সব সমস্যা solve করে। তোমার entire application — code, dependencies, system libraries, configuration — সব কিছু একটি container এ pack হয়। এই container যেকোনো machine এ same way তে run হবে — guaranteed!

### Docker দিয়ে কী শিখবো?

এই চ্যাপ্টারে আমরা:

1. **Dockerfile লিখবো** — আমাদের Flask app এর জন্য complete Docker image definition
2. **Docker image build করবো** — Dockerfile থেকে executable image তৈরি
3. **Container run করবো** — Image থেকে running container
4. **Production deployment শিখবো** — Gunicorn WSGI server, cloud platforms, security best practices

শেষে তোমার কাছে এমন একটি setup থাকবে যেটা একটি command দিয়ে যেকোনো server এ deploy করা যাবে — আর কোনো "it works on my machine" problem থাকবে না!
