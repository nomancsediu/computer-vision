## Flask অ্যাপ বিল্ডিং

এই সেকশনে আমরা আমাদের bone fracture classifier কে Flask web app হিসেবে গড়ে তুলবো — step by step। Flask হলো Python এর একটি lightweight web framework যেটা দিয়ে দ্রুত web application তৈরি করা যায়। Django এর মতো "batteries included" framework এর বিপরীতে Flask "micro" framework — মানে শুধু প্রয়োজনীয় features দেয়, বাকি সব তুমি নিজের মতো করো। ML model deployment এর জন্য এটিই ideal।

### Application Architecture

আমাদের Flask app এর architecture বেশ simple:

```
Client (Browser) → Flask Server → Model (VGG16) → Response (HTML)
     ↑                                                    |
     └──────────── Result Page ←──────────────────────────┘
```

ইউজার browser এ image upload করে → Flask server request receive করে → image preprocess করে → pre-trained VGG16 model এ predict করে → result HTML page এ render করে → ইউজার দেখতে পায়। পুরো process কয়েক সেকেন্ডের মধ্যে হয়!

### Project Directory Structure

শুরু করার আগে project structure টা বুঝে নিই:

```
bone_fracture_app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── model/                 # Model weights directory
│   └── (weights downloaded from Google Drive)
├── static/                # Static files (CSS, JS, uploaded images)
│   └── style.css
└── templates/             # HTML templates (Jinja2)
    └── index.html
```

`templates/` folder এ Jinja2 HTML templates থাকে — Flask automatically এই folder থেকে template search করে। `static/` folder এ CSS, JavaScript ও uploaded images রাখা হয়। `model/` folder এ model weights file থাকবে যেটা Google Drive থেকে download হবে।

### Complete app.py — Flask Application Code

এবার মূল Flask application code দেখি — সম্পূর্ণ `app.py`:

```python
import os
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import gdown

# ============================================================
# Configuration
# ============================================================

app = Flask(__name__)
app.secret_key = 'bone_fracture_classifier_secret_key'

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Google Drive model weights URL
MODEL_WEIGHTS_URL = 'https://drive.google.com/uc?id=YOUR_FILE_ID_HERE'
MODEL_WEIGHTS_PATH = 'model/model_vgg16_weights.h5'

# Class labels
CLASS_NAMES = {0: 'Oblique Fracture', 1: 'Spiral Fracture'}

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ============================================================
# Helper Functions
# ============================================================

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_model_weights():
    """Download model weights from Google Drive using gdown."""
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print("Downloading model weights from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
        gdown.download(MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH, quiet=False)
        print("Download complete!")
    else:
        print("Model weights already exist.")


def build_model():
    """Reconstruct VGG16 model architecture and load weights."""
    # VGG16 base model (same architecture as training)
    base_model = VGG16(
        weights=None,              # We'll load our own weights
        include_top=False,         # No ImageNet classification head
        input_shape=(224, 224, 3)
    )

    # Custom classification head (must match training architecture)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)

    # Complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Load trained weights
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("Model loaded successfully!")

    return model


def predict_fracture(img_path, model):
    """Predict fracture type from X-ray image."""
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])

    # Get all class probabilities
    all_probs = {
        CLASS_NAMES[i]: float(predictions[0][i])
        for i in range(len(CLASS_NAMES))
    }

    return CLASS_NAMES[class_idx], confidence, all_probs


# ============================================================
# Model Initialization (runs once at startup)
# ============================================================

print("=" * 50)
print("Initializing Bone Fracture Classifier...")
print("=" * 50)

download_model_weights()
model = build_model()

print("Application ready!")
print("=" * 50)


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        flash('কোনো ফাইল আপলোড করা হয়নি!', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('কোনো ফাইল সিলেক্ট করা হয়নি!', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('শুধুমাত্র PNG, JPG, JPEG, GIF ফাইল আপলোড করা যাবে!', 'error')
        return redirect(url_for('index'))

    # Save uploaded file
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Make prediction
        predicted_class, confidence, all_probs = predict_fracture(filepath, model)

        # Format confidence for display
        confidence_pct = f"{confidence:.2%}"

        return render_template(
            'index.html',
            prediction=predicted_class,
            confidence=confidence_pct,
            all_probs=all_probs,
            image_path=filepath,
            filename=filename
        )

    except Exception as e:
        flash(f'প্রেডিকশনে সমস্যা হয়েছে: {str(e)}', 'error')
        return redirect(url_for('index'))


# ============================================================
# Run Application
# ============================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Code বিশ্লেষণ

#### Flask App Setup

```python
app = Flask(__name__)
app.secret_key = 'bone_fracture_classifier_secret_key'
```

`Flask(__name__)` দিয়ে Flask application instance তৈরি হয়। `secret_key` দরকার flash messages ও session এর জন্য — production এ এটা environment variable থেকে নিতে হবে, hardcode করা উচিত নয়।

#### Model Loading — Google Drive থেকে

```python
def download_model_weights():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        gdown.download(MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH, quiet=False)
```

`gdown` library দিয়ে Google Drive থেকে model weights download করি। এটা `wget` এর মতো কাজ করে কিন্তু Google Drive এর special URL format handle করতে পারে। যদি weights file আগেই download করা থাকে তাহলে আবার download করবে না — `os.path.exists()` check করে।

#### Model Reconstruction

```python
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
```

খুব গুরুত্বপূর্ণ পয়েন্ট: `weights=None` দেওয়া হয়েছে, `weights='imagenet'` নয়! কারণ আমরা ImageNet weights নয় বরং আমাদের নিজেদের trained weights load করবো। Model architecture অবশ্যই training এর সাথে exactly match করতে হবে — VGG16 base + Flatten + Dense(256) + Dropout(0.5) + Dense(2, softmax)। একটা layer ও আলাদা হলে `load_weights()` fail করবে!

#### File Upload Handling

```python
if 'file' not in request.files:
    flash('কোনো ফাইল আপলোড করা হয়নি!', 'error')
```

`request.files` dictionary তে uploaded file গুলো থাকে। `'file'` key টা HTML form এর `<input name="file">` এর সাথে match করতে হবে। যদি ইউজার submit করে কোনো file select না করে, তাহলে error message flash হবে।

#### File Validation

```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

শুধু image file ই accept করবো — অন্য কিছু upload করলে reject হবে। `rsplit('.', 1)[1]` দিয়ে extension বের করা হয়। Security এর জন্য এই validation খুবই জরুরি — নাহলে কেউ malicious file upload করতে পারে।

#### Image Preprocessing ও Prediction

```python
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
predictions = model.predict(img_array, verbose=0)
```

Training এর সময় যেভাবে preprocess করেছিলাম — ঠিক সেভাবেই করতে হবে: 224×224 resize, 255 দিয়ে normalize, batch dimension add। `np.expand_dims(img_array, axis=0)` দিয়ে (224, 224, 3) shape কে (1, 224, 224, 3) করা হয় — model batch input expect করে।

### requirements.txt

```
tensorflow==2.19.0
Flask==3.0.0
numpy==1.26.4
Pillow==10.2.0
gdown==5.0.0
```

Version pin করা খুব জরুরি — নাহলে এক machine এ কাজ করলেও অন্য machine এ fail করতে পারে। TensorFlow 2.19.0 আমাদের training environment এর সাথে compatible।

### App চালানো

```bash
# 1. Dependencies install করো
pip install -r requirements.txt

# 2. Flask app চালাও
python app.py
```

App চালু হলে browser এ `http://localhost:5000` যাও — তোমার bone fracture classifier web app দেখতে পাবে! প্রথমবার run করলে Google Drive থেকে model weights download হবে — একটু সময় লাগবে। এরপর থেকে cached weights ব্যবহার হবে, instantly model load হবে।

### সারসংক্ষেপ

এই সেকশনে আমরা সম্পূর্ণ Flask application তৈরি করলাম — Google Drive থেকে model weights download, model reconstruction, file upload handling, validation, preprocessing, prediction, এবং result rendering। Architecture সহজ: Client → Flask → Model → Response। সবচেয়ে গুরুত্বপূর্ণ বিষয় হলো model architecture exactly match করা training এর সাথে এবং preprocessing pipeline একই রাখা। পরবর্তী সেকশনে আমরা HTML template ও CSS styling দেখবো — কারণ একটা সুন্দর UI ছাড়া app টা তো আর incomplete!
