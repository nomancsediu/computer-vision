## HTML টেমপ্লেট ও স্টাইলিং

এই সেকশনে আমরা Flask app এর frontend তৈরি করবো — HTML template ও CSS styling। Flask Jinja2 templating engine ব্যবহার করে — এটা Python data কে HTML এর মধ্যে dynamically render করতে দেয়। আমাদের template এ থাকবে file upload form, error message display, prediction result display, এবং uploaded image preview।

### Jinja2 Templating — কিভাবে কাজ করে?

Flask এ `render_template()` function ব্যবহার করলে এটা automatically `templates/` folder এ search করে। Jinja2 template এ `{{ variable }}` syntax দিয়ে Python variable display করা যায়, আর `{% if condition %}` syntax দিয়ে logic লেখা যায়। এটা HTML এ Python এর power যোগ করে — কিন্তু browser এ pure HTML ই যায়, server side এ render হয়।

### Complete index.html

`templates/index.html` ফাইলের সম্পূর্ণ কোড:

```html
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bone Fracture Classifier — AI X-ray Analysis</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🦴 Bone Fracture Classifier</h1>
            <p class="subtitle">AI-powered X-ray Fracture Detection — VGG16 Transfer Learning</p>
        </div>

        <!-- Flash Messages — Error/Warning Display -->
        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
        <div class="flash-messages">
            {% for category, message in messages %}
            <div class="alert alert-{{ category }}">
                {{ message }}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endwith %}

        <!-- Upload Form -->
        <div class="upload-section">
            <h2>X-ray Image Upload করুন</h2>
            <form action="/predict" method="POST" enctype="multipart/form-data" id="upload-form">
                <div class="file-input-wrapper">
                    <input type="file" name="file" id="file-input" accept="image/*" required>
                    <label for="file-input" class="file-label">
                        📁 Choose X-ray Image
                    </label>
                    <span id="file-name" class="file-name">কোনো ফাইল সিলেক্ট করা হয়নি</span>
                </div>
                <button type="submit" class="predict-btn">
                    🔍 Analyze X-ray
                </button>
            </form>
            <p class="help-text">Supported formats: PNG, JPG, JPEG, GIF</p>
        </div>

        <!-- Prediction Result -->
        {% if prediction %}
        <div class="result-section">
            <h2>📋 Prediction Result</h2>
            <div class="result-card">
                <div class="result-class">
                    <span class="label">Fracture Type:</span>
                    <span class="value">{{ prediction }}</span>
                </div>
                <div class="result-confidence">
                    <span class="label">Confidence:</span>
                    <span class="value confidence">{{ confidence }}</span>
                </div>
                <div class="probabilities">
                    <h3>Class Probabilities:</h3>
                    {% for class_name, prob in all_probs.items() %}
                    <div class="prob-row">
                        <span class="prob-label">{{ class_name }}</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar" style="width: {{ '%.1f'|format(prob * 100) }}%"></div>
                        </div>
                        <span class="prob-value">{{ '%.2f'|format(prob * 100) }}%</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Image Preview -->
        {% if image_path %}
        <div class="preview-section">
            <h2>🖼️ Uploaded X-ray</h2>
            <div class="image-preview">
                <img src="{{ image_path }}" alt="Uploaded X-ray: {{ filename }}">
            </div>
            <p class="image-name">{{ filename }}</p>
        </div>
        {% endif %}

        <!-- Footer -->
        <div class="footer">
            <p>Bone Fracture Classifier — VGG16 Transfer Learning Model</p>
            <p>⚠️ এটি শুধুমাত্র educational purpose। চিকিৎসকীয় সিদ্ধান্তের জন্য ব্যবহার করবেন না।</p>
        </div>
    </div>

    <!-- JavaScript: File name display -->
    <script>
        const fileInput = document.getElementById('file-input');
        const fileName = document.getElementById('file-name');

        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                fileName.textContent = this.files[0].name;
            } else {
                fileName.textContent = 'কোনো ফাইল সিলেক্ট করা হয়নি';
            }
        });
    </script>
</body>
</html>
```

### HTML Template বিশ্লেষণ

#### enctype="multipart/form-data"

```html
<form action="/predict" method="POST" enctype="multipart/form-data">
```

ফাইল upload করার সময় `enctype="multipart/form-data"` অবশ্যই দিতে হবে! এটা না দিলে browser file এর data server এ পাঠাবে না — শুধু filename যাবে। এটা file upload form এর সবচেয়ে common mistake — ভুলে গেলে `request.files` empty থাকবে।

#### Flash Messages

```html
{% with messages = get_flashed_messages(with_categories=true) %}
{% if messages %}
    {% for category, message in messages %}
    <div class="alert alert-{{ category }}">{{ message }}</div>
    {% endfor %}
{% endif %}
{% endwith %}
```

`get_flashed_messages()` দিয়ে Flask এর flash messages display হয়। যখন ইউজার ভুল file upload করে বা কোনো error হয়, তখন `flash()` function দিয়ে message পাঠানো হয় এবং এখানে display হয়। `with_categories=true` দিলে message type (error, success, warning) ও পাওয়া যায় — CSS class dynamically set করা যায়।

#### Conditional Result Display

```html
{% if prediction %}
    <div class="result-section">...</div>
{% endif %}
```

Jinja2 তে `{% if %}` block দিয়ে conditional rendering হয়। প্রথমবার page load হলে `prediction` variable define থাকে না — তাই result section hidden থাকে। Prediction হওয়ার পর Flask থেকে `prediction` variable pass হলে এই section visible হয়। একটি template দিয়েই upload form ও result display হয়!

#### Probability Bar Chart

```html
<div class="prob-bar" style="width: {{ '%.1f'|format(prob * 100) }}%"></div>
```

Jinja2 filter `'|format'` দিয়ে number formatting করা যায়। `prob * 100` করে percentage এ convert করে CSS width হিসেবে set করা হয় — ফলে visual bar chart তৈরি হয়। JavaScript বা external library ছাড়াই simple CSS দিয়ে progress bar বানানো যায়!

### Complete style.css

`static/style.css` ফাইলের সম্পূর্ণ কোড:

```css
/* ============================================================
   Bone Fracture Classifier — Stylesheet
   ============================================================ */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    min-height: 100vh;
    color: #e0e0e0;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
.header {
    text-align: center;
    padding: 30px 20px;
    margin-bottom: 30px;
}

.header h1 {
    font-size: 2.5em;
    color: #00d4ff;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 1.1em;
    color: #8892b0;
}

/* Flash Messages */
.flash-messages {
    margin-bottom: 20px;
}

.alert {
    padding: 12px 20px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 0.95em;
}

.alert-error {
    background: rgba(255, 71, 87, 0.2);
    border: 1px solid #ff4757;
    color: #ff6b81;
}

.alert-success {
    background: rgba(46, 213, 115, 0.2);
    border: 1px solid #2ed573;
    color: #7bed9f;
}

/* Upload Section */
.upload-section {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 30px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.upload-section h2 {
    color: #ffffff;
    margin-bottom: 20px;
    font-size: 1.5em;
}

/* Custom File Input */
.file-input-wrapper {
    margin-bottom: 20px;
}

.file-input-wrapper input[type="file"] {
    display: none;
}

.file-label {
    display: inline-block;
    padding: 12px 30px;
    background: #0f3460;
    color: #00d4ff;
    border: 2px dashed #00d4ff;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1em;
    transition: all 0.3s ease;
}

.file-label:hover {
    background: #16213e;
    border-style: solid;
}

.file-name {
    display: block;
    margin-top: 10px;
    color: #8892b0;
    font-size: 0.9em;
}

/* Predict Button */
.predict-btn {
    padding: 14px 40px;
    background: linear-gradient(135deg, #00d4ff, #0099cc);
    color: #1a1a2e;
    border: none;
    border-radius: 8px;
    font-size: 1.1em;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}

.predict-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
}

.help-text {
    margin-top: 15px;
    font-size: 0.85em;
    color: #636e82;
}

/* Result Section */
.result-section {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 30px;
    border: 1px solid rgba(0, 212, 255, 0.3);
}

.result-section h2 {
    color: #00d4ff;
    margin-bottom: 20px;
}

.result-card {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 20px;
}

.result-class, .result-confidence {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.label {
    color: #8892b0;
    font-size: 1em;
}

.value {
    color: #ffffff;
    font-size: 1.2em;
    font-weight: bold;
}

.confidence {
    color: #2ed573;
}

/* Probability Bars */
.probabilities {
    margin-top: 20px;
}

.probabilities h3 {
    color: #cccccc;
    font-size: 1em;
    margin-bottom: 15px;
}

.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
}

.prob-label {
    width: 180px;
    font-size: 0.9em;
    color: #b0b0b0;
}

.prob-bar-container {
    flex: 1;
    height: 20px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    overflow: hidden;
    margin: 0 12px;
}

.prob-bar {
    height: 100%;
    background: linear-gradient(90deg, #00d4ff, #2ed573);
    border-radius: 10px;
    transition: width 0.5s ease;
    min-width: 2px;
}

.prob-value {
    width: 60px;
    text-align: right;
    font-size: 0.9em;
    color: #ffffff;
    font-weight: bold;
}

/* Image Preview */
.preview-section {
    text-align: center;
    margin-bottom: 30px;
}

.preview-section h2 {
    color: #00d4ff;
    margin-bottom: 15px;
}

.image-preview {
    display: inline-block;
    border-radius: 12px;
    overflow: hidden;
    border: 2px solid rgba(0, 212, 255, 0.3);
    max-width: 100%;
}

.image-preview img {
    display: block;
    max-width: 400px;
    max-height: 400px;
    object-fit: contain;
}

.image-name {
    margin-top: 10px;
    color: #636e82;
    font-size: 0.85em;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #636e82;
    font-size: 0.85em;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.footer p {
    margin-bottom: 5px;
}
```

### Flask কিভাবে Template Render করে

```python
# Home page — template কে variable ছাড়া render
@app.route('/')
def index():
    return render_template('index.html')

# Prediction — template কে data সহ render
@app.route('/predict', methods=['POST'])
def predict():
    return render_template(
        'index.html',
        prediction=predicted_class,      # {{ prediction }}
        confidence=confidence_pct,        # {{ confidence }}
        all_probs=all_probs,              # {% for class_name, prob in all_probs.items() %}
        image_path=filepath,              # {{ image_path }}
        filename=filename                 # {{ filename }}
    )
```

`render_template()` Flask কে `templates/` folder এ `index.html` খুঁজতে বলে। Keyword arguments হিসেবে পাস করা variables Jinja2 template এ `{{ variable_name }}` দিয়ে access করা যায়। প্রথমবার (GET request) কোনো variable পাস হয় না — তাই শুধু upload form দেখায়। Prediction এর পর (POST request) সব data পাস হয় — তখন result ও preview দেখায়।

### Static File Serving

```html
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
```

`url_for('static', filename='style.css')` দিয়ে Flask automatically `static/style.css` এর URL generate করে। Direct path (`/static/style.css`) দেওয়ার বদলে `url_for()` ব্যবহার করা best practice — কারণ যদি static folder এর location change হয়, সব link automatically update হবে। JavaScript, images, fonts — সব static file এভাবেই serve হয়।

### Responsive Design Basics

আমাদের CSS এ কিছু responsive design element আছে:

- `max-width: 800px` + `margin: 0 auto` — container center এ থাকে, বড় screen এ পুরো width নেয় না
- `max-width: 400px` image preview — বড় image ও reasonable size এ দেখায়
- `viewport` meta tag — mobile device এ properly scale হয়

আরও better responsive design এর জন্য CSS media query ব্যবহার করা যায়:

```css
@media (max-width: 600px) {
    .header h1 {
        font-size: 1.8em;
    }
    .prob-row {
        flex-direction: column;
        align-items: flex-start;
    }
    .prob-label {
        width: auto;
        margin-bottom: 5px;
    }
}
```

### সারসংক্ষেপ

এই সেকশনে আমরা সম্পূর্ণ frontend তৈরি করলাম — Jinja2 HTML template দিয়ে dynamic content rendering, CSS দিয়ে professional dark theme styling, flash message display, file upload form with custom styling, probability bar chart, এবং image preview। সবচেয়ে গুরুত্বপূর্ণ জিনিসগুলো: `enctype="multipart/form-data"` file upload এ বাধ্যতামূলক, `url_for()` দিয়ে static file reference করা best practice, এবং Jinja2 conditional block দিয়ে একটি template এ upload form ও result display দুটোই করা যায়। এখন তোমার Flask app frontend + backend সম্পূর্ণ — browser এ চালাও এবং X-ray image test করো!
