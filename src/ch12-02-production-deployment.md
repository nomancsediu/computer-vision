## প্রোডাকশন ডিপ্লয়মেন্ট

এই সেকশনে আমরা শিখবো কিভাবে আমাদের Flask app কে প্রোডাকশন এ ডিপ্লয় করতে হয় — Gunicorn WSGI server, cloud platforms, security best practices, এবং monitoring। এটা হলো সেই পার্ট যেখানে "works on my machine" থেকে "works for everyone" এ রূপান্তর হয়!

### Flask Dev Server — প্রোডাকশন এ নয়!

Flask এর built-in development server (`app.run()`) শুধু development এর জন্য। Flask documentation নিজেই বলে:

> "While lightweight and easy to use, Flask's built-in server is not suitable for production as it doesn't scale well."

কেন নয়?

- **Single-threaded:** একসাথে একটাই request handle করতে পারে — 10 জন একসাথে request পাঠালে 9 জন অপেক্ষা করবে
- **No security hardening:** Debug mode enabled থাকলে arbitrary code execute করা যায় — massive security risk
- **No load balancing:** CPU multiple cores থাকলেও একটা core ই ব্যবহার হবে
- **Unstable:** Memory leak, crash recovery নেই

প্রোডাকশন এ চাই WSGI server — Gunicorn, uWSGI, বা Waitress।

### Gunicorn — Production WSGI Server

Gunicorn (Green Unicorn) হলো Python এর সবচেয়ে popular WSGI HTTP server। এটা pre-fork worker model ব্যবহার করে — multiple worker process একসাথে request handle করতে পারে।

#### Gunicorn Install ও Run

```bash
# Install Gunicorn
pip install gunicorn

# 4 worker দিয়ে run করো
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Command breakdown:
- `-w 4` — 4টি worker process (request একসাথে handle করবে)
- `-b 0.0.0.0:5000` — সব network interface এর 5000 port এ bind করো
- `app:app` — `app.py` ফাইল এর `app` variable (Flask instance)

#### Worker Count Formula

কতগুলো worker দরকার? Gunicorn documentation suggest:

```
workers = (2 × CPU_cores) + 1
```

4-core machine এ: `(2 × 4) + 1 = 9` workers। কিন্তু ML model prediction memory-intensive — তাই বেশি worker দিলে memory শেষ হয়ে যাবে! ML app এর জন্য 2-4 worker ই reasonable — model প্রতিটি worker এ load হবে (memory per worker ~2-4GB for VGG16)।

#### Gunicorn Configuration File

```python
# gunicorn_config.py
import multiprocessing

# Worker count (ML model: use fewer workers due to memory)
workers = 4

# Worker class
worker_class = 'sync'

# Bind address
bind = '0.0.0.0:5000'

# Timeout (ML prediction takes longer)
timeout = 120

# Maximum concurrent requests per worker
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
```

```bash
# Config file দিয়ে run
gunicorn -c gunicorn_config.py app:app
```

### Dockerfile Update — Gunicorn Entry

```dockerfile
# Updated Dockerfile for production
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Gunicorn added)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p static/uploads

# Non-root user for security
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 5000

# Gunicorn entry point (production WSGI server)
ENTRYPOINT ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

**গুরুত্বপূর্ণ changes:**
- `gdown` ও `gunicorn` requirements.txt তে add করতে ভুলো না!
- `--timeout 120` — model prediction এ বেশি সময় লাগতে পারে, default 30 second timeout হলে kill হয়ে যাবে
- `useradd -m appuser` + `USER appuser` — non-root user তে run (security best practice)

### Cloud Deployment Options

Container ready হলে এখন সেটা cloud এ deploy করতে হবে। বেশ কিছু option আছে:

#### AWS ECS/Fargate

```bash
# ECR তে image push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com
docker tag bone-fracture-classifier <account_id>.dkr.ecr.us-east-1.amazonaws.com/bone-fracture-classifier
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/bone-fracture-classifier

# ECS Fargate তে deploy — serverless, no EC2 management
aws ecs create-service --cli-input-json file://ecs-service.json
```

AWS ECS (Elastic Container Service) production grade container orchestration। Fargate হলো serverless option — তুমি server manage করো না, শুধু container define করো, AWS run করে দেয়। Auto-scaling, load balancing সব built-in।

#### Google Cloud Run

```bash
# gcloud CLI দিয়ে deploy
gcloud run deploy bone-fracture-classifier \
    --source . \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --allow-unauthenticated
```

Google Cloud Run সবচেয়ে simple deployment option! একটা command দিয়ে deploy — Dockerfile থেকে automatically image build হয়, container run হয়, HTTPS URL পাও। Serverless — request না আসলে cost নেই। ML model এর জন্য `--memory 4Gi` দিতে হবে (VGG16 বেশ memory খায়)।

#### Azure Container Instances

```bash
# Azure CLI দিয়ে deploy
az container create \
    --resource-group myResourceGroup \
    --name bone-fracture-classifier \
    --image bone-fracture-classifier \
    --ports 5000 \
    --cpu 2 \
    --memory 4
```

Azure Container Instances (ACI) ও simple — quick deployment, per-second billing। Kubernetes এর complexity ছাড়াই container run করা যায়।

#### Heroku

```bash
# Heroku deploy
heroku create bone-fracture-classifier
heroku container:login
heroku container:push web
heroku container:release web
heroku open
```

Heroku সবচেয়ে beginner-friendly platform — Git push করলেই deploy! Container deployment ও support করে। Free tier নেই এখন, কিন্তু minimal cost এ শুরু করা যায়।

### Security Best Practices

Production deployment এ security অত্যন্ত গুরুত্বপূর্ণ — বিশেষ করে medical AI app হলে:

#### 1. Model Files Commit করবে না

```bash
# .gitignore তে add করো
model/
*.h5
*.pkl
*.hdf5
static/uploads/
```

Model weights file গুলো Git এ commit করবে না — এগুলো Google Drive, AWS S3 বা অন্য cloud storage থেকে download করবে। কারণ: (a) file বড় — repository size বাড়িয়ে দেবে, (b) model তে training data এর information leak থাকতে পারে, (c) version control system এ binary file রাখা bad practice।

#### 2. Upload Validation শক্ত করো

```python
import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_upload(file):
    """Strict file validation for security."""
    # Extension check
    if not allowed_file(file.filename):
        raise ValueError("Invalid file extension")

    # Secure filename — prevent directory traversal
    filename = secure_filename(file.filename)

    # File size check
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        raise ValueError("File too large")

    # MIME type verification
    mime_type = file.content_type
    if not mime_type.startswith('image/'):
        raise ValueError("Invalid MIME type")

    return filename
```

`secure_filename()` দিয়ে malicious filename (যেমন `../../etc/passwd`) sanitize হয়। MIME type check দিয়ে renamed file detect করা যায়। File size limit দিয়ে DoS attack prevent হয়।

#### 3. CORS Policy

```python
from flask_cors import CORS

# Specific origin allow করো
CORS(app, origins=['https://yourdomain.com'])
```

Cross-Origin Resource Sharing (CORS) নিয়ন্ত্রণ করে কোন domain থেকে তোমার API access হবে। `origins='*'` (সব কে allow) production এ কখনো করবে না — specific domain গুলোই allow করো।

#### 4. Environment Variables for Secrets

```python
import os

# Environment variable থেকে sensitive data নাও
app.secret_key = os.environ.get('SECRET_KEY', 'dev-fallback-key')
model_url = os.environ.get('MODEL_WEIGHTS_URL')
debug_mode = os.environ.get('FLASK_ENV') == 'development'
```

```bash
# Docker run এ environment variable pass
docker run -p 5000:5000 \
    -e SECRET_KEY=your_production_secret \
    -e MODEL_WEIGHTS_URL=https://drive.google.com/... \
    -e FLASK_ENV=production \
    bone-fracture-classifier
```

Secret key, API key, model URL — কিছুই code এ hardcode করবে না! `.env` file বা environment variable ব্যবহার করো। `.env` file ও Git এ commit করবে না।

#### 5. Non-Root User তে Run

```dockerfile
# Docker image তে non-root user create
RUN useradd -m -r appuser
USER appuser
```

Container default root user তে run হয় — এটা security risk। Attacker container এ access পেলে root privilege পাবে! `USER appuser` দিলে restricted permission এ run হবে — damage limited থাকবে।

### Monitoring ও Logging

Production এ app চালু থাকলে জানা দরকার কী হচ্ছে — error, performance, usage pattern:

```python
import logging
from logging.handlers import RotatingFileHandler

# Logging configuration
log_handler = RotatingFileHandler(
    'app.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))

app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info(f"Prediction request from {request.remote_addr}")
    try:
        # ... prediction logic ...
        app.logger.info(f"Prediction: {predicted_class} ({confidence:.2%})")
        return render_template(...)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        flash('প্রেডিকশনে সমস্যা হয়েছে।', 'error')
        return redirect(url_for('index'))
```

**RotatingFileHandler:** Log file 10MB হলে automatically rotate হয় — 5টি backup file রাখে। এটা ছাড়া log file infinite বড় হয়ে disk full করে দিতে পারে!

**Logging levels:**
- `DEBUG` — detailed debugging info
- `INFO` — general events (prediction made, user visited)
- `WARNING` — potential problems
- `ERROR` — errors that don't crash the app
- `CRITICAL` — app can't continue

Production এ `INFO` level appropriate — `DEBUG` তে অতিরিক্ত log হবে, performance impact হবে।

### সারসংক্ষেপ

এই সেকশনে আমরা production deployment এর সম্পূর্ণ workflow শিখলাম। Flask dev server প্রোডাকশন এ ব্যবহার করা যাবে না — Gunicorn WSGI server দিয়ে 4 worker এ run করতে হবে। Dockerfile এ Gunicorn entry point ও non-root user configuration add করতে হবে। Cloud deployment options: AWS ECS/Fargate, Google Cloud Run, Azure Container Instances, Heroku — প্রতিটার নিজস্ব advantage। Security best practices: model files commit না করা, upload validation শক্ত করা, CORS policy, environment variables, non-root user। Monitoring ও logging দিয়ে app এর health track করা। এই সব মিলিয়ে তোমার bone fracture classifier app production-ready! 🚀
