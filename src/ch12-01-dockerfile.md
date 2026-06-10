## Dockerfile ও কন্টেইনার বিল্ড

এই সেকশনে আমরা Docker এর fundamentals বুঝবো এবং আমাদের Flask app এর জন্য complete Dockerfile লিখবো। Docker হলো containerization platform — এটা তোমার application কে একটি self-contained unit এ pack করে যেটা যেকোনো environment এ consistently run হয়।

### Docker Concepts — Image vs Container

Docker এ দুটি core concept:

- **Image:** এটা একটি read-only template — তোমার application এর code, dependencies, OS libraries সব কিছুর blueprint। Python image, Node image — এগুলো base image। তুমি তোমার application এর উপর custom layers add করে নতুন image বানাও। Image হলো class, container হলো object — OOP এর analogy তে!

- **Container:** Image থেকে create হওয়া running instance। একটি image থেকে অনেক container চালানো যায়। Container isolated — নিজের filesystem, network, process space আছে। একটা container এ TensorFlow আর অন্যটায় PyTorch — কোনো conflict নেই!

### কেন Docker? — Real-World Problems

Docker ছাড়া যে সমস্যাগুলো হয়:

1. **"It works on my machine" syndrome:** Developer এর machine এ run হয়, production server এ crash করে — কারণ environment আলাদা
2. **Dependency hell:** Project A তে TensorFlow 2.19, Project B তে TensorFlow 2.15 — same machine এ conflict
3. **Onboarding time:** নতুন developer join করলে environment setup এ ২-৩ দিন কাটে
4. **Deployment inconsistency:** Development, staging, production — তিনটিতে আলাদা behavior

Docker এই সব সমাধান করে — "build once, run anywhere"!

### Complete Dockerfile

আমাদের bone fracture classifier Flask app এর জন্য Dockerfile:

```dockerfile
# ============================================================
# Bone Fracture Classifier — Docker Image
# ============================================================

# Base image: Python 3.12 on Debian
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p static/uploads

# Expose Flask port
EXPOSE 5000

# Run the application
ENTRYPOINT ["python3", "app.py"]
```

### Dockerfile বিশ্লেষণ — প্রতিটি Instruction

#### FROM — Base Image

```dockerfile
FROM python:3.12-slim
```

`FROM` দিয়ে base image specify হয়। `python:3.12-slim` হলো official Python image এর lightweight version — Debian based, Python 3.12 pre-installed। `slim` variant শুধু essential packages রাখে, `full` variant এর তুলনায় অনেক ছোট (~150MB vs ~900MB)। ML app এর জন্য `slim` ই sufficient — TensorFlow নিজেই যা দরকার সব install করে নেবে।

#### WORKDIR — Working Directory

```dockerfile
WORKDIR /app
```

Container এর ভিতর `/app` directory কে working directory হিসেবে set করে। এরপর থেকে সব `RUN`, `COPY`, `CMD` command এই directory তে execute হবে। `cd /app` এর equivalent কিন্তু persistent — প্রতিটি layer এ working directory same থাকে।

#### RUN apt-get — System Dependencies

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    awscli \
    && rm -rf /var/lib/apt/lists/*
```

`awscli` install করা হচ্ছে — যদি ভবিষ্যতে model weights AWS S3 থেকে download করতে হয়। `--no-install-recommends` দিয়ে unnecessary packages বাদ দেওয়া হয় — image size ছোট হয়। `rm -rf /var/lib/apt/lists/*` দিয়ে apt cache clean করা হয় — এটা Docker best practice, নাহলে cache layer image size বাড়িয়ে দেয়।

#### COPY requirements.txt — Layer Caching Trick

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

এটা Docker এর সবচেয়ে important optimization! কেন পুরো code একসাথে copy করছি না? Docker layer caching এর কারণে। Docker প্রতিটি instruction কে একটি layer হিসেবে save করে। যদি কোনো layer change না হয়, Docker সেটা rebuild করে না — cache থেকে নেয়।

- `requirements.txt` change হলে: pip install আবার run হবে (5-10 minutes!)
- `app.py` change হলে: pip install cache থেকে নেওয়া হবে (instant!)

তাই requirements.txt আলাদাভাবে আগে copy করে pip install করা হয় — তারপর বাকি code copy হয়। Code change হলে dependency installation skip হয় — development এ অনেক সময় বাঁচে!

#### EXPOSE — Port Documentation

```dockerfile
EXPOSE 5000
```

`EXPOSE` দিয়ে container কোন port এ listen করবে তা document করা হয়। এটা actually port publish করে না — শুধু documentation ও hint। Actual port mapping `docker run` command এ `-p` flag দিয়ে হয়। কিন্তু `EXPOSE` লিখা good practice — অন্য developer বুঝতে পারবে app কোন port ব্যবহার করে।

#### ENTRYPOINT — Application Run

```dockerfile
ENTRYPOINT ["python3", "app.py"]
```

`ENTRYPOINT` দিয়ে container start হওয়ার সময় কোন command execute হবে তা specify করা হয়। `CMD` এর বিপরীতে `ENTRYPOINT` override করা কঠিন — container টা specifically এই app এর জন্যই তৈরি, অন্য কিছু run করার সুযোগ নেই। Exec form `["python3", "app.py"]` ব্যবহার করা best practice — shell form (`python3 app.py`) এর চেয়ে signal handling ভালো।

### Docker Image Build করা

```bash
# Docker image build করো
docker build -t bone-fracture-classifier .
```

- `docker build` — image build করার command
- `-t bone-fracture-classifier` — image কে একটি নাম (tag) দেওয়া
- `.` — current directory তে Dockerfile আছে

Build process: Docker Dockerfile এর প্রতিটি instruction step by step execute করে একটি layer create করে। প্রথমবার build করলে 5-10 মিনিট লাগতে পারে (pip install এর জন্য)। পরবর্তী build গুলো fast হবে — cache থেকে layers নেওয়া হবে।

### Docker Container Run করা

```bash
# Container run করো
docker run -p 5000:5000 bone-fracture-classifier
```

- `docker run` — image থেকে container create ও start করে
- `-p 5000:5000` — port mapping: host:container
- `bone-fracture-classifier` — image এর নাম

#### Port Mapping Explained

```
Host Machine (your computer)          Docker Container
    |                                      |
    |  localhost:5000  ----->  5000        |
    |                                      |
```

`-p 5000:5000` এর মানে: host machine এর port 5000 কে container এর port 5000 এর সাথে map করো। তুমি browser এ `localhost:5000` গেলে request container এর port 5000 তে forward হবে। প্রথম 5000 হলো host port, দ্বিতীয় 5000 হলো container port। এগুলো আলাদা ও হতে পারে:

```bash
# Host এর 8080 port কে container এর 5000 তে map
docker run -p 8080:5000 bone-fracture-classifier
# এখন browser এ localhost:8080 যাও
```

Container run হলে তুমি browser এ `http://localhost:5000` গিয়ে app টা দেখতে পাবে — ঠিক যেভাবে locally Flask run করলে দেখতে!

### Useful Docker Commands

```bash
# Running containers দেখো
docker ps

# সব containers দেখো (stopped গুলো সহ)
docker ps -a

# Container stop করো
docker stop <container_id>

# Container remove করো
docker rm <container_id>

# Image গুলো দেখো
docker images

# Container এর logs দেখো
docker logs <container_id>

# Container এর ভিতরে shell এ যাও
docker exec -it <container_id> /bin/bash
```

### Docker Compose — Multi-Service Orchestration

যদি তোমার app এ একাধিক service থাকে (Flask app + PostgreSQL database + Redis cache), Docker Compose দিয়ে সব একসাথে manage করা যায়:

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./static/uploads:/app/static/uploads
    environment:
      - FLASK_ENV=production
    restart: unless-stopped

  # ভবিষ্যতে database add করতে চাইলে:
  # db:
  #   image: postgres:15
  #   environment:
  #     POSTGRES_DB: fracture_db
  #     POSTGRES_PASSWORD: secret
```

```bash
# Compose দিয়ে সব service start
docker-compose up -d

# সব service stop
docker-compose down

# Logs দেখো
docker-compose logs -f
```

আমাদের single-service Flask app এর জন্য Docker Compose strictly necessary নয়, কিন্তু যখন app grow করবে এবং database, cache, queue service add হবে — তখন Compose অপরিহার্য হয়ে যাবে।

### সারসংক্ষেপ

এই সেকশনে আমরা Docker fundamentals শিখলাম — image vs container concept, "it works on my machine" problem এর solution, এবং complete Dockerfile লিখলাম। Key takeaways: `python:3.12-slim` base image ব্যবহার, requirements.txt আগে copy করে pip install (layer caching optimization), `EXPOSE 5000` documentation, `ENTRYPOINT` দিয়ে app execute। Docker build + Docker run দিয়ে locally test করা যায়। Docker Compose দিয়ে multi-service orchestration করা যায়। পরবর্তী সেকশনে আমরা production deployment শিখবো — Flask dev server নয়, Gunicorn WSGI server দিয়ে আসল production setup!
