## বাইনারি ইমেজ ক্লাসিফিকেশন

এই সেকশনে আমরা augmented data দিয়ে একটি সম্পূর্ণ binary image classification pipeline বানাবো  model design থেকে শুরু করে training, evaluation, এবং নতুন image তে prediction পর্যন্ত। Binary classification মানে হলো image কে দুটি class এর একটিতে classify করা  যেমন cat vs dog, fracture vs normal, spam vs not-spam। এটি computer vision এর সবচেয়ে common task গুলোর একটি।

### CNN Model Architecture ডিজাইন

Binary classification এর জন্য আমরা একটি custom CNN architecture বানাবো। Architecture টি হবে: Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Dense → Output। চলো দেখি কেন এই design:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    # Block 1: 32 filters  low-level features (edges, corners)
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    # Block 2: 64 filters  mid-level features (textures, patterns)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Block 3: 128 filters  high-level features (object parts)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten: 3D feature maps → 1D vector
    Flatten(),

    # Fully connected layers
    Dense(128, activation='relu'),
    Dropout(0.5),                    # 50% neurons randomly off  reduces overfitting
    Dense(64, activation='relu'),
    Dropout(0.3),

    # Output layer: 1 neuron + sigmoid = binary classification
    Dense(1, activation='sigmoid')
])

model.summary()
```

Architecture design এর পেছনে কিছু logic:

- **Filter সংখ্যা বাড়ানো (32→64→128):** Early layers simple features detect করে (edges, corners)  তাই কম filter দরকার। Later layers complex features detect করে (object parts)  তাই বেশি filter দরকার। এটি CNN architecture এর standard practice।

- **Spatial dimension কমানো:** প্রতি MaxPooling2D layer spatial dimension অর্ধেক করে দেয়: 224→112→56→28। Feature map ছোট হলেও feature এর depth (filter সংখ্যা) বাড়ে  তাই information compress হয় না, বরং আরও abstract হয়।

- **Output layer: 1 neuron + sigmoid:** Binary classification এ শুধু একটি probability দরকার  class 1 হওয়ার probability। Sigmoid 0 থেকে 1 এর মধ্যে value দেয়। 0.5 এর উপরে হলে class 1, নিচে হলে class 0। এটি 2 neuron + softmax এর চেয়ে computationally efficient।

- **Dropout:** Training এর সময় random neuron off করে দেয়  এতে model কোনো specific neuron এর উপর over-rely করতে পারে না, overfitting কমে। Dropout শুধু training এ apply হয়, inference এ নয়।

### Model Compile করা

```python
model.compile(
    optimizer='adam',                  # Adaptive learning rate optimizer
    loss='binary_crossentropy',        # Binary classification loss function
    metrics=['accuracy']               # Track accuracy during training
)
```

**কেন binary_crossentropy?** Binary classification এ sigmoid output থাকে  probability value 0 থেকে 1। `binary_crossentropy` এই probability কে true label এর সাথে compare করে loss calculate করে। সূত্র: `-[y*log(p) + (1-y)*log(1-p)]`। যদি prediction correct ও confident হয়, loss কম; prediction wrong হলে, loss অনেক বেশি। এটি binary classification এর standard loss function।

**কেন adam?** Adam optimizer automatically learning rate adjust করে  শুরুতে বড় step নেয় (fast convergence), পরে ছোট step নেয় (fine-tuning)। বেশিরভাগ ক্ষেত্রে default `lr=0.001` ভালো কাজ করে, extra tuning লাগে না।

### Data Preparation  ImageDataGenerator ও flow_from_directory

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data generator  with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Validation data generator  only rescale
val_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directory
train_generator = train_datagen.flow_from_directory(
    'dataset/train',              # Training folder path
    target_size=(224, 224),       # Resize to 224x224
    batch_size=32,                # 32 image per batch
    class_mode='binary'           # Binary classification (0 or 1)
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# View class label mapping
print(f"Class indices: {train_generator.class_indices}")
# Example output: {'cats': 0, 'dogs': 1}
```

**class_mode='binary':** এটি image গুলোকে 0 বা 1 label দেয়। Folder name alphabetical order এ label assign হয়  যেমন `cats` folder → 0, `dogs` folder → 1। তাই `class_indices` check করে নিশ্চিত হও কোন class কোন label পেয়েছে।

**Directory structure:** `flow_from_directory()` নিচের মতো folder structure expect করে:

```
dataset/
├── train/
│   ├── class_a/    ← class 0
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── class_b/    ← class 1
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── val/
    ├── class_a/
    └── class_b/
```

### Training  model.fit() দিয়ে ট্রেইনিং

```python
# Training
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)
```

`steps_per_epoch` হিসাব করা হয়: total training samples ÷ batch_size। যেমন 640 training image ও batch_size=32 হলে, steps_per_epoch = 640/32 = 20। অর্থাৎ প্রতি epoch এ 20 batch process হবে, সব image একবার করে দেখা হবে।

### Training Curve প্লট করা

Training শেষে loss ও accuracy curve plot করে model এর training quality check করতে হবে:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy curve
axes[0].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss curve
axes[1].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('binary_classification_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

**কী দেখবে curve এ:**
- Training ও validation accuracy দুটোই বাড়ছে এবং কাছাকাছি → **Good fit** ✅
- Training accuracy অনেক বেশি, validation accuracy অনেক কম → **Overfitting** ❌
- দুটোই কম → **Underfitting** ❌

Augmentation use করার কারণে overfitting অনেক কম হবে  কারণ model প্রতি epoch এ নতুন augmented image দেখে, একই image বারবার দেখে না।

### নতুন ইমেজে Prediction

Trained model দিয়ে নতুন, unseen image তে prediction করা যাক:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_binary(img_path, model, target_size=(224, 224), threshold=0.5):
    """
    Binary classification prediction function।

    Parameters:
        img_path: file path of the image
        model: trained Keras model
        target_size: resize dimension
        threshold: classification threshold (default 0.5)

    Returns:
        predicted_class: class name
        confidence: prediction confidence
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0                        # Normalize
    img_array = np.expand_dims(img_array, axis=0)        # Batch dimension add

    # Prediction
    probability = model.predict(img_array, verbose=0)[0][0]   # Sigmoid output

    # Class mapping (obtained from train_generator.class_indices)
    class_names = {0: 'cats', 1: 'dogs'}

    if probability > threshold:
        predicted_class = class_names[1]
        confidence = probability
    else:
        predicted_class = class_names[0]
        confidence = 1 - probability

    return predicted_class, float(confidence)

# Prediction example
result, conf = predict_binary('test_cat.jpg', model)
print(f"Prediction: {result} (Confidence: {conf:.2%})")

result, conf = predict_binary('test_dog.jpg', model)
print(f"Prediction: {result} (Confidence: {conf:.2%})")
```

**Threshold adjustment:** Default threshold 0.5, কিন্তু task এর উপর ভিত্তি করে এটি change করা যায়। Medical diagnosis এ false negative খুব dangerous  তাই threshold কমানো যায় (যেমন 0.3) যাতে কম confidence তেও positive class predict হয়। আবার spam detection এ false positive বেশি problematic  তাই threshold বাড়ানো যায় (যেমন 0.7)।

### মডেল Save করা

```python
from tensorflow.keras.models import save_model, load_model

# Save model
model.save('binary_classifier.h5')
print("Model saved: binary_classifier.h5")

# Load model
loaded_model = load_model('binary_classifier.h5')
print("Model loaded!")

# Verify loaded model
result, conf = predict_binary('test_cat.jpg', loaded_model)
print(f"Loaded model prediction: {result} (Confidence: {conf:.2%})")
```

### Complete End-to-End Code

সব একসাথে  complete pipeline একটি script এ:

```python
# ========================================================
# Binary Image Classification  Complete Pipeline
# ========================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image

# --- 1. Data Preparation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

print(f"Class indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# --- 2. Model Building ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- 3. Training ---
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# --- 4. Training Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], 'b-', label='Train Acc')
axes[0].plot(history.history['val_accuracy'], 'r-', label='Val Acc')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], 'b-', label='Train Loss')
axes[1].plot(history.history['val_loss'], 'r-', label='Val Loss')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 5. Save Model ---
model.save('binary_classifier.h5')
print("Model saved: binary_classifier.h5")

# --- 6. Prediction Function ---
def predict_image(img_path, model, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    probability = model.predict(img_array, verbose=0)[0][0]

    class_names = {0: 'class_a', 1: 'class_b'}  # Put your own class names
    if probability > 0.5:
        return class_names[1], float(probability)
    else:
        return class_names[0], float(1 - probability)

# Prediction example
# result, conf = predict_image('test.jpg', model)
# print(f"Prediction: {result} (Confidence: {conf:.2%})")
```

### সারসংক্ষেপ

এই সেকশনে আমরা একটি সম্পূর্ণ binary image classification pipeline বানালাম। CNN model design (Conv2D → MaxPool → Dense → sigmoid output), compile (adam + binary_crossentropy), augmented data দিয়ে training, training curve evaluation, নতুন image তে prediction  সব কভার করলাম। এই pipeline টি যেকোনো binary classification problem এ reuse করা যাবে  শুধু dataset change করো আর class names update করো। পরের চ্যাপ্টারে আমরা শিখবো কিভাবে প্রিট্রেইন্ড model ব্যবহার করে আরও ভালো accuracy পাওয়া যায়  transfer learning!
