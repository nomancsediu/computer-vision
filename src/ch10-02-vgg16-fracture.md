## VGG16 দিয়ে বোন ফ্র্যাকচার ক্লাসিফিকেশন

এই সেকশনে আমরা transfer learning এর সবচেয়ে practical application দেখবো — ImageNet pretrained **VGG16** model কে feature extractor হিসেবে ব্যবহার করে **bone fracture classification** করবো। আমাদের task: X-ray image দেখে নির্ধারণ করা ফ্র্যাকচারটি **Oblique** নাকি **Spiral**। এটি একটি domain-specific medical imaging task — ImageNet এ এর কোনো class নেই, তাই pretrained model সরাসরি predict করতে পারবে না। কিন্তু transfer learning দিয়ে আমরা VGG16 এর learned features কাজে লাগাবো!

### Step 1: VGG16 Base Model Load করা

প্রথমে VGG16 model load করবো — কিন্তু শুধু convolutional base, classification head ছাড়া:

```python
from tensorflow.keras.applications import VGG16

# VGG16 base model load — classification head ছাড়া
base_model = VGG16(
    weights='imagenet',          # ImageNet pretrained weights
    include_top=False,           # Fully connected layers (top) বাদ
    input_shape=(224, 224, 3)    # আমাদের input size
)

# Base model summary দেখা
base_model.summary()
```

**include_top=False কেন?** ImageNet VGG16 এর fully connected layers 1000 class এর জন্য trained — আমাদের দরকার মাত্র 2 class (Oblique vs Spiral)। তাই original classification head বাদ দিয়ে শুধু convolutional feature extraction layers রাখি — নিচে আমাদের নিজস্ব custom head add করবো।

**input_shape=(224, 224, 3):** VGG16 default ভাবে 224×224 RGB image expect করে। আমরা এটাই ব্যবহার করবো।

### Step 2: Base Model এর Layers Freeze করা

Freezing মানে হলো — training এর সময় এই layer গুলোর weights update হবে না। ImageNet এ শেখা features ঠিক রেখে, শুধু নতুন head train হবে:

```python
# সব layer freeze করা
for layer in base_model.layers:
    layer.trainable = False

# Verify — কোন layer trainable আর কোনটি নয়
print(f"Total layers: {len(base_model.layers)}")
print(f"Trainable layers: {sum(1 for l in base_model.layers if l.trainable)}")
print(f"Frozen layers: {sum(1 for l in base_model.layers if not l.trainable)}")
```

**Layer freeze কেন দরকার?** দুটি কারণ:

1. **Pretrained features রক্ষা:** ImageNet এ শেখা general features (edges, textures) নষ্ট হয়ে যাবে যদি training এ random update হয়। Freeze করলে এই features intact থাকে।

2. **Training efficiency:** VGG16 তে ~14.7M parameters। সব trainable হলে training অনেক slow হবে এবং ছোট dataset এ overfitting হবে। Freeze করলে শুধু ~3.3M parameter train হয় — fast ও safe!

### Step 3: Custom Classification Head Add করা

Frozen base model এর উপরে আমাদের নিজস্ব classification layer যোগ করবো:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Base model এর output নিই
x = base_model.output

# Flatten: 3D feature maps → 1D vector
x = Flatten()(x)

# Fully connected layers
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer: 2 class → 2 neuron + softmax
predictions = Dense(2, activation='softmax')(x)

# Final model: base_model.input → predictions
model = Model(inputs=base_model.input, outputs=predictions)

# Parameter count
total_params = model.count_params()
trainable_params = sum(np.prod(w.shape) for w in model.trainable_weights)
frozen_params = total_params - trainable_params

print(f"\n{'='*50}")
print(f"Total parameters:     {total_params:>12,}")
print(f"Trainable parameters: {trainable_params:>12,}")
print(f"Frozen parameters:    {frozen_params:>12,}")
print(f"{'='*50}")
# Output approximately:
# Total parameters:     ~18,000,000
# Trainable parameters: ~3,300,000  (custom head only)
# Frozen parameters:    ~14,700,000 (VGG16 base)
```

**কেন Dense(2, activation='softmax')?** আমাদের 2টি class: Oblique ও Spiral। Multi-class classification এ `softmax` use হয় — এটি দুটি class এর probability দেয় যার যোগফল 1। `[0.85, 0.15]` মানে 85% Oblique, 15% Spiral। Binary classification এর মতো 1 neuron + sigmoid ও ব্যবহার করা যায়, কিন্তু 2 neuron + softmax interpretation সহজ।

### Step 4: Data Preparation — ImageDataGenerator + Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,            # X-ray তে বেশি rotation logical নয়
    zoom_range=0.15,              # Slight zoom acceptable
    width_shift_range=0.1,        # Small shift
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.9, 1.1],  # Slight brightness variation
    fill_mode='nearest',
    horizontal_flip=False,        # ❌ X-ray flip করা যাবে না!
    vertical_flip=False           # ❌ Vertical flip ও নয়!
)

# Validation data — rescale only
val_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    'fracture_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'      # Multi-class: 2 neuron output
)

val_generator = val_datagen.flow_from_directory(
    'fracture_dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(f"Class indices: {train_generator.class_indices}")
# Output: {'Oblique': 0, 'Spiral': 1}
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
```

**গুরুত্বপূর্ণ:** X-ray image তে `horizontal_flip=False` ও `vertical_flip=False` রাখা হয়েছে! কারণ bone fracture এর orientation medically important — flip করলে fracture type এর appearance change হতে পারে। Rotation ও কম রাখা হয়েছে (15°) — বেশি rotation unrealistic। এটি পূর্বের চ্যাপ্টারে শেখা "task-specific augmentation" এর perfect example!

### Step 5: Model Compile

```python
import numpy as np
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.001),    # Standard learning rate
    loss='categorical_crossentropy',         # Multi-class loss (2 class)
    metrics=['accuracy']
)
```

**categorical_crossentropy:** Multi-class classification (2+ class) এ এটি standard loss function। Softmax output probability গুলোকে true label এর সাথে compare করে। Binary classification (1 neuron + sigmoid) এ `binary_crossentropy` use হতো, কিন্তু আমরা 2 neuron + softmax ব্যবহার করছি তাই `categorical_crossentropy` দরকার।

### Step 6: Training — 10 Epochs

```python
# Training
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)
```

শুধু 10 epoch! কারণ transfer learning এ base model ইতিমধ্যে features শিখে আছে — শুধু custom head train করলেই হয়। Scratch training এ 50-100 epoch লাগতে পারে, কিন্তু transfer learning এ 5-15 epoch এই good accuracy আসে!

### Step 7: Evaluation ও Model Save

```python
import matplotlib.pyplot as plt

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], 'b-', label='Train Acc', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
axes[0].set_title('VGG16 Transfer Learning — Accuracy', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[1].set_title('VGG16 Transfer Learning — Loss', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vgg16_fracture_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Model save
model.save('model_vgg16.h5')
print("Model saved: model_vgg16.h5")
```

### Step 8: নতুন X-ray Image তে Prediction

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Model load
model = load_model('model_vgg16.h5')

# Class names
class_names = {0: 'Oblique Fracture', 1: 'Spiral Fracture'}

def predict_fracture(img_path, model):
    """
    X-ray image তে bone fracture type predict করে।
    """
    # Image load ও preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    predicted_class = class_names[class_idx]

    # Detailed result
    print(f"\n{'='*40}")
    print(f"Image: {img_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nClass Probabilities:")
    for idx, prob in enumerate(predictions[0]):
        print(f"  {class_names[idx]}: {prob:.2%}")
    print(f"{'='*40}")

    return predicted_class, float(confidence)

# Prediction examples
predict_fracture('test_xray_1.jpg', model)
predict_fracture('test_xray_2.jpg', model)
```

### Fine-tuning — Last Layers Unfreeze

Feature extraction দিয়ে ভালো accuracy পেলেও, আরও improvement চাইলে **fine-tuning** করা যায়। Fine-tuning এ pretrained model এর শেষের কয়েকটি layer unfreeze করে নতুন task এর জন্য adjust করা হয়:

```python
from tensorflow.keras.optimizers import Adam

# --- Fine-tuning: Last convolutional block unfreeze ---

# প্রথমে সব layer freeze (আগের training এর weights protect)
for layer in base_model.layers:
    layer.trainable = False

# শেষ block (block5) এর layers unfreeze
for layer in base_model.layers:
    if 'block5' in layer.name:
        layer.trainable = True

# Trainable layers check
print("Fine-tuning trainable layers:")
for layer in base_model.layers:
    if layer.trainable:
        print(f"  ✓ {layer.name}")

# Recompile with LOWER learning rate — অত্যন্ত গুরুত্বপূর্ণ!
model.compile(
    optimizer=Adam(learning_rate=1e-5),    # 0.001 → 0.00001 (100x কম!)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune training
fine_tune_history = model.fit(
    train_generator,
    epochs=5,                               # আরও 5 epoch
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Save fine-tuned model
model.save('model_vgg16_finetuned.h5')
print("Fine-tuned model saved: model_vgg16_finetuned.h5")
```

**Fine-tuning এ Lower Learning Rate কেন অত্যন্ত গুরুত্বপূর্ণ?** এটি fine-tuning এর সবচেয়ে critical rule! Pretrained weights এ large update করলে সেগুলো নষ্ট হয়ে যাবে —catastrophic forgetting। Learning rate 1e-5 (0.00001) দিলে weights এ খুব ছোট পরিবর্তন হয় — pretrained features retain হয় এবং নতুন task এ ধীরে ধীরে adapt হয়। Standard learning rate (0.001) দিলে pretrained features এক epoch এই নষ্ট হয়ে যাবে!

**কোন layers unfreeze করবে?** General rule:
- ছোট dataset + ImageNet এর সাথে similar task → শুধু last 1-2 block unfreeze
- বড় dataset + ImageNet এর থেকে আলাদা task → বেশি block unfreeze
- সব কিছু unfreeze করলে effectively scratch training — overfitting risk বেশি

### Transfer Learning Decision Framework

কখন transfer learning করবে, কোন strategy follow করবে — তার একটি decision framework:

| Dataset Size | ImageNet এর সাথে Similarity | Strategy |
|---|---|---|
| ছোট ( < 1K) | Similar (natural images) | Feature extraction only — শুধু custom head train করো |
| ছোট ( < 1K) | Different (medical/satellite) | Feature extraction + শেষ 1 block fine-tune |
| মাঝারি (1K-10K) | Similar | Feature extraction + শেষ 2-3 block fine-tune |
| মাঝারি (1K-10K) | Different | বেশি block fine-tune, data augmentation জোরদার |
| বড় ( > 10K) | Similar | বেশি block fine-tune, বা full training from scratch |
| বড় ( > 10K) | Different | Full training from scratch ই best |

**Bone fracture classification এর ক্ষেত্রে:** Dataset সাধারণত ছোট (কয়েক শত X-ray), এবং X-ray ImageNet থেকে অনেক আলাদা — তাই Feature extraction + সামান্য fine-tuning সবচেয়ে appropriate strategy।

### Complete End-to-End Code

সব একসাথে — VGG16 transfer learning complete pipeline:

```python
# ========================================================
# VGG16 Transfer Learning — Bone Fracture Classification
# ========================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image

# --- Step 1: Load VGG16 base ---
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# --- Step 2: Freeze all layers ---
for layer in base_model.layers:
    layer.trainable = False

# --- Step 3: Add custom head ---
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- Step 4: Data preparation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest',
    horizontal_flip=False,
    vertical_flip=False
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'fracture_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'fracture_dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# --- Step 5: Compile ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Step 6: Train ---
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# --- Step 7: Evaluate & Save ---
# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], 'b-', label='Train Acc')
axes[0].plot(history.history['val_accuracy'], 'r-', label='Val Acc')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], 'b-', label='Train Loss')
axes[1].plot(history.history['val_loss'], 'r-', label='Val Loss')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vgg16_fracture_curves.png', dpi=150, bbox_inches='tight')
plt.show()

model.save('model_vgg16.h5')
print("Model saved: model_vgg16.h5")

# --- Step 8: Fine-tuning (optional) ---
for layer in base_model.layers:
    if 'block5' in layer.name:
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

model.save('model_vgg16_finetuned.h5')
print("Fine-tuned model saved: model_vgg16_finetuned.h5")

# --- Prediction on new X-ray ---
class_names = {0: 'Oblique Fracture', 1: 'Spiral Fracture'}

def predict_fracture(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    return class_names[class_idx], float(confidence)

# Usage example
# result, conf = predict_fracture('test_xray.jpg', model)
# print(f"Prediction: {result} (Confidence: {conf:.2%})")
```

### সারসংক্ষেপ

এই সেকশনে আমরা transfer learning এর সম্পূর্ণ workflow শিখলাম — VGG16 pretrained model কে feature extractor হিসেবে ব্যবহার করে bone fracture classification করলাম। মূল steps: (1) `include_top=False` দিয়ে base model load, (2) সব layer freeze, (3) custom classification head add, (4) augmented data দিয়ে train, (5) evaluate ও save, (6) fine-tuning with lower learning rate। আমরা শিখলাম 14.7M frozen + 3.3M trainable parameter এর concept, এবং transfer learning decision framework — dataset size ও ImageNet similarity অনুযায়ী কোন strategy best তা। Transfer learning small dataset এ deep learning apply করার সবচেয়ে practical approach — এটি ছাড়া medical imaging, satellite imaging এর মতো domain এ deep model train করা প্রায় অসম্ভব!
