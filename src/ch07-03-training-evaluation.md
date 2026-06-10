## ট্রেনিং ও ইভ্যালুয়েশন

মডেল বানানো ও compile করার পর সবচেয়ে গুরুত্বপূর্ণ ধাপ হলো training ও evaluation। এই সেকশনে আমরা `model.fit()` দিয়ে মডেল train করবো, training curve analyze করে overfitting/underfitting নির্ণয় করবো, মডেল save/load করবো, এবং নতুন ইমেজে inference করবো। Training হলো সেই প্রক্রিয়া যেখানে মডেল data থেকে pattern শিখে — এবং evaluation হলো সেই প্রক্রিয়া যেখানে আমরা যাচাই করি মডেল সত্যিই শিখেছে কিনা।

### model.fit() দিয়ে ট্রেইনিং

`model.fit()` method দিয়ে মডেল train করা হয়। যেহেতু আমরা `ImageDataGenerator` ব্যবহার করছি, training data generator pass করতে হবে। একটি গুরুত্বপূর্ণ concept হলো `steps_per_epoch` — এটি নির্দেশ করে প্রতি epoch এ কতগুলো batch process হবে। Calculation: `steps_per_epoch = total_training_samples / batch_size`। একইভাবে `validation_steps = total_validation_samples / batch_size`।

```python
# ট্রেইনিং
history = model.fit(
    train_generator,                              # Training data generator
    epochs=20,                                    # 20 epoch train করবো
    validation_data=val_generator,                # Validation data generator
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Training history তে কী কী আছে?
print(f"Available metrics: {history.history.keys()}")
```

`history` object টrainin এর সম্পূর্ণ ইতিহাস ধরে রাখে — প্রতি epoch এ training loss, training accuracy, validation loss, validation accuracy। এই data দিয়েই আমরা training curve plot করবো।

কয়টি epoch রাখবে? এটি dataset ও task এর উপর নির্ভর করে। সাধারণত 10-50 epoch দিয়ে শুরু করা উচিত, তারপর training curve দেখে decide করা উচিত। Early stopping ব্যবহার করলে epoch সংখ্যা বেশি রাখলেও problem নেই — validation loss improve না করলে training automatically বন্ধ হবে।

### model.evaluate() দিয়ে টেস্ট ইভ্যালুয়েশন

Training শেষে test set এ final evaluation করতে হবে:

```python
# Test set এ evaluation
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"\n{'='*50}")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"{'='*50}")
```

Test accuracy হলো আমাদের মডেলের "real" performance — এটি unseen data তে কতটুকু ভালো কাজ করে তার unbiased estimate। যদি test accuracy training accuracy এর অনেক কম হয়, তার মানে overfitting হয়েছে। যদি দুটোই কম হয়, underfitting।

### Training Curve প্লট করা

Training curve হলো মডেলের "health report"। এই curve দেখে আমরা বুঝতে পারি training smoothly হচ্ছে কিনা, overfitting হচ্ছে কিনা, learning rate ঠিক আছে কিনা। আসুন accuracy ও loss curve দুটোই plot করি:

```python
import matplotlib.pyplot as plt

# Figure তৈরি
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy curve
axes[0].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss curve
axes[1].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Overfitting, Underfitting ও Good Fit নির্ণয়

Training curve দেখে তিনটি scenario identify করা যায়:

**1. Overfitting (অতি-অভ্যস্ততা):**
- Training accuracy অনেক বেশি (যেমন 98%), কিন্তু validation accuracy অনেক কম (যেমন 75%)
- Training loss ক্রমাগত কমে, কিন্তু validation loss কমার পর বাড়তে শুরু করে
- এর মানে: মডেল training data "মুখস্থ" করে ফেলেছে, কিন্তু generalization করতে পারছে না
- সমাধান: Data augmentation বাড়াও, Dropout যোগ করো, model complexity কমাও, L2 regularization ব্যবহার করো, আরও data collect করো

**2. Underfitting (অনুভ্যস্ততাহীনতা):**
- Training accuracy ও validation accuracy দুটোই কম (যেমন 60%)
- Training loss অনেক বেশি থাকে, epoch বাড়লেও কমে না
- এর মানে: মডেল এতই simple যে data এর pattern শিখতে পারছে না
- সমাধান: Model complexity বাড়াও (আরও layer, আরও filter), training বেশি epoch চালাও, learning rate কমাও

**3. Good Fit (সঠিক অভ্যস্ততা):**
- Training accuracy ও validation accuracy দুটোই বেশি এবং কাছাকাছি
- Validation loss training loss এর কাছাকাছি, ক্রমাগত কমে
- এর মানে: মডেল training data থেকে pattern শিখেছে এবং unseen data তেও ভালো কাজ করছে
- এটিই আমাদের লক্ষ্য!

একটি practical tip: validation loss curve এ যে epoch এ minimum loss হয়, সেই epoch এর model weights সবচেয়ে ভালো — এটিকে best model বলে। `ModelCheckpoint` callback দিয়ে best model automatically save করা যায়।

### মডেল Save ও Load করা

Training শেষে মডেল save করা অত্যন্ত গুরুত্বপূর্ণ — না হলে প্রতিবার নতুন করে train করতে হবে! Keras দুটি format এ model save করতে পারে:

```python
from tensorflow.keras.models import save_model, load_model

# মডেল save করা (H5 format)
model.save('apple_tomato_model.h5')
print("মডেল save হয়েছে: apple_tomato_model.h5")

# মডেল load করা
loaded_model = load_model('apple_tomato_model.h5')
print("মডেল load হয়েছে!")

# Load করা মডেল verify করা
loaded_loss, loaded_acc = loaded_model.evaluate(test_generator)
print(f"Loaded model accuracy: {loaded_acc:.4f}")
```

H5 format (`model.h5`) একটি single file এ মডেলের architecture, weights, ও training configuration সব save করে। এটি portable — এক machine এ train করে অন্য machine এ load করে inference করা যায়। বড় মডেল এর H5 file অনেক বড় হতে পারে (শত MB থেকে GB) — সেক্ষেত্রে শুধু weights save করতে পারো `model.save_weights('weights.h5')`।

### নতুন ইমেজে Inference (predict_image function)

মডেলের মূল উদ্দেশ্য হলো নতুন, unseen ইমেজে prediction করা। আসুন একটি reusable inference function বানাই:

```python
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# মডেল load
model = load_model('apple_tomato_model.h5')

def predict_image(img_path, model, target_size=(224, 224), threshold=0.5):
    """
    একটি ইমেজে prediction করে।

    Parameters:
        img_path: ইমেজের file path
        model: trained Keras model
        target_size: ইমেজ resize এর সাইজ
        threshold: binary classification threshold

    Returns:
        predicted_class: 'apple' বা 'tomato'
        confidence: prediction এর confidence score
    """

    # ইমেজ লোড ও preprocess
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0           # Normalize (0-255 → 0-1)
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension যোগ

    # Prediction
    prediction = model.predict(img_array, verbose=0)
    probability = prediction[0][0]          # Sigmoid output (0-1)

    # Threshold অনুযায়ী class নির্ধারণ
    if probability > threshold:
        predicted_class = 'tomato'
        confidence = probability
    else:
        predicted_class = 'apple'
        confidence = 1 - probability

    return predicted_class, confidence

# ব্যবহারের উদাহরণ
result, conf = predict_image('test_apple.jpg', model)
print(f"Prediction: {result} (Confidence: {conf:.2%})")

result, conf = predict_image('test_tomato.jpg', model)
print(f"Prediction: {result} (Confidence: {conf:.2%})")
```

এই function এ কিছু গুরুত্বপূর্ণ বিষয় আছে:

- **`np.expand_dims(img_array, axis=0)`:** মডেল batch input expect করে — shape `(batch_size, 224, 224, 3)`। একটি ইমেজ এর shape হলো `(224, 224, 3)`, তাই batch dimension যোগ করে `(1, 224, 224, 3)` করতে হয়।

- **`threshold=0.5`:** Sigmoid output 0.5 এর উপরে হলে class 1 (tomato), নিচে হলে class 0 (apple)। এই threshold adjust করা যায় — যদি false negative কমানো দরকার হয়, threshold কমাও; false positive কমানো দরকার হলে, threshold বাড়াও।

- **Confidence calculation:** যদি prediction "apple" হয় (probability < 0.5), তাহলে confidence = 1 - probability। কারণ probability যত কম, apple হওয়ার confidence তত বেশি।

### একাধিক ইমেজে Batch Prediction

একসাথে অনেক ইমেজে prediction করতে চাইলে:

```python
import os

def predict_batch(folder_path, model):
    """একটি folder এর সব ইমেজে prediction করে।"""
    results = []

    for img_name in sorted(os.listdir(folder_path)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_name)
            pred_class, confidence = predict_image(img_path, model)
            results.append({
                'image': img_name,
                'prediction': pred_class,
                'confidence': confidence
            })

    # Result দেখানো
    print(f"\n{'Image':<30} {'Prediction':<12} {'Confidence':<12}")
    print('-' * 54)
    for r in results:
        print(f"{r['image']:<30} {r['prediction']:<12} {r['confidence']:.2%}")

    return results

# ব্যবহার
results = predict_batch('test_images/', model)
```

### সারসংক্ষেপ

এই সেকশনে আমরা CNN মডেল training এর সম্পূর্ণ workflow শিখলাম। `model.fit()` দিয়ে training, `model.evaluate()` দিয়ে test evaluation, training curve plot করে overfitting/underfitting/good fit নির্ণয় — সব কভার করলাম। আমরা শিখলাম কিভাবে `model.save()` ও `load_model()` দিয়ে মডেল save/load করতে হয়, এবং একটি `predict_image()` function বানালাম যা নতুন ইমেজে inference করে। সবচেয়ে গুরুত্বপূর্ণ শিক্ষা: training curve হলো মডেলের health report — এটি ভালো করে analyze করতে পারলে মডেলের problem diagnose করা সহজ হয়। এই চ্যাপ্টার শেষে তুমি end-to-end custom CNN training pipeline সম্পূর্ণভাবে বুঝতে পারবে — Kaggle থেকে ডেটা ডাউনলোড থেকে শুরু করে নতুন ইমেজে prediction পর্যন্ত!
