## ResNet50 দিয়ে প্রেডিকশন

Transfer learning এর সবচেয়ে সহজ ও powerful demonstration হলো  একটি pretrained model directly use করে prediction করা। কোনো training নেই, কোনো dataset নেই  শুধু model load করো আর image দিয়ে predict করো! এই সেকশনে আমরা ImageNet এ pretrained **ResNet50** model দিয়ে exactly এটাই করবো।

### Transfer Learning এর Key Insight

Transfer learning কেন কাজ করে? এর পেছনে একটি গভীর insight আছে। যখন একটি CNN model ImageNet (14M+ images, 1000 classes) এ train হয়, তখন এর বিভিন্ন layer বিভিন্ন level এর feature শেখে:

- **Early layers (প্রথম কয়েকটি Conv layer):** খুব general features শেখে  edges, corners, color gradients, simple textures। এই features যেকোনো visual task এ useful  কারণ প্রায় সব image তেই edge ও color gradient থাকে।

- **Middle layers:** একটু বেশি complex features শেখে  textures, patterns, simple shapes (circle, rectangle), repeating patterns। এই features ও অনেক task এ transferable।

- **Later layers (শেষের Conv layers):** Task-specific features শেখে  নির্দিষ্ট object parts (wheel, eye, leaf, paw) যা ImageNet class identify করতে সাহায্য করে। এই features তুলনামূলকভাবে কম transferable।

- **Fully connected layers (Classification head):** Completely task-specific  ImageNet এর 1000 class এর decision boundary। নতুন task এ এটি replace করতে হয়।

এই insight টি অত্যন্ত গুরুত্বপূর্ণ  কারণ এর মানে হলো, ImageNet এ শেখা early ও middle features নতুন task এও কাজে লাগানো যায়! একটি medical image classification model ও একটি satellite image classification model  দুটোরই early layers একই রকম edge detector শিখবে। এটাই transfer learning এর মূল শক্তি।

### ResNet50  একটি Quick Overview

ResNet50 হলো Microsoft Research এর 2015 সালের গ্রাউন্ডব্রেকিং architecture। 50 layer deep এই model ImageNet এ 76% top-1 accuracy ও 93% top-5 accuracy অর্জন করেছে। এর মূল innovation হলো **Residual Connection (Skip Connection)**  যা vanishing gradient problem solve করে এবং 50+ layer train করা সম্ভব করে।

Keras এ `keras.applications` module এ ResNet50 pretrained weights সহ available আছে  মাত্র কয়েক লাইন code এ load করা যায়!

### Pretrained ResNet50 Load ও Prediction

চলো ResNet50 model load করে কয়েকটি image তে prediction করি:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# --- 1. Pretrained ResNet50 Load ---
    # weights='imagenet' → Loads weights trained on ImageNet
model = ResNet50(weights='imagenet')

    # View model summary (optional)
model.summary()
```

**weights='imagenet':** এটি ImageNet dataset (1.2M training images, 1000 classes) এ pretrained weights download করে। প্রথমবার run করলে weights download হবে (~98MB), এরপর cache থেকে load হবে।

### Image Preprocess ও Prediction

```python
    # --- 2. Image Load and Preprocess ---
def predict_with_resnet(img_path):
    """
    Classifies image using ResNet50.

    Parameters:
        img_path: File path of the image

    Returns:
        top predictions with labels and confidence
    """
    # Image load  ResNet50 expects 224x224 input
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert to numpy array
    img_array = image.img_to_array(img)

    # Batch dimension add: (224,224,3) → (1,224,224,3)
    img_array = np.expand_dims(img_array, axis=0)

    # ResNet50 specific preprocessing
    # preprocess_input() converts pixel values to ImageNet training format
    # (0-255 → centered around 0, BGR channel order, etc.)
    img_array = preprocess_input(img_array)

    # --- 3. Prediction ---
    predictions = model.predict(img_array)

    # --- 4. Decode Predictions ---
    # decode_predictions() converts raw prediction vector to human-readable label
    # top=5 means top 5 most likely classes will be returned
    top_predictions = decode_predictions(predictions, top=5)[0]

    # Result display
    print(f"\n{'Class':<35} {'Description':<30} {'Confidence':<12}")
    print('=' * 77)
    for class_id, description, confidence in top_predictions:
        print(f"{class_id:<35} {description:<30} {confidence:.2%}")

    return top_predictions

# Prediction examples
print("=== Image 1 ===")
predict_with_resnet('elephant.jpg')

print("\n=== Image 2 ===")
predict_with_resnet('car.jpg')

print("\n=== Image 3 ===")
predict_with_resnet('pizza.jpg')
```

**preprocess_input() কেন দরকার?** প্রতিটি pretrained model নির্দিষ্ট preprocessing expect করে। ResNet50 ImageNet training এর সময় যে preprocessing use হয়েছিল, `preprocess_input()` ঠিক সেটাই করে  pixel value scale করা, channel order conversion (RGB→BGR), mean subtraction ইত্যাদি। এই preprocessing না করলে prediction ভুল হবে!

**decode_predictions() কী করে?** ResNet50 এর output হলো 1000-length probability vector। `decode_predictions()` এই vector কে ImageNet class label (যেমন 'African_elephant', 'sports_car') এ convert করে। `top=5` দিলে সবচেয়ে probable 5টি class return হয়  একে "top-5 accuracy" বলে।

### Prediction Results Visualize

Prediction গুলো image এর সাথে visualize করলে বেশি বোঝা যায়:

```python
def predict_and_visualize(img_path):
    """Visualizes image and prediction result together."""
    # Image load
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    top_preds = decode_predictions(predictions, top=5)[0]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Original image
    ax1.imshow(image.load_img(img_path, target_size=(224, 224)))
    ax1.set_title('Input Image', fontsize=14)
    ax1.axis('off')

    # Prediction bar chart
    labels = [pred[1] for pred in top_preds]
    confidences = [pred[2] for pred in top_preds]
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(5)]

    ax2.barh(range(len(labels)), confidences, color=colors)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_title('Top-5 Predictions', fontsize=14)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'prediction_{img_path.split("/")[-1]}', dpi=150, bbox_inches='tight')
    plt.show()

# Usage
predict_and_visualize('elephant.jpg')
```

### Pretrained Model এর Limitations

ResNet50 ImageNet এর 1000 class এ trained  এটি everyday objects (animals, vehicles, food, furniture) খুব ভালো recognize করে। কিন্তু এর সীমাবদ্ধতাও আছে:

- **Domain-specific classes নেই:** Medical images (X-ray, MRI), satellite images, microscope images  এসব ImageNet class এ নেই। তাই ResNet50 একটি brain tumor MRI বা একটি satellite image তে residential area ঠিকমতো classify করতে পারবে না।

- **Fine-grained classification কঠিন:** 200 প্রজাতির bird distinguish করা, বা 196 প্রজাতির car model identify করা  এসব fine-grained task এ ImageNet pretrained model এর feature গুলো sufficient নয়।

- **Novel classes:** ImageNet training data তে যে class ছিল না (যেমন কোনো rare disease, নতুন পণ্য), সেটি predict করা অসম্ভব।

এই limitations এর কারণেই **transfer learning with custom head** দরকার  pretrained convolutional base কে feature extractor হিসেবে ব্যবহার করে, উপরে নতুন classification layer add করে নতুন task এ train করা। এটিই আমরা পরের সেকশনে VGG16 দিয়ে শিখবো!

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম transfer learning এর core insight  early/middle CNN layers general features (edges, textures) শেখে যা যেকোনো visual task এ useful। আমরা ImageNet pretrained ResNet50 model load করে সরাসরি prediction করলাম  `preprocess_input()` দিয়ে preprocessing, `decode_predictions()` দিয়ে result interpret। এটি ImageNet এর 1000 class এ দারুণ কাজ করে, কিন্তু domain-specific tasks (medical, satellite) এ সরাসরি কাজ করে না। পরের সেকশনে আমরা শিখবো কিভাবে pretrained model কে feature extractor হিসেবে ব্যবহার করে নতুন domain এ train করা যায়  VGG16 দিয়ে bone fracture classification!
