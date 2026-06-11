## ফিচার ম্যাপ ভিজুয়ালাইজেশন

ফিল্টার visualization আমাদের দেখায় নেটওয়ার্ক কী কী pattern খুঁজছে  কিন্তু একটি নির্দিষ্ট ইমেজে সেই pattern গুলো কোথায় পাওয়া গেছে, সেটি বোঝায় না। ফিচার ম্যাপ (feature map) ঠিক এটাই দেখায়! ফিচার ম্যাপ হলো convolution operation এর output  এটি একটি spatial map যা দেখায় ইমেজের কোন region এ কোন filter strongly activate হয়েছে। যদি একটি edge-detecting filter horizontal edge খুঁজে পায়, তাহলে feature map এ সেই horizontal edge এর position গুলো bright দেখাবে। এই সেকশনে আমরা VGG16 এর বিভিন্ন layer এর feature map visualize করে দেখবো কীভাবে নেটওয়ার্ক একটি ইমেজ বোঝে  shallow layer থেকে deep layer পর্যন্ত।

### ফিচার ম্যাপ কী?

Feature map হলো convolution operation এর সরাসরি output। যখন একটি ইমেজ কোনো convolution layer এর মধ্য দিয়ে যায়, প্রতিটি filter ইমেজের উপর scan করে এবং একটি 2D activation map তৈরি করে  এটিই feature map। Feature map এর প্রতিটি pixel value indicate করে সেই spatial location এ filter এর pattern কতটুকু match পাওয়া গেছে। High value = strong match, low value = weak match।

Filter visualization আর feature map visualization এর মূল পার্থক্য:

| Aspect | ফিল্টার | ফিচার ম্যাপ |
|--------|---------|-------------|
| কী দেখায় | নেটওয়ার্ক কী খুঁজছে | নেটওয়ার্ক কী পেয়েছে |
| Input-dependent? | না (weights fixed) | হ্যাঁ (নির্দিষ্ট ইমেজে) |
| Interpretation | "আমি horizontal edge খুঁজছি" | "এই ইমেজে horizontal edge এখানে আছে" |

### Feature Map Model তৈরি করা

Keras এর `Model` class ব্যবহার করে আমরা একটি intermediate model বানাবো যা শুধু একটি নির্দিষ্ট layer এর output return করবে। এটি একটি sub-model  same input, কিন্তু output হবে intermediate layer এর feature map:

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Load VGG16
vgg16 = VGG16(weights='imagenet', include_top=False)

# Model to extract feature maps from a specific layer
layer_name = 'block1_conv2'
feature_map_model = Model(
    inputs=vgg16.inputs,
    outputs=vgg16.get_layer(layer_name).output
)

print(f"Feature map model created: {layer_name}")
print(f"Input shape: {feature_map_model.input_shape}")
print(f"Output shape: {feature_map_model.output_shape}")
```

এই `feature_map_model` একটি ইমেজ input নেবে এবং `block1_conv2` layer এর feature map return করবে। আমরা যেকোনো layer এর feature map extract করতে পারি  শুধু `layer_name` change করলেই হবে।

### ইমেজ লোড ও প্রিপ্রসেস

VGG16 এর জন্য ইমেজ preprocess করার একটি নির্দিষ্ট নিয়ম আছে। `preprocess_input()` function pixel value গুলো RGB থেকে BGR তে convert করে এবং ImageNet mean subtract করে  এটি VGG16 training এর সময় যে preprocessing হয়েছিল, ঠিক সেটাই apply করে:

```python
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load image
img_path = 'sample_image.jpg'  # Your image path
img = image.load_img(img_path, target_size=(224, 224))

# Convert to array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

# Preprocess for VGG16
img_preprocessed = preprocess_input(img_array)

print(f"Image shape: {img_preprocessed.shape}")
print(f"Value range: {img_preprocessed.min():.2f} to {img_preprocessed.max():.2f}")
```

`preprocess_input()` এর কাজ অত্যন্ত গুরুত্বপূর্ণ  যদি এটি না করো, feature map গুলো ভুল আসবে! কারণ VGG16 training এর সময় এই specific preprocessing ব্যবহার হয়েছিল  inference এও same preprocessing apply করতে হবে।

### Feature Map Extract ও Plot করা

এখন আমরা feature map extract করে plot করবো:

```python
# Extract feature maps
feature_maps = feature_map_model.predict(img_preprocessed)

print(f"Feature map shape: {feature_maps.shape}")
# Output: (1, 224, 224, 64)  64 feature maps, each 224×224

# Plot first 16 feature maps
n_features = 16
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

for i in range(n_features):
    ax = axes[i // 8, i % 8]
    ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
    ax.set_title(f'Map {i}', fontsize=9)
    ax.axis('off')

plt.suptitle(f'Feature Maps  {layer_name}', fontsize=14)
plt.tight_layout()
plt.savefig('feature_maps_single_layer.png', dpi=150, bbox_inches='tight')
plt.show()
```

Feature map plot করার সময় `cmap='viridis'` ব্যবহার করা হয়েছে  এটি একটি perceptually uniform colormap যেখানে low value বেগুনি ও high value হলুদ দেখায়। তুমি `'gray'` (grayscale), `'hot'` (heat map), বা `'magma'` ও ব্যবহার করতে পারো  choose করো কোনটিতে pattern সবচেয়ে clearly visible হয়।

### মাল্টি-লেয়ার Feature Map Extraction

সবচেয়ে চমকপ্রদ visualization পাওয়া যায় যখন আমরা একই ইমেজের feature map VGG16 এর বিভিন্ন depth এ compare করি। আমরা 5টি layer থেকে feature map extract করবো  প্রতিটি block এর শেষ conv layer:

```python
# Layers whose feature maps we will visualize
layer_names = [
    'block1_conv2',   # Block 1 end (shallow)
    'block2_conv2',   # Block 2 end
    'block3_conv3',   # Block 3 end (middle)
    'block4_conv3',   # Block 4 end
    'block5_conv3'    # Block 5 end (deep)
]

# Create multi-output model
layer_outputs = [vgg16.get_layer(name).output for name in layer_names]
multi_output_model = Model(inputs=vgg16.inputs, outputs=layer_outputs)

# Feature maps from all layers at once
feature_maps_list = multi_output_model.predict(img_preprocessed)

# Plot 8 feature maps from each layer
for idx, (fmap, layer_name) in enumerate(zip(feature_maps_list, layer_names)):
    n_show = 8
    fig, axes = plt.subplots(1, n_show, figsize=(16, 2))

    for i in range(n_show):
        axes[i].imshow(fmap[0, :, :, i], cmap='viridis')
        axes[i].axis('off')

    fig.suptitle(f'{layer_name}  Feature Maps (shape: {fmap.shape[1:]}x{fmap.shape[-1]})',
                 fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(f'feature_maps_{layer_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Shallow → Deep: Feature Map এর Transformation

যখন তুমি বিভিন্ন depth এর feature map compare করবে, একটি সুন্দর hierarchical pattern দেখতে পাবে:

**Shallow Layers (Block 1-2):**
- Feature map গুলো মূল ইমেজের মতোই দেখায়  তুমি এখনও object এর outline recognize করতে পারবে
- কিছু feature map edge detect করে  ইমেজের boundary গুলো bright দেখায়
- কিছু feature map color contrast detect করে  dark ও bright region আলাদা হয়
- অনেক feature map active  information dense
- এগুলো মূলত low-level feature: edges, corners, color blobs, gradients

**Middle Layers (Block 3):**
- Feature map আর মূল ইমেজের মতো দেখায় না  অনেক abstract হয়ে গেছে
- Texture pattern detect হচ্ছে  যেমন fur texture, leaf pattern, brick pattern
- কিছু feature map specific shape part detect করে  circular curve, rectangular corner
- Spatial resolution কমে গেছে (224→112→56→28), কিন্তু channel বেশি (128-256)
- Object এর exact position আর নেই, কিন্তু "কোন ধরনের pattern কোন region এ" তা আছে

**Deep Layers (Block 4-5):**
- Feature map অত্যন্ত abstract  মানুষের চোখে বোঝা প্রায় অসম্ভব
- অনেক feature map sparse  শুধু কয়েকটি location এ activation, বাকি সব dark
- যেখানে activation আছে, সেখানে নেটওয়ার্ক "interested"  সেই region এ high-level semantic feature আছে
- Spatial resolution অনেক ছোট (14×14 বা 7×7)  পুরো ইমেজ একটি compact representation এ compress হয়েছে
- এখানকার feature গুলো object category specific  "এটি একটি dog shape" বা "এখানে wheel আছে" ইত্যাদি

এই hierarchical transformation কে বলা যায় **"pixels → edges → textures → parts → objects"**  এটি CNN এর সবচেয়ে গভীর insight। একটি মডেন্ট মডেল 14 million parameter শেখে, কিন্তু এর মূল essence হলো এই hierarchical feature extraction  যা biological vision system ও follow করে।

### সব লেয়ার একসাথে Comprehensive Plot

সব layer এর feature map একটি single figure তে compare করার জন্য:

```python
# Comprehensive visualization
fig, axes = plt.subplots(len(layer_names), 8, figsize=(16, 10))

for row, (fmap, layer_name) in enumerate(zip(feature_maps_list, layer_names)):
    for col in range(8):
        ax = axes[row, col]
        ax.imshow(fmap[0, :, :, col], cmap='viridis')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(layer_name, fontsize=10, rotation=0, labelpad=70)

plt.suptitle('VGG16 Feature Maps  Shallow to Deep', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('vgg16_feature_maps_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### প্র্যাকটিক্যাল অ্যাপ্লিকেশন

Feature map visualization শুধু academic exercise না  এর real-world application আছে:

**1. Debugging:** মডেল ভুল prediction দিলে feature map দেখে বোঝা যায় কোন layer এ problem। যদি shallow layer এর feature map ঠিক থাকে কিন্তু deep layer এ garbage আসে, তাহলে training এ deep layer properly optimize হয়নি। যদি shallow layer এই meaningless হয়, তাহলে input preprocessing এ problem আছে।

**2. Architecture Design:** Feature map analyze করে architecture improve করা যায়। যদি দেখা যায় অনেক feature map dead (সব zero), তাহলে সেই layer এ বেশি filter দরকার নেই  parameter বাঁচানো যায়। যদি feature map অত্যন্ত noisy হয়, batch normalization বা more regularization দরকার।

**3. Model Interpretation:** ক্লায়েন্ট বা stakeholder কে explain করতে হলে feature map visualization অত্যন্ত helpful। "দেখুন, মডেল tumor region এ strongly activate করছে"  এটি একটি convincing evidence যে মডেল সঠিক feature দেখছে।

**4. Transfer Learning Decision:** Pretrained model এর feature map দেখে decide করা যায় কোন layer freeze করবো, কোন layer fine-tune করবো। Shallow layer এর feature map general (edges, textures)  এগুলো freeze করা safe। Deep layer এর feature map task-specific  নতুন task এ fine-tune করা দরকার।

### সারসংক্ষেপ

এই সেকশনে আমরা শিখলাম কিভাবে CNN এর feature map visualize করতে হয়। আমরা `Model(inputs=model.inputs, outputs=model.layers[i].output)` দিয়ে intermediate model বানালাম, `preprocess_input()` দিয়ে ইমেজ preprocess করলাম, এবং `.predict()` দিয়ে feature map extract করলাম। VGG16 এর 5টি block ([2,5,9,13,17] layers) থেকে feature map compare করে আমরা দেখলাম কীভাবে representation shallow থেকে deep এ transform হয়: edges → textures → abstract/sparse। আমরা feature map visualization এর practical application ও শিখলাম  debugging, architecture design, model interpretation, আর transfer learning decision। এই চ্যাপ্টার শেষে CNN আর তোমার কাছে "ব্ল্যাক বক্স" না  তুমি জানো কোন layer কী শেখে, কেন shallow layer এ edge detect হয়, আর deep layer এ abstract concept form হয়।
