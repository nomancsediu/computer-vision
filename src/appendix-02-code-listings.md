## কমপ্লিট কোড লিস্টিং

এই সেকশনে বই এর সবচেয়ে important functions ও models এর complete code দেওয়া হলো — copy-paste ready। বই এর বিভিন্ন চ্যাপ্টারে যে concepts গুলো শিখেছি, তাদের complete implementation এক জায়গায় পাবে। প্রতিটি code block self-contained — শুধু dependency install করা থাকলে সরাসরি run করা যাবে।

### 1. simple_conv() — Scratch Implementation

কনভোলিউশন অপারেশন এর scratch implementation — কোনো library ছাড়া pure NumPy দিয়ে:

```python
import numpy as np

def simple_conv(image, kernel):
    """
    Perform 2D convolution on a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        2D array representing the input grayscale image.
        Shape: (height, width)
    kernel : numpy.ndarray
        2D array representing the convolution filter/kernel.
        Shape: (k_height, k_width)

    Returns
    -------
    numpy.ndarray
        2D array representing the convolved output image.
        Shape: (output_height, output_width)

    Notes
    -----
    - Uses 'valid' padding (no padding, output shrinks)
    - Kernel is NOT flipped (correlation, not strict convolution)
    - For strict mathematical convolution, flip kernel first:
      kernel = kernel[::-1, ::-1]
    - Output size formula: (H - kH + 1) x (W - kW + 1)
    """
    # Get dimensions
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # Calculate output dimensions (valid padding)
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1

    # Initialize output array
    output = np.zeros((out_h, out_w), dtype=np.float64)

    # Slide kernel across the image
    for i in range(out_h):
        for j in range(out_w):
            # Extract region of interest
            region = image[i:i + k_h, j:j + k_w]

            # Element-wise multiply and sum
            output[i, j] = np.sum(region * kernel)

    return output


# ============================================================
# Usage Example
# ============================================================

if __name__ == '__main__':
    # Create a simple 6x6 test image
    image = np.array([
        [10, 10, 10, 20, 20, 20],
        [10, 10, 10, 20, 20, 20],
        [10, 10, 10, 20, 20, 20],
        [30, 30, 30, 40, 40, 40],
        [30, 30, 30, 40, 40, 40],
        [30, 30, 30, 40, 40, 40]
    ], dtype=np.float64)

    # Horizontal edge detection kernel (Sobel)
    kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    # Apply convolution
    result = simple_conv(image, kernel)

    print("Input Image (6x6):")
    print(image)
    print("\nKernel (3x3 Sobel Horizontal):")
    print(kernel)
    print("\nConvolution Output (4x4):")
    print(result)
    print(f"\nOutput shape: {result.shape}")
    print(f"Expected: ({image.shape[0] - kernel.shape[0] + 1}, "
          f"{image.shape[1] - kernel.shape[1] + 1})")
```

### 2. LeNet-5 — Keras Implementation

Yann LeCun এর historical LeNet-5 architecture — প্রথম successful CNN:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

def build_lenet5(input_shape=(32, 32, 1), num_classes=10):
    """
    Build LeNet-5 architecture (LeCun et al., 1998).

    Parameters
    ----------
    input_shape : tuple
        Input image shape. Default (32, 32, 1) for MNIST.
    num_classes : int
        Number of output classes. Default 10.

    Returns
    -------
    tensorflow.keras.Model
        Compiled LeNet-5 model.

    Architecture
    ------------
    Layer 1: Conv2D (6 filters, 5x5) → AvgPool (2x2)
    Layer 2: Conv2D (16 filters, 5x5) → AvgPool (2x2)
    Layer 3: Flatten → Dense(120) → Dense(84) → Dense(num_classes)

    Total parameters: ~60,000
    """
    model = Sequential([
        # Layer 1: C1 — Convolutional Layer
        Conv2D(
            filters=6,
            kernel_size=(5, 5),
            activation='tanh',
            input_shape=input_shape,
            name='C1_Conv'
        ),
        # Layer 2: S2 — Average Pooling
        AveragePooling2D(
            pool_size=(2, 2),
            name='S2_Pool'
        ),

        # Layer 3: C3 — Convolutional Layer
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            activation='tanh',
            name='C3_Conv'
        ),
        # Layer 4: S4 — Average Pooling
        AveragePooling2D(
            pool_size=(2, 2),
            name='S4_Pool'
        ),

        # Flatten for fully connected layers
        Flatten(name='Flatten'),

        # Layer 5: C5 — Fully Connected
        Dense(units=120, activation='tanh', name='C5_FC'),

        # Layer 6: F6 — Fully Connected
        Dense(units=84, activation='tanh', name='F6_FC'),

        # Output Layer
        Dense(units=num_classes, activation='softmax', name='Output')
    ], name='LeNet-5')

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# Usage Example — MNIST
# ============================================================

if __name__ == '__main__':
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess: reshape, normalize, one-hot encode
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Resize to 32x32 (LeNet-5 expects 32x32 input)
    import tensorflow as tf
    x_train = tf.image.resize(x_train, [32, 32]).numpy()
    x_test = tf.image.resize(x_test, [32, 32]).numpy()

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build and train
    model = build_lenet5()
    model.summary()

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(x_test, y_test)
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
```

### 3. AlexNet — Keras Implementation

2012 ImageNet competition বিজয়ী AlexNet — deep learning revolution এর সূচনা:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)

def build_alexnet(input_shape=(227, 227, 3), num_classes=1000):
    """
    Build AlexNet architecture (Krizhevsky et al., 2012).

    Parameters
    ----------
    input_shape : tuple
        Input image shape. Default (227, 227, 3) for ImageNet.
    num_classes : int
        Number of output classes. Default 1000 for ImageNet.

    Returns
    -------
    tensorflow.keras.Model
        Compiled AlexNet model.

    Architecture
    ------------
    5 Convolutional layers + 3 Fully Connected layers
    Key innovations: ReLU activation, Dropout, Data Augmentation
    Total parameters: ~60 million
    """
    model = Sequential([
        # Layer 1: Conv + MaxPool
        Conv2D(96, (11, 11), strides=4, activation='relu',
               input_shape=input_shape, name='Conv1'),
        MaxPooling2D((3, 3), strides=2, name='Pool1'),

        # Layer 2: Conv + MaxPool
        Conv2D(256, (5, 5), padding='same', activation='relu', name='Conv2'),
        MaxPooling2D((3, 3), strides=2, name='Pool2'),

        # Layer 3: Conv (no pooling)
        Conv2D(384, (3, 3), padding='same', activation='relu', name='Conv3'),

        # Layer 4: Conv (no pooling)
        Conv2D(384, (3, 3), padding='same', activation='relu', name='Conv4'),

        # Layer 5: Conv + MaxPool
        Conv2D(256, (3, 3), padding='same', activation='relu', name='Conv5'),
        MaxPooling2D((3, 3), strides=2, name='Pool5'),

        # Flatten
        Flatten(name='Flatten'),

        # FC Layer 6
        Dense(4096, activation='relu', name='FC6'),
        Dropout(0.5, name='Dropout6'),

        # FC Layer 7
        Dense(4096, activation='relu', name='FC7'),
        Dropout(0.5, name='Dropout7'),

        # Output Layer
        Dense(num_classes, activation='softmax', name='Output')
    ], name='AlexNet')

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# Usage Example
# ============================================================

if __name__ == '__main__':
    model = build_alexnet(input_shape=(227, 227, 3), num_classes=10)
    model.summary()

    # Parameter count
    total = model.count_params()
    print(f"\nTotal parameters: {total:,}")
    print(f"Approximately {total / 1e6:.1f}M parameters")
```

### 4. Transfer Learning Pipeline — VGG16

আমাদের bone fracture classification এর সম্পূর্ণ transfer learning pipeline:

```python
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_transfer_model(num_classes=2, input_shape=(224, 224, 3)):
    """
    Build VGG16-based transfer learning model for bone fracture classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 2 (Oblique, Spiral).
    input_shape : tuple
        Input image shape. Default (224, 224, 3).

    Returns
    -------
    tuple
        (model, base_model) — complete model and base model reference
        base_model needed for later fine-tuning
    """
    # Load VGG16 base (without classification head)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze all base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def create_data_generators(train_dir, val_dir, batch_size=32):
    """
    Create data generators with augmentation for training.

    Parameters
    ----------
    train_dir : str
        Path to training data directory.
    val_dir : str
        Path to validation data directory.
    batch_size : int
        Batch size for generators.

    Returns
    -------
    tuple
        (train_generator, val_generator)
    """
    # Training augmentation (conservative for medical images)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest',
        horizontal_flip=False,  # X-ray: no flip!
        vertical_flip=False
    )

    # Validation: rescale only
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_gen, val_gen


def fine_tune_model(model, base_model, unfreeze_from='block5',
                    learning_rate=1e-5):
    """
    Fine-tune the model by unfreezing last convolutional block.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The complete transfer learning model.
    base_model : tensorflow.keras.Model
        The VGG16 base model reference.
    unfreeze_from : str
        Layer name prefix to unfreeze. Default 'block5'.
    learning_rate : float
        Learning rate for fine-tuning. Must be LOW (e.g., 1e-5).

    Returns
    -------
    tensorflow.keras.Model
        Recompiled model with unfrozen layers.
    """
    # Unfreeze specified layers
    for layer in base_model.layers:
        if unfreeze_from in layer.name:
            layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def predict_image(img_path, model, class_names):
    """
    Predict fracture type from a single X-ray image.

    Parameters
    ----------
    img_path : str
        Path to the X-ray image file.
    model : tensorflow.keras.Model
        Trained classification model.
    class_names : dict
        Mapping of class index to class name.

    Returns
    -------
    tuple
        (predicted_class_name, confidence, all_probabilities)
    """
    # Load and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])

    all_probs = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }

    return class_names[class_idx], confidence, all_probs


# ============================================================
# Complete Pipeline Usage
# ============================================================

if __name__ == '__main__':
    # Configuration
    TRAIN_DIR = 'fracture_dataset/train'
    VAL_DIR = 'fracture_dataset/val'
    CLASS_NAMES = {0: 'Oblique Fracture', 1: 'Spiral Fracture'}

    # Step 1: Build model
    print("Building transfer learning model...")
    model, base_model = build_transfer_model(num_classes=2)
    model.summary()

    # Step 2: Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    # Step 3: Train (feature extraction)
    print("\nTraining — Feature Extraction Phase...")
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # Step 4: Fine-tune
    print("\nFine-tuning — Unfreezing block5...")
    model = fine_tune_model(model, base_model, unfreeze_from='block5')

    fine_tune_history = model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # Step 5: Save final model
    model.save('model_vgg16_final.h5')
    print("Model saved: model_vgg16_final.h5")

    # Step 6: Predict on new image
    # result, conf, probs = predict_image('test_xray.jpg', model, CLASS_NAMES)
    # print(f"Prediction: {result} (Confidence: {conf:.2%})")
    # for cls, prob in probs.items():
    #     print(f"  {cls}: {prob:.2%}")
```

### সারসংক্ষেপ

এই সেকশনে বই এর চারটি core implementation এর complete code দেওয়া হলো: (1) `simple_conv()` — NumPy দিয়ে scratch convolution, (2) LeNet-5 — historical CNN architecture এর Keras implementation, (3) AlexNet — ImageNet 2012 winner, (4) Transfer Learning pipeline — VGG16 feature extraction + fine-tuning complete workflow। প্রতিটি function এ detailed docstring আছে — parameter, return value, usage example সহ। এই codes গুলো directly run করে experiment করো — এটাই শেখার সবচেয়ে ভালো উপায়!
