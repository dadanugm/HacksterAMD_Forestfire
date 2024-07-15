import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers

# Load label map from pbtxt file
def load_label_map(pbtxt_file):
    label_map = {}
    with open(pbtxt_file, 'r') as f:
        label_id = None
        for line in f:
            if 'id' in line:
                _, id_str = line.strip().split(':')
                label_id = int(id_str.strip().strip(','))
            elif 'name' in line and label_id is not None:
                _, name = line.strip().split(':')
                class_name = name.strip().strip("'")
                label_map[label_id] = class_name
                label_id = None
    return label_map

label_map = load_label_map('datasets/train/fire-smoke_label_map.pbtxt')

# Function to parse TFRecord examples
def parse_tfrecord_fn(example_proto):
    feature_description = {
        'image_path': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image_path = example['image_path']
    label = example['label']
    return image_path, label

# Function to load and preprocess images
def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # Resize to match model input size
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image
    return image, label

# Example dataset loading and preprocessing
def load_dataset(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_tfrecord_fn)
    dataset = dataset.map(lambda image_path, label: load_and_preprocess_image(image_path, label))
    return dataset

# Load TFRecord dataset
#tfrecord_file = 'path_to_tfrecord.tfrecord'
#dataset = load_dataset(tfrecord_file)

# Load train and validation datasets
train_dataset = load_dataset('datasets/train/fire-smoke.tfrecord')
val_dataset = load_dataset('datasets/valid/fire-smoke.tfrecord')

train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))  # Normalize images
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))  # Normalize images

batch_size = 32
# Data augmentation function
def augment_image(image):
    # Apply data augmentation techniques here (e.g., random flip, random crop, etc.)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image

# Apply data augmentation to the dataset
train_dataset = train_dataset.map(lambda x, y: (augment_image(x), y))
# Apply data augmentation to the validation dataset
val_dataset = val_dataset.map(lambda x, y: (augment_image(x), y))

# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)
# Shuffle and batch the validation dataset
val_dataset = val_dataset.batch(batch_size)

# Example model definition
base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False

num_classes = 3
# Add custom layers for object detection
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Define the detection loss function
def detection_loss(y_true, y_pred):
    # Calculate classification loss
    class_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true[:, 0], y_pred)

    # Calculate localization loss (e.g., smooth L1 loss)
    bbox_true = y_true[:, 1:]
    bbox_pred = y_pred[:, 1:]
    localization_loss = tf.keras.losses.Huber()(bbox_true, bbox_pred)

    # Total loss
    total_loss = class_loss + localization_loss
    return total_loss

# Compile the model
model.compile(optimizer=Adam(), loss=detection_loss, metrics=['accuracy'])

# Example training loop with validation
epochs = 5
for epoch in range(epochs):
    # Training loop
    for images, labels in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(images)
            loss_value = detection_loss(labels, logits)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss_value.numpy():.4f}")

    # Validation loop
    val_loss = 0.0
    val_steps = 0
    for val_images, val_labels in val_dataset:
        val_logits = model(val_images)
        val_loss += detection_loss(val_labels, val_logits)
        val_steps += 1
    val_loss /= val_steps
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss.numpy():.4f}")

# Save the trained model
model.save('path_to_saved_model')
