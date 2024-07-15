import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import io
import tensorflow as tf


# Define a custom dataset class
class TFRecordDataset(torch.utils.data.Dataset):
    def __init__(self, tfrecord_file, transform=None):
        self.tfrecord_file = tfrecord_file
        self.transform = transform

        # Load the TFRecord file
        self.dataset = tf.data.TFRecordDataset(tfrecord_file)

    def __len__(self):
        # Return the total number of examples in the dataset
        return sum(1 for _ in self.dataset)

    def __getitem__(self, idx):
        # Parse the TFRecord example at the given index
        raw_record = next(iter(self.dataset.skip(idx).take(1)))
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Check if the 'image' feature is present and non-empty
        if 'images' not in example.features.feature or not example.features.feature['images'].bytes_list.value:
            raise ValueError('Images feature not found or empty in TFRecord example')

        # Extract and decode the image
        image_bytes = example.features.feature['images'].bytes_list.value[0]
        image = Image.open(io.BytesIO(image_bytes))

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Extract and return the label
        label = example.features.feature['labels'].int64_list.value[0]

        return image, label


# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Create train, validation, and test datasets
train_dataset = TFRecordDataset('datasets/train-forest-fire.tfrecord', transform=transform)
valid_dataset = TFRecordDataset('datasets/valid-forest-fire.tfrecord', transform=transform)
test_dataset = TFRecordDataset('datasets/test-forest-fire.tfrecord', transform=transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes for the final layer

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Valid Loss: {valid_loss / len(valid_loader):.4f}, Valid Acc: {(correct / total) * 100:.2f}%")

print('Finished Training')
