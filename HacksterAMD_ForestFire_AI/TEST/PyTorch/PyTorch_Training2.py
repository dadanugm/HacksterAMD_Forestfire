import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import TFRecordLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Create train, validation, and test loaders
train_loader = TFRecordLoader(
    'datasets/train-forest-fire.tfrecord',
    transform=transform,
    records_batch_size=32,
    shuffle=False
)
valid_loader = TFRecordLoader(
    'datasets/valid-forest-fire.tfrecord',
    transform=transform,
    records_batch_size=32,
    shuffle=False
)
test_loader = TFRecordLoader(
    'datasets/test-forest-fire.tfrecord',
    transform=transform,
    records_batch_size=32,
    shuffle=False
)

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
    for data in train_loader:
        images, labels = data['image'], data['label']

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
        for data in valid_loader:
            images, labels = data['image'], data['label']

            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Valid Loss: {valid_loss / len(valid_loader):.4f}, Valid Acc: {(correct / total) * 100:.2f}%"
    )

print('Finished Training')
