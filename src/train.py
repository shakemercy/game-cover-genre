import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loaders
from models.baseline_cnn import BaselineCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

data_dir = "data/raw"
batch_size = 32
epochs = 5

loader, num_classes = get_loaders(data_dir, batch_size=batch_size)

model = BaselineCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
